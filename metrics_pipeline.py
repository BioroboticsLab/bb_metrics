"""
Helpers for step 2 (calculate metrics): pairing trajectory files, running metrics,
and computing feeder/exit visit tables.
"""

import glob
import multiprocessing
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import get_config
from . import datafunctions as dfunc
from . import metricsfunctions as mfunc
from .uid import assign_uid, build_reuse_intervals


def build_pairs_from_traj(all_datafiles: Iterable[Path], cam_hive_map: dict) -> Tuple[list, list]:
    """
    Group trajectory parquet files into pairs per hive/start/end based on filename.
    Returns (pairs, unmatched).
    Each pair entry: (hive, startdt, enddt, [file_cam_left, file_cam_right])
    """
    from bb_binary.parsing import parse_video_fname

    buckets = defaultdict(list)
    for fp in all_datafiles:
        base = Path(fp).name
        cam, startdt, enddt = parse_video_fname(base)
        hive = cam_hive_map[cam]
        buckets[(hive, startdt, enddt)].append((cam, fp))

    pairs, unmatched = [], []
    for (hive, startdt, enddt), items in buckets.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        if len(items_sorted) >= 2:
            pairs.append((hive, startdt, enddt, [items_sorted[0][1], items_sorted[1][1]]))
            if len(items_sorted) > 2:
                unmatched.append((hive, startdt, enddt, [p for _, p in items_sorted[2:]]))
        else:
            unmatched.append((hive, startdt, enddt, [p for _, p in items_sorted]))
    return pairs, unmatched


def run_metrics_from_pairs(
    pairs: List[tuple],
    *,
    reprocess: bool = False,
    update: bool = True,
    time_division: str = "1min",
    min_num_detections: Optional[int] = None,
    save_xy_hist: bool = True,
    metrics_dir: Optional[Path] = None,
    num_processes: int = 6,
    grid_lookup=None,
    comb_label_order=None,
):
    """
    Dispatch metric computation across pairs using mfunc.datafile_to_metrics.
    """
    cfg = get_config()
    if metrics_dir is None:
        metrics_dir = getattr(cfg, "metrics_dir", None)
    if metrics_dir is None:
        raise ValueError("metrics_dir not set (pass metrics_dir or set cfg.metrics_dir)")

    # set defaults for min_num_detections
    if min_num_detections is None:
        if time_division == "60min":
            min_num_detections = 90
        elif time_division == "5min":
            min_num_detections = 30
        elif time_division == "1min":
            min_num_detections = 12
        else:
            min_num_detections = 12

    args = [
        (
            hive,
            start,
            end,
            datafiles,
            reprocess,
            time_division,
            min_num_detections,
            save_xy_hist,
            str(metrics_dir),
            update,
            grid_lookup,
            comb_label_order,
        )
        for hive, start, end, datafiles in pairs
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(mfunc.datafile_to_metrics, args)


def define_feeder_visits(df: pd.DataFrame, visit_gap_seconds: int = 15, confidence_threshold: float = 0.01) -> pd.DataFrame:
    """
    Build visit table from feeder/exit detections.
    """
    df_tagged = df[df["detection_type"] == "TaggedBee"].copy()
    df_tagged = df_tagged[df_tagged["bee_id_confidence"] > confidence_threshold]
    if len(df_tagged) == 0:
        return pd.DataFrame()

    df_tagged = df_tagged.sort_values(by=["bee_id", "timestamp"])
    df_tagged["time_diff"] = df_tagged.groupby("bee_id")["timestamp"].diff().fillna(pd.Timedelta(seconds=0))
    df_tagged["time_diff"] = df_tagged["time_diff"].dt.total_seconds()
    df_tagged["new_visit"] = (df_tagged["time_diff"] > visit_gap_seconds).astype(int)
    df_tagged["visit_id"] = df_tagged.groupby("bee_id")["new_visit"].cumsum()

    visit_df = (
        df_tagged.groupby(["bee_id", "visit_id"], as_index=False)
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            detection_count=("timestamp", "count"),
            cam_id=("cam_id", lambda x: x.mode().iloc[0]),
        )
    )
    visit_df["duration_seconds"] = (visit_df["end_time"] - visit_df["start_time"]).dt.total_seconds()
    return visit_df


def pair_by_date(datadir: Path, pattern_c: str, pattern_nc: str) -> list:
    files_c = sorted(datadir.glob(pattern_c))
    files_nc = sorted(datadir.glob(pattern_nc))
    c_by_date = {p.stem.split("_", 1)[0]: p for p in files_c}
    nc_by_date = {p.stem.split("_", 1)[0]: p for p in files_nc}
    return [[c_by_date[d], nc_by_date[d]] for d in sorted(c_by_date.keys() & nc_by_date.keys())]


def process_visit_pairs(
    file_pairs: Iterable[list],
    *,
    visit_gap_seconds: int = 15,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    visits = []
    for filec, filenc in file_pairs:
        df_combined = pd.concat((pd.read_parquet(filec), pd.read_parquet(filenc)))
        df_combined = df_combined.drop_duplicates(subset=["timestamp", "cam_id", "bee_id"], keep="first")
        visit_df = define_feeder_visits(df_combined, visit_gap_seconds=visit_gap_seconds, confidence_threshold=confidence_threshold)
        visits.append(visit_df)
    if not visits:
        return pd.DataFrame()
    return pd.concat(visits).reset_index(drop=True)


def load_visit_tables(metrics_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_dir = Path(metrics_dir)
    df_feedervisits = pd.read_parquet(metrics_dir / "df_feedervisits.parquet")
    df_exitvisits = pd.read_parquet(metrics_dir / "df_exitvisits.parquet")
    return df_feedervisits, df_exitvisits


def load_metrics_files(metrics_dir: Path, pattern: str = "metrics-60min*") -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    files = sorted(metrics_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files)).reset_index(drop=True)


def build_day_data_matrix(
    df_metrics: pd.DataFrame,
    day_to_number: dict,
    *,
    dftags: Optional[pd.DataFrame] = None,
    tz: str = "Europe/Berlin",
    daysine_peaktime: Optional[float] = None,
    sumqs: Optional[List[str]] = None,
    weightedmeanqs: Optional[List[str]] = None,
) -> pd.DataFrame:
    if df_metrics.empty:
        return pd.DataFrame()

    if sumqs is None:
        sumqs = ["num_detections", "num_trips", "inplace_events", "burst_events", "large_turn_events"]

    df_metrics = df_metrics.copy()
    df_metrics["timestamp_start_cest"] = df_metrics["timestamp_start"].dt.tz_convert(tz)
    df_metrics["timestamp_end_cest"] = df_metrics["timestamp_end"].dt.tz_convert(tz)
    df_metrics["day"] = df_metrics["timestamp_start_cest"].dt.date
    df_metrics["daynum"] = df_metrics["day"].map(day_to_number)

    # Assign uid if tag introduction data is available
    has_uid = dftags is not None and not dftags.empty
    if has_uid:
        df_metrics["uid"] = assign_uid(df_metrics, dftags).values

    def weighted_mean(values, weights):
        total_weight = np.nansum(weights)
        if total_weight == 0:
            return np.nan
        return np.nansum(values * weights) / total_weight

    df_metrics["Hour"] = df_metrics["timestamp_start_cest"].dt.hour + df_metrics["timestamp_start_cest"].dt.minute / 60.0
    if daysine_peaktime is None:
        df_metrics["daysinevals"] = np.sin(2 * np.pi * df_metrics["Hour"] / 24)
    else:
        # Peak of sine wave at daysine_peaktime in local hours.
        middle_offset = daysine_peaktime - 6.0
        df_metrics["daysinevals"] = np.sin((df_metrics["Hour"] - middle_offset) / 24 * (2 * np.pi))

    if weightedmeanqs is None:
        numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
        exclude = set(sumqs + ["bee_id", "uid", "daynum", "Hour", "daysinevals"])
        weightedmeanqs = [col for col in numeric_cols if col not in exclude]

    group_keys = ["hive", "bee_id", "uid", "daynum", "day"] if has_uid else ["hive", "bee_id", "daynum", "day"]
    grouped = df_metrics.groupby(group_keys)

    results = []
    for name, group in grouped:
        if has_uid:
            hive, bee_id, uid, day_num, day = name
            result = {"hive": hive, "bee_id": bee_id, "uid": uid, "daynum": day_num, "day": day}
        else:
            hive, bee_id, day_num, day = name
            result = {"hive": hive, "bee_id": bee_id, "daynum": day_num, "day": day}

        for q in sumqs:
            result[q] = group[q].sum()

        for q in weightedmeanqs:
            result[q] = weighted_mean(group[q], group["num_detections"])

        speed_median = group["speed_median"]
        daysinevals = group["daysinevals"]
        weights = group["num_detections"]
        numerator = np.nansum(speed_median * daysinevals * weights)
        denominator = np.nansum(speed_median * weights)
        result["speed_circadian_coeff"] = numerator / denominator if denominator != 0 else np.nan

        results.append(result)

    out_cols = group_keys + sumqs + weightedmeanqs + ["speed_circadian_coeff"]
    return pd.DataFrame(results, columns=out_cols)


def _build_daily_visit_summary(
    df_visits: pd.DataFrame,
    day_to_number: dict,
    visit_label: str,
    dftags: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df_visits.empty:
        return pd.DataFrame(columns=["hive", "bee_id", "daynum", "day"])

    df_visits = df_visits.copy()
    df_visits["day"] = df_visits["start_time"].dt.date
    df_visits["daynum"] = df_visits["day"].map(day_to_number)
    df_visits = df_visits[df_visits["daynum"] >= 0]
    df_visits["hive"] = df_visits["cam_id"].apply(lambda x: x[-1])

    has_uid = dftags is not None and not dftags.empty
    if has_uid:
        df_visits["uid"] = assign_uid(df_visits, dftags).values

    group_keys = ["hive", "bee_id", "uid", "daynum", "day"] if has_uid else ["hive", "bee_id", "daynum", "day"]

    return (
        df_visits.groupby(group_keys, as_index=False)
        .agg(
            **{
                f"num_{visit_label}_visits": ("visit_id", "count"),
                f"{visit_label}_visit_duration_mean": ("duration_seconds", "mean"),
            }
        )
    )


def merge_daily_visit_metrics(
    dfday: pd.DataFrame,
    df_feedervisits: pd.DataFrame,
    df_exitvisits: pd.DataFrame,
    day_to_number: dict,
    dftags: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df_feeder_daily = _build_daily_visit_summary(df_feedervisits, day_to_number, "feeder", dftags=dftags)
    df_exit_daily = _build_daily_visit_summary(df_exitvisits, day_to_number, "exit", dftags=dftags)

    has_uid = "uid" in dfday.columns and "uid" in df_feeder_daily.columns
    merge_keys = ["hive", "bee_id", "uid", "daynum", "day"] if has_uid else ["hive", "bee_id", "daynum", "day"]

    dfday = pd.merge(dfday, df_feeder_daily, on=merge_keys, how="left")
    dfday["num_feeder_visits"] = dfday["num_feeder_visits"].fillna(0)
    dfday["feeder_visit_duration_mean"] = dfday["feeder_visit_duration_mean"].fillna(0)

    has_uid_exit = "uid" in dfday.columns and "uid" in df_exit_daily.columns
    merge_keys_exit = ["hive", "bee_id", "uid", "daynum", "day"] if has_uid_exit else ["hive", "bee_id", "daynum", "day"]

    dfday = pd.merge(dfday, df_exit_daily, on=merge_keys_exit, how="left")
    dfday["num_exit_visits"] = dfday["num_exit_visits"].fillna(0)
    dfday["exit_visit_duration_mean"] = dfday["exit_visit_duration_mean"].fillna(0)

    return dfday


def save_day_data_matrix(dfday: pd.DataFrame, metrics_dir: Path, filename: str = "daydatamat.csv") -> Path:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / filename
    dfday.to_csv(output_path, index=False)
    return output_path


def build_day_xyhist(
    xyhist_dir: Path,
    *,
    output_hdf5_file: Optional[Path] = None,
    tz: str = "Europe/Berlin",
) -> Optional[Path]:
    import h5py
    import pytz

    xyhist_dir = Path(xyhist_dir)
    if output_hdf5_file is None:
        output_hdf5_file = xyhist_dir / "dayxyhist.h5"

    xyhist_files = sorted(xyhist_dir.glob("xyhist-*.h5"))
    if not xyhist_files:
        print(f"No xyhist files found under {xyhist_dir}")
        return None

    day_to_files = defaultdict(set)
    target_tz = pytz.timezone(tz)

    for filepath in xyhist_files:
        filename = filepath.name
        try:
            start_timestamp, end_timestamp = dfunc.parse_data_file_timestamps(filename)
            start_local = start_timestamp.astimezone(target_tz)
            end_local = end_timestamp.astimezone(target_tz)
            current_day = start_local.date()
            end_day = end_local.date()
            while current_day <= end_day:
                day_to_files[current_day].add(filepath)
                current_day += timedelta(days=1)
        except ValueError as exc:
            print(exc)
            continue

    with h5py.File(output_hdf5_file, "w") as output_hdf:
        for day in sorted(day_to_files.keys()):
            print(f"Processing day {day}")
            per_day_xyhist = {}
            files_for_day = day_to_files[day]
            for filepath in files_for_day:
                filename = filepath.name
                try:
                    hive_name = dfunc.parse_hive_name(filename)
                    with h5py.File(filepath, "r") as hdf_file:
                        for bee_id in hdf_file.keys():
                            bee_group = hdf_file[bee_id]
                            for day_str in bee_group.keys():
                                day_group = bee_group[day_str]
                                for hour_str in day_group.keys():
                                    hour_group = day_group[hour_str]
                                    xyhist = hour_group["xyhist"][:]
                                    timestamp_start_str = hour_group.attrs["timestamp_start"]
                                    timestamp_start = pd.to_datetime(timestamp_start_str)
                                    if timestamp_start.tzinfo is None:
                                        timestamp_start = timestamp_start.tz_localize("UTC")
                                    timestamp_start_local = timestamp_start.tz_convert(target_tz)
                                    data_day = timestamp_start_local.date()
                                    if data_day != day:
                                        continue
                                    key = (hive_name, bee_id)
                                    if key not in per_day_xyhist:
                                        per_day_xyhist[key] = xyhist.copy()
                                    else:
                                        per_day_xyhist[key] += xyhist
                except Exception as exc:
                    print(f"Error processing file {filepath}: {exc}")
                    continue

            for (hive_name, bee_id), xyhist in per_day_xyhist.items():
                hive_group = output_hdf.require_group(hive_name)
                bee_group = hive_group.require_group(str(bee_id))
                day_str = day.strftime("%Y%m%d")
                dataset_name = f"day_{day_str}"
                if dataset_name in bee_group:
                    del bee_group[dataset_name]
                bee_group.create_dataset(dataset_name, data=xyhist, compression="gzip", compression_opts=9)

    return output_hdf5_file


class LifetimeEstimator:
    def __init__(
        self,
        mu_days_alive: int = 21,
        sigma_days_alive: int = 25,
        min_detections: int = 1000,
        max_detections: int = 3000,
        dead_rate_beta: float = 25,
    ):
        self.mu_days_alive = mu_days_alive
        self.sigma_days_alive = sigma_days_alive
        self.min_detections = min_detections
        self.max_detections = max_detections
        self.dead_rate_beta = dead_rate_beta

    def fit(
        self,
        num_detect,
        *,
        switchpoint_emerged: int = 0,
        num_tune: int = 2000,
        num_draws: int = 5000,
        target_accept: float = 0.95,
        progress: bool = False,
        cores: int = 12,
        chains: int = 12,
    ):
        import pymc as pm
        import scipy.stats

        num_detections = num_detect.copy()
        num_detections -= self.min_detections
        num_detections = np.clip(num_detections, 0, self.max_detections)
        num_detections = num_detections / self.max_detections

        observed_data = (num_detections > 0.5).astype(int)
        days = np.arange(num_detections.shape[0]).astype(np.float64)

        model = pm.Model()
        with model:
            p_days_alive = scipy.stats.norm.pdf(np.arange(len(days)), self.mu_days_alive, self.sigma_days_alive)
            p_days_alive /= p_days_alive.sum()

            days_alive = pm.Categorical("days_alive", p=p_days_alive)
            switchpoint_died = pm.Deterministic("switchpoint_died", switchpoint_emerged + days_alive)

            threshold = pm.Beta("dead_rate", alpha=1, beta=self.dead_rate_beta)
            probability_higher = pm.Beta("probability_higher", alpha=5, beta=1)
            probability_lower = pm.Beta("probability_lower", alpha=1, beta=5)

            rate = pm.math.switch(
                (days >= switchpoint_emerged) & (days <= switchpoint_died),
                probability_higher,
                probability_lower,
            )

            pm.Bernoulli("detections", rate, observed=observed_data)

            trace = pm.sample(
                tune=num_tune,
                draws=num_draws,
                progressbar=progress,
                target_accept=target_accept,
                cores=cores,
                chains=chains,
            )

        return model, trace, num_detections


def estimate_death_days(
    dfday: pd.DataFrame,
    *,
    hives: Optional[Iterable[str]] = None,
    estimator: Optional[LifetimeEstimator] = None,
    extra_days_after: int = 25,
    progress: bool = False,
) -> pd.DataFrame:
    if estimator is None:
        estimator = LifetimeEstimator()
    if hives is None:
        cfg = get_config()
        hives = cfg.hives

    has_uid = "uid" in dfday.columns

    results = []
    for hive in hives:
        print(hive)
        dfsel = dfday[dfday["hive"] == hive]

        if has_uid:
            id_col = "uid"
        else:
            id_col = "bee_id"

        ids = dfsel[id_col].unique()
        for i, current_id in enumerate(ids):
            if i % 100 == 0:
                print(i, len(ids))

            bee_data = dfsel[dfsel[id_col] == current_id].sort_values(by="daynum").copy()
            if bee_data.empty:
                continue
            hive_mode = bee_data["hive"].mode().values[0]
            bee_data = bee_data[bee_data["hive"] == hive_mode]

            full_days = np.arange(bee_data["daynum"].min(), bee_data["daynum"].max() + 1)
            bee_data = bee_data.set_index("daynum").reindex(full_days, fill_value=0).reset_index()

            num_detect = bee_data["num_detections"].to_numpy()
            num_detect = np.concatenate((num_detect, np.zeros(extra_days_after)))
            if len(num_detect) == 0:
                continue

            model, trace, num_detections = estimator.fit(num_detect, progress=progress)

            death_day = trace.posterior["switchpoint_died"].mean().item()
            death_day_index = int(np.round(death_day))
            if death_day_index >= len(bee_data):
                estimated_death_daynum = bee_data.loc[int(np.round(death_day)) - 1, "daynum"] + 1
            else:
                estimated_death_daynum = bee_data.loc[death_day_index, "daynum"]

            row = {
                "bee_id": bee_data["bee_id"].mode().values[0] if has_uid else current_id,
                "hive": hive,
                "estimated_death_daynum": estimated_death_daynum,
            }
            if has_uid:
                row["uid"] = current_id
            results.append(row)

    return pd.DataFrame(results)

def create_birth_df(df_tags):
    birth_records = []

    for _, row in df_tags.iterrows():
        hive = row['Hive']
        birthdate = row['Date'].date()  # Convert to date
        tag_ranges = []

        # Process first tag range
        if not pd.isnull(row['tag_start']) and not pd.isnull(row['tag_end']):
            tag_start = int(row['tag_start'])
            tag_end = int(row['tag_end'])
            tag_ranges.append(range(tag_start, tag_end + 1))  # Inclusive range

        # Process second tag range if present
        if not pd.isnull(row['tag_start2']) and not pd.isnull(row['tag_end2']):
            tag_start2 = int(row['tag_start2'])
            tag_end2 = int(row['tag_end2'])
            tag_ranges.append(range(tag_start2, tag_end2 + 1))  # Inclusive range

        # Combine all bee_ids
        for tag_range in tag_ranges:
            for bee_id in tag_range:
                birth_records.append({
                    'bee_id': float(bee_id),
                    'birthdate': birthdate,
                    'hive': hive
                })

    df_birth = pd.DataFrame(birth_records)

    # Assign uid to handle reused tags
    if not df_birth.empty:
        intervals = build_reuse_intervals(df_tags)
        if not intervals.empty:
            # Match each birth record to its generation via (hive, bee_id, intro_date)
            intervals_match = intervals.copy()
            intervals_match["birthdate"] = intervals_match["intro_date"].dt.date
            df_birth = df_birth.merge(
                intervals_match[["hive", "bee_id", "birthdate", "uid"]],
                on=["hive", "bee_id", "birthdate"],
                how="left",
            )
            # Fill any unmatched with bee_id
            df_birth["uid"] = df_birth["uid"].fillna(df_birth["bee_id"]).astype(int)
        else:
            df_birth["uid"] = df_birth["bee_id"].astype(int)

    return df_birth

def create_death_df(df_beedeath):
    # Drop rows where 'estimated_death_daynum' is NaN
    df_beedeath_clean = df_beedeath.dropna(subset=['estimated_death_daynum']).copy()

    # Convert 'estimated_death_daynum' to integer
    df_beedeath_clean['estimated_death_daynum'] = df_beedeath_clean['estimated_death_daynum'].astype(int)

    # Map day numbers to dates using bd.number_to_day
    cfg = get_config()
    df_beedeath_clean['deathdate'] = df_beedeath_clean['estimated_death_daynum'].apply(lambda x: cfg.number_to_day.get(x))

    # Keep only relevant columns
    cols = ['hive', 'bee_id', 'deathdate']
    if 'uid' in df_beedeath_clean.columns:
        cols = ['hive', 'bee_id', 'uid', 'deathdate']
    df_death = df_beedeath_clean[cols]

    return df_death