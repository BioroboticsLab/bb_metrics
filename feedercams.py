"""
Feeder/exit cam detection processing (step 1 companion).

Functions:
 - get_df_feedercam: clean/standardize per-detection DataFrame.
 - get_average_counts_daily: average counts per 30s chunk.
 - process_datedir: combine per-video parquet into one per-day file.
 - process_daily_files: compute average counts for daily files.

Plotting companions (step 3 "Feederplots"):
 - cam_hive_map: build {f"{which}{hive}": hive} from cfg.hives.
 - get_dfcounts_from_avgdir: glob + concat avgcount parquets into one DataFrame.
 - prep_dfcounts: add tz-aware timestamp, hive, date, hour columns.
 - get_feedercam_hourly_average: time-weighted hourly average counts per hive (+weather).
 - load_avgcounts: one-call load + prep of avgcounts for a cam type (convenience).
"""

import glob
import os
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bb_utils.ids import BeesbookID

from . import get_config


def _output_is_current(outfile: Path, input_files: Iterable) -> bool:
    """
    True iff `outfile` exists and no input is strictly newer (newest input
    mtime <= output mtime; equality counts as up to date). False if `outfile`
    is missing. If `input_files` is empty, True when `outfile` exists.

    Matches the package convention in trajectories.py / datafunctions.py /
    metricsfunctions.py. Limitation: an mtime-max check does NOT detect input
    DELETIONS (a removed source with nothing else touched leaves a stale output
    that this reports as current) -- same limitation as the other sites.
    """
    outfile = Path(outfile)
    if not outfile.is_file():
        return False
    out_ts = outfile.stat().st_mtime
    mtimes = [Path(f).stat().st_mtime for f in input_files]
    if not mtimes:
        return True
    return max(mtimes) <= out_ts


def _validate_recalc(recalc):
    """Accept True / False / 'ifnewer'; raise ValueError otherwise.

    Returns the value unchanged so callers use `is True` / `is False` /
    `== 'ifnewer'` (no bool/str cross-matching, and ints like 0/1 are rejected).
    """
    if recalc is True or recalc is False or recalc == "ifnewer":
        return recalc
    raise ValueError(f"recalc must be True, False, or 'ifnewer'; got {recalc!r}")


def get_df_feedercam(df: pd.DataFrame, timezone: str = "Europe/Berlin") -> pd.DataFrame:
    df_feedercam = df.copy()
    # add absolute timestamps
    df_feedercam["timestamp"] = df_feedercam["video_start_timestamp"] + pd.to_timedelta(
        df["timestamp"], unit="s"
    )
    for col in ["timestamp", "video_start_timestamp"]:
        df_feedercam[col] = df_feedercam[col].dt.tz_localize(timezone)

    # convert beeID for tagged bees
    sel = df_feedercam["detection_type"] == "TaggedBee"
    df_feedercam.loc[sel, "beeID"] = [
        BeesbookID.from_bb_binary((bitprobs > 0.5).astype(int)).as_ferwar()
        for bitprobs in df_feedercam.loc[sel, "beeID"]
    ]

    # rename columns
    df_feedercam = df_feedercam.rename(
        columns={
            "xpos": "x_pixels",
            "ypos": "y_pixels",
            "zrotation": "orientation",
            "camID": "cam_id",
            "confidence": "bee_id_confidence",
            "beeID": "bee_id",
        }
    )

    df_feedercam = df_feedercam[
        [
            "timestamp",
            "video_start_timestamp",
            "x_pixels",
            "y_pixels",
            "orientation",
            "detection_type",
            "cam_id",
            "bee_id",
            "bee_id_confidence",
            "localizerSaliency",
        ]
    ]

    for col in ["x_pixels", "y_pixels", "orientation", "bee_id"]:
        df_feedercam[col] = pd.to_numeric(df_feedercam[col])

    df_feedercam = df_feedercam.sort_values(by="timestamp").reset_index(drop=True)
    return df_feedercam


def get_average_counts_daily(
    df: pd.DataFrame,
    *,
    localizer_threshold: float = 0.1,
    bee_id_confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["cam_id", "video_start_timestamp", "totalcounts", "untaggedcounts", "taggedcounts"]
        )

    filtered = df[df["localizerSaliency"] > localizer_threshold].copy()
    low_conf_tagged = (filtered["detection_type"] == "TaggedBee") & (
        filtered["bee_id_confidence"] < bee_id_confidence_threshold
    )
    filtered = filtered[~low_conf_tagged]

    counts_per_frame = (
        filtered.groupby(["cam_id", "video_start_timestamp", "timestamp"])
        .agg(
            totalcounts=("detection_type", "size"),
            untaggedcounts=("detection_type", lambda x: (x == "UnmarkedBee").sum()),
            taggedcounts=("detection_type", lambda x: (x == "TaggedBee").sum()),
        )
        .reset_index()
    )

    avg_counts = (
        counts_per_frame.groupby(["cam_id", "video_start_timestamp"])
        .agg(
            totalcounts=("totalcounts", "mean"),
            untaggedcounts=("untaggedcounts", "mean"),
            taggedcounts=("taggedcounts", "mean"),
        )
        .reset_index()
    )
    return avg_counts


# Output columns produced by get_df_feedercam, in order. Used for the empty
# "no_data" parquet so the file exists with the canonical schema.
_FEEDERCAM_OUTPUT_COLUMNS = [
    "timestamp", "video_start_timestamp", "x_pixels", "y_pixels", "orientation",
    "detection_type", "cam_id", "bee_id", "bee_id_confidence", "localizerSaliency",
]

# Numeric output columns cast to float64 before writing so that int64 (TaggedBee
# bee_id from as_ferwar) vs float64 (NaN bee_id for UnmarkedBee-only files) never
# yields mismatched row-group schemas across per-video files.
_FLOAT_COLS = [
    "x_pixels", "y_pixels", "orientation", "bee_id", "bee_id_confidence", "localizerSaliency",
]


def _slim_to_table(df: pd.DataFrame, schema: Optional["pa.Schema"] = None) -> "pa.Table":
    """A get_df_feedercam'd frame -> pyarrow Table.

    Casts the numeric output columns to float64 so int-vs-float bee_id (and other
    numerics) cannot break ParquetWriter's single-schema requirement. When `schema`
    is given, the table is cast to it so every row group shares the schema locked
    from the first written frame.
    """
    df = df.copy()
    for col in _FLOAT_COLS:
        df[col] = df[col].astype("float64")
    table = pa.Table.from_pandas(df, preserve_index=False)
    if schema is not None:
        table = table.cast(schema)
    return table


def process_datedir(
    datedir: Path,
    whichpi: str,
    *,
    outputdir: Optional[Path] = None,
    recalc: Union[bool, str] = True,
    clahepostfix: str = "-c",
) -> Optional[str]:
    """
    Combine per-video parquet files for one date dir into a single per-day file.

    recalc:
        True       -> always recompute (default; legacy behavior).
        False      -> skip with "skip_existing" if the output already exists
                      (legacy behavior).
        'ifnewer'  -> recompute only if a globbed source file is newer than the
                      output; otherwise return "skip_up_to_date".

    Returns one of: "skip_existing", "skip_up_to_date", "no_files",
    "no_data", "ok".
    """
    recalc = _validate_recalc(recalc)

    cfg = get_config()
    if outputdir is None:
        outputdir = getattr(cfg, "feedercam_daily_dir", None)
    if outputdir is None:
        raise ValueError("outputdir not set (pass outputdir or set cfg.feedercam_daily_dir)")

    postfix = clahepostfix + ".parquet"
    outfile = Path(outputdir) / f"{Path(datedir).name}_{whichpi}{postfix}"

    # Legacy fast path: recalc=False skips an existing output without globbing.
    if recalc is False and outfile.is_file():
        return "skip_existing"

    files = glob.glob(str(Path(datedir) / f"{whichpi}*/*{postfix}"))
    if len(files) == 0:
        return "no_files"

    # 'ifnewer': skip only when the output exists and no input is newer.
    if recalc == "ifnewer" and _output_is_current(outfile, files):
        return "skip_up_to_date"

    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Stream each per-video file -> transform -> append as a parquet row group, so
    # only one per-video file is resident at a time (high-traffic exitcam days hold
    # the heavy raw beeID bit-prob arrays for all videos at once otherwise -> OOM).
    # Write to a temp file in the SAME dir and os.replace() into place only after a
    # clean close, so a mid-loop crash never leaves a partial output (whose mtime
    # would otherwise make recalc='ifnewer' skip it forever).
    tmpfile = outfile.with_name(outfile.name + ".__tmp__.parquet")
    writer = None
    schema = None
    try:
        for f in files:
            df = pd.read_parquet(f)
            if len(df) == 0:
                continue
            df = get_df_feedercam(df)
            # drop bad cam_id rows (non-string) -- same filter as legacy
            df = df[df["cam_id"].apply(lambda x: isinstance(x, str))]
            if len(df) == 0:
                continue
            if writer is None:
                table = _slim_to_table(df)            # lock schema from first frame
                schema = table.schema
                writer = pq.ParquetWriter(tmpfile, schema, compression="snappy")
            else:
                table = _slim_to_table(df, schema)    # cast every later frame to it
            writer.write_table(table)

        if writer is not None:
            writer.close()
            writer = None
            os.replace(tmpfile, outfile)              # atomic on same filesystem
            return "ok"
    except BaseException:
        # On any failure, close+delete the temp file and re-raise so the caller's
        # try/except can log it; never leave a partial output.
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            os.remove(tmpfile)
        except OSError:
            pass
        raise

    # No file produced any rows -> canonical empty-columns parquet, "no_data".
    pd.DataFrame(columns=_FEEDERCAM_OUTPUT_COLUMNS).to_parquet(outfile)
    return "no_data"


def _process_daily_file_worker(args):
    """
    Worker function for parallel processing of daily files.
    Must be at module level to be picklable by multiprocessing.
    """
    file, avg_dir, recalc, localizer_threshold, bee_id_confidence_threshold = args
    outfile = avg_dir / Path(file).name

    if recalc is False:
        try:
            if outfile.is_file():
                pd.read_parquet(outfile)            # validate readable
                return ("skip", str(file))
        except Exception:
            pass                                    # unreadable -> recompute
    elif recalc == "ifnewer":
        try:
            if _output_is_current(outfile, [file]):
                pd.read_parquet(outfile)            # validate readable (parity)
                return ("skip_up_to_date", str(file))
        except Exception:
            pass                                    # unreadable/unstattable -> recompute
    # recalc is True -> always recompute (neither branch taken)

    try:
        df = pd.read_parquet(file)
        avg = get_average_counts_daily(
            df,
            localizer_threshold=localizer_threshold,
            bee_id_confidence_threshold=bee_id_confidence_threshold,
        )
        avg.to_parquet(outfile)
        return ("ok", str(file))
    except Exception as e:
        return ("error", f"{file}: {e}")


def process_daily_files(
    daily_files: Iterable[Path],
    *,
    avg_dir: Optional[Path] = None,
    recalc: Union[bool, str] = True,
    clahepostfix: str = "-c",
    localizer_threshold: float = 0.1,
    bee_id_confidence_threshold: float = 0.01,
    processes: int = 2,
):
    """Compute average counts for a list of daily parquet files.

    recalc: True (always recompute), False (skip readable existing outputs,
    status "skip"), or 'ifnewer' (recompute only when the source file is newer
    than the output, status "skip_up_to_date").
    """
    from multiprocessing import Pool

    recalc = _validate_recalc(recalc)  # fail fast in the parent process

    cfg = get_config()
    if avg_dir is None:
        avg_dir = getattr(cfg, "feedercam_avg_dir", None)
    if avg_dir is None:
        raise ValueError("avg_dir not set (pass avg_dir or set cfg.feedercam_avg_dir)")
    avg_dir = Path(avg_dir)
    avg_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for worker function
    worker_args = [
        (file, avg_dir, recalc, localizer_threshold, bee_id_confidence_threshold)
        for file in daily_files
    ]

    with Pool(processes=processes) as pool:
        results = pool.map(_process_daily_file_worker, worker_args)
    return results


# ---------------------------------------------------------------------------
# avgcounts loading / hourly aggregation (step 3 "Feederplots" companion)
# ---------------------------------------------------------------------------

def cam_hive_map(which: str = "feedercam", cfg=None) -> dict:
    """Map feeder/exit cam_id strings to hive labels from the active config.

    Returns ``{f"{which}{hive}": hive for hive in cfg.hives}``, e.g.
    ``{"feedercamA": "A", ...}`` for a 4-hive Berlin config or
    ``{"feedercamA": "A"}`` for a single-hive Konstanz config.

    Assumes the in-row ``cam_id`` strings follow ``f"{which}{hive}"``. Where that
    does not hold (e.g. Konstanz "outdoorcam" files whose in-row cam_id may differ
    from the filename token), pass an explicit ``cam_to_hive`` override to
    ``prep_dfcounts`` / ``load_avgcounts`` instead.
    """
    cfg = cfg or get_config()
    hives = getattr(cfg, "hives", None)
    if hives is None:
        raise ValueError("cfg.hives not set (needed to build cam_hive_map)")
    return {f"{which}{h}": h for h in hives}


def get_dfcounts_from_avgdir(avgdir, globpattern: str) -> pd.DataFrame:
    """Load and concatenate avgcount parquet files matching a glob pattern.

    Reads every ``avgdir/globpattern`` parquet (sorted), concatenates them, and
    coerces ``video_start_timestamp`` / ``timestamp`` columns to datetime.
    Raises ``FileNotFoundError`` if nothing matches.
    """
    files = sorted(glob.glob(os.path.join(str(avgdir), globpattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {globpattern} in {avgdir}")
    dfcounts = pd.concat(pd.read_parquet(f) for f in files).reset_index(drop=True)
    for col in ["video_start_timestamp", "timestamp"]:
        if col in dfcounts.columns:
            dfcounts[col] = pd.to_datetime(dfcounts[col])
    return dfcounts


def prep_dfcounts(dfcounts: pd.DataFrame, cam_to_hive=None,
                  tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Add tz-aware timestamps and ``hive`` / ``date`` / ``hour`` columns to avgcounts.

    ``video_start_timestamp`` is localized to ``tz`` if naive, else converted.
    ``hive`` is ``cam_id`` mapped through ``cam_to_hive`` (unmapped cam_ids pass
    through unchanged via ``fillna``). ``date`` is the normalized (midnight)
    timestamp; ``hour`` is the integer hour of day.
    """
    cam_to_hive = cam_to_hive or {}
    df = dfcounts.copy()
    ts = df["video_start_timestamp"]
    df["video_start_timestamp"] = (
        ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    )
    df["hive"] = df["cam_id"].map(cam_to_hive).fillna(df["cam_id"])
    df["date"] = df["video_start_timestamp"].dt.normalize()
    df["hour"] = df["video_start_timestamp"].dt.hour
    return df


def get_feedercam_hourly_average(dfcounts: pd.DataFrame, hourdata: pd.DataFrame,
                                 video_duration_seconds: int = 30,
                                 hives=None) -> pd.DataFrame:
    """Time-weighted hourly average counts per hive.

    For each hour in ``hourdata.index``, each video contributes
    ``(avg_count * seconds_within_hour) / 3600`` summed per hive (videos clipped
    at hour boundaries). ``hourdata.index`` defines both the hour grid and the
    timezone. Weather columns present on ``hourdata`` (temp/prcp/tsun/rhum/wspd)
    are merged in by hour. Adds ``date`` and ``hour_integer``.

    Returns a long DataFrame with columns ``[hive, hour, totalcounts,
    taggedcounts, untaggedcounts, <weather...>, date, hour_integer]``.

    Note: uses ``groupby(...).apply(..., include_groups=False)`` (pandas >= 2.2).
    """
    df = dfcounts.copy()
    tz = hourdata.index.tz
    df["timestamp_start"] = pd.to_datetime(df["video_start_timestamp"]).dt.tz_convert(tz)
    df["timestamp_end"] = df["timestamp_start"] + pd.to_timedelta(video_duration_seconds, unit="s")

    if "hive" not in df.columns:
        df["hive"] = df["cam_id"].str.extract(r"([A-D])", expand=False)

    hours = hourdata.index
    HIVE_LIST = hives or sorted(df["hive"].dropna().unique().tolist())
    num_cols = ["totalcounts", "taggedcounts", "untaggedcounts"]
    rows = []

    for hour_start in hours:
        hour_end = hour_start + pd.Timedelta(hours=1)
        mask = (df["timestamp_start"] < hour_end) & (df["timestamp_end"] > hour_start)
        sub = df.loc[mask].copy()
        if sub.empty:
            rows.extend({"hive": h, "hour": hour_start, **{c: 0.0 for c in num_cols}} for h in HIVE_LIST)
            continue

        sub["clip_start"] = sub["timestamp_start"].clip(lower=hour_start)
        sub["clip_end"] = sub["timestamp_end"].clip(upper=hour_end)
        sub["seg_sec"] = (sub["clip_end"] - sub["clip_start"]).dt.total_seconds()

        def weighted_for_group(g):
            return {col: (g[col] * g["seg_sec"]).sum() / 3600.0 for col in num_cols}

        weighted = sub.groupby("hive").apply(weighted_for_group, include_groups=False)
        for h in HIVE_LIST:
            vals = weighted.get(h, {col: 0.0 for col in num_cols})
            rows.append({"hive": h, "hour": hour_start, **vals})

    df_hour = pd.DataFrame(rows)

    weather_cols = [c for c in ("temp", "prcp", "tsun", "rhum", "wspd") if c in hourdata.columns]
    if weather_cols:
        weather = hourdata.reset_index()
        weather.columns = ["hour"] + list(hourdata.columns)
        df_hour = df_hour.merge(weather[["hour", *weather_cols]], on="hour", how="left")

    df_hour["date"] = df_hour["hour"].dt.normalize()
    df_hour["hour_integer"] = df_hour["hour"].dt.hour
    return df_hour


def load_avgcounts(which: str = "feedercam", clahe: bool = True, cfg=None,
                   globpattern=None, cam_to_hive=None,
                   tz: str = "Europe/Berlin") -> pd.DataFrame:
    """One-call load of analysis-ready avgcounts for a cam type.

    Globs ``cfg.feedercam_avg_dir`` for ``f"{cfg.year}*{which}{postfix}.parquet"``
    (``postfix`` = ``-c`` if ``clahe`` else ``-nc``) unless ``globpattern`` is
    given, then concatenates, preps (tz/hive/date/hour), and maps cam_id -> hive.

    ``cam_to_hive`` defaults to ``cam_hive_map(which, cfg)``; pass an explicit dict
    where in-row ``cam_id`` strings do not match ``f"{which}{hive}"`` (e.g. the
    Konstanz "outdoorcam" files).
    """
    cfg = cfg or get_config()
    avgdir = getattr(cfg, "feedercam_avg_dir", None)
    if avgdir is None:
        raise ValueError("cfg.feedercam_avg_dir not set")
    if globpattern is None:
        postfix = "-c" if clahe else "-nc"
        globpattern = f"{getattr(cfg, 'year', '')}*{which}{postfix}.parquet"
    if cam_to_hive is None:
        cam_to_hive = cam_hive_map(which, cfg)
    dfcounts = get_dfcounts_from_avgdir(avgdir, globpattern)
    return prep_dfcounts(dfcounts, cam_to_hive=cam_to_hive, tz=tz)
