"""Loaders for the binned per-bee metrics (1 min, 5 min, 60 min).

Analyses read binned metrics only through ``load_metrics``, never by globbing files. The files are
an implementation detail: today they are the bb_metrics per-hour parquets
``metrics-<period>-Hive_<H>_<start>--<end>.parquet`` in ``cfg.metrics_dir``; later the same table
may come from a Mosaic run. The returned columns (see METRICS.md) are the contract, so switching the
source changes this function and nothing in the analyses.

    from bb_metrics import tables
    m60 = tables.load_metrics("60min", cfg)
    m1 = tables.load_metrics("1min", cfg, start="2025-10-01", end="2025-10-04",
                             columns=["num_detections", "speed_median"])

Rows are keyed by (hive, bee_id, timestamp_start); ``uid`` is added when a tag table is found.
"""
import re
from pathlib import Path

import pandas as pd

KEYS = ["hive", "bee_id", "timestamp_start", "timestamp_end"]
_FNAME = re.compile(r"metrics-(?P<period>[^-]+)-Hive_(?P<hive>[^_]+)_"
                    r"(?P<start>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)--"
                    r"(?P<end>\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}Z)\.parquet")


def _ts(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _file_time(s):
    return pd.Timestamp(s[:10] + " " + s[11:19].replace("_", ":"), tz="UTC")


def metric_files(period, cfg=None, start=None, end=None, hives=None, metrics_dir=None):
    """The per-hour metric files of `period` that overlap [start, end), as a DataFrame
    (path, hive, file_start, file_end), sorted by hive and time."""
    if metrics_dir is None:
        if cfg is None:
            from . import get_config
            cfg = get_config()
        metrics_dir = cfg.metrics_dir
    start, end = _ts(start), _ts(end)
    rows = []
    for p in Path(metrics_dir).glob(f"metrics-{period}-*.parquet"):
        m = _FNAME.fullmatch(p.name)
        if not m or m["period"] != period:
            continue
        fs, fe = _file_time(m["start"]), _file_time(m["end"])
        if (start is not None and fe <= start) or (end is not None and fs >= end):
            continue
        if hives is not None and m["hive"] not in set(hives):
            continue
        rows.append((p, m["hive"], fs, fe))
    return (pd.DataFrame(rows, columns=["path", "hive", "file_start", "file_end"])
            .sort_values(["hive", "file_start"]).reset_index(drop=True))


def _find_dftags(cfg):
    from . import saved_output as so
    for folder in ("tags", "."):
        try:
            p = so.find("dftags", folder, cfg, verbose=False, missing_ok=True)
        except Exception:
            p = None
        if p is not None:
            return p
    return None


def load_metrics(period, cfg=None, start=None, end=None, hives=None, columns=None,
                 add_uid=True, use_combined="auto", metrics_dir=None, verbose=True):
    """Binned per-bee metrics of `period` ('1min', '5min', '60min') as one DataFrame.

    start/end    restrict to bins with timestamp_start in [start, end) (UTC if naive).
    hives        restrict to these hives.
    columns      metric columns to read; the key columns are always included.
    add_uid      add `uid` from the tag table (saved_output/tags/dftags.csv, else the flat
                 saved_output/dftags.csv); skipped with a message if there is none.
    use_combined 'auto': read metrics_dir/allmetrics-<period>.parquet if it is newer than every
                 per-hour file, else the per-hour files. True/False forces it.
    """
    if metrics_dir is None:
        if cfg is None:
            from . import get_config
            cfg = get_config()
        metrics_dir = cfg.metrics_dir
    metrics_dir = Path(metrics_dir)
    files = metric_files(period, cfg, start, end, hives, metrics_dir=metrics_dir)
    cols = None if columns is None else list(dict.fromkeys(KEYS + list(columns)))

    combined = metrics_dir / f"allmetrics-{period}.parquet"
    use = False
    if use_combined is True:
        use = combined.exists()
    elif use_combined == "auto" and combined.exists():
        all_files = metric_files(period, cfg, metrics_dir=metrics_dir)
        newest = max((p.stat().st_mtime for p in all_files["path"]), default=0)
        use = combined.stat().st_mtime >= newest

    if use:
        df = pd.read_parquet(combined, columns=cols)
        src = combined.name
    elif len(files):
        df = pd.concat((pd.read_parquet(p, columns=cols) for p in files["path"]), ignore_index=True)
        src = f"{len(files)} {period} files"
    else:
        if verbose:
            print(f"no metrics-{period} files in {metrics_dir} for the requested range")
        return pd.DataFrame(columns=cols or KEYS)

    if start is not None:
        df = df[df["timestamp_start"] >= _ts(start)]
    if end is not None:
        df = df[df["timestamp_start"] < _ts(end)]
    if hives is not None:
        df = df[df["hive"].isin(list(hives))]
    df = df.reset_index(drop=True)

    if add_uid and cfg is not None and len(df):
        p = _find_dftags(cfg)
        if p is None:
            if verbose:
                print("no dftags found: uid not added")
        else:
            from .uid import assign_uid
            dftags = pd.read_csv(p)
            dftags = dftags[~dftags["tag_start"].isna()] if "tag_start" in dftags else dftags
            df["uid"] = assign_uid(df, dftags, date_col="timestamp_start").values
    if verbose:
        print(f"metrics {period}: {len(df):,} rows from {src}")
    return df
