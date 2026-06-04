"""
Feeder/exit cam detection processing (step 1 companion).

Functions:
 - get_df_feedercam: clean/standardize per-detection DataFrame.
 - get_average_counts_daily: average counts per 30s chunk.
 - process_datedir: combine per-video parquet into one per-day file.
 - process_daily_files: compute average counts for daily files.
"""

import glob
import os
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
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

    df = pd.concat([pd.read_parquet(f) for f in files])
    if len(df)==0:
        # uncommon but possible case - video files but no detections for a day
        # create and save an empty dateframe
        pd.DataFrame(columns=["timestamp",
            "video_start_timestamp",
            "x_pixels",
            "y_pixels",
            "orientation",
            "detection_type",
            "cam_id",
            "bee_id",
            "bee_id_confidence",
            "localizerSaliency"]).to_parquet(outfile)
        return 'no_data'

    df = get_df_feedercam(df)

    # drop bad cam_id rows (non-string)
    df = df[df["cam_id"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outfile)
    return "ok"


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
