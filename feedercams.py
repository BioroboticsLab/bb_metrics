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
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from bb_utils.ids import BeesbookID

from . import get_config


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
    recalc: bool = True,
    clahepostfix: str = "-c",
) -> Optional[str]:
    cfg = get_config()
    if outputdir is None:
        outputdir = getattr(cfg, "feedercam_daily_dir", None)
    if outputdir is None:
        raise ValueError("outputdir not set (pass outputdir or set cfg.feedercam_daily_dir)")

    postfix = clahepostfix + ".parquet"
    outfile = Path(outputdir) / f"{Path(datedir).name}_{whichpi}{postfix}"

    if outfile.is_file() and not recalc:
        return "skip_existing"

    files = glob.glob(str(Path(datedir) / f"{whichpi}*/*{postfix}"))
    if len(files) == 0:
        return "no_files"

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

    try:
        if outfile.is_file() and not recalc:
            pd.read_parquet(outfile)
            return ("skip", str(file))
    except Exception:
        pass

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
    recalc: bool = True,
    clahepostfix: str = "-c",
    localizer_threshold: float = 0.1,
    bee_id_confidence_threshold: float = 0.01,
    processes: int = 2,
):
    """Compute average counts for a list of daily parquet files."""
    from multiprocessing import Pool

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
