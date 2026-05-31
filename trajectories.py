"""
Trajectory processing helpers (step 1).

Key functions:
 - filter_speed_jumps: drop implausible within-camera jumps per bee.
 - get_valid_tags: return allowed tag ids for a hive/date (or None if no filtering).
 - process_file: load a dill track file, filter, and save parquet.
"""

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from bb_binary.parsing import parse_video_fname

from . import get_config
from . import datafunctions as dfunc


def filter_speed_jumps(df: pd.DataFrame, min_num_obs: int, max_speed: float = 3.0) -> pd.DataFrame:
    """Remove per-bee segments with speed jumps > max_speed (same camera)."""
    bee_groups = df.groupby("bee_id")
    filtered_bees = []
    for _, dfbee in bee_groups:
        # Reset index to avoid index alignment warnings
        dfbee = dfbee.reset_index(drop=True)
        numobs = len(dfbee)
        if numobs < min_num_obs:
            continue

        numerrorframes = 1
        while numobs >= min_num_obs and numerrorframes > 0:
            dtimes = dfbee["timestamp"].diff().dt.total_seconds()
            speed = np.sqrt(dfbee["x_hive"].diff() ** 2 + dfbee["y_hive"].diff() ** 2) / dtimes
            samecamera = dfbee["cam_id"].diff().fillna(0) == 0
            error_frames = (speed > max_speed) & samecamera
            # Reset index after filtering to keep indices aligned
            dfbee = dfbee[~error_frames].reset_index(drop=True)
            numerrorframes = error_frames.sum()
            numobs = len(dfbee)

        if not dfbee.empty:
            filtered_bees.append(dfbee)

    if not filtered_bees:
        return df.iloc[0:0]
    return pd.concat(filtered_bees, ignore_index=True)


def get_valid_tags(dftags: Optional[pd.DataFrame], camdate, hive: str) -> Optional[list]:
    """
    Return sorted list of valid tag numbers for a given hive and date.
    If dftags is None or empty, return None (means no filtering).
    """
    if dftags is None or len(dftags) == 0:
        return None

    camdate = pd.to_datetime(camdate)

    if not pd.api.types.is_datetime64_any_dtype(dftags["Date"]):
        dftags = dftags.copy()
        dftags["Date"] = pd.to_datetime(dftags["Date"], errors="coerce")

    valid_rows = dftags[
        (dftags["Hive"].astype(str).str.upper() == str(hive).upper())
        & (dftags["Date"] <= camdate)
    ]

    tags = []
    for _, row in valid_rows.iterrows():
        for a, b in (("tag_start", "tag_end"), ("tag_start2", "tag_end2")):
            if pd.notna(row.get(a)) and pd.notna(row.get(b)):
                try:
                    start = int(row[a])
                    end = int(row[b])
                    if end >= start:
                        tags.extend(range(start, end + 1))
                except (ValueError, TypeError):
                    continue

    return sorted(set(tags))


def process_file(
    file: Path,
    df_cornerpoints: pd.DataFrame,
    df_px_per_cm: pd.DataFrame,
    dftags: Optional[pd.DataFrame],
    *,
    update: bool = True,
    confidence_threshold: float = 0.8,
    min_num_obs_in_1hrs_tracked: int = 15,
    max_speed: float = 5.0,
    outdir: Optional[Path] = None,
    cam_hive_map: Optional[dict] = None,
) -> Tuple[str, str]:
    """
    Process a single dill file -> parquet.
    Returns (filename, status).
    """
    cfg = get_config()
    if outdir is None:
        outdir = getattr(cfg, "traj_outdir", None)
    if cam_hive_map is None:
        cam_hive_map = getattr(cfg, "cam_hive_map", None)
    outdir = Path(outdir) if outdir is not None else None
    if outdir is None:
        raise ValueError("Output directory not set (pass outdir or set cfg.traj_outdir)")
    if cam_hive_map is None:
        raise ValueError("cam_hive_map not set in config")

    file = Path(file)
    outfile = outdir / (file.stem + ".parquet")

    try:
        if outfile.exists():
            if not update:
                return (str(file), "skip_existing")
            if file.stat().st_mtime <= outfile.stat().st_mtime:
                return (str(file), "skip_up_to_date")

        cam, startdt, _ = parse_video_fname(str(file))
        hive = cam_hive_map[cam]
        df = dfunc.get_tracked_dataframe(str(file), df_cornerpoints, df_px_per_cm)
        if len(df) == 0:
            return (str(file), "no_tracks")

        # remove low-confidence identified bees
        df = df[(df["bee_id_confidence"] > confidence_threshold) & (df["detection_type"] == 1)]
        if len(df) == 0:
            return (str(file), "no_tracks")

        # filter bees by tag introduction dates (optional)
        valid_tags_list = get_valid_tags(dftags, startdt, hive)
        if valid_tags_list:
            df = df[df["bee_id"].isin(valid_tags_list)]
            if len(df) == 0:
                return (str(file), "no_tracks")

        # remove duplicates (same timestamp+cam+bee_id)
        df = (
            df.sort_values(
                by=["timestamp", "cam_id", "bee_id", "bee_id_confidence"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(subset=["timestamp", "cam_id", "bee_id"], keep="first")
        )

        # filter out speed jumps
        df = filter_speed_jumps(df, min_num_obs_in_1hrs_tracked, max_speed=max_speed)
        if len(df) == 0:
            return (str(file), "no_tracks")

        # drop unnecessary columns
        cols_to_drop = ["frame_id", "track_id", "detection_index", "detection_type"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        outdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(outfile, compression="snappy")
        return (str(file), "ok")

    except Exception:
        return (str(file), "error")


def process_files(
    dillfiles: Iterable[Path],
    df_cornerpoints: pd.DataFrame,
    df_px_per_cm: pd.DataFrame,
    dftags: Optional[pd.DataFrame],
    *,
    update: bool = True,
    confidence_threshold: float = 0.8,
    min_num_obs_in_1hrs_tracked: int = 15,
    max_speed: float = 5.0,
    outdir: Optional[Path] = None,
    cam_hive_map: Optional[dict] = None,
    max_workers: int = 4,
) -> list:
    """Process multiple dill files in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for f in dillfiles:
            tasks.append(
                ex.submit(
                    process_file,
                    f,
                    df_cornerpoints,
                    df_px_per_cm,
                    dftags,
                    update=update,
                    confidence_threshold=confidence_threshold,
                    min_num_obs_in_1hrs_tracked=min_num_obs_in_1hrs_tracked,
                    max_speed=max_speed,
                    outdir=outdir,
                    cam_hive_map=cam_hive_map,
                )
            )
        for fut in as_completed(tasks):
            results.append(fut.result())
    return results
