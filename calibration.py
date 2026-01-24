"""
Calibration helpers for corner points and pixel-to-cm factors.

These functions are designed to be called from a notebook so you can
visualize and validate results. They take a config object (or use the
active bb_metrics config) for dataset-specific parameters.
"""

from pathlib import Path
import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import get_config

def parse_annotation_xml(xmlfilename: Path) -> pd.DataFrame:
    """Parse a CVAT-style XML file into a DataFrame of boxes/points."""
    import xml.etree.ElementTree as ET

    xml_path = Path(xmlfilename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []

    for track in root.findall("track"):
        label = track.attrib["label"]
        tid = int(track.attrib["id"])

        # Bounding boxes
        for box in track.findall("box"):
            records.append(
                {
                    "track_id": tid,
                    "label": label,
                    "frame": int(box.attrib["frame"]),
                    "type": "box",
                    "xtl": float(box.attrib["xtl"]),
                    "ytl": float(box.attrib["ytl"]),
                    "xbr": float(box.attrib["xbr"]),
                    "ybr": float(box.attrib["ybr"]),
                    "points": None,
                }
            )

        # Points (may contain one or more coordinate pairs)
        for pts in track.findall("points"):
            frame = int(pts.attrib["frame"])
            raw_points = pts.attrib["points"]

            coords = []
            for pair in raw_points.split(";"):
                x, y = map(float, pair.split(","))
                coords.append((x, y))

            records.append(
                {
                    "track_id": tid,
                    "label": label,
                    "frame": frame,
                    "type": "points",
                    "xtl": None,
                    "ytl": None,
                    "xbr": None,
                    "ybr": None,
                    "points": coords,
                }
            )

    return pd.DataFrame(records)


def build_cam_timestamps(base_dir: Path, prefix: str = "background_") -> pd.DataFrame:
    """
    Walk a directory tree of PNGs and extract camera id + timestamp from filenames.
    Searches subdirectories (e.g., cam-*/...) and uses bb_binary.parse_image_fname
    after stripping an optional prefix.
    """
    from bb_binary.parsing import parse_image_fname

    rows = []
    base_dir = Path(base_dir)
    all_pngs = sorted(p for p in base_dir.rglob("*.png"))
    if not all_pngs:
        # explicit cam-* glob (in case the filesystem behaves oddly)
        all_pngs = sorted(base_dir.glob("cam-*/*.png"))
    for p in all_pngs:
        name = p.name
        if prefix and prefix in name:
            name = name.split(prefix, 1)[1]
        try:
            camera, ts = parse_image_fname(name)
        except Exception:
            continue
        rows.append(
            {
                "filename": p.name,
                "camera": int(camera),
                "timestamp": ts,
            }
        )

    if not rows:
        print(f"ERROR: No matching PNG files found under {base_dir}.")
        raise FileNotFoundError(
            f"No matching PNG files under {base_dir}. Expected filenames like "
            f"'{prefix}cam-<n>_<TS>Z[--<TS>Z].png'."
        )

    df_cam_timestamps = (
        pd.DataFrame(rows)
        .sort_values(["camera", "timestamp"])
        .reset_index(drop=True)
    )
    return df_cam_timestamps


def infer_hive_from_path(p: Path) -> Optional[str]:
    """Guess hive letter from a filename/path."""
    m = re.search(r"hive[_\- ]?([A-D])", str(p), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _get_cfg_param(cfg, name: str, default=None):
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if default is not None:
        return default
    raise AttributeError(f"Config missing required attribute '{name}'")


def corner_points_from_annotations(
    xml_dfs: List[pd.DataFrame],
    xmlfiles: Iterable[Path],
    cam_timestamps: pd.DataFrame,
    cfg=None,
) -> pd.DataFrame:
    """
    Extract left/right corner points from annotations and attach timestamps.
    Returns DataFrame with columns: hive, cam, timestamp, corner_x, corner_y, midday_utc.
    """
    if cfg is None:
        cfg = get_config()

    SCALE_FACTOR = _get_cfg_param(cfg, "SCALE_FACTOR", 2.0)
    XPIXELS = _get_cfg_param(cfg, "xpixels")
    YPIXELS = _get_cfg_param(cfg, "ypixels")
    hive_cam_map = _get_cfg_param(cfg, "hive_cam_map")
    annot_stitched = getattr(cfg, "annot_stitched", True)  # if annotations are on stitched images

    # Frame index per camera
    # Create separate frame indices for each camera to handle different image counts
    cam_frames_list = []
    for cam in cam_timestamps["camera"].unique():
        cam_data = (
            cam_timestamps[cam_timestamps["camera"] == cam]
            .sort_values("timestamp")
            .reset_index(drop=True)
            .assign(frame=lambda d: d.index, cam=cam)
            [["cam", "frame", "timestamp"]]
        )
        cam_frames_list.append(cam_data)

    cam_frames = pd.concat(cam_frames_list, ignore_index=True)

    records = []
    xmlfiles = list(xmlfiles)

    # Get max frame count per camera for validation
    cam_max_frames = cam_frames.groupby("cam")["frame"].max().to_dict()

    for i, xdf in enumerate(xml_dfs):
        hive = infer_hive_from_path(xmlfiles[i]) if i < len(xmlfiles) else None
        if hive is None:
            hive = chr(ord("A") + i)
        if hive not in hive_cam_map:
            continue

        left_cam, right_cam = hive_cam_map[hive]

        pts_df = xdf.loc[xdf["type"].str.lower() == "points", ["frame", "points"]].copy()

        for _, row in pts_df.iterrows():
            frame = int(row["frame"])
            pts = row["points"]
            if not isinstance(pts, list) or len(pts) < 2:
                continue

            pts_sorted = sorted(pts, key=lambda t: t[0])
            (x_left, y_left), (x_right, y_right) = pts_sorted[0], pts_sorted[-1]
            x_left, y_left, x_right, y_right = np.array(
                [x_left, y_left, x_right, y_right]
            ) * SCALE_FACTOR
            # For stitched annotations: shift right cam by width if coords exceed midline
            if annot_stitched and x_right > (XPIXELS / 2):
                x_right = x_right - XPIXELS

            # Corner points are kept in top-left origin (image convention)
            # to match trajectory coordinates and all other pixel data

            # Only add records for cameras that have this frame
            if left_cam in cam_max_frames and frame <= cam_max_frames[left_cam]:
                records.append(
                    {
                        "hive": hive,
                        "frame": frame,
                        "cam": left_cam,
                        "corner_x": float(x_left),
                        "corner_y": float(y_left),
                    }
                )

            if right_cam in cam_max_frames and frame <= cam_max_frames[right_cam]:
                records.append(
                    {
                        "hive": hive,
                        "frame": frame,
                        "cam": right_cam,
                        "corner_x": float(x_right),
                        "corner_y": float(y_right),
                    }
                )

    points_by_frame = pd.DataFrame.from_records(records)
    if points_by_frame.empty:
        print("ERROR: No usable point annotations found in XMLs.")
        return pd.DataFrame(
            columns=["hive", "cam", "timestamp", "corner_x", "corner_y", "midday_utc"]
        )

    merged = (
        points_by_frame
        .merge(cam_frames, on=["cam", "frame"], how="left")
        .sort_values(["hive", "cam", "frame"])
        [["hive", "cam", "frame", "timestamp", "corner_x", "corner_y"]]
        .reset_index(drop=True)
    )

    # Check for missing timestamps (annotations on frames without corresponding images)
    missing_ts = merged["timestamp"].isna()
    if missing_ts.any():
        missing_details = merged[missing_ts][["hive", "cam", "frame"]].drop_duplicates()
        print(f"WARNING: {missing_ts.sum()} annotations could not be matched to image timestamps:")
        for _, row in missing_details.iterrows():
            print(f"  Hive {row['hive']}, cam-{row['cam']}, frame {row['frame']}")
        print("  This typically means annotations reference frames beyond available images for that camera.")
        # Drop rows with missing timestamps
        merged = merged[~missing_ts].copy()

    if merged.empty:
        print("ERROR: No annotations could be matched to image timestamps.")
        return pd.DataFrame(
            columns=["hive", "cam", "timestamp", "corner_x", "corner_y", "midday_utc"]
        )

    merged["midday_utc"] = merged["timestamp"].dt.floor("D") + pd.Timedelta(hours=12)
    merged = merged[["hive", "cam", "timestamp", "corner_x", "corner_y", "midday_utc"]]
    return merged


def pixels_per_cm_from_boxes(
    xml_dfs: List[pd.DataFrame],
    xmlfiles: Iterable[Path],
    cfg=None,
) -> pd.DataFrame:
    """
    Compute pixels-per-cm per hive/cam using calibration boxes in the last frame.
    """
    if cfg is None:
        cfg = get_config()

    SCALE_FACTOR = _get_cfg_param(cfg, "SCALE_FACTOR", 1.0)
    KNOWN_LENGTH_CM = _get_cfg_param(cfg, "KNOWN_LENGTH_CM")
    hive_cam_map = _get_cfg_param(cfg, "hive_cam_map")

    records = []
    xmlfiles = list(xmlfiles)

    for i, xdf in enumerate(xml_dfs):
        hive = infer_hive_from_path(xmlfiles[i]) if i < len(xmlfiles) else None
        if hive is None:
            hive = chr(ord("A") + i)
        if hive not in hive_cam_map:
            continue

        left_cam, right_cam = hive_cam_map[hive]
        boxes = xdf.loc[xdf["type"].str.lower() == "box", ["frame", "xtl", "ytl", "xbr", "ybr"]].copy()
        if boxes.empty:
            continue

        last_frame = boxes["frame"].max()
        last_boxes = boxes.loc[boxes["frame"] == last_frame].sort_values("xtl").reset_index(drop=True)

        for j, (_, row) in enumerate(last_boxes.iterrows()):
            w = abs(row["xbr"] - row["xtl"])
            h = abs(row["ybr"] - row["ytl"])
            long_dim_px = max(w, h)

            cam = left_cam if j == 0 else right_cam
            px_per_cm = long_dim_px / KNOWN_LENGTH_CM * SCALE_FACTOR
            records.append({"hive": hive, "cam": cam, "pixels_per_cm": px_per_cm})

    df_px_per_cm = (
        pd.DataFrame(records)
        .sort_values(["hive", "cam"])
        .reset_index(drop=True)
    )
    return df_px_per_cm
