"""Comb label grids (datafunctions.process_one_annotation / GridLookup), synthetic data only.

Annotations are drawn on recording-frame backgrounds, trajectories are in the analysis
frame. A grid built with rot_cfg must answer lookups made with trajectory coordinates;
a legacy grid (no rot_cfg) stays in the recording frame and is refused by check_frame.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bb_metrics import datafunctions as dfunc
from bb_metrics.rotation import get_rotation_config

# Konstanz/Berlin 2026: cfg.xpixels/ypixels are the analysis frame, cameras rotated cw90
CFG = SimpleNamespace(camera_rotation="cw90", xpixels=4608, ypixels=5312)
RAW_W, RAW_H = CFG.ypixels, CFG.xpixels
DS = 8
LABELS = ["empty_cell", "open_brood", "capped_honey", "capped_brood", "other_cell"]
KEEP, OTHER = set(LABELS[:4]), LABELS[4]

# recording-frame cell centres, far apart so each owns its neighbourhood
CELLS = [
    {"id": "a", "center_x": 400, "center_y": 300, "radius": 24, "label": "capped_brood"},
    {"id": "b", "center_x": 5000, "center_y": 600, "radius": 24, "label": "capped_honey"},
    {"id": "c", "center_x": 2600, "center_y": 4300, "radius": 24, "label": "open_brood"},
    {"id": "d", "center_x": 1200, "center_y": 2400, "radius": 24, "label": "pollen"},
]


def _build(tmp_path, rot_cfg, recalc=True, as_dict=False):
    ann = tmp_path / "cam-0_background_20260623T000000.000000.000Z.json"
    ann.write_text(json.dumps({"annotations": CELLS} if as_dict else CELLS))
    out, status = dfunc.process_one_annotation(
        ann, tmp_path / "grids", RAW_W, RAW_H, DS, LABELS, KEEP, OTHER,
        recalc=recalc, off_comb_label="off_comb", off_comb_threshold=50, rot_cfg=rot_cfg,
    )
    df = pd.DataFrame({
        "cam": [0], "timestamp": [pd.Timestamp("2026-06-23", tz="UTC")], "grid_path": [out],
    })
    return out, status, dfunc.GridLookup(df)


def _lookup(lookup, x, y):
    n = len(x)
    return lookup.lookup_labels(
        np.zeros(n, dtype=int), np.repeat(pd.Timestamp("2026-06-24", tz="UTC"), n), x, y
    )


def test_analysis_frame_grid_matches_trajectory_coords(tmp_path):
    rot_cfg = get_rotation_config(CFG)
    out, status, lookup = _build(tmp_path, rot_cfg)
    assert status == "saved"
    assert dfunc.read_grid_frame(out) == "analysis"
    grid, ds, w, h, _ = lookup._load_grid(out)
    assert grid.shape == (RAW_W // DS, RAW_H // DS) == (664, 576)
    assert (w, h) == (CFG.xpixels, CFG.ypixels)

    # bee positions as the trajectories store them: recording coords rotated cw90
    cx = np.array([c["center_x"] for c in CELLS], dtype=float)
    cy = np.array([c["center_y"] for c in CELLS], dtype=float)
    x_a, y_a = rot_cfg.transform_detections(cx, cy)
    expected = ["capped_brood", "capped_honey", "open_brood", "other_cell"]
    assert list(_lookup(lookup, x_a, y_a)) == expected

    # the analysis frame is 5312 tall: y >= 4608 must not be dropped
    assert y_a.max() >= RAW_H
    lookup.check_frame("analysis")


def test_legacy_grid_stays_recording_frame_and_is_refused(tmp_path):
    out, _, lookup = _build(tmp_path, rot_cfg=None)
    assert dfunc.read_grid_frame(out) == "recording"
    grid, _, w, h, _ = lookup._load_grid(out)
    assert grid.shape == (RAW_H // DS, RAW_W // DS) == (576, 664)
    assert (w, h) == (RAW_W, RAW_H)
    with pytest.raises(ValueError, match="not in the analysis frame"):
        lookup.check_frame("analysis")


def test_old_npz_without_frame_key_counts_as_recording(tmp_path):
    path = tmp_path / "old_ds_8.npz"
    np.savez_compressed(
        path, label_grid=np.zeros((576, 664), dtype=np.int16), ds=np.array(8),
        raw_w=np.array(RAW_W), raw_h=np.array(RAW_H), label_order=np.array(LABELS, dtype=object),
    )
    assert dfunc.read_grid_frame(path) == "recording"
    _, _, w, h, _ = dfunc.GridLookup._load_grid(str(path))
    assert (w, h) == (RAW_W, RAW_H)


def test_recording_grid_is_rebuilt_when_analysis_frame_requested(tmp_path):
    out, _, _ = _build(tmp_path, rot_cfg=None)
    _, status, _ = _build(tmp_path, get_rotation_config(CFG), recalc=False)
    assert status == "saved"
    assert dfunc.read_grid_frame(out) == "analysis"
    _, status, _ = _build(tmp_path, get_rotation_config(CFG), recalc=False)
    assert status == "skip"


def test_dict_form_annotation_file(tmp_path):
    out, _, lookup = _build(tmp_path, get_rotation_config(CFG), as_dict=True)
    x_a, y_a = get_rotation_config(CFG).transform_detections(np.array([400.0]), np.array([300.0]))
    assert list(_lookup(lookup, x_a, y_a)) == ["capped_brood"]


def test_swapped_raw_dims_rejected():
    with pytest.raises(ValueError, match="does not match"):
        dfunc.grid_frame_size(RAW_H, RAW_W, get_rotation_config(CFG))
