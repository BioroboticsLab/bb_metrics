"""Corner points from CVAT annotations on the side-by-side (stitched) hive videos.

The two cameras' corners can be annotated as one two-point shape per frame
(Berlin/Konstanz 2025) or as a separate single-point track per camera (Berlin 2026);
both must give the same corner table.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from bb_metrics import calibration as cal

CFG = SimpleNamespace(
    SCALE_FACTOR=2, xpixels=4608, ypixels=5312, hive_cam_map={"A": (0, 1)}
)
N_FRAMES = 4
# half-resolution video coordinates; the right camera starts at x = 4608 / 2 = 2304
LEFT, RIGHT = (120.0, 2606.0), (2450.0, 2636.0)
EXPECTED = {0: (240.0, 5212.0), 1: (2 * 2450.0 - 4608, 5272.0)}


def _points(frame, pts, outside=0):
    coords = ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return (
        f'<points frame="{frame}" keyframe="1" outside="{outside}" occluded="0" '
        f'points="{coords}" z_order="0"></points>'
    )


def _write_xml(path, tracks):
    body = "".join(
        f'<track id="{i}" label="exit/corner" source="manual">{"".join(shapes)}</track>'
        for i, shapes in enumerate(tracks)
    )
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><annotations><version>1.1</version>{body}</annotations>'
    )
    return path


def _cam_ts():
    return pd.DataFrame(
        [
            {"camera": cam, "timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=d)}
            for cam in (0, 1)
            for d in range(N_FRAMES)
        ]
    )


def _corners(xml):
    return cal.corner_points_from_annotations(
        [cal.parse_annotation_xml(xml)], [xml], _cam_ts(), cfg=CFG
    )


def _check(corner_df):
    assert len(corner_df) == 2 * N_FRAMES
    for cam, (x, y) in EXPECTED.items():
        rows = corner_df[corner_df["cam"] == cam]
        assert len(rows) == N_FRAMES
        assert rows["corner_x"].tolist() == pytest.approx([x] * N_FRAMES)
        assert rows["corner_y"].tolist() == pytest.approx([y] * N_FRAMES)


def test_two_point_shape_per_frame(tmp_path):
    xml = _write_xml(tmp_path / "hive_A.xml", [[_points(f, [LEFT, RIGHT]) for f in range(N_FRAMES)]])
    _check(_corners(xml))


def test_one_track_per_camera(tmp_path):
    xml = _write_xml(
        tmp_path / "hive_A.xml",
        [
            [_points(f, [RIGHT]) for f in range(N_FRAMES)],  # track order must not matter
            [_points(f, [LEFT]) for f in range(N_FRAMES)],
        ],
    )
    _check(_corners(xml))


def test_outside_points_are_ignored(tmp_path):
    # a corner track that was ended in CVAT, then replaced by a new one
    ended = [_points(0, [(999.0, 999.0)]), _points(1, [(999.0, 999.0)], outside=1)]
    xml = _write_xml(
        tmp_path / "hive_A.xml",
        [
            ended,
            [_points(f, [LEFT]) for f in range(1, N_FRAMES)],
            [_points(f, [RIGHT]) for f in range(N_FRAMES)],
        ],
    )
    corner_df = _corners(xml)
    left = corner_df[corner_df["cam"] == 0].sort_values("timestamp")
    # frame 0 comes from the ended track, frames 1.. from its replacement
    assert left["corner_x"].tolist() == pytest.approx([1998.0] + [240.0] * (N_FRAMES - 1))


def test_two_points_for_one_camera_raises(tmp_path):
    xml = _write_xml(
        tmp_path / "hive_A.xml",
        [
            [_points(f, [LEFT]) for f in range(N_FRAMES)],
            [_points(f, [(LEFT[0] + 50, LEFT[1])]) for f in range(N_FRAMES)],
            [_points(f, [RIGHT]) for f in range(N_FRAMES)],
        ],
    )
    with pytest.raises(ValueError, match="More than one corner point"):
        _corners(xml)
