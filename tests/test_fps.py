"""fps inference + fps-aware helpers (no hardcoded frame rate).

Covers infer_fps recovering 3/6/14 fps and a 6->14 burst (per-window), and that
assign_integer_framenums / filter_df_by_numdetections scale with fps instead of a
hardcoded constant.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from bb_metrics import datafunctions as dfunc


def _detection_times(fps, n_frames, start="2025-08-01T09:00:00", dup=2, tz=None):
    """n_frames frames at `fps`, each repeated `dup` times (multiple detections per
    frame), as a tz-aware-or-naive Series of timestamps."""
    base = pd.Timestamp(start, tz=tz)
    frame_idx = np.repeat(np.arange(n_frames), dup)
    return pd.Series(base + pd.to_timedelta(frame_idx / fps, unit="s"))


@pytest.mark.parametrize("fps", [3.0, 6.0, 14.0])
def test_infer_fps_recovers_constant_rate(fps):
    ts = _detection_times(fps, 300, dup=3)
    assert dfunc.infer_fps(ts) == pytest.approx(fps, rel=1e-6)


def test_infer_fps_tz_aware():
    ts = _detection_times(6.0, 200, dup=2, tz="UTC")
    assert dfunc.infer_fps(ts) == pytest.approx(6.0, rel=1e-6)


def test_infer_fps_robust_to_gaps():
    # drop a chunk of frames (tracking gap) -> median of consecutive gaps unaffected
    ts = _detection_times(6.0, 300, dup=1)
    ts = pd.concat([ts.iloc[:100], ts.iloc[200:]]).reset_index(drop=True)
    assert dfunc.infer_fps(ts) == pytest.approx(6.0, rel=1e-6)


def test_infer_fps_fallback_and_warns():
    with pytest.warns(UserWarning):
        assert dfunc.infer_fps([]) == dfunc._FALLBACK_FPS
    with pytest.warns(UserWarning):
        assert dfunc.infer_fps([pd.Timestamp("2025-08-01")]) == dfunc._FALLBACK_FPS
    # warn=False stays silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dfunc.infer_fps([], warn=False) == dfunc._FALLBACK_FPS


def test_infer_fps_per_window_recovers_burst():
    # 6 fps then 14 fps, concatenated -> per-window slices recover each rate
    a = _detection_times(6.0, 200, start="2025-08-01T09:00:00", dup=1)
    b = _detection_times(14.0, 200, start="2025-08-01T09:01:00", dup=1)
    assert dfunc.infer_fps(a) == pytest.approx(6.0, rel=1e-6)
    assert dfunc.infer_fps(b) == pytest.approx(14.0, rel=1e-6)


@pytest.mark.parametrize("fps", [3, 6, 14])
def test_assign_integer_framenums_scales_with_fps(fps):
    times = pd.to_datetime(
        ["2025-08-01T09:00:00.000", "2025-08-01T09:00:00.500", "2025-08-01T10:15:30.250"]
    )
    sec = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in times])
    expected = np.floor(sec * fps).astype(int)
    np.testing.assert_array_equal(dfunc.assign_integer_framenums(times, fps=fps), expected)


def test_assign_integer_framenums_infers_when_fps_none():
    times = _detection_times(6.0, 50, dup=1)
    explicit = dfunc.assign_integer_framenums(times, fps=dfunc.infer_fps(times))
    inferred = dfunc.assign_integer_framenums(times)  # fps=None -> infer
    np.testing.assert_array_equal(explicit, inferred)


def test_assign_integer_framenums_hourminsec_uses_fps():
    assert int(dfunc.assign_integer_framenums_hourminsec(0, 1, 0, fps=6)) == 360
    assert int(dfunc.assign_integer_framenums_hourminsec(0, 1, 0, fps=3)) == 180


@pytest.mark.parametrize("fps", [3, 6, 14])
def test_filter_df_by_numdetections_threshold_scales(fps):
    # threshold == round(min_minutes*60*fps); rows at/above are kept
    df = pd.DataFrame({"num_detections": [0, 59 * fps, 60 * fps, 120 * fps]})
    out = dfunc.filter_df_by_numdetections(df, min_time_detection_minutes=1, fps=fps)
    # 1 min at `fps` -> threshold 60*fps; rows >= that survive (2 rows)
    assert set(out["num_detections"]) == {60 * fps, 120 * fps}


def test_filter_df_by_numdetections_infers_fps_from_timestamps():
    # df carries per-detection timestamps -> fps inferred (6), threshold = 360 for 1 min
    ts = _detection_times(6.0, 600, dup=1)
    df = pd.DataFrame({"timestamp": ts})
    df["num_detections"] = 360  # exactly the 1-min-at-6fps threshold
    out = dfunc.filter_df_by_numdetections(df, min_time_detection_minutes=1)
    assert len(out) == len(df)  # 360 >= 360 kept
    out2 = dfunc.filter_df_by_numdetections(df.assign(num_detections=359),
                                            min_time_detection_minutes=1)
    assert len(out2) == 0
