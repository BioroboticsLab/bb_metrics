"""Gap-boundary columns of get_metrics (exit_edge_dist_*, t_first_s/t_last_s, cam_first/cam_last), synthetic data only.

They exist only with the 4-frame geometry (compute_frame_exit_metrics), come after the existing columns, and are
left out of the day matrix.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bb_metrics
from bb_metrics import datafunctions as dfunc
from bb_metrics import metricsfunctions as mfunc
from bb_metrics.metrics_pipeline import build_day_data_matrix

# column order before the gap-boundary columns were added (downstream notebooks rely on it)
COLS_4FRAME = [
    'bee_id', 'num_detections', 'dispersion', 'speed_median', 'speed_iqr', 'num_trips',
    'fraction_squares_visited', 'exit_distance_median',
    'frame_0_hist', 'frame_1_hist', 'frame_2_hist', 'frame_3_hist',
    'frame_0_centermedian', 'frame_1_centermedian', 'frame_2_centermedian', 'frame_3_centermedian',
    'inplace_events', 'burst_events', 'large_turn_events',
]
COLS_2FRAME = [
    'bee_id', 'num_detections', 'dispersion', 'speed_median', 'speed_iqr', 'fraction_squares_visited',
    'inplace_events', 'burst_events', 'large_turn_events',
]
NEW_COLS = ['exit_edge_dist_first', 'exit_edge_dist_last', 'exit_edge_dist_min',
            't_first_s', 't_last_s', 'cam_first', 'cam_last']

T0 = pd.Timestamp('2026-07-01T10:00:00', tz='UTC')


def _cfg(frame_exit=True):
    xpixels, ypixels, ppb = 4608, 5312, 225
    numxbins, numybins = 42, 24
    return SimpleNamespace(
        offset_div_cm=23.2, frame_width_cm=37.6, xpixels=xpixels, ypixels=ypixels,
        numxbins=numxbins, numybins=numybins,
        x_edges=np.linspace(0, 2 * xpixels, numxbins + 1), y_edges=np.linspace(0, ypixels, numybins + 1),
        cam_hive_map={2: 'B', 3: 'B'}, hive_cam_map={'B': (2, 3)},
        compute_frame_exit_metrics=frame_exit,
    )


@pytest.fixture
def use_cfg(monkeypatch):
    def _set(cfg):
        monkeypatch.setattr(bb_metrics, '_CFG', cfg)
        monkeypatch.setattr(dfunc, 'bd', None)  # datafunctions caches the config
    return _set


def _dfbee(t_s, y_hive, cam_id):
    n = len(t_s)
    x_hive = np.linspace(5, 15, n)
    df = pd.DataFrame({
        'bee_id': 7,
        'timestamp': T0 + pd.to_timedelta(t_s, unit='s'),
        'cam_id': cam_id,
        'x_hive': x_hive,
        'y_hive': y_hive,
        'x_pixels': x_hive * 100,
        'y_pixels': -np.asarray(y_hive, dtype=float) * 100,
        'orientation': np.linspace(0, 1, n),
    })
    df['x_hive_flat'] = np.where(df['cam_id'] == 3, 37.6 - df['x_hive'], df['x_hive'])
    return df


def test_gap_boundary_values(use_cfg):
    use_cfg(_cfg())
    t_s = np.arange(20) * (1 / 6) + 3.0  # 3.0 s .. 6.17 s after the segment start
    y = np.linspace(-10, -30, 20)
    y[7] = 0.4  # in the bottom slot: negative distance, kept
    cam = np.r_[np.full(10, 3), np.full(10, 2)]
    df = _dfbee(t_s, y, cam).sample(frac=1, random_state=0)  # unsorted input
    m, _ = mfunc.get_metrics(df, T0 + pd.Timedelta('60s'), timestamp_start=T0)

    assert m['exit_edge_dist_first'] == pytest.approx(10.0)
    assert m['exit_edge_dist_last'] == pytest.approx(30.0)
    assert m['exit_edge_dist_min'] == pytest.approx(-0.4)
    assert m['t_first_s'] == pytest.approx(3.0)
    assert m['t_last_s'] == pytest.approx(3.0 + 19 / 6)
    assert (m['cam_first'], m['cam_last']) == (1, 0)


def test_no_timestamp_start_gives_nan(use_cfg):
    use_cfg(_cfg())
    m, _ = mfunc.get_metrics(_dfbee(np.arange(5.0), np.full(5, -5.0), np.full(5, 2)), T0 + pd.Timedelta('60s'))
    assert np.isnan(m['t_first_s']) and np.isnan(m['t_last_s'])
    assert m['exit_edge_dist_first'] == pytest.approx(5.0)


def test_single_detection(use_cfg):
    use_cfg(_cfg())
    m, _ = mfunc.get_metrics(_dfbee([12.5], [-8.0], [3]), T0 + pd.Timedelta('60s'), timestamp_start=T0)
    assert m['exit_edge_dist_first'] == m['exit_edge_dist_last'] == m['exit_edge_dist_min'] == pytest.approx(8.0)
    assert m['t_first_s'] == m['t_last_s'] == pytest.approx(12.5)
    assert m['cam_first'] == m['cam_last'] == 1


def test_column_order_4frame(use_cfg):
    use_cfg(_cfg())
    m, _ = mfunc.get_metrics(_dfbee(np.arange(10.0), np.full(10, -5.0), np.full(10, 2)), T0, timestamp_start=T0)
    assert list(m) == COLS_4FRAME + NEW_COLS
    assert tuple(NEW_COLS) == mfunc.GAP_BOUNDARY_COLS


def test_2frame_has_no_new_columns(use_cfg):
    use_cfg(_cfg(frame_exit=False))
    m, _ = mfunc.get_metrics(_dfbee(np.arange(10.0), np.full(10, -5.0), np.full(10, 2)), T0, timestamp_start=T0)
    assert list(m) == COLS_2FRAME


def test_day_matrix_excludes_new_columns():
    starts = T0 + pd.to_timedelta([0, 60, 120], unit='s')
    df = pd.DataFrame({
        'hive': 'B', 'bee_id': 7, 'timestamp_start': starts, 'timestamp_end': starts + pd.Timedelta('60s'),
        'num_detections': [12, 20, 30], 'num_trips': 0, 'inplace_events': 0, 'burst_events': 0,
        'large_turn_events': 0, 'speed_median': [0.1, 0.2, 0.3], 'dispersion': [1.0, 2.0, 3.0],
        **{c: [1.0, 2.0, 3.0] for c in NEW_COLS},
    })
    day = df['timestamp_start'].dt.tz_convert('Europe/Berlin').dt.date.iloc[0]
    out = build_day_data_matrix(df, {day: 0})
    assert len(out) == 1
    assert 'dispersion' in out.columns
    assert not set(NEW_COLS) & set(out.columns)
