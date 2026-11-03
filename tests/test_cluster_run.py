"""Cluster-run path (prepare_metrics_run -> python -m bb_metrics.run_metrics -> check_metrics_complete), synthetic data only.

The worker must write exactly what datafile_to_metrics writes when called directly (the local path).
"""
import json

import numpy as np
import pandas as pd
import pytest

import bb_metrics
from bb_metrics import datafunctions as dfunc
from bb_metrics import metricsfunctions as mfunc
from bb_metrics import metrics_pipeline as mp
from bb_metrics import run_metrics

T0 = pd.Timestamp('2026-07-01T10:00:00', tz='UTC')
T1 = T0 + pd.Timedelta('1h')

CFG_TEXT = """
import numpy as np
from pathlib import Path
offset_div_cm = 23.2
frame_width_cm = 37.6
xpixels, ypixels = 4608, 5312
numxbins, numybins = 42, 24
x_edges = np.linspace(0, 2 * xpixels, numxbins + 1)
y_edges = np.linspace(0, ypixels, numybins + 1)
cam_hive_map = {2: 'B', 3: 'B'}
hive_cam_map = {'B': (2, 3)}
compute_frame_exit_metrics = True
metrics_dir = Path(__file__).parent / 'metrics'
traj_outdir = Path(__file__).parent / 'traj'
"""


def _traj(cam_id, n_bees, seconds, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for bee in range(n_bees):
        t = np.arange(0, seconds, 1 / 6)
        x = 10 + np.cumsum(rng.normal(0, 0.05, len(t)))
        y = -15 + np.cumsum(rng.normal(0, 0.05, len(t)))
        rows.append(pd.DataFrame({
            'bee_id': bee, 'timestamp': T0 + pd.to_timedelta(t, unit='s'), 'cam_id': cam_id,
            'x_hive': x, 'y_hive': y, 'x_pixels': x * 100, 'y_pixels': -y * 100,
            'orientation': rng.uniform(-np.pi, np.pi, len(t)),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / 'traj').mkdir()
    cfg_path = tmp_path / 'cfg.py'
    cfg_path.write_text(CFG_TEXT)
    cfg = bb_metrics.load_config(cfg_path, reload=True)
    monkeypatch.setattr(bb_metrics, '_CFG', cfg)
    monkeypatch.setattr(dfunc, 'bd', None)

    def pair(hour, n_bees=3, seconds=150):
        start, end = T0 + pd.Timedelta(hours=hour), T1 + pd.Timedelta(hours=hour)
        stem = f"{start:%Y-%m-%dT%H_%M_%S}Z--{end:%Y-%m-%dT%H_%M_%S}Z.parquet"
        files = []
        for cam in (2, 3):
            f = tmp_path / 'traj' / f"Cam_{cam}_{stem}"
            df = _traj(cam, n_bees, seconds, seed=cam + 10 * hour)
            df['timestamp'] += pd.Timedelta(hours=hour)
            df.to_parquet(f)
            files.append(str(f))
        return ('B', start, end, files)

    return cfg_path, cfg, pair


def test_worker_matches_local(setup, tmp_path):
    cfg_path, cfg, pair = setup
    p = pair(0)
    local_dir = tmp_path / 'local'
    local_dir.mkdir()
    assert mfunc.datafile_to_metrics(*p, False, '1min', None, 2.0, False, str(local_dir)) == 'written'

    run_dir = mp.prepare_metrics_run(cfg_path, '1min', pairs=[p])
    assert run_metrics.main(['--run-dir', str(run_dir), str(run_dir / 'pairs.jsonl')]) == 0

    out = mfunc.metrics_file_for('B', p[3], '1min', cfg.metrics_dir)
    ref = mfunc.metrics_file_for('B', p[3], '1min', local_dir)
    pd.testing.assert_frame_equal(pd.read_parquet(out), pd.read_parquet(ref))
    run = json.loads((run_dir / 'RUN.json').read_text())
    assert run['min_track_seconds'] == 2.0 and run['min_num_detections'] is None and run['n_pairs'] == 1


def test_pending_and_complete(setup):
    cfg_path, cfg, pair = setup
    full, empty = pair(0), pair(1, seconds=0.5)  # 0.5 s per cam (1 s in total) < 2 s gate -> no output
    pairs = [full, empty]
    run_dir = mp.prepare_metrics_run(cfg_path, '1min', pairs=pairs)
    assert run_metrics.main(['--run-dir', str(run_dir), str(run_dir / 'pairs.jsonl')]) == 0

    df = mp.check_metrics_complete(pairs, '1min', cfg.metrics_dir)
    assert list(df['state']) == ['ok', 'empty']
    assert mp.pending_pairs(pairs, '1min', cfg.metrics_dir) == []
    assert mp.prepare_metrics_run(cfg_path, '1min', pairs=pairs) is None

    mfunc.metrics_file_for('B', full[3], '1min', cfg.metrics_dir).unlink()
    assert mp.pending_pairs(pairs, '1min', cfg.metrics_dir) == [full]
    assert len(mp.pending_pairs(pairs, '1min', cfg.metrics_dir, reprocess=True)) == 2


def test_error_fails_loudly(setup):
    cfg_path, cfg, pair = setup
    p = pair(0)
    bad = ('B', p[1], p[2], [p[3][0], p[3][1] + '.missing'])
    run_dir = mp.prepare_metrics_run(cfg_path, '1min', pairs=[bad])
    assert run_metrics.main(['--run-dir', str(run_dir), str(run_dir / 'pairs.jsonl')]) == 1
    df = mp.check_metrics_complete([bad], '1min', cfg.metrics_dir)
    assert list(df['state']) == ['error']
