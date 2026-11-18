import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bb_metrics import events as ev
from bb_metrics import tables
from bb_metrics import calibration as cal


def _events(**over):
    d = dict(hive=["A", "A"], event_id=["1_pierce", "1_removal"], kind=["pierce", "removal"],
             t_start=["2025-10-01 11:50:39+00:00", "2025-10-01 17:10:25+00:00"],
             t_end=["2025-10-01 11:50:39+00:00", "2025-10-01 17:13:01+00:00"],
             cam=[1, 1], cell_id=[None, None], x_px=[10.0, 10.0], y_px=[20.0, 20.0],
             notes=["", ""], cell_no=[1, 1])
    d.update(over)
    return pd.DataFrame(d)


def test_validate_events_coerces_and_keeps_extras():
    d = ev.validate_events(_events())
    assert str(d["t_start"].dtype) == "datetime64[ns, UTC]"
    assert str(d["cam"].dtype) == "Int64"
    assert list(d.columns)[: len(ev.COLUMNS)] == list(ev.COLUMNS) and "cell_no" in d


@pytest.mark.parametrize("bad", [
    dict(event_id=["x", "x"]),
    dict(t_end=["2025-10-01 11:00:00+00:00", "2025-10-01 17:13:01+00:00"]),
    dict(kind=[None, "removal"]),
])
def test_validate_events_rejects(bad):
    with pytest.raises(ValueError):
        ev.validate_events(_events(**bad))


def test_snap_to_cells_per_camera():
    e = ev.validate_events(_events(cam=[1, 0]))
    cells = pd.DataFrame(dict(cell_id=["a", "b"], x=[11.0, 10.0], y=[20.0, 20.0], cam=[1, 0]))
    s = ev.snap_to_cells(e, cells)
    assert s["cell_id"].tolist() == ["a", "b"]
    s = ev.snap_to_cells(e, cells, max_dist_px=0.5)
    assert pd.isna(s.loc[0, "cell_id"]) and s.loc[1, "cell_id"] == "b"


def _write_metrics(d, hive, start, n_bees=2):
    t0 = pd.Timestamp(start, tz="UTC")
    rows = [dict(hive=hive, bee_id=b, timestamp_start=t0 + pd.Timedelta(minutes=m),
                 timestamp_end=t0 + pd.Timedelta(minutes=m + 1), num_detections=10 + b)
            for b in range(n_bees) for m in range(60)]
    name = (f"metrics-1min-Hive_{hive}_{t0:%Y-%m-%dT%H_%M_%SZ}--"
            f"{t0 + pd.Timedelta(hours=1):%Y-%m-%dT%H_%M_%SZ}.parquet")
    pd.DataFrame(rows).to_parquet(d / name)


def test_load_metrics_range_hives_uid(tmp_path):
    md = tmp_path / "metrics"; md.mkdir()
    so_dir = tmp_path / "saved_output" / "tags"; so_dir.mkdir(parents=True)
    for h in ("A", "B"):
        for hr in ("2025-10-01 10:00", "2025-10-01 11:00", "2025-10-02 10:00"):
            _write_metrics(md, h, hr)
    pd.DataFrame(dict(Date=["2025-07-01"], Hive=["A"], tag_start=[0], tag_end=[10])).to_csv(
        so_dir / "dftags.csv", index=False)
    cfg = SimpleNamespace(metrics_dir=md, saved_output_dir=tmp_path / "saved_output")

    files = tables.metric_files("1min", cfg, start="2025-10-01 10:30", end="2025-10-02")
    assert len(files) == 4                                     # 2 hives x hours 10 and 11
    m = tables.load_metrics("1min", cfg, start="2025-10-01 10:30", end="2025-10-02",
                            hives=["A"], columns=["num_detections"], verbose=False)
    assert set(m["hive"]) == {"A"} and len(m) == 2 * 90
    assert m["timestamp_start"].min() == pd.Timestamp("2025-10-01 10:30", tz="UTC")
    assert "uid" in m and set(m.columns) >= set(tables.KEYS + ["num_detections"])
    assert tables.load_metrics("60min", cfg, verbose=False).empty


def test_load_metrics_combined_only_when_fresh(tmp_path):
    md = tmp_path / "metrics"; md.mkdir()
    _write_metrics(md, "A", "2025-10-01 10:00")
    cfg = SimpleNamespace(metrics_dir=md, saved_output_dir=tmp_path / "so")
    stale = pd.DataFrame(dict(hive=["A"], bee_id=[99], timestamp_start=[pd.Timestamp("2025-10-01", tz="UTC")],
                              timestamp_end=[pd.Timestamp("2025-10-01", tz="UTC")], num_detections=[1]))
    stale.to_parquet(md / "allmetrics-1min.parquet")
    import os
    old = (md / "allmetrics-1min.parquet").stat().st_mtime - 100
    os.utime(md / "allmetrics-1min.parquet", (old, old))
    assert 99 not in tables.load_metrics("1min", cfg, add_uid=False, verbose=False)["bee_id"].values
    assert 99 in tables.load_metrics("1min", cfg, add_uid=False, use_combined=True,
                                     verbose=False)["bee_id"].values


def test_calibrate_orientation_recovers_sign_and_offset():
    rng = np.random.default_rng(0)
    rows = []
    for b in range(40):
        x, y = 1000.0, 1000.0
        for k in range(60):
            psi = rng.uniform(-np.pi, np.pi)
            x, y = x + 20 * np.cos(psi), y + 20 * np.sin(psi)
            ori = -(psi - np.radians(30)) + rng.normal(0, 0.2)     # sign -1, offset +30 deg
            rows.append(dict(bee_id=b, timestamp=pd.Timestamp("2025-10-01", tz="UTC")
                             + pd.Timedelta(seconds=k / 3), x_pixels=x, y_pixels=y, orientation=ori))
    # heading at step k must describe the step INTO detection k
    r = cal.calibrate_orientation(pd.DataFrame(rows))
    assert r["sign"] == -1 and abs(r["offset_deg"] - 30) < 2 and r["R"] > 0.9


def test_load_orientation(tmp_path):
    d = tmp_path / "saved_output" / "calibration"; d.mkdir(parents=True)
    (d / "orientation.json").write_text(json.dumps({"cams": {"1": {"sign": 1, "offset_rad": -3.07}}}))
    cfg = SimpleNamespace(saved_output_dir=tmp_path / "saved_output")
    assert cal.load_orientation(cfg, cam=1)["sign"] == 1
    assert list(cal.load_orientation(cfg)) == [1]
