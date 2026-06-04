"""Offline synthetic-data unit tests for the feedercams plotting-companion helpers."""
import types

import numpy as np
import pandas as pd
import pytest

from bb_metrics import feedercams as fc


def _cfg(hives=("A", "B", "C", "D"), **kw):
    ns = types.SimpleNamespace(hives=hives, year=2026, **kw)
    return ns


# --------------------------- cam_hive_map ---------------------------

def test_cam_hive_map_four_hive():
    assert fc.cam_hive_map("feedercam", _cfg()) == {
        "feedercamA": "A", "feedercamB": "B", "feedercamC": "C", "feedercamD": "D"
    }


def test_cam_hive_map_single_hive_and_exit_token():
    assert fc.cam_hive_map("feedercam", _cfg(hives=("A",))) == {"feedercamA": "A"}
    assert fc.cam_hive_map("exitcam", _cfg(hives=("A",))) == {"exitcamA": "A"}


def test_cam_hive_map_missing_hives_raises():
    bad = types.SimpleNamespace()  # no .hives
    with pytest.raises(ValueError):
        fc.cam_hive_map("feedercam", bad)


# --------------------------- get_dfcounts_from_avgdir ---------------------------

def _write_parquet(path, n, cam="feedercamA"):
    df = pd.DataFrame({
        "cam_id": [cam] * n,
        "video_start_timestamp": pd.date_range("2026-06-02", periods=n, freq="30min", tz="UTC"),
        "totalcounts": np.arange(n, dtype=float),
        "taggedcounts": np.zeros(n),
        "untaggedcounts": np.arange(n, dtype=float),
    })
    df.to_parquet(path)


def test_get_dfcounts_from_avgdir_concats(tmp_path):
    _write_parquet(tmp_path / "2026a-feedercam-c.parquet", 3)
    _write_parquet(tmp_path / "2026b-feedercam-c.parquet", 2)
    out = fc.get_dfcounts_from_avgdir(tmp_path, "2026*feedercam-c.parquet")
    assert len(out) == 5
    assert pd.api.types.is_datetime64_any_dtype(out["video_start_timestamp"])


def test_get_dfcounts_from_avgdir_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fc.get_dfcounts_from_avgdir(tmp_path, "nope*.parquet")


# --------------------------- prep_dfcounts ---------------------------

def test_prep_dfcounts_naive_localized_no_wallclock_shift():
    df = pd.DataFrame({
        "cam_id": ["feedercamA"],
        "video_start_timestamp": [pd.Timestamp("2026-06-02 08:30:00")],  # naive
        "totalcounts": [1.0], "taggedcounts": [0.0], "untaggedcounts": [1.0],
    })
    out = fc.prep_dfcounts(df, {"feedercamA": "A"})
    ts = out["video_start_timestamp"].iloc[0]
    assert str(ts.tz) == "Europe/Berlin"
    assert (ts.hour, ts.minute) == (8, 30)  # localized, wall-clock unchanged
    assert out["hive"].iloc[0] == "A"
    assert out["hour"].iloc[0] == 8
    assert out["date"].iloc[0] == pd.Timestamp("2026-06-02", tz="Europe/Berlin")


def test_prep_dfcounts_tzaware_converted():
    df = pd.DataFrame({
        "cam_id": ["feedercamA"],
        "video_start_timestamp": [pd.Timestamp("2026-06-02 08:30:00", tz="UTC")],
        "totalcounts": [1.0], "taggedcounts": [0.0], "untaggedcounts": [1.0],
    })
    out = fc.prep_dfcounts(df, {"feedercamA": "A"})
    ts = out["video_start_timestamp"].iloc[0]
    assert str(ts.tz) == "Europe/Berlin"
    assert ts.hour == 10  # 08:30 UTC -> 10:30 Berlin (summer +2)


def test_prep_dfcounts_unmapped_cam_passthrough():
    df = pd.DataFrame({
        "cam_id": ["weirdcamX"],
        "video_start_timestamp": [pd.Timestamp("2026-06-02 08:30:00", tz="UTC")],
        "totalcounts": [1.0], "taggedcounts": [0.0], "untaggedcounts": [1.0],
    })
    out = fc.prep_dfcounts(df, {"feedercamA": "A"})
    assert out["hive"].iloc[0] == "weirdcamX"  # fillna keeps raw cam_id


# --------------------------- get_feedercam_hourly_average ---------------------------

def _hourdata(n=3, with_weather=False):
    idx = pd.date_range("2026-06-02 00:00", periods=n, freq="h", tz="Europe/Berlin")
    df = pd.DataFrame(index=idx)
    if with_weather:
        df["temp"] = np.arange(n, dtype=float)
    return df


def test_hourly_average_time_weighting_and_boundary():
    tz = "Europe/Berlin"
    df = pd.DataFrame({
        "cam_id": ["feedercamA", "feedercamA"],
        "hive": ["A", "A"],
        # video1: fully inside hour 00:00 (30s); video2: straddles 00:00/01:00 (10s | 20s)
        "video_start_timestamp": [
            pd.Timestamp("2026-06-02 00:00:10", tz=tz),
            pd.Timestamp("2026-06-02 00:59:50", tz=tz),
        ],
        "totalcounts": [12.0, 6.0],
        "taggedcounts": [0.0, 0.0],
        "untaggedcounts": [12.0, 6.0],
    })
    out = fc.get_feedercam_hourly_average(df, _hourdata(), video_duration_seconds=30, hives=["A", "B"])

    def val(hive, hour, col="totalcounts"):
        m = (out["hive"] == hive) & (out["hour"] == pd.Timestamp(hour, tz=tz))
        return out.loc[m, col].iloc[0]

    assert val("A", "2026-06-02 00:00") == pytest.approx(12 * 30 / 3600 + 6 * 10 / 3600)
    assert val("A", "2026-06-02 01:00") == pytest.approx(6 * 20 / 3600)
    assert val("A", "2026-06-02 02:00") == pytest.approx(0.0)
    # empty hive present in `hives` -> all zeros
    assert val("B", "2026-06-02 00:00") == pytest.approx(0.0)
    assert "hour_integer" in out.columns and "date" in out.columns


def test_hourly_average_merges_weather():
    tz = "Europe/Berlin"
    df = pd.DataFrame({
        "cam_id": ["feedercamA"], "hive": ["A"],
        "video_start_timestamp": [pd.Timestamp("2026-06-02 00:00:10", tz=tz)],
        "totalcounts": [1.0], "taggedcounts": [0.0], "untaggedcounts": [1.0],
    })
    out = fc.get_feedercam_hourly_average(df, _hourdata(with_weather=True), hives=["A"])
    assert "temp" in out.columns
    row0 = out[out["hour"] == pd.Timestamp("2026-06-02 00:00", tz=tz)]
    assert row0["temp"].iloc[0] == 0.0


# --------------------------- load_avgcounts ---------------------------

def test_load_avgcounts_derived_map(tmp_path, monkeypatch):
    cfg = _cfg(hives=("A", "B"), feedercam_avg_dir=tmp_path)
    monkeypatch.setattr(fc, "get_config", lambda: cfg)
    _write_parquet(tmp_path / "2026-feedercam-c.parquet", 2, cam="feedercamA")
    out = fc.load_avgcounts("feedercam", clahe=True)
    assert out["hive"].iloc[0] == "A"
    assert str(out["video_start_timestamp"].dt.tz.zone if hasattr(out["video_start_timestamp"].dt.tz, "zone") else out["video_start_timestamp"].dt.tz) == "Europe/Berlin"


def test_load_avgcounts_explicit_override_beats_derived(tmp_path, monkeypatch):
    cfg = _cfg(hives=("A",), feedercam_avg_dir=tmp_path)
    monkeypatch.setattr(fc, "get_config", lambda: cfg)
    # konstanz-like: file token outdoorcam, in-row cam_id feedercamA
    _write_parquet(tmp_path / "2026-outdoorcam-c.parquet", 2, cam="feedercamA")
    out = fc.load_avgcounts("outdoorcam", clahe=True, cam_to_hive={"feedercamA": "A"})
    assert set(out["hive"]) == {"A"}


def test_load_avgcounts_missing_avgdir_raises(monkeypatch):
    monkeypatch.setattr(fc, "get_config", lambda: types.SimpleNamespace(hives=("A",)))
    with pytest.raises(ValueError):
        fc.load_avgcounts("feedercam")
