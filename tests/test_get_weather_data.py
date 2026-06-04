"""Tests for the get_weather_data tz= enhancement (backward compatible)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import bb_metrics.datafunctions as dfunc


class _FakeHourly:
    last = {}

    def __init__(self, station, start, end):
        _FakeHourly.last = {"station": station, "start": start, "end": end}

    def fetch(self):
        idx = pd.date_range("2026-06-01 00:00", periods=3, freq="h")  # naive (UTC)
        return pd.DataFrame({"temp": [10.0, 11.0, 12.0]}, index=idx)


class _FakeDaily:
    used = False

    def __init__(self, station, start, end):
        _FakeDaily.used = True

    def fetch(self):
        idx = pd.date_range("2026-06-01", periods=2, freq="D")
        return pd.DataFrame({"temp": [10.0, 11.0]}, index=idx)


@pytest.fixture(autouse=True)
def _patch_meteostat(monkeypatch):
    monkeypatch.setattr(dfunc, "_MSHourly", _FakeHourly)
    monkeypatch.setattr(dfunc, "_MSDaily", _FakeDaily)
    _FakeDaily.used = False


def test_invalid_data_type_returns_nan():
    assert np.isnan(dfunc.get_weather_data("X", datetime(2026, 6, 1), datetime(2026, 6, 2), data_type="weekly"))


def test_tz_none_keeps_naive_index_unchanged():
    out = dfunc.get_weather_data("X", datetime(2026, 6, 1), datetime(2026, 6, 2))
    assert out.index.tz is None  # legacy naive-UTC behavior preserved


def test_naive_input_passed_through():
    start = datetime(2026, 6, 1, 0, 0)
    dfunc.get_weather_data("ST", start, datetime(2026, 6, 2))
    assert _FakeHourly.last["station"] == "ST"
    assert pd.Timestamp(_FakeHourly.last["start"]) == pd.Timestamp(start)
    assert pd.Timestamp(_FakeHourly.last["start"]).tz is None


def test_tzaware_input_coerced_to_naive_utc():
    # 00:00 Berlin (summer +2) -> 22:00 UTC the previous day, naive
    start = pd.Timestamp("2026-06-02 00:00", tz="Europe/Berlin")
    dfunc.get_weather_data("X", start, pd.Timestamp("2026-06-03 00:00", tz="Europe/Berlin"))
    got = pd.Timestamp(_FakeHourly.last["start"])
    assert got.tz is None
    assert got == pd.Timestamp("2026-06-01 22:00")


def test_tz_localizes_output_index():
    out = dfunc.get_weather_data("X", datetime(2026, 6, 1), datetime(2026, 6, 2), tz="Europe/Berlin")
    assert str(out.index.tz) == "Europe/Berlin"
    # 00:00 naive-UTC -> 02:00+02:00 Berlin
    assert out.index[0] == pd.Timestamp("2026-06-01 00:00", tz="UTC").tz_convert("Europe/Berlin")


def test_daily_selects_daily_class():
    dfunc.get_weather_data("X", datetime(2026, 6, 1), datetime(2026, 6, 3), data_type="daily")
    assert _FakeDaily.used
