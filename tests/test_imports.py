"""Smoke test: the new public API exists after the feedercams/weather refactor."""


def test_feedercams_public_api():
    from bb_metrics import feedercams as fc

    for name in [
        "cam_hive_map",
        "get_dfcounts_from_avgdir",
        "prep_dfcounts",
        "get_feedercam_hourly_average",
        "load_avgcounts",
    ]:
        assert hasattr(fc, name), f"feedercams.{name} missing"


def test_get_weather_data_has_tz_param():
    import bb_metrics.datafunctions as dfunc

    assert "tz" in dfunc.get_weather_data.__code__.co_varnames
