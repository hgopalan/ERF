"""test_ucm_from_gis_smoke.py — smoke tests for ucm_from_gis module.

Skipped by default. Only runs when SLUCM_GIS_TESTS=1.
"""
import pytest
import os

skip_unless_gis = pytest.mark.skipif(
    os.environ.get("SLUCM_GIS_TESTS") != "1",
    reason="Skipped unless SLUCM_GIS_TESTS=1 (GIS deps not installed)",
)


@skip_unless_gis
def test_abort_if_not_utm_epsg_4326_raises():
    """_abort_if_not_utm("EPSG:4326") raises SystemExit."""
    from ucm_from_gis import _abort_if_not_utm
    with pytest.raises(SystemExit) as exc_info:
        _abort_if_not_utm("EPSG:4326")
    assert "UTM" in str(exc_info.value)


@skip_unless_gis
def test_abort_if_not_utm_epsg_3857_raises():
    """_abort_if_not_utm("EPSG:3857") (Web Mercator) raises."""
    from ucm_from_gis import _abort_if_not_utm
    with pytest.raises(SystemExit):
        _abort_if_not_utm("EPSG:3857")


@skip_unless_gis
def test_abort_if_not_utm_epsg_32619_returns_crs():
    """_abort_if_not_utm("EPSG:32619") returns a CRS object."""
    from ucm_from_gis import _abort_if_not_utm
    from pyproj import CRS
    crs = _abort_if_not_utm("EPSG:32619")
    assert isinstance(crs, CRS)
    assert crs.to_epsg() == 32619


@skip_unless_gis
def test_suggest_utm_zone_epsg_boston():
    """suggest_utm_zone_epsg(-71.06, 42.36) == 32619."""
    from ucm_from_gis import suggest_utm_zone_epsg
    epsg = suggest_utm_zone_epsg(-71.06, 42.36)
    assert epsg == 32619


@skip_unless_gis
def test_suggest_utm_zone_epsg_sao_paulo():
    """suggest_utm_zone_epsg(-46.63, -23.55) == 32723."""
    from ucm_from_gis import suggest_utm_zone_epsg
    epsg = suggest_utm_zone_epsg(-46.63, -23.55)
    assert epsg == 32723


@skip_unless_gis
def test_abort_if_not_utm_message_contains_fix():
    """_abort_if_not_utm error message contains fix-it instructions."""
    from ucm_from_gis import _abort_if_not_utm
    with pytest.raises(SystemExit) as exc_info:
        _abort_if_not_utm("EPSG:4326")
    message = str(exc_info.value)
    assert "HOW TO FIX" in message or "how to fix" in message.lower()
    assert "zone" in message.lower()
