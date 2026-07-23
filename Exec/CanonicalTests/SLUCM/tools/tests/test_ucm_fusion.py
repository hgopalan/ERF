"""Minimal tests for ucm_fusion."""

import pytest

gpd = pytest.importorskip("geopandas")
_ = pytest.importorskip("shapely")

from shapely.geometry import box

from ucm_fusion import fuse_footprints


def test_fuse_footprints_prefers_osm():
    osm = gpd.GeoDataFrame({"name": ["osm"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
    microsoft = gpd.GeoDataFrame({"name": ["ms"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")

    fused = fuse_footprints(osm=osm, microsoft=microsoft)

    assert len(fused) == 1
    assert fused.iloc[0]["source"] == "OSM"
