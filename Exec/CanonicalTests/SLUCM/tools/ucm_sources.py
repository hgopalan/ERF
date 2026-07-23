"""Adapters for remote urban canopy data sources.

Each adapter uses :func:`ucm_fetch.fetch_with_failover` and degrades
gracefully by returning ``None`` on any failure.
"""

from __future__ import annotations

import gzip
import io
import json
import warnings
from typing import Iterable, Optional, Sequence
from urllib.parse import quote

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from rasterio.io import MemoryFile
except Exception:
    rasterio = None
    raster_shapes = None
    MemoryFile = None

try:
    from shapely.geometry import Polygon, shape
    from shapely import wkt as shapely_wkt
except Exception:
    Polygon = None
    shape = None
    shapely_wkt = None

from ucm_endpoints import (
    COPERNICUS_GHS_BUILT_H,
    GOOGLE_OPEN_BUILDINGS,
    MICROSOFT_ML,
    OSM_OVERPASS,
    WUDAPT_LCZ,
)
from ucm_fetch import FetchAllFailed, fetch_with_failover

_OSM_ATTRIBUTION = "© OpenStreetMap contributors"
_MICROSOFT_ATTRIBUTION = "Microsoft Global ML Building Footprints"
_GOOGLE_ATTRIBUTION = "Google Open Buildings"
_COPERNICUS_ATTRIBUTION = "Copernicus GHSL Built-H"
_WUDAPT_ATTRIBUTION = "WUDAPT LCZ"


def _empty_gdf(crs: Optional[str] = None):
    if gpd is None:
        return None
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _clip_to_bbox(gdf, domain_bbox_m):
    if gdf is None or len(gdf) == 0 or domain_bbox_m is None or Polygon is None:
        return gdf
    try:
        from shapely.geometry import box

        clipped = gdf.clip(box(*domain_bbox_m))
        return clipped.reset_index(drop=True)
    except Exception:
        return gdf


def _ensure_geospatial():
    return all(mod is not None for mod in (gpd, Polygon))


def _bbox_to_latlon(domain_bbox_m, crs) -> Optional[Sequence[float]]:
    if Transformer is None or domain_bbox_m is None or crs is None:
        return None
    try:
        x_min, y_min, x_max, y_max = domain_bbox_m
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon0, lat0 = transformer.transform(x_min, y_min)
        lon1, lat1 = transformer.transform(x_max, y_max)
        west, east = sorted((lon0, lon1))
        south, north = sorted((lat0, lat1))
        return south, west, north, east
    except Exception:
        return None


def _normalize_polygons(gdf, crs=None):
    if gdf is None or len(gdf) == 0:
        return gdf
    out = gdf.copy()
    if crs is not None and out.crs is not None and str(out.crs) != str(crs):
        out = out.to_crs(crs)
    out = out[out.geometry.notnull()]
    out = out[~out.geometry.is_empty]
    out = out[out.geom_type.isin(["Polygon", "MultiPolygon"])]
    out = out.reset_index(drop=True)
    return out


def _print_attribution(source: str, attribution: str) -> None:
    print(f"[ucm_sources] {source} attribution: {attribution}")


def fetch_osm_buildings(
    domain_bbox_m,
    crs,
    timeout: int = 60,
    max_retries: int = 0,
):
    """Fetch OSM building polygons inside a UTM bbox."""
    if not _ensure_geospatial():
        print("[ucm_sources] OSM unavailable: geopandas/shapely missing")
        return None
    latlon_bbox = _bbox_to_latlon(domain_bbox_m, crs)
    if latlon_bbox is None:
        print("[ucm_sources] OSM unavailable: could not convert bbox to EPSG:4326")
        return None
    try:
        south, west, north, east = latlon_bbox
        query = (
            "[out:json][timeout:60];"
            f'(way["building"]({south},{west},{north},{east});'
            f'relation["building"]({south},{west},{north},{east}););'
            "out geom;"
        )
        endpoints = [f"{endpoint}?data={quote(query, safe='')}" for endpoint in OSM_OVERPASS]
        payload = fetch_with_failover(
            endpoints=endpoints,
            source="OSM",
            timeout=timeout,
            max_retries=max_retries,
        )
        doc = json.loads(payload.decode("utf-8"))
        rows = []
        for element in doc.get("elements", []):
            coords = element.get("geometry") or []
            if len(coords) < 3:
                continue
            polygon = Polygon([(pt["lon"], pt["lat"]) for pt in coords])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            row = dict(element.get("tags") or {})
            row["osm_id"] = element.get("id")
            row["geometry"] = polygon
            rows.append(row)
        if not rows:
            print("[ucm_sources] OSM returned no building polygons")
            return None
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(crs)
        gdf = _clip_to_bbox(_normalize_polygons(gdf, crs=crs), domain_bbox_m)
        _print_attribution("OSM", _OSM_ATTRIBUTION)
        return gdf if len(gdf) else None
    except FetchAllFailed as exc:
        print(f"[ucm_sources] OSM failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ucm_sources] OSM parse failed: {exc}")
        return None


def _geojsonl_bytes_to_gdf(payload: bytes, crs=None):
    if not _ensure_geospatial():
        return None
    text = gzip.decompress(payload).decode("utf-8") if payload[:2] == b"\x1f\x8b" else payload.decode("utf-8")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        feature = json.loads(line)
        geometry = feature.get("geometry")
        if geometry is None or shape is None:
            continue
        row = dict(feature.get("properties") or {})
        row["geometry"] = shape(geometry)
        rows.append(row)
    if not rows:
        return None
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return _normalize_polygons(gdf, crs=crs)


def fetch_microsoft_ml_footprints(
    country: str,
    domain_bbox_m=None,
    crs=None,
    timeout: int = 60,
    max_retries: int = 0,
):
    """Fetch Microsoft ML building footprints for a country code."""
    if not _ensure_geospatial():
        print("[ucm_sources] MICROSOFT unavailable: geopandas/shapely missing")
        return None
    try:
        payload = fetch_with_failover(
            endpoints=MICROSOFT_ML,
            source="MICROSOFT",
            timeout=timeout,
            max_retries=max_retries,
            format_kwargs={"country": country.upper()},
        )
        gdf = _geojsonl_bytes_to_gdf(payload, crs=crs)
        gdf = _clip_to_bbox(gdf, domain_bbox_m)
        if gdf is None or len(gdf) == 0:
            print("[ucm_sources] MICROSOFT returned no usable building polygons")
            return None
        _print_attribution("MICROSOFT", _MICROSOFT_ATTRIBUTION)
        return gdf
    except FetchAllFailed as exc:
        print(f"[ucm_sources] MICROSOFT failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ucm_sources] MICROSOFT parse failed: {exc}")
        return None


def _google_csv_to_gdf(payload: bytes, crs=None):
    if not all(mod is not None for mod in (gpd, pd, shapely_wkt)):
        return None
    text = gzip.decompress(payload).decode("utf-8") if payload[:2] == b"\x1f\x8b" else payload.decode("utf-8")
    frame = pd.read_csv(io.StringIO(text))
    geometry_col = next(
        (col for col in ("geometry", "wkt", "polygon_wkt", "footprint_wkt") if col in frame.columns),
        None,
    )
    if geometry_col is None:
        return None
    rows = frame.copy()
    rows["geometry"] = rows[geometry_col].map(shapely_wkt.loads)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return _normalize_polygons(gdf, crs=crs)


def fetch_google_open_buildings(
    s2_cell: str,
    domain_bbox_m=None,
    crs=None,
    timeout: int = 60,
    max_retries: int = 0,
):
    """Fetch Google Open Buildings polygons for an S2 cell identifier."""
    if not _ensure_geospatial():
        print("[ucm_sources] GOOGLE unavailable: geopandas/shapely missing")
        return None
    try:
        payload = fetch_with_failover(
            endpoints=GOOGLE_OPEN_BUILDINGS,
            source="GOOGLE",
            timeout=timeout,
            max_retries=max_retries,
            format_kwargs={"s2_cell": str(s2_cell)},
        )
        gdf = _google_csv_to_gdf(payload, crs=crs)
        gdf = _clip_to_bbox(gdf, domain_bbox_m)
        if gdf is None or len(gdf) == 0:
            print("[ucm_sources] GOOGLE returned no usable building polygons")
            return None
        _print_attribution("GOOGLE", _GOOGLE_ATTRIBUTION)
        return gdf
    except FetchAllFailed as exc:
        print(f"[ucm_sources] GOOGLE failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ucm_sources] GOOGLE parse failed: {exc}")
        return None


def _raster_bytes_to_polygons(
    payload: bytes,
    value_name: str,
    domain_bbox_m=None,
    crs=None,
    max_features: int = 5000,
):
    if not all(mod is not None for mod in (rasterio, raster_shapes, MemoryFile, gpd)):
        return None
    with MemoryFile(payload) as memfile:
        with memfile.open() as src:
            band = src.read(1, masked=True)
            rows = []
            for geom, value in raster_shapes(band.filled(0), mask=~band.mask, transform=src.transform):
                if len(rows) >= max_features:
                    warnings.warn(f"Raster polygonization truncated at {max_features} features")
                    break
                if value is None:
                    continue
                value = float(value)
                if value <= 0.0:
                    continue
                rows.append({value_name: value, "geometry": shape(geom)})
            if not rows:
                return None
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=src.crs)
    gdf = _normalize_polygons(gdf, crs=crs)
    return _clip_to_bbox(gdf, domain_bbox_m)


def fetch_copernicus_ghs_built_h(
    domain_bbox_m=None,
    crs=None,
    timeout: int = 120,
    max_retries: int = 0,
):
    """Fetch Copernicus GHSL Built-H and polygonize positive-height pixels."""
    try:
        payload = fetch_with_failover(
            endpoints=COPERNICUS_GHS_BUILT_H,
            source="COPERNICUS_GHS_BUILT_H",
            timeout=timeout,
            max_retries=max_retries,
        )
        gdf = _raster_bytes_to_polygons(payload, "height_m", domain_bbox_m=domain_bbox_m, crs=crs)
        if gdf is None or len(gdf) == 0:
            print("[ucm_sources] COPERNICUS returned no usable polygons")
            return None
        _print_attribution("COPERNICUS", _COPERNICUS_ATTRIBUTION)
        return gdf
    except FetchAllFailed as exc:
        print(f"[ucm_sources] COPERNICUS failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ucm_sources] COPERNICUS parse failed: {exc}")
        return None


def fetch_wudapt_lcz(
    city_slug: str,
    domain_bbox_m=None,
    crs=None,
    timeout: int = 120,
    max_retries: int = 0,
):
    """Fetch WUDAPT LCZ raster and polygonize positive LCZ classes."""
    try:
        payload = fetch_with_failover(
            endpoints=WUDAPT_LCZ,
            source="WUDAPT",
            timeout=timeout,
            max_retries=max_retries,
            format_kwargs={"city_slug": city_slug},
        )
        gdf = _raster_bytes_to_polygons(payload, "lcz", domain_bbox_m=domain_bbox_m, crs=crs)
        if gdf is None or len(gdf) == 0:
            print("[ucm_sources] WUDAPT returned no usable polygons")
            return None
        _print_attribution("WUDAPT", _WUDAPT_ATTRIBUTION)
        return gdf
    except FetchAllFailed as exc:
        print(f"[ucm_sources] WUDAPT failed: {exc}")
        return None
    except Exception as exc:
        print(f"[ucm_sources] WUDAPT parse failed: {exc}")
        return None
