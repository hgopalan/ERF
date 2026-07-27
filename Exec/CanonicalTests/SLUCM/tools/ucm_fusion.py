"""Footprint fusion helpers."""

from __future__ import annotations

try:
    import geopandas as gpd
except Exception:
    gpd = None

_PRIORITY = ("OSM", "MICROSOFT", "GOOGLE")


def _empty_gdf(crs=None):
    if gpd is None:
        return None
    return gpd.GeoDataFrame({"source": [], "geometry": []}, geometry="geometry", crs=crs)


def _prepare(gdf, source, target_crs):
    if gdf is None or gpd is None or len(gdf) == 0:
        return _empty_gdf(target_crs)
    out = gdf.copy()
    if out.crs is not None and target_crs is not None and str(out.crs) != str(target_crs):
        out = out.to_crs(target_crs)
    out = out[out.geometry.notnull()]
    out = out[~out.geometry.is_empty]
    out = out[out.geom_type.isin(["Polygon", "MultiPolygon"])]
    if "source" not in out.columns:
        out["source"] = source
    else:
        out["source"] = out["source"].fillna(source)
    return out.reset_index(drop=True)


def fuse_footprints(osm=None, microsoft=None, google=None, overlap_threshold: float = 0.7):
    """Fuse footprint layers with priority OSM > MICROSOFT > GOOGLE."""
    if gpd is None:
        print("[ucm_fusion] geopandas missing; returning None")
        return None

    first = next((g for g in (osm, microsoft, google) if g is not None and len(g) > 0), None)
    target_crs = None if first is None else first.crs

    ordered = [
        _prepare(osm, "OSM", target_crs),
        _prepare(microsoft, "MICROSOFT", target_crs),
        _prepare(google, "GOOGLE", target_crs),
    ]
    frames = [frame for frame in ordered if frame is not None and len(frame) > 0]
    if not frames:
        return _empty_gdf(target_crs)

    kept = []
    kept_geoms = []
    for frame in frames:
        for _, row in frame.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.area <= 0.0:
                continue
            duplicate = False
            for kept_geom in kept_geoms:
                intersection = geom.intersection(kept_geom).area
                denom = min(geom.area, kept_geom.area)
                if denom > 0.0 and intersection / denom >= overlap_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(row.drop(labels=["geometry"]).to_dict())
            kept_geoms.append(geom)

    if not kept:
        return _empty_gdf(target_crs)
    return gpd.GeoDataFrame(kept, geometry=kept_geoms, crs=target_crs).reset_index(drop=True)
