"""ucm_endpoints.py — Single source of truth for remote data endpoint URLs.

Phase 2.9: Centralizes all remote data source URLs for SLUCM toolchain.
Supports override via ucm_endpoints_override.py placed alongside this module.

URLs verified: 2026-07
Maintainer: Update annually or when URLs change.
"""

# Primary endpoints (in preference order, tried left-to-right on failover)
OSM_OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]

MICROSOFT_ML = [
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/{country}.geojsonl.gz",
    "https://github.com/microsoft/GlobalMLBuildingFootprints/releases/latest/download/{country}.geojsonl.gz",
]

GOOGLE_OPEN_BUILDINGS = [
    "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_6_gzip/{s2_cell}_buildings.csv.gz",
]

COPERNICUS_GHS_BUILT_H = [
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0.tif",
]

WUDAPT_LCZ = [
    "https://lcz-generator.rub.de/api/city/{city_slug}/lcz.tif",
]

# Try to load user overrides
try:
    import importlib.util
    import os
    override_path = os.path.join(os.path.dirname(__file__), "ucm_endpoints_override.py")
    if os.path.exists(override_path):
        spec = importlib.util.spec_from_file_location("ucm_endpoints_override", override_path)
        override_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(override_module)
        
        # Apply overrides
        if hasattr(override_module, "OSM_OVERPASS"):
            OSM_OVERPASS = override_module.OSM_OVERPASS
        if hasattr(override_module, "MICROSOFT_ML"):
            MICROSOFT_ML = override_module.MICROSOFT_ML
        if hasattr(override_module, "GOOGLE_OPEN_BUILDINGS"):
            GOOGLE_OPEN_BUILDINGS = override_module.GOOGLE_OPEN_BUILDINGS
        if hasattr(override_module, "COPERNICUS_GHS_BUILT_H"):
            COPERNICUS_GHS_BUILT_H = override_module.COPERNICUS_GHS_BUILT_H
        if hasattr(override_module, "WUDAPT_LCZ"):
            WUDAPT_LCZ = override_module.WUDAPT_LCZ
except Exception as e:
    # Silently ignore override load failures
    pass
