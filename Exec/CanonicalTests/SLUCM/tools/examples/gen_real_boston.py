#!/usr/bin/env python3
"""Generate CSV for a 2.4 km x 2.4 km domain centered on downtown Boston.

This example demonstrates real-city ingestion from OpenStreetMap.
Requires: pip install -r ../requirements-gis.txt

Usage:
  python3 gen_real_boston.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ucm_from_gis import build_ucm_from_location, suggest_utm_zone_epsg

lon_c, lat_c = -71.06, 42.36
epsg = suggest_utm_zone_epsg(lon_c, lat_c)
print(f"[gen_real_boston] Using UTM EPSG:{epsg}")

half = 1200.0
build_ucm_from_location(
    domain_bbox_m=(330000 - half, 4691000 - half,
                   330000 + half, 4691000 + half),
    crs=f"EPSG:{epsg}",
    ucm_dx_m=75.0,
    output_dir=".",
    lcz_source="manual",
    lcz_manual_class=2,
)
