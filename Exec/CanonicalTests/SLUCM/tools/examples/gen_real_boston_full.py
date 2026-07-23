#!/usr/bin/env python3
"""Pattern stub for a fuller Boston workflow."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ucm_from_gis import build_ucm_from_location, suggest_utm_zone_epsg


def main():
    lon_c, lat_c = -71.06, 42.36
    epsg = suggest_utm_zone_epsg(lon_c, lat_c)
    half = 1200.0
    build_ucm_from_location(
        domain_bbox_m=(330000 - half, 4691000 - half, 330000 + half, 4691000 + half),
        crs=f"EPSG:{epsg}",
        ucm_dx_m=75.0,
        output_dir=".",
        footprints_source="osm",
        lcz_source="manual",
        lcz_manual_class=2,
    )


if __name__ == "__main__":
    main()
