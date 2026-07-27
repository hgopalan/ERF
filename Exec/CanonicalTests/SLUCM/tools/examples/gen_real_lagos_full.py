#!/usr/bin/env python3
"""Pattern stub for a fuller Lagos workflow."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ucm_from_gis import build_ucm_from_location, suggest_utm_zone_epsg


def main():
    lon_c, lat_c = 3.3792, 6.5244
    epsg = suggest_utm_zone_epsg(lon_c, lat_c)
    half = 1200.0
    build_ucm_from_location(
        domain_bbox_m=(542000 - half, 721000 - half, 542000 + half, 721000 + half),
        crs=f"EPSG:{epsg}",
        ucm_dx_m=75.0,
        output_dir=".",
        footprints_source="osm",
        lcz_source="manual",
        lcz_manual_class=3,
    )


if __name__ == "__main__":
    main()
