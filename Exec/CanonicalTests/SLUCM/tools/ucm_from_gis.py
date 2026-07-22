"""ucm_from_gis.py — real-city UCM CSV generation from OSM + WUDAPT LCZ.

REQUIREMENTS
  pip install -r requirements-gis.txt
  Domain CRS MUST be UTM (EPSG:326xx northern hemisphere, 327xx southern).
  Passing a geographic CRS (EPSG:4326 lat/lon) or a non-UTM projected CRS
  will abort with a message telling you how to fix it.

WHY UTM
  The UCM grid is a Cartesian mesh in meters. All morphology aggregates
  (plan area fraction, mean building height, frontal area) are area-weighted
  in meters. Lat/lon degrees are NOT constant-length; using them silently
  produces area errors of 10–30% at mid-latitudes and much worse at high
  latitudes. UTM zones are constant-metric within the zone and are the
  correct choice for a domain up to ~500 km wide.
"""
from pyproj import CRS
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import csv
from ucm_csv import write_layout, write_materials

UTM_NORTH_EPSG_RANGE = range(32601, 32661)
UTM_SOUTH_EPSG_RANGE = range(32701, 32761)
_UTM_EPSGS = set(UTM_NORTH_EPSG_RANGE) | set(UTM_SOUTH_EPSG_RANGE)


def _abort_if_not_utm(crs_input) -> CRS:
    """Validate CRS is UTM. Abort with a fix-it message if not.

    Args:
        crs_input: CRS specification (string, int, or CRS object).

    Returns:
        Validated CRS object.

    Raises:
        SystemExit: If CRS is not UTM.
    """
    crs = CRS.from_user_input(crs_input)
    epsg = crs.to_epsg()
    if epsg in _UTM_EPSGS:
        return crs
    msg = [
        "",
        "=" * 72,
        "[ucm_from_gis] ABORTING: CRS is not UTM.",
        "=" * 72,
        f"  You passed CRS: {crs.name} (EPSG:{epsg})",
        "  The UCM grid is Cartesian in meters, so the domain MUST be "
        "in UTM.",
        "",
        "  HOW TO FIX:",
        "    1. Determine your domain's UTM zone. Rule of thumb:",
        "         zone = int((longitude + 180) / 6) + 1",
        "         For Boston (long -71): zone 19. EPSG = 32619 (N hemi).",
        "         For Singapore (long 104): zone 48. EPSG = 32648 (N).",
        "         For Sao Paulo (long -46, S): zone 23. EPSG = 32723 (S).",
        "    2. Pass crs='EPSG:326XX' (N hemi) or 'EPSG:327XX' (S hemi).",
        "    3. If you passed a lat/lon bbox, reproject it first:",
        "         from pyproj import Transformer",
        "         t = Transformer.from_crs('EPSG:4326', "
        "'EPSG:32619', always_xy=True)",
        "         x_min, y_min = t.transform(lon_min, lat_min)",
        "         x_max, y_max = t.transform(lon_max, lat_max)",
        "",
        "  See https://en.wikipedia.org/wiki/"
        "Universal_Transverse_Mercator_coordinate_system",
        "=" * 72,
        "",
    ]
    raise SystemExit("\n".join(msg))


def suggest_utm_zone_epsg(lon_center: float, lat_center: float) -> int:
    """Return the EPSG code of the UTM zone containing the given lat/lon.

    Args:
        lon_center: Longitude in degrees (-180 to 180).
        lat_center: Latitude in degrees (-90 to 90).

    Returns:
        EPSG code for the UTM zone.
    """
    zone = int((lon_center + 180.0) / 6.0) + 1
    if lat_center >= 0.0:
        return 32600 + zone
    return 32700 + zone


def _load_materials_lcz(csv_path: str) -> dict:
    """Load materials_lcz.csv into a dict keyed by mat_id.

    Args:
        csv_path: Path to materials_lcz.csv.

    Returns:
        Dict mapping mat_id to material properties dict.
    """
    materials = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat_id = int(row["mat_id"])
            materials[mat_id] = row
    return materials


def build_ucm_from_location(
    domain_bbox_m,
    crs,
    ucm_dx_m,
    output_dir=".",
    footprints_source="osm",
    lcz_source="wudapt",
    lcz_manual_class=6,
) -> None:
    """Build building_layout.csv + materials.csv for a real city.

    Steps:
      1. Validate CRS is UTM (aborts if not).
      2. Compute UCM grid (nx, ny) from bbox and ucm_dx_m.
      3. Fetch OSM building footprints (osmnx) within the bbox.
      4. For each UCM cell:
         - plan_area_frac = sum(footprint intersection area) / cell_area
         - H_bldg = area-weighted mean of building:height (fallback:
                    building:levels * 3.0; final fallback: 0)
         - W_road, W_roof approximated from cell size and plan_frac
      5. Fetch LCZ raster (WUDAPT) OR use manual LCZ class.
      6. For each cell, look up material properties from
         materials_lcz.csv.
      7. Write both CSVs via ucm_csv.write_layout / write_materials.

    Args:
        domain_bbox_m: (x_min, y_min, x_max, y_max) in UTM meters.
        crs: UTM CRS, e.g., "EPSG:32619".
        ucm_dx_m: UCM cell size in meters.
        output_dir: Output directory for CSVs (default ".").
        footprints_source: "osm" (only source in Phase 2.9).
        lcz_source: "wudapt" or "manual".
        lcz_manual_class: LCZ class (1-10) if lcz_source=="manual".
    """
    crs = _abort_if_not_utm(crs)
    print(f"[ucm_from_gis] CRS validated: {crs.name}")

    x_min, y_min, x_max, y_max = domain_bbox_m
    print(f"[ucm_from_gis] domain bbox (UTM): "
          f"({x_min}, {y_min}) to ({x_max}, {y_max})")

    nx_ucm = max(1, int(np.round((x_max - x_min) / ucm_dx_m)))
    ny_ucm = max(1, int(np.round((y_max - y_min) / ucm_dx_m)))
    cell_area_m2 = ucm_dx_m * ucm_dx_m
    print(f"[ucm_from_gis] UCM grid: {nx_ucm} x {ny_ucm} cells "
          f"(dx={ucm_dx_m:.1f} m)")

    if footprints_source == "osm":
        print("[ucm_from_gis] fetching OSM buildings (this may take a minute)...")
        try:
            import osmnx as ox
            tags = {"building": True}
            buildings = ox.features_from_bbox(
                north=y_max, south=y_min, east=x_max, west=x_min, tags=tags
            )
            buildings = gpd.GeoDataFrame(buildings, crs="EPSG:4326")
            buildings = buildings[buildings.geom_type == "Polygon"]
            buildings = buildings.to_crs(crs)
            print(f"[ucm_from_gis] fetched {len(buildings)} OSM buildings")
        except Exception as e:
            print(f"[ucm_from_gis] WARNING: OSM fetch failed ({e}). "
                  f"Using zero buildings.")
            buildings = gpd.GeoDataFrame(geometry=[], crs=crs)
    else:
        raise ValueError(f"Unknown footprints_source: {footprints_source}")

    print("[ucm_from_gis] rasterizing buildings into UCM grid...")
    morphology = {}
    for j in range(ny_ucm):
        for i in range(nx_ucm):
            x0 = x_min + i * ucm_dx_m
            y0 = y_min + j * ucm_dx_m
            x1 = x0 + ucm_dx_m
            y1 = y0 + ucm_dx_m

            from shapely.geometry import box
            cell_box = box(x0, y0, x1, y1)

            footprint_area = 0.0
            total_height = 0.0
            num_bldgs = 0

            for _, bldg in buildings.iterrows():
                geom = bldg.geometry
                if geom.intersects(cell_box):
                    intersection = geom.intersection(cell_box)
                    area = intersection.area
                    footprint_area += area

                    h = 0.0
                    if "building:height" in bldg:
                        try:
                            h_str = str(bldg["building:height"])
                            h = float(h_str.replace(" m", "").strip())
                        except (ValueError, AttributeError):
                            pass
                    if h <= 0.0 and "building:levels" in bldg:
                        try:
                            levels = int(bldg["building:levels"])
                            h = levels * 3.0
                        except (ValueError, TypeError):
                            pass

                    if h > 0.0:
                        total_height += h * area
                        num_bldgs += 1

            plan_area_frac = footprint_area / cell_area_m2
            plan_area_frac = min(1.0, max(0.0, plan_area_frac))

            if num_bldgs > 0 and footprint_area > 0.0:
                height_m = total_height / footprint_area
            else:
                height_m = 0.0

            morphology[(i, j)] = {
                "plan_area_frac": plan_area_frac,
                "height_m": height_m,
            }

    print(f"[ucm_from_gis] morphology rasterization complete")

    if lcz_source == "manual":
        print(f"[ucm_from_gis] using manual LCZ class {lcz_manual_class}")
        lcz_map = {(i, j): lcz_manual_class for i in range(nx_ucm)
                   for j in range(ny_ucm)}
    elif lcz_source == "wudapt":
        print("[ucm_from_gis] TODO: WUDAPT LCZ raster download not yet "
              "implemented in Phase 2.9. Using manual fallback LCZ=6.")
        lcz_map = {(i, j): 6 for i in range(nx_ucm) for j in range(ny_ucm)}
    else:
        raise ValueError(f"Unknown lcz_source: {lcz_source}")

    materials_lcz_path = os.path.join(
        os.path.dirname(__file__), "materials_lcz.csv"
    )
    materials_dict = _load_materials_lcz(materials_lcz_path)
    print(f"[ucm_from_gis] loaded {len(materials_dict)} materials from "
          f"{materials_lcz_path}")

    print("[ucm_from_gis] generating cell function...")

    def cell_fn(i, j):
        plan_frac = morphology[(i, j)]["plan_area_frac"]
        h_bldg = morphology[(i, j)]["height_m"]
        lcz_class = lcz_map[(i, j)]

        mat_id = lcz_class
        if mat_id not in materials_dict:
            print(f"[ucm_from_gis] WARNING: LCZ {lcz_class} not in "
                  f"materials_lcz.csv, using mat_id=1")
            mat_id = 1

        is_urban = 0 if plan_frac < 0.05 or lcz_class > 10 else 1

        W_road_m = ucm_dx_m * (1.0 - np.sqrt(plan_frac))
        W_roof_m = ucm_dx_m * np.sqrt(plan_frac)

        return dict(
            i=i, j=j, bldg_id=1, height_m=max(0.0, h_bldg),
            plan_area_frac=plan_frac,
            W_road_m=W_road_m, W_roof_m=W_roof_m,
            roof_mat_id=mat_id, wall_mat_id=mat_id, road_mat_id=mat_id,
            orientation_deg=0.0, ah_profile_id=0, is_urban=is_urban,
        )

    layout_path = os.path.join(output_dir, "building_layout.csv")
    write_layout(layout_path, nx_ucm, ny_ucm, cell_fn)

    materials_list = [materials_dict[m] for m in sorted(materials_dict.keys())]
    materials_path = os.path.join(output_dir, "materials.csv")
    write_materials(materials_path, materials_list)

    print(f"[ucm_from_gis] complete. CSVs written to {output_dir}/")
