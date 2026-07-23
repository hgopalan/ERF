"""Utilities for extracting anthropogenic heat from LUCY NetCDF output."""

from __future__ import annotations

import warnings

try:
    import numpy as np
except Exception:
    np = None

try:
    import xarray as xr
except Exception:
    xr = None

_SEASONS = {
    "djf": (12, 1, 2),
    "mam": (3, 4, 5),
    "jja": (6, 7, 8),
    "son": (9, 10, 11),
}


def _detect_variable(ds, variable_name=None):
    if variable_name and variable_name in ds.data_vars:
        return variable_name
    candidates = [
        "QF",
        "qf",
        "anthropogenic_heat",
        "anthropogenic_heat_flux",
        "AH_Wm2",
        "total_anthropogenic_heat_flux",
    ]
    for name in candidates:
        if name in ds.data_vars:
            return name
    for name, data in ds.data_vars.items():
        long_name = str(data.attrs.get("long_name", "")).lower()
        std_name = str(data.attrs.get("standard_name", "")).lower()
        if "anthropogenic" in long_name or "anthropogenic" in std_name or name.lower().endswith("qf"):
            return name
    return None


def _detect_time_dim(ds, data_array):
    for dim in data_array.dims:
        if dim.lower() == "time":
            return dim
        coord = ds.coords.get(dim)
        if coord is not None and getattr(coord.dtype, "kind", "") == "M":
            return dim
    for name, coord in ds.coords.items():
        if name.lower() == "time" and name in data_array.dims:
            return name
    return None


def _detect_crs(ds, data_array):
    for key in ("crs", "spatial_ref", "grid_mapping"):
        value = data_array.attrs.get(key, ds.attrs.get(key))
        if value:
            return value
    spatial_ref = ds.get("spatial_ref")
    if spatial_ref is not None:
        return spatial_ref.attrs.get("spatial_ref") or spatial_ref.attrs.get("crs_wkt")
    return None


def _apply_time_filters(data_array, time_coord, season=None, hour_window=None):
    filtered = data_array
    if season and str(season).lower() not in ("annual", "all"):
        months = _SEASONS.get(str(season).lower())
        if months is None:
            raise ValueError(f"Unsupported season: {season}")
        filtered = filtered.where(time_coord.dt.month.isin(months), drop=True)
    if hour_window is not None:
        start_hour, end_hour = hour_window
        hours = time_coord.dt.hour
        if start_hour <= end_hour:
            mask = (hours >= start_hour) & (hours < end_hour)
        else:
            mask = (hours >= start_hour) | (hours < end_hour)
        filtered = filtered.where(mask, drop=True)
    return filtered


def _convert_to_wm2(values, units: str):
    units_norm = (units or "W m-2").strip().lower()
    if units_norm in {"w/m2", "w m-2", "w m^-2", "wm-2", "w m**-2"}:
        return values
    if units_norm in {"kw/m2", "kw m-2", "kw m^-2"}:
        return values * 1000.0
    if units_norm in {"mw/km2", "mw km-2", "mw km^-2"}:
        return values
    warnings.warn(f"Unknown AH units '{units}'; assuming W/m²")
    return values


def parse_lucy_qf_netcdf(path, season: str = "annual", hour_window=None, variable_name: str = None):
    """Read a LUCY NetCDF file and return a 2-D mean AH field in W/m²."""
    if xr is None or np is None:
        warnings.warn("xarray/numpy unavailable; cannot parse LUCY NetCDF")
        return None
    try:
        with xr.open_dataset(path) as ds:
            var_name = _detect_variable(ds, variable_name=variable_name)
            if var_name is None:
                warnings.warn(f"No AH-like variable found in {path}")
                return None
            data = ds[var_name]
            time_dim = _detect_time_dim(ds, data)
            crs = _detect_crs(ds, data)
            if crs:
                print(f"[ucm_ah_from_lucy] detected CRS: {crs}")
            if time_dim is not None:
                time_coord = ds[time_dim]
                filtered = _apply_time_filters(data, time_coord, season=season, hour_window=hour_window)
                if filtered.sizes.get(time_dim, 0) == 0:
                    warnings.warn("No time slices matched requested averaging window")
                    return None
                data = filtered.mean(dim=time_dim, skipna=True)
            values = np.asarray(data.squeeze(), dtype=float)
            while values.ndim > 2 and 1 in values.shape:
                values = np.squeeze(values)
            if values.ndim != 2:
                warnings.warn(f"Expected 2-D AH field after averaging; got shape {values.shape}")
                return None
            values = _convert_to_wm2(values, str(data.attrs.get("units", "")))
            if np.nanmax(values) > 500.0:
                warnings.warn("Clamping AH values above 500 W/m²")
                values = np.where(values > 500.0, 500.0, values)
            return np.where(np.isfinite(values), values, 0.0)
    except Exception as exc:
        warnings.warn(f"Failed to parse {path}: {exc}")
        return None
