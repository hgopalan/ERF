"""Plot ERF-SLUCM CSV products."""

from __future__ import annotations

import argparse
import os
import warnings

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

try:
    import folium
except Exception:
    folium = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None


def _grid(df, column, nx, ny, fill_value=0.0):
    array = np.full((ny, nx), fill_value, dtype=float)
    for _, row in df.iterrows():
        array[int(row["j"]), int(row["i"])] = float(row[column])
    return array


def _warn_anomalies(df):
    if ((df["plan_area_frac"] < 0.0) | (df["plan_area_frac"] > 1.0)).any():
        warnings.warn("plan_area_frac outside [0, 1] detected")
    if (df["height_m"] < 0.0).any():
        warnings.warn("Negative building heights detected")
    if "AH_Wm2" in df.columns and (df["AH_Wm2"] > 500.0).any():
        warnings.warn("AH_Wm2 above 500 W/m² detected")
    if ((df["is_urban"] == 1) & (df["roof_mat_id"] < 1)).any():
        warnings.warn("Urban cells with invalid material IDs detected")


def _save_panel(array, title, path, cmap="viridis", discrete=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(array, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    if discrete:
        fig.colorbar(image, ax=ax, shrink=0.8)
    else:
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_optional_folium(output_dir, panels):
    if folium is None:
        warnings.warn("folium not installed; skipping HTML map")
        return None
    fmap = folium.Map(location=[0.5, 0.5], zoom_start=2, tiles="CartoDB positron")
    html = ["<h3>ERF-SLUCM plot summary</h3><ul>"]
    for name in panels:
        html.append(f"<li>{name}.png</li>")
    html.append("</ul>")
    folium.Marker([0.5, 0.5], popup="".join(html)).add_to(fmap)
    path = os.path.join(output_dir, "summary_map.html")
    fmap.save(path)
    return path


def plot_all(layout_csv, materials_csv, output_dir, folium_html: bool = False):
    """Generate six PNG summaries from layout/materials CSV files."""
    if any(mod is None for mod in (np, pd, plt)):
        warnings.warn("numpy/pandas unavailable; cannot generate plots")
        return []

    os.makedirs(output_dir, exist_ok=True)
    layout = pd.read_csv(layout_csv)
    _ = pd.read_csv(materials_csv)
    nx = int(layout["i"].max()) + 1
    ny = int(layout["j"].max()) + 1

    _warn_anomalies(layout)

    if "AH_Wm2" not in layout.columns:
        warnings.warn("AH_Wm2 column missing; plotting zeros")
        layout["AH_Wm2"] = 0.0

    denom = layout["W_road_m"].astype(float) + layout["W_roof_m"].astype(float)
    lambda_f = np.where(denom > 0.0, layout["height_m"].astype(float) / denom, 0.0)
    layout = layout.copy()
    layout["lambda_f"] = lambda_f

    panels = {
        "urban_mask": (_grid(layout, "is_urban", nx, ny), "gray"),
        "height": (_grid(layout, "height_m", nx, ny), "viridis"),
        "plan_area_frac": (_grid(layout, "plan_area_frac", nx, ny), "magma"),
        "lambda_f": (_grid(layout, "lambda_f", nx, ny), "plasma"),
        "materials": (_grid(layout, "roof_mat_id", nx, ny), "tab20"),
        "AH_Wm2": (_grid(layout, "AH_Wm2", nx, ny), "inferno"),
    }

    outputs = []
    for name, (array, cmap) in panels.items():
        path = os.path.join(output_dir, f"{name}.png")
        _save_panel(array, name, path, cmap=cmap, discrete=name in {"urban_mask", "materials"})
        outputs.append(path)

    if folium_html:
        html_path = _write_optional_folium(output_dir, list(panels))
        if html_path:
            outputs.append(html_path)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot ERF-SLUCM layout/material CSVs")
    parser.add_argument("layout_csv")
    parser.add_argument("materials_csv")
    parser.add_argument("--output", default="plots")
    parser.add_argument("--folium-html", action="store_true")
    args = parser.parse_args(argv)
    plot_all(args.layout_csv, args.materials_csv, args.output, folium_html=args.folium_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
