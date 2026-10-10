# ERF-Hazard Visualisation Scripts

Two Python scripts for visualising ERF-Hazard canonical test outputs.
No C++ build required — run directly after an ERF-Hazard simulation.

---

## Scripts

| Script | Input | Requires |
|---|---|---|
| `plot_hazard_fields.py` | AMReX plotfile directory (`plt_1_NNNNN`) | `yt`, `matplotlib`, `numpy` |
| `plot_hazard_timeseries.py` | Plain-text CSV/DAT diagnostic files | `matplotlib`, `pandas`, `numpy` |

Install dependencies:
```bash
pip install yt matplotlib pandas numpy
```

---

## plot_hazard_fields.py — AMReX Plotfile Figures

Reads ERF plotfiles and produces PNG slice plots.

```bash
# Smoke plume + terrain + wind for HaboobFireHill
python plot_hazard_fields.py --plotdir path/to/plt_1_00020 --case HaboobFireHill

# Dust emission asymmetry on Gaussian hill
python plot_hazard_fields.py --plotdir path/to/plt_1_00020 --case DustGaussianHill

# Wind in the x-z plane across the Gaussian pit
python plot_hazard_fields.py --plotdir path/to/plt_1_00020 --case HaboobFirePit
```

### Output files per case

| Case | Outputs |
|---|---|
| `HaboobFireHill` | `_smoke_plan.png`, `_smoke_xz.png`, `_wind_sfc.png`, `_dust_emission.png`, `_dust_xz.png`, `_theta_xz.png` |
| `HaboobFireFlat` | Same as HaboobFireHill |
| `HaboobFirePit` | `_wind_recirculation.png`, `_dust_emission.png`, `_smoke_xz.png`, `_dust_xz.png`, `_theta_xz.png` |
| `DustGaussianHill` | `_dust_emission.png`, `_wind_sfc.png`, `_dust_xz.png` |
| `DustGaussianPit` | `_wind_recirculation.png`, `_dust_emission.png`, `_dust_xz.png` |

`dust_emission_flux` lives on the dust grid, so `_dust_emission.png` is made
only from a dust plotfile (`--plotdir path/to/plt_dust_NNNNN`); the other
figures need the atmosphere plotfile (`plt_1_NNNNN`). A figure whose field
is missing is skipped with a message. The slices are in computational
coordinates, so terrain cases are drawn on the unmapped mesh.

### Required plotfile fields

Make sure the ERF `inputs` file includes these in `erf.plot_vars_1`:
```
erf.plot_vars_1 = density x_velocity y_velocity z_velocity theta smoke rhoadv_dust
```
Smoke (`smoke`, fire builds) and dust (`rhoadv_dust`, dust builds) are written
only when listed, as mass concentrations in kg/m³, and `erf.plot_int_1` (or
`erf.plot_per_1`) must be positive for any plotfile to be written.

---

## plot_hazard_timeseries.py — Diagnostic Time Series

Reads `dust_diag.dat` (plain CSV written by the dust module, no yt needed).

### Terrain amplification comparison
```bash
python plot_hazard_timeseries.py --mode terrain_amplification \
    --flat  HaboobFireFlat/dust_diag.dat \
    --hill  HaboobFireHill/dust_diag.dat \
    --pit   HaboobFirePit/dust_diag.dat
# Output: terrain_amplification.png
```

### Single dust diagnostic
```bash
python plot_hazard_timeseries.py --mode dust_diag \
    --file HaboobFireHill/dust_diag.dat \
    --label "HaboobFireHill"
# Output: dust_diag_HaboobFireHill.png
```

### Smoke diagnostic

No ERF output writes `smoke_diag.dat`; this mode plots a CSV you write
yourself with a `time_s` column and any of `smoke_src_max`,
`smoke_conc_max_k0` and `smoke_total_mass`.
```bash
python plot_hazard_timeseries.py --mode smoke_diag \
    --file HaboobFireHill/smoke_diag.dat \
    --label "HaboobFireHill"
# Output: smoke_diag_HaboobFireHill.png
```

### Fire-dust coupling interaction contributions
```bash
python plot_hazard_timeseries.py --mode coupling \
    --baseline     FireDustBaseline/dust_diag.dat \
    --interaction1 FireDustInteraction1/dust_diag.dat \
    --interaction2 FireDustInteraction2/dust_diag.dat \
    --interaction3 FireDustInteraction3/dust_diag.dat \
    --all          FireDustInteractions123/dust_diag.dat
# Output: fire_dust_coupling.png
```

---

## Terrain comparison

The terrain amplification plot compares the domain-total dust emission and
the maximum u* of `HaboobFireFlat`, `HaboobFireHill` and `HaboobFirePit`
from `dust_diag.dat` alone. In the full-length runs the time-averaged totals
of the hill and the pit are within 1 % and 7 % of the flat case's (step by
step the ratios range from 0.93 to 1.19): the terrain moves where dust is
raised more than how much, which the `_dust_emission.png` maps from
`plt_dust_*` plotfiles show.
