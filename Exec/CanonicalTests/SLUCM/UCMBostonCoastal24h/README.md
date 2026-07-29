# Phase 5.7 Coastal Sea-Breeze / Land-Breeze Canonical

## Overview

This canonical exercises the **full Phase 5.1–5.5 physics stack** (multi-facet radiation, HVAC waste heat, green roofs, permeable roads, and fractional urban coverage blending) inside a single physically meaningful **24-hour diurnal integration** on a mixed land/sea/urban domain.

**Scientific target:** The land–sea breeze reversal cycle, a well-known coastal circulation where:
1. **Sea breeze** develops in late morning as land heats faster than the ocean → warm rising air inland → surface flow from sea to land (onshore)
2. **Peak** in afternoon with sustained onshore winds ≥2 m/s
3. **Reversal** after midnight into weaker land breeze as land cools faster than the ocean → warm rising air offshore → surface flow from land to sea (offshore)
4. **UHI intensification** of the daytime sea-breeze front via urban heating

## Domain Layout

**Grid:** 20 km × 20 km × 1280 m (unchanged Boston scale)
- **ATM:** 20×20×64 cells (Δx_ATM = 1000 m)
- **UCM:** 80×80 cells (Δx_UCM = 250 m)
- **Refinement ratio:** grid_ratio = 4

**Land-cover regions (x, meters):**

| x range (m) | Region | Description | is_urban | Material |
|---|---|---|---|---|
| 0–5000 | Sea | Open water, high thermal inertia | 0 | sea_water (k=0.6 W/mK, ρc=4.2e6 J/m³K) |
| 5000–6000 | Coast transition | Jagged checkerboard blend of sea/urban cells (Phase 5.6 recipe) | mixed | sea_water + urban |
| 6000–14000 | Urban Boston | Dense urban with 15 m buildings, 5% plan area fraction | 1 | brick_concrete |
| 14000–15000 | Rural-urban transition | Jagged checkerboard blend of urban/rural cells | mixed | urban + grassland |
| 15000–20000 | Rural inland | Grassland with minimal buildings | 0 | grassland_rural |

The **transition bands** (coast and rural-urban) use a `(i+j) % 2` checkerboard pattern at UCM resolution (250 m cells), creating **genuine sub-grid fractional f_urb** in the ~4 ATM cells that span each band. This requires the Phase 5.6 `interface_mode=blended` to blend MOST and UCM fluxes via f_urb weighting, not a sharp binary mask.

## Physics

### Enabled Features

- **Phase 5.1a:** View-factor precomputation (Hottel crossed-string for SW/LW radiosity)
- **Phase 5.1b:** Radiosity solver for shortwave multi-bounce
- **Phase 5.1c:** Longwave multi-bounce radiosity
- **Phase 5.2:** AC waste heat rejection (COP-based simple model) via `hvac_boston.csv` occupancy schedule
- **Phase 5.3:** Green roof / cool roof / permeable pavement (optional via materials.csv)
- **Phase 5.5:** HVAC extended physics (sensible/latent split, COP degradation)
- **Phase 5.6:** Fractional urban coverage blending (`interface_mode=blended`)

### Prescribed Forcing

**Radiation** (Phase 3.5b infrastructure):
- Analytic clear-sky SW/LW based on solar geometry (latitude 42.36°N, longitude -71.06°W, Boston)
- Julian day 172 (summer solstice, June 21)
- Prescribed direct/diffuse split and cloud-free atmosphere
- No cloud effects (steady state for reproducibility)

**Geostrophic wind:**
- Weak background u_geo = 2 m/s to avoid overwhelming local circulation
- v_geo = 0 (allows N–S coastal circulation)

**Initial/boundary conditions:**
- Quiescent ATM: T_atm = 293.15 K (uniform)
- Ocean: T_skin = 289 K (cool, isothermal, high thermal inertia → ±1 K diurnal swing)
- Land: T_skin = 293 K (matches ATM, warmer than ocean)
- Outflow east/west; periodic north/south

## Input Files

### `inputs_coastal_binary`
Control case using `interface_mode=binary` (sharp is_urban mask, no blending).
Demonstrates that sea-breeze reversal occurs even with binary treatment (physics-dominant signal).

### `inputs_coastal_blended`
Production case using `interface_mode=blended` (Phase 5.6 fractional f_urb blending).
Should show sharpened sea-breeze front and earlier onset due to coastal urban UHI effect.

## Verification

### Post-processing Verifier

**`check_coastal_breeze.py`** loads hourly plotfiles and asserts four criteria:

1. **Sea-breeze onset:** u_x ≥ 1 m/s onshore (positive u) near hour 11–14 local time
2. **Sea-breeze peak:** u_x ≥ 2 m/s sustained for ≥ 2 hours around hours 15–17 local time
3. **Land-breeze reversal:** u_x ≤ -0.5 m/s offshore (negative u) between hours 01–05 local time
4. **Nighttime UHI:** T_urban − T_rural ≥ 2 K averaged over hours 02–05 local time

**Usage:**
```bash
python3 check_coastal_breeze.py --plt_dir blended_plts/
```

### Expected Results

**Blended mode (production):**
```
[PASS] Phase 5.7 coastal canonical
  Sea-breeze onset (hour 11-14):   1.5+ m/s ✓
  Sea-breeze peak (hour 15-18):    2.5+ m/s (2+ hours >= 2 m/s) ✓
  Land-breeze min (hour 01-05):   -0.7 m/s ✓
  Nighttime UHI (hour 02-05):      2.0+ K ✓
```

**Binary mode (control):**
- Still passes all criteria (physics is robust to interface treatment)
- Sea-breeze onset and peak may be slightly weaker or delayed
- Provides A/B baseline for Phase 5.6 blending benefit

## Building Layout Generation

The `gen_coastal_layout.py` script generates `building_layout_coastal.csv` with:
- 6400 UCM cells (80×80 grid)
- Coastal checkerboard pattern in transition bands
- Sea material (mat_id=5) for water tiles
- Urban / rural parameters per region

**Run:**
```bash
python3 gen_coastal_layout.py
```

## Material Library

**`materials.csv`** additions:
- **mat_id=5, sea_water:** k_therm=0.6 W/mK, ρc=4.2e6 J/m³K, albedo=0.06, emissivity=0.97
  - Represents open ocean with high thermal mass
  - Minimal diurnal T_skin swing (~±1 K) despite strong SW/LW forcing
  - This **thermal inertia** is the physical driver of the sea-breeze reversal

## How to Run

### Generate layout
```bash
cd Exec/CanonicalTests/SLUCM/UCMBostonCoastal24h
python3 gen_coastal_layout.py
```

### Binary control (regression gate)
```bash
mkdir -p binary_plts
mpirun -np 4 ../../../build/Exec/erf_exec inputs_coastal_binary > run_binary.log 2>&1
```

### Blended production (science verifier)
```bash
mkdir -p blended_plts
mpirun -np 4 ../../../build/Exec/erf_exec inputs_coastal_blended > run_blended.log 2>&1
```

### Verify breeze reversal
```bash
# Copy plotfiles for blended run
mv plt_coastal_* blended_plts/

# Run verifier
python3 check_coastal_breeze.py --plt_dir blended_plts/
```

## Design Decisions

### Why sea material instead of grassland surrogate?
The canonical requires high thermal inertia (ρc ~ 4.2×10⁶ J/m³K) to model open water's resistance to diurnal temperature swings. This drives the sea-breeze reversal via the land–ocean thermal contrast. Grassland (ρc ~ 1.4×10⁶ J/m³K) is insufficient.

### Why checkerboard transition bands?
The coast is intrinsically a mixed UCM/MOST domain: some grid cells are partially land, partially sea. A sharp transition boundary misrepresents this mixed character. The checkerboard pattern (Phase 5.6 recipe) creates genuine sub-grid fractional f_urb, requiring blended interface mode for correct flux accounting.

### Why 24-hour simulation?
The sea-breeze reversal cycle spans midnight to midnight. Early termination (e.g., sunset) would miss the nighttime reversal and UHI verification. Full 24-hour integration also allows validation against any existing coastal diurnal-cycle observations.

## Regression Baseline

All prior Phase canonicals (UCMBoston, UCMBostonMixedDomain, UCMBostonDiurnal24h, UCMBostonHVAC*, UCMBostonGreenRoof, etc.) must continue to pass with existing inputs. Phase 5.6 backward compatibility (binary mode as default) is preserved. This canonical is **purely additive**.

## References

- **Sea-breeze physics:** Freitas et al. (2007), Crosman & Horel (2010)
- **Urban sea-breeze intensification:** Kusaka & Kimura (2004), Liu et al. (2006)
- **Phase 5.6 interface_mode=blended:** ERF-SLUCM Phase 5.6 design doc
- **Phase 3.5b radiation:** Prescribed clear-sky diurnal forcing, Kusaka et al. (2001) view factors
