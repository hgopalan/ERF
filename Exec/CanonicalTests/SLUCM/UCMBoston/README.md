# UCMBoston (Phase 2.11)

Canonical test establishing the **first real-city baseline** for SLUCM using a Boston-stylized concentric urban layout. This is the single-level one-way baseline that Phase 3.6 (multi-level one-way) and Phase 3.10 (multi-level two-way) will compare against.

## Domain geometry

20 km × 20 km × 1280 m, ATM 20×20×64 (Δx=1000 m, Δz=20 m), UCM 80×80 (grid_ratio=4).

Along x and y, the domain contains a **city with concentric rings**, using Chebyshev (L-infinity) distance from center (i=39.5, j=39.5):

| Ring | d (UCM cells) | Extent ≈ | Type | λ_p | H (m) | AH (W/m²) | Material |
|------|---------------|-----------| ----|-----|-------|----------|----------|
| Downtown core | 0–7 | 0–7 km from center | Financial District | 0.55 | 100 | 60 | glass/steel (mat_id=1) |
| Dense mid-rise | 8–15 | 8–15 km | Back Bay / Beacon Hill | 0.50 | 40 | 45 | brick/concrete (mat_id=2) |
| Residential dense | 16–24 | 16–24 km | South End / Cambridge | 0.35 | 15 | 30 | brick/concrete (mat_id=2) |
| Residential sparse | 25–32 | 25–32 km | Somerville / Brookline | 0.20 | 8 | 15 | wood/vinyl (mat_id=3) |
| Suburban / rural | 33–39 | 33–39 km | Newton / outer metro | 0.05 | 5 | 5 | wood/vinyl (mat_id=3) |

**Geographic context:** Notionally centered at 42.3601° N, 71.0589° W (Boston Common), but geographic accuracy is not required — this is a **stylized synthetic layout**, not a GIS-authentic WUDAPT/OSM reproduction.

## Physics under test

- **Full Phase 2 stack:** Phase 2.1 CSV input → 2.9 per-cell AH override
- **Phase 2.7** facet3D injection (continuous BEP heat distribution)
- **Phase 2.8** BEP momentum drag (`wall_drag_mode = "explicit"`, `Cd_wall = 0.4`)
- **Multi-material heterogeneity** via WUDAPT-LCZ-inspired stripes
- **Urban-rural contrast** — city core surrounded by suburbs and outer rural
- **Compressible dycore + MRF PBL** (validated Phase 2.8 path)
- **One-way coupling only** (`atm_feedback = 0.0`)
- **Inflow/outflow BCs** with log-law profile (z₀ = 1.5 m urban roughness)

## Not tested here

- Two-way ATM→UCM feedback (Phase 3.2)
- Radiation coupling (Phase 4.2)
- Multi-level AMR (Phase 3.1, reserved for Phase 3.6 and 3.10)
- Quantitative match to Boston field campaigns (Phase 4+)

## Input files

### `inputs_singlelevel` — Single-level one-way (this phase)
- Working canonical test for Phase 2.11
- ATM grid: 20×20×64, UCM grid: 80×80
- Simulation time: 3600 s (1 hour) at CFL=0.5
- Outputs main ATM plotfiles `plt_NNNNN` every 600 time steps
- Expected runtime: ~5 min on 4 CPU cores

### `inputs_multilevel_oneway` — Multi-level one-way (Phase 3.6)
- Placeholder reserved by Phase 2.11
- Will add 3-level AMR configuration with downtown refinement
- Downtown core (UCM 32–47 / ATM 8–11) refined at level=1
- Requires Phase 3.1 (level-aware allocation)

### `inputs_multilevel_twoway` — Multi-level two-way (Phase 3.10)
- Placeholder reserved by Phase 2.11
- Will add full Part 3 stack: 3-level AMR + `atm_feedback = 1.0`
- Innermost 4×4 downtown block (UCM 36–43 / ATM 9–10) refined at level=2
- Requires Phases 3.1 through 3.9

## Workflow

```bash
# Generate or verify CSV files
python3 gen_boston.py

# Run the single-level baseline
./erf_ucm_boston inputs_singlelevel

# Verify output
python3 check_boston_singlelevel.py
```

## Verification and expected output

The check script (`check_boston_singlelevel.py`) verifies:

1. **Plotfile discoverable** — a `plt_NNNNN` file exists and can be loaded with yt
2. **Concentric UHI structure** — near-surface θ at downtown core (i≈10) is ≥0.05 K warmer than at domain edge (i=0)
3. **Canopy wind reduction** — wind speed at downtown is ≥10% less than upwind rural reference
4. **No NaN in output** — all θ, u, v fields are finite
5. **Diagnostic profiles** — vertical θ profile and ring temperature summary (informational only)

Expected results (loose assertions):
- Positive urban–rural ΔT at ~30 m AGL (~0.5–2.0 K for typical diurnal noon LST)
- 15–30% canopy wind reduction in downtown core
- Monotonic θ increase from upwind edge to downtown
- Smooth vertical structure (no spurious oscillations)

## Shared infrastructure

### `gen_boston.py`
- Pure-Python generator (no external data pulls)
- Produces `building_layout.csv` (80×80 UCM grid, 6400 cells)
- Produces `materials.csv` (4 material types)
- Uses `sys.path.insert()` to import `ucm_csv` tools from `../../tools/`

### `building_layout.csv`
- Header: i, j, bldg_id, height_m, plan_area_frac, W_road_m, W_roof_m, roof_mat_id, wall_mat_id, road_mat_id, orientation_deg, ah_profile_id, AH_Wm2, is_urban
- All 6400 cells marked `is_urban=1` (synthetic-only domain for Phase 2.11 baseline)
- Height varies: 100 m downtown, 5 m outer edge
- Plan area fraction: 0.55 downtown, 0.05 outer edge

### `materials.csv`
- 4 materials: glass/steel (downtown), brick/concrete (urban), wood/vinyl (suburban), grassland (reserved)
- Albedo: 0.30 downtown glass, 0.20 urban brick, 0.25 residential wood, 0.20 grassland
- Thermal: varying conductivity (50 W/mK glass → 0.3 W/mK grassland) and volumetric heat capacity

### `sounding_boston`
- Neutral θ profile: θ_ref = 295 K (mid-summer New England), gradient = 0.001 K/m to top
- Wind from west at constant 5 m/s at 30 m height (hub height reference)
- ERF sounding format: pressure_Pa, θ_K, q_kg/kg; then z_m, θ_K, q_kg/kg, u_m/s, v_m/s for levels

### `inflow_boston.txt`
- Four-column Askervein format: z_m, u_m/s, speed_m/s, w_m/s
- Log-law profile: U = (U_ref / ln(z_ref/z₀)) × ln(z/z₀)
- z₀ = 1.5 m (urban Boston roughness), U_ref = 5 m/s at z_ref = 30 m
- 80 levels, geometric spacing from 1.0 m to 1280 m
- Used for inflow BC and sponge damping

## Phase 3.6 and 3.10 references

**Downtown core extent for future AMR refinement:**
- **UCM grid:** i, j ∈ [32, 47] (16×16 UCM cells)
- **ATM grid (level=0):** i, j ∈ [8, 11] (4×4 ATM cells, refined from 80×80 UCM)
- **ATM grid (level=1, Phase 3.6):** will refine downtown to 2× resolution
- **ATM grid (level=2, Phase 3.10):** will further refine innermost 4×4 UCM block to 3× resolution

## Known limitations

- **Synthetic layout:** This is a stylized Boston-inspired concentric domain, not a real WUDAPT or OSM pull. Phase 2.9's `gen_real_boston_full.py` script can regenerate with actual GIS data for manual QA in Phase 4+.
- **No radiation:** Urban canopy albedo and emissivity are set constant; no shortwave/longwave radiation coupling (Phase 4.2).
- **One-way coupling only:** UCM does not feed back to atmosphere (Phase 2.11 baseline design).
- **Synthetic initialization:** Neutral sounding; no actual Boston radiosonde data.
- **Loose check assertions:** The verification script uses physically motivated but non-quantitative assertions suitable for baseline validation, not field campaign comparison.

## References

- **Salamanca et al. (2011),** *Theor. Appl. Climatol.* 99:331–344 — reference for canonical urban canopy parameterization
- **Phase 2.10:** UCMSalamancaMadrid — predecessor with upwind/downwind 1D stripe layout
- **Phase 2.8:** UCMBEPMomentumDrag — validation of compressible dycore + MRF PBL + explicit drag
- **Real_Terrain/Askervein:** Reference for inflow/outflow BC architecture and log-law profile format

---

**Phase 2.11 closes Part 2** by establishing this baseline. Phase 3.6 will add 3-level AMR with downtown refinement (1-way), and Phase 3.10 will complete the full two-way multi-level stack.
