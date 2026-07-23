# UCMSalamancaMadrid (Phase 2.10)

Inflow/outflow canonical test comparing ERF-SLUCM urban heat island
against Salamanca et al. (2011) Madrid summer measurements.

## Domain geometry

5 km × 500 m × 1280 m, ATM 10×10×64 (Δx=500 m, Δz=20 m), UCM 40×40 (grid_ratio=4).

Along x, the domain contains a **city with rural surroundings**:

| UCM i     | Extent (km) | Type      | λ_p  | H (m) | AH (W/m²) |
|-----------|-------------|-----------|------|-------|-----------|
| 0..7      | 0.0 – 1.0   | Rural     | 0.0  | 0     | 0         |
| 8..15     | 1.0 – 2.0   | Suburban  | 0.25 | 5     | 20        |
| 16..23    | 2.0 – 3.0   | Urban core| 0.55 | 20    | 40        |
| 24..31    | 3.0 – 4.0   | Suburban  | 0.25 | 5     | 20        |
| 32..39    | 4.0 – 5.0   | Rural     | 0.0  | 0     | 0         |

**Rationale:** The Salamanca 2011 comparison requires an upwind rural
reference so `T_urban − T_rural` can be measured. A uniform-urban domain
has no rural surrogate. The 5-zone layout mirrors the Madrid–Meseta
transition observed in the paper's measurement campaign.

## Physics under test

- Phase 2.2 heterogeneous morphology (λ_p, H varying by cell)
- Phase 2.3–2.6 facet-split heat + anthropogenic heat
- Phase 2.7 facet3D BEP-continuous injection
- Phase 2.9 per-cell `AH_Wm2` override
- Compressible dycore + MRF PBL (matches Phase 2.8 validated path)
- Inflow/outflow BC with Askervein-style sponge damping at inflow

## Not tested here

- Two-way ATM→UCM feedback (Phase 3.2)
- Radiation coupling (Phase 4.2)
- Diurnal cycle (fixed at noon LST for strongest daytime UHI signal)
- Quantitative match to Salamanca Fig 5 (Phase 3+)

## Workflow

```bash
python3 gen_csv.py              # produces building_layout.csv + materials.csv
./erf_ucm_salamanca_madrid inputs   # ~5 min on 4 cores
python3 check_salamanca.py       # loose physics assertions
```

Expected output: positive urban–rural ΔT at ~10 m AGL, canopy wind
reduction in the urban core, monotonic θ recovery downwind of the city.

## References

- Salamanca, F., Krpo, A., Martilli, A., & Clappier, A. (2011).
  "A new Building Energy Model coupled with an Urban Canopy
  Parameterization for urban climate simulations—part I. formulation,
  verification, and sensitivity analysis of the model."
  *Theor. Appl. Climatol.* 99:331–344. doi:10.1007/s00704-009-0142-9

- Wagenbrenner et al. (2019). *Atmosphere.* Askervein hill validation —
  reference for inflow/outflow architecture template.