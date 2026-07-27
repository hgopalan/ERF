# UCMBostonMixedDomain — Phase 3.8 Non-Urban Partial-Domain Regression

## Purpose

Verify SLUCM handles mixed urban / non-urban domains correctly:
- Urban cells (is_urban=1) receive full SEB, slab conduction, drag, AH
- Non-urban cells (is_urban=0) bypass UCM (LSM/MOST path, though full LSM wiring is Phase 4.1)
- Interface between urban and non-urban cells produces physical wind/theta gradient

## Configuration

- Domain: 20 km × 20 km × 1280 m (same as UCMBoston)
- Base grid: 20 × 20 × 64
- UCM anchor_level = 0
- Duration: 600 steps (~14 min simulated) — this is a regression test
- One-way coupling only (atm_feedback_heat = 0, atm_feedback_momentum = 1)
- Custom CSV: `building_layout_mixed.csv`
  - Left half of domain (x < 10 km): urban (uniform Boston 5-zone morphology)
  - Right half of domain (x >= 10 km): non-urban (is_urban=0, grassland_rural material)

## Data Files

Required data files must be symlinked or copied:

```bash
ln -sf ../UCMBoston/materials.csv .
ln -sf ../UCMBoston/inflow_boston.txt .
ln -sf ../UCMBoston/sounding_boston .
python3 gen_mixed_layout.py    # generates building_layout_mixed.csv
```

## Validation Metrics (check_mixed_domain.py)

1. Mixed domain confirmed (is_urban=0 count > 0 AND is_urban=1 count > 0)
2. No assertion failures or aborts
3. No NaN or Inf in fields
4. Zero Newton clamps
5. SEB solver called at least once
6. Wind reduction over urban half > 10% (drag active on urban)
7. Wind reduction over non-urban half < 5% (drag NOT active on rural)

## Running

```bash
cd /path/to/UCMBostonMixedDomain

# Generate CSV
python3 gen_mixed_layout.py

# Link/copy required data files
ln -sf ../UCMBoston/materials.csv .
ln -sf ../UCMBoston/inflow_boston.txt .
ln -sf ../UCMBoston/sounding_boston .

# Run simulation (from ERF top-level Build directory)
../../../Build/erf_slucm inputs_mixed_domain > run.log 2>&1

# Validate results
python3 check_mixed_domain.py
```

Exit code 0 = PASS, 1 = FAIL. Should complete in ~15 seconds.

## Prerequisites

- Phase 3.7 (physical-coordinate CSV) must be merged first
- `building_layout_mixed.csv` uses Phase 3.7 physical-coordinate schema

## Notes

This is an **integration test** (no new physics). It validates:
- CSV reader handles mixed is_urban columns correctly
- Drag is applied only to is_urban=1 cells
- Temperature and other fields remain physical
- SEB solver executes without numerical failures

Differences from UCMBoston single-level:
- **CSV**: Left half urban (uniform 15 m, 10×10 m canyon), right half rural (grassland_rural mat)
- **Timestepping**: 600 steps (shorter for CI) vs 3600 in other canonicals
- **Validation**: Checks wind reduction differential, not absolute UHI magnitude
