# UCMScaleAwareAggregation

Canonical test for Phase 2.5 follow-up. Exercises urban-fraction-weighted flux coarsening
when ATM cells partially contain urban UCM cells.

## Grid
- ATM 4x4, dz-resolved column (64 vertical cells).
- grid_ratio=4 -> UCM 16x16.
- Diagonal urban wedge: cells with (i+j) < 12 are urban. Result: f_urb
  spans [0, 1] across the 16 ATM cells.

## Setup
```
python3 gen_csv.py           # produces building_layout.csv and materials.csv
```

## Pass criteria
1. `mpirun -n 1 erf_ucm_scale_aware_aggregation inputs` exits 0.
2. Repeat on 2 MPI ranks. Exits 0.
3. `[UCM][2.5-followup][BANNER]` prints `f_urb=[0,1]`, `H_bldg_mean=[0,10] m`,
   `H_bldg_std=[0,~0] m`, `lambda_f max >= 0`.
4. `plt_ucm_atm_00100/` exists.
5. `python3 check_conservation.py` prints `PASS`.
6. Regression: `UCMShadowCanyon` still exits 0 (if it exists).

## Physics
- Neutral ABL with MRF PBL (same as UCMHomogeneousGrid).
- Urban pattern is diagonal wedge: creates variety of f_urb values
  in ATM cells (0 to 1) for testing weighted aggregation.
- Phase 2.5 aggregation convention (weighted-divide): `Q_atm = sum(is_urban*Q_ucm) / f_urb`.
