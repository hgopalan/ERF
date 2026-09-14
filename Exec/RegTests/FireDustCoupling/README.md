# FireDustCoupling

## Purpose
A 40-step coupled fire + dust run whose check script proves the three fire-dust
couplings are applied once per step and in the right order. It was written for
the three defects found in the September 2026 audit:

| coupling | defect before the fix | check |
|---|---|---|
| burned area removes crust | applied every step on top of the previous step (crust 0.2^n) because the reset path in `DustLayer::advance` was never registered | u*_t burned / unburned = (1 + a c (1-r)) / (1 + a c) at steps 20 and 40 |
| fire wind raises the dust u* | applied before `DustLayer::advance`, whose surface-layer fill overwrote it | dust u* >= log-law u* of the fire wind in every cell |
| deposition accumulator | added at every RK stage with that stage's dt (1.83 dt per step) | deposition_total at step 40 within 15 % of the reference |

## Running
```bash
mpirun -np 1 ../../../build/Exec/erf_exec inputs   # one rank: 20 cells do not split by the grid ratio
python3 check_firedust.py            # needs erf_plotfile.py from Exec/CanonicalTests/Canonical_RANS
```
CTest runs it as `FireDustCoupling_check`.

## Expected results
See the check script's output; the deposition reference is recorded in
`check_firedust.py` (`DEP_REF`) with the measured pre-fix value.
