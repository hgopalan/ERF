# UCMSalamancaMadrid (Phase 2.10 canonical inflow/outflow test)

This case adds a non-periodic x inflow/outflow SLUCM canonical test for daytime urban heat island behavior, following the Askervein inflow/outflow layout but using the compressible dycore + MRF PBL stack validated in Phase 2.8.

- Domain: 5 km × 0.5 km × 1.28 km, periodic in y (`geometry.is_periodic = 0 1 0`)
- Surface: MOST (`zlo.type = "surface_layer"`), top slip wall
- Terrain: disabled (`erf.terrain_type = None`)
- Urban forcing: uniform Salamanca morphology with per-cell `AH_Wm2=40` and facet3d injection enabled
- Focus hour: noon LST (12:00), chosen for strongest daytime UHI signal in a first canonical validation

## Workflow

```bash
python3 gen_csv.py
./erf_ucm_salamanca_madrid inputs
python3 check_salamanca.py
```

## Expected verification outcome

`check_salamanca.py` computes the near-surface urban-minus-upwind temperature signal over available plotfiles and asserts:

- `T_urban > T_rural` (basic UHI presence)

It also prints a diagnostic center-column theta profile.

## Notes

- This is a qualitative validation target only.
- Quantitative agreement is expected to improve in later phases with two-way feedback and radiation coupling.

## References

- Salamanca, F., Krpo, A., Martilli, A., & Clappier, A. (2011). *Theor. Appl. Climatol.* 99:331–344.
- Wagenbrenner et al. (2019), Askervein inflow/outflow architectural template.
