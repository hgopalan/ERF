# UCMKandaWindTunnel (Phase 2.10 canonical inflow/outflow test)

This case adds a non-periodic x inflow/outflow SLUCM canonical test for momentum drag across four packing densities, following the Askervein inflow/outflow layout while preserving the compressible + MRF path from Phase 2.8.

- Domain: 5 km × 0.5 km × 1.28 km, periodic in y (`geometry.is_periodic = 0 1 0`)
- Surface: MOST with neutral flux (`erf.most.surf_temp_flux = 0.0`)
- Top boundary: slip wall
- Terrain: disabled (`erf.terrain_type = None`)
- Four x-stripes in UCM grid for `lambda_p = 0.11, 0.25, 0.33, 0.44`
- Heat forcing disabled (`AH_Wm2=0`, `use_facet3d_injection=0`)

## Boundary-condition deviation note

Kanda et al. (2004) LES used a slip lower boundary. This canonical ERF case keeps MOST at the lower boundary with neutral heat flux to preserve MRF `u_star` evaluation, while leaving the physics under test (canopy momentum drag) unchanged.

## Workflow

```bash
python3 gen_csv.py
./erf_ucm_kanda_wind_tunnel inputs
python3 check_kanda.py
```

## Expected verification outcome

`check_kanda.py` prints normalized profiles `U(z)/U_H` at z/H = {0.25, 0.5, 0.75, 1.0} for all four `lambda_p` stripes and compares against approximate Kanda Fig. 6 values.

Assertion is intentionally loose and qualitative:

- canopy-interior `U/U_H` decreases monotonically as `lambda_p` increases.

## References

- Kanda, M., Moriwaki, R., & Kasamatsu, F. (2004). *Boundary-Layer Meteorol.* 112:343–368.
- Wagenbrenner et al. (2019), Askervein inflow/outflow architectural template.
