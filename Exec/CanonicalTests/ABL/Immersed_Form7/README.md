# Immersed-terrain form 7 wall law: flat ABL cases

Four flat atmospheric boundary layers run two ways: on flat ground with the surface layer at
`z = 0`, and on an immersed terrain plane (`erf.terrain_type = ImmersedForcing`) with the form 7
wall law (`erf.if_wall_form = fraction_stress`). Because the immersed plane is flat, the flat-ground
run is the reference: once the heights are shifted by `h`, the two profiles should agree.

| Folder                | Case                                             | Run time |
|-----------------------|--------------------------------------------------|----------|
| `neutral`             | neutral, geostrophic wind 10 m/s                 | 4 h      |
| `heat_flux`           | unstable, surface heat flux 0.05 K m/s, 15 m/s   | 4 h      |
| `surface_temperature` | unstable, surface temperature 303 K, 15 m/s      | 4 h      |
| `gabls1`              | GABLS1 stable, cooling 0.25 K/h from 265 K, 8 m/s | 9 h      |

Each case folder holds four decks:

| Deck          | Grid                                                                            |
|---------------|---------------------------------------------------------------------------------|
| `flat_1lev`   | flat ground, 160 x 160 m periodic, one level, dz = 20 m                         |
| `form7_1lev`  | immersed plane at h = 45 m (a quarter of a cell into the solid), one level     |
| `flat_patch`  | flat ground, 480 x 80 m, level 1 patch over x = 160-320 m, up to 200 m, 20 -> 10 m |
| `form7_patch` | immersed plane at h = 42.5 m, same patch; the patch sides cross the surface    |

All decks use the k-eqn RANS closure, anelastic `MidPoint` time stepping, implicit vertical
diffusion (`erf.vert_implicit`) and the FFT Poisson solver (`erf.use_fft = true`), so ERF must be
built with `ERF_ENABLE_FFT=ON`. Run each deck from its own folder:

```
cd neutral/form7_1lev
mpiexec -n 1 /path/to/erf_exec inputs
```

The runs write `mean_profiles.dat`, `flux_profiles.dat` and `sfs_profiles.dat` every 10 minutes.
Compare the horizontally averaged wind speed, potential temperature and TKE against the flat deck,
with the form 7 heights shifted down by `h`.

## Keys the form 7 decks set

- `erf.if_wall_form = fraction_stress`: the form 7 wall law, which needs `erf.if_use_most = true`
  and a constant dz.
- `erf.if_implicit_drag = true`: the wall stress is applied exactly in time,
  `C = (1 - exp(-r dt)) / dt`.
- `erf.if_implicit_projection = true` (patch decks only): the wall drag is applied inside the
  anelastic projection.
- `erf.if_wall_tke = true`: the wall-cell TKE relaxes to its wall value. This is k-eqn only.
- `erf.wall_dist_type = terrain_height`: the k-eqn length scale is measured from the immersed wall.
- The heat boundary condition is one of `erf.if_surf_temp_flux`, `erf.if_init_surf_temp`
  (with `erf.if_surf_heating_rate`) or `erf.if_Olen`. Leave all of them out for a neutral run.

The two-level decks use the per-level input sounding (already on development) and the immersed
average-down and coarse/fine flux balance on this branch. `erf.cf_loglaw_fill = true` switches on
the opt-in log-law fill of the lateral coarse/fine faces next to the wall. It improves the
flat-ground patch runs; the immersed runs are limited by the missing apertures of partial cells.

## Going to LES

For LES, set `erf.rans_type = "None"` and `erf.les_type = "Deardorff"` (or `"Smagorinsky"`),
and remove `erf.if_wall_tke`, `erf.dirichlet_k` and `prob.KE_*`. The domain also needs enough
resolution and horizontal extent to resolve the turbulence. The friction velocity, the wall
stress and the wall heat flux of form 7 do not depend on the closure. The TKE relaxation and
`erf.if_flat_wall_dissipation` are RANS-specific. These decks have only been validated with the
k-eqn closure.
