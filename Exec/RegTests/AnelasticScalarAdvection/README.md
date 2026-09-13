# Anelastic scalar advection on stretched meshes and with map factors

With the anelastic integrator, `erf_slow_rhs_post` sets the momenta that advect
the slow scalars (TKE, moisture, passive scalars, dust, smoke) from the projected
momenta. `AdvectionSrcForScalars` expects them in the form `AdvectionSrcForRho`
builds, `ax rho u / mf_uy`, `ay rho v / mf_vx` and `az Omega / (mf_mx mf_my)`,
and divides the flux differences by `detJ`. The anelastic branch copied the raw
momenta instead, which scaled horizontal scalar advection by `1 / h_zeta` on
stretched and terrain-fitted meshes and dropped the map factors. On a uniform mesh
with unit map factors the two forms are identical.

Each case advects a passive-scalar blob (`prob_type = 11`, an x-y cosine bump,
uniform in z) in a uniform wind.

| deck | mesh | check | CTest |
|---|---|---|---|
| `inputs_fitted` | `StaticFittedMesh` on flat ground (`VariableDz`), 12 levels from 20 m stretched by 1.2 | `check_scalar_centroid.py`: x-centroid on every level at `x_c + U t` within one cell | `AnelasticScalarAdvection_Fitted` (FFT builds) |
| `inputs_stretched` | the same stretching without terrain (`StretchedDz`) | as above | `AnelasticScalarAdvection_Stretched` (FFT builds) |
| `inputs_mapfac_anelastic` | uniform mesh, `erf.test_mapfactor = true`, MLMG projection | `check_mapfac_parity.py`: x- and y-centroids on every level within a quarter cell of the compressible run and of `U t m / dx`, `V t m / dy` (m = 0.5), peak within 5 % | `AnelasticScalarAdvection_MapFactor` (every build) |
| `inputs_mapfac_compressible` | the reference for the map-factor case: compressible, no substepping | | |

The anelastic projection on `StretchedDz` and `VariableDz` meshes needs a build
with FFT, which no CI configuration enables, so the first two tests are registered
only when `ERF_ENABLE_FFT` is on. The map-factor case runs everywhere.

Without the fix, after 50 steps (500 m of travel), the stretched blob moved
`U / h_zeta`: about 1650 m in the 20 m bottom level (`h_zeta` 0.30) and 220 m in the
149 m top level (`h_zeta` 2.25), 40 to 457 m from the expected centroid on every
level. With the fix every level is within 0.3 m. In the map-factor case the
anelastic blob trailed the compressible one by 2.4 cells in x and 1.2 cells in y;
with the fix both sit on the analytic centroid.
