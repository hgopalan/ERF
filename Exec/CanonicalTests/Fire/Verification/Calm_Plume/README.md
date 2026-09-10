# Calm_Plume

A steady prescribed heat source under still air, and the plume it raises. The
heat the coupling injects is checked against the atmosphere's heat content; the
plume is compared with plume theory, and that comparison is reported rather than
checked, for the reason below.

```
MPIRUN="mpirun -np 8" ./run_calm_plume.sh /path/to/erf_exec
SKIP_RUN=1 ./run_calm_plume.sh x      # checks only
```

## The case

1.6 x 1.6 x 1.2 km, 40 m cells, periodic in x and y, Smagorinsky LES, gravity, an
adiabatic ground (`erf.most.surf_temp_flux = 0`) and damping of the vertical
velocity only in the top 200 m, so heat is conserved. `erf.fire.prescribed_heat.flux`
= 1.1e3 W/m2 over a 120 m disc at the centre, 49.3 MW over the fire cells inside
it, injected with the coupling's exp(-z / 45 m) profile; nothing burns.

| deck | air | run |
|---|---|---|
| `neutral` | theta = 300 K at every height | 1200 s |
| `stable` | N = 0.01 1/s (theta rising 3.06 K/km) | 1500 s |

## What is checked

The heat budget: Cp times the change of the integral of rho theta equals the
placed power, 1 - exp(-H/alfg) of the source, times the heated time. Within
0.01 % at every plotfile of both decks.

## What is reported

Heskestad's far-field centreline correlations for the neutral plume, and the
Morton, Taylor and Turner (1956) maximum height 5.0 F^(1/4) N^(-3/4) (F = 440
m4/s3 here) with its spreading level at 0.76 of it for the stable one, both from
Heskestad's virtual origin z0 = -239 m of this broad source:

| | measured | theory |
|---|---|---|
| neutral, centreline dT at 260 / 340 / 460 / 540 m | 1.27 / 1.15 / 1.02 / 0.95 K | x1.14 / x1.32 / x1.58 / x1.76 of Heskestad |
| neutral, centreline w at the same heights | 4.23 / 4.72 / 5.34 / 5.68 m/s | x0.88 / x1.03 / x1.24 / x1.36 of Heskestad |
| neutral, axis w from 260 to 740 m | rises from 4.2 to 6.1 m/s | falls as (z - z0)^(-1/3) |
| stable, strongest radial outflow | 383 m | 312 m (x1.23) |
| stable, axis neutrally buoyant | about 460 m | |
| stable, mean axis w reaches zero | about 935 m | 486 m (x1.93) |

The plume is under-entrained. The source is six cells across and broad against
its height, so within the domain the plume stays in its near field (a lazy plume
accelerating as it contracts) and the grid does not resolve the eddies that should
mix ambient air into it: the axis velocity grows with height where the far-field
law has it fall, and the stable plume passes its level of neutral buoyancy with
4 m/s and overshoots, up to 1.6 K colder than its surroundings, to nearly twice the
theoretical top, while its outflow sits within a quarter of the spreading level.
A validation of the plume needs a smaller source on 10 to 20 m cells.

## References

Briggs, G. A. (1975). Plume rise predictions. In Lectures on Air Pollution and
Environmental Impact Analyses, American Meteorological Society, 59-111.

Heskestad, G. (2016). Fire plumes, flame height, and air entrainment. In SFPE
Handbook of Fire Protection Engineering, 5th ed., Springer, 396-428.

Morton, B. R., G. I. Taylor and J. S. Turner (1956). Turbulent gravitational
convection from maintained and instantaneous sources. Proceedings of the Royal
Society A, 234, 1-23.
