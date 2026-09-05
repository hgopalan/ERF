# FireWindSampling

The Community Fire Behavior Model's second wind interpolation option
(Jiménez y Muñoz et al. 2026): sample the atmospheric wind at a height above
the flames and bring it down to the target height with a neutral log
profile, so the fire is driven by ambient wind rather than by air its own
plume has accelerated. `erf.fire.wind_sample_ht` sets the sampling height and
`erf.fire.wind_sample_z0` the roughness of the profile; 0 (default) samples at
`erf.fire.wind_ref_ht` as before.

A grass fire on flat ground, one-way so the atmosphere is identical in every
deck:

- `off`: sampled at 6.1 m (historical); `off_key` writes the key out and must
  reproduce it line for line.
- `sample20`: sampled at 20 m, brought to 6.1 m with z0 = 0.1 m, factor
  ln(6.1/0.1)/ln(20/0.1) = 0.77589.
- `ref20`: wind taken at 20 m directly, the reference for the factor check.

```
MPIRUN="mpirun -np 4" ./run_wind.sh /path/to/erf_exec
python3 plot_wind.py
```

The script reads the largest reference wind the fire grid sees each step
(before the wind adjustment factor) and checks that the sampled deck is the
factor times the direct 20 m deck at every step, and that the factor the code
prints is the expected one. With the uniform 8 m/s sounding the wind at 20 m
is barely higher than at 6.1 m early in the run, so the sampled deck sees
about 78 % of the historical wind and burns less; that comparison is
tabulated, not checked.
