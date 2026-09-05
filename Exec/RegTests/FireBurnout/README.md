# FireBurnout

How long a cell keeps releasing heat once the front has passed. The
historical form ties the fuel's e-folding time to the cell crossing time,
`max(dx/R, tau_SAV)`, so timber litter burns out as fast as grass. WRF-SFIRE
and the Community Fire Behavior Model give each fuel model a burn time
(7 s grass, 180 s chaparral, 100 s brush, 900 s timber litter and slash;
Jiménez y Muñoz et al. 2026, Table 1) divided by 0.8514 for the e-folding
time. `erf.fire.burnout_model = sfire` selects that; `residence` (default)
keeps the crossing time. `erf.fire.burnout_times_s` (13 values) overrides
the table and `erf.fire.burnout_time_to_efold` the divisor.

Six one-way decks, 60 s: grass with the crossing time (`residence`, and
`residence_key` with the key written out), grass with the SFIRE time
(`sfire_grass`, 8.2 s) and with the table overridden (`sfire_override`,
16.4 s), and hardwood litter (Anderson 9) under both (`residence_litter`,
`sfire_litter`, 1057 s).

```
MPIRUN="mpirun -np 4" ./run_burnout.sh /path/to/erf_exec
python3 plot_burnout.py
```

The script checks that the key written out reproduces the historical deck
line for line; that in every sfire deck the largest heat flux over the run
is exactly the fresh-cell value `w0 h / tau` the code prints and never
exceeds it; that the energy released over the run equals `h` times the fuel
consumed (to 2 % on grass, where holding the flux over a 0.125 s step
against an 8 s e-folding is a bias the historical form shares, and 0.1 % on
litter; the power is integrated from the second log line because each line
prints the power before that step's depletion and the fuel after it); and that on litter the SFIRE deck leaves more fuel at 60 s. The
plot shows the total power and the fuel consumed against time: the same
energy, released over seconds on grass and over a quarter of an hour on
litter.
