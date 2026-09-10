# FireEmcModel

The equilibrium moisture curves the dead fuel relaxes toward,
`erf.fire.emc_model = legacy` (default) or `van_wagner`.

- `legacy`: the adsorption and desorption quartics in the relative humidity
  fraction carried over from wildfire_levelset. They were attributed to
  Nelson (2000), which does not contain them; their constant and linear terms
  are Simard's (1968) RH < 10 % equation with the percent units dropped, and
  no source is known for the rest. At 40 % RH they give 0.166 and 0.189, about
  twice the published curves, and both reach the 0.35 cap above 70 % RH. With
  no atmospheric moisture the relative humidity is clamped to 1 %, where they
  give 0.035 and 0.060.
- `van_wagner`: the drying and wetting equilibrium moisture of the Canadian
  Fine Fuel Moisture Code (Van Wagner and Pickett 1985; Van Wagner 1987 eqs.
  2a-2b), temperature dependent, the pair WRF-SFIRE uses: 0.089 and 0.105 at
  40 % RH and 27 C, zero in dry air (the fuel stops at the 0.01 floor).

Five one-way decks of a grass fire, 60 s, every dead class starting at 0.08:
dry air with the legacy curves (the historical deck), the same with the key
written out, dry air with `van_wagner`, and both curves in air at about 40 %
RH (a `Kessler_NoRain` atmosphere with 9 g/kg of vapour at 300 K).

```
MPIRUN="mpirun -np 4" ./run_emc.sh /path/to/erf_exec
```

`check_emc.py` reads the `[FIRE DEBUG]` lines of the logs (domain-average dead
classes, the Rothermel no-wind rate `R0`, the surface RH and temperature) and
the fire plotfiles at 60 s, and checks that the key written out reproduces the
historical deck bit for bit, that `van_wagner` differs, that dry air reaches
the fuel as zero RH and the humid sounding at 30-50 %, that each deck moves
the classes in lag order, and that `van_wagner` leaves the fuel drier than
`legacy` in dry air and less wet in humid air, with `R0` ordered the other way.
Those orderings hold whichever hysteresis branch the kernel takes. Sixty
seconds is a small fraction of the one-hour lag, so the differences here are
in the third or fourth decimal; the unit test `ERF_GTestFuelMoistureEMC` runs
the curves for hours.

## Expected Results

On four ranks (macOS Release, built at 3a7b36680; the dead-class moisture
path is unchanged through 08678c8e2), surface temperature 300 K:

```
variant             RH max  T max K     M_1hr    M_10hr   M_100hr     R0 m/s   cells
legacy               0.000   300.00  0.079179  0.079917  0.079992    0.02038     774
legacy_key           0.000   300.00  0.079179  0.079917  0.079992    0.02038     774
van_wagner           0.000   300.00  0.078536  0.079852  0.079985    0.02052     774
humid_legacy         0.396   300.00  0.081958  0.080197  0.080020    0.01975     774
humid_van_wagner     0.396   300.00  0.080448  0.080045  0.080005    0.02010     774
```

and every check passes. The 1-h changes over 60 s follow forward Euler at
tau_eff = 0.902 h under the hysteresis rule as coded (a fuel above the
adsorption curve relaxes toward it, one below the desorption curve toward
that):

- dry legacy, toward E_w = 0.035: -8.2e-4;
- dry van_wagner, toward E_w = 0: -1.46e-3;
- humid legacy, toward E_d = 0.187: +1.96e-3;
- humid van_wagner, toward E_d = 0.104: +4.5e-4.

The burning-cell count is the same in all five decks: rate-of-spread
differences of 0.7-1.8 % over 60 s move the front by less than a fire cell.
A change to the hysteresis rule changes these numbers (legacy dry air would
relax toward E_d = 0.060, both humid decks toward E_w) but not the checks.
