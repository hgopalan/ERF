# Moisture_Relaxation

Dead fuel drying in still, dry air, and the fire it releases once it is dry
enough. Each dead class relaxes towards the equilibrium moisture of the air with
its own time lag, dM/dt = (M_e - M) / tau_eff, tau_eff = tau exp(-0.015 (T - 20 C))
with tau = 1, 10, 100 h (Nelson 2000). Rothermel is rebuilt from the moisture every
step, so the rate of spread is zero until the 1-hour class falls below the 12 %
moisture of extinction of short grass, and follows it after that.

```
MPIRUN="mpirun -np 4" ./run_moisture_relaxation.sh /path/to/erf_exec
```

## The case

400 x 400 m, still atmosphere at 300 K with no moisture model, so the relative
humidity the fuel sees is zero (clamped to 1 %) and nothing changes the air.
Anderson fuel model 1 (all of its dead load is 1-hour fuel), every dead class
starting at 20 %, `moisture_dynamic = true`, an 8 m ignition disc, two hours in
5 s steps, a fire plotfile every 10 minutes. `check_moisture_relaxation.py`
checks at every plotfile each class against the model's forward-Euler steps and
against the closed form, the rate of spread against Rothermel at the 1-hour
moisture, and the burned radius against r_ig plus the integral of that rate.

Nelson's equilibrium moisture has an adsorption (wetting) and a desorption
(drying) curve, E_w = 0.0351 and E_d = 0.0600 here. As implemented
(`compute_emc_with_hysteresis` in `Source/Fire/ERF_FuelMoisture.H`) the choice is
the reverse of Nelson's: fuel above the adsorption curve relaxes towards it and
fuel below the desorption curve towards that, so a fuel drying from 20 % heads
for E_w and then holds at E_d,

    M(t) = max(E_w + (M0 - E_w) exp(-t / tau_eff), E_d),

where Nelson's hysteresis gives E_d + (M0 - E_d) exp(-t / tau_eff). The check
follows the code and prints Nelson's value next to it; the branch choice is a
separate fix.

## Expected Results

On four ranks, T = 26.85 C and tau_eff = 0.9024 h:

- every class matches the stepwise solution to 1e-16 and the closed form to
  5e-5 at every plotfile;
- the 1-hour class reads 0.0895 at one hour (Nelson's curve: 0.106) and holds at
  0.0600 from about 6000 s;
- the rate of spread is zero (Rothermel's floor) until the 1-hour class crosses
  12 % at t = 2157 s, and matches Rothermel at the 1-hour moisture to six digits
  at every plotfile after that;
- the burned radius follows r_ig plus the integral of the rate to 0.08 cells,
  reaching 102.7 m at two hours.
