# FireStickMoisture

The dead-fuel moisture advanced as diffusion through a cylindrical stick,
`erf.fire.moisture_model = stick` (default `timelag`), the framework of
Nelson (2000): per class and per fire cell, radial diffusion on
`erf.fire.stick.n_shells` shells with the surface held at the air's
equilibrium moisture (or the rain value), the diffusivity set so the
slowest mode has the class's time lag, and the volume average handed to the
rate-of-spread models. The shells are checkpointed.

Five one-way decks of a grass fire in dry air, 60 s: time-lag (and with
the key written out), stick straight, stick to a checkpoint at 30 s, and
the restart from it.

```
MPIRUN="mpirun -np 4" ./run_stick.sh /path/to/erf_exec
```

The script compares the fire plotfiles at 60 s: the key written out
reproduces the historical deck exactly; the stick classes stay within
bounds and dry from the deck values in lag order; the stick deck differs
from the time-lag deck; and the restarted stick run reproduces the straight
one exactly, which only holds if the shells come back from the checkpoint.
The physics of the stick (relaxation, the calibrated lag, surface before
core, rain) is in the unit test `ERF_GTestFuelMoistureStick`, where hours
of stick time cost nothing.
