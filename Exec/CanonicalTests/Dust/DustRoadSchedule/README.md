# DustRoadSchedule

## Purpose
The haul-road source alone: `erf.dust.test_ustar = 0` switches the wind source off, so every kilogram of dust comes from `road_schedule.csv` through the EPA AP-42 13.2.2 unpaved-road factor, E = 423 (s/12)^0.9 (W/3)^0.45 g per vehicle-kilometre (PM-10 constants, industrial sites). The case checks the schedule reader, the activation window and the mass rate.

## Setup
| Item | Value |
|------|-------|
| Domain, grid | 3000 x 3000 x 1024 m, 8 x 8 x 64 cells (375 x 375 x 16 m), periodic in x and y, flat; the dust grid is the atmosphere's columns (`erf.dust.grid_ratio = 1`) |
| Run length | `max_step = 5` at `erf.fixed_dt = 0.5` s: a start-up regression run |
| Source | `road_schedule.csv` only (`test_ustar = 0`): the columns are road name, bounding box, `road_width_m` (carried for the format, unused), vehicle weight [t], silt [%], vehicle kilometres per hour, start and end times (`-1` = whole run) |
| Bins | {7, 2.5, 50} um as one scalar; the road mass is the PM-10 factor and lands on bin 0, which must be a PM-10 size (start-up aborts otherwise) |

## What to look at
- `dust_road_diag.csv`: one row per active road and step with its mass rate M = 1e-3 E VKT/h / 3600 [kg/s], the per-cell flux M / (n A_cell) and the cell count n; the flux times n times the cell area is the AP-42 rate whatever cells the box covers (until October 2026 the flux per unit road area was stamped on every covered cell, 37.5x the AP-42 mass for a 20 m road on 375 m cells).
- `dust_diag.dat`: `emission_total_kg_s` equals the sum of the active roads' M; it drops to zero outside the schedule windows.
- The PM shares in `dust_naaqs.csv`: with the road the only source the emitted mass is all bin 0, so PM10 is the whole surface dust and PM2.5 is zero (the shares follow the emitted mass since October 2026).

## References
- EPA AP-42, section 13.2.2, Unpaved roads (2006).
