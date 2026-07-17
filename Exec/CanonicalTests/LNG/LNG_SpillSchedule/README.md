# ERF-LNG Phase 8: Spill Scheduling — Canonical Test

## Purpose

This test validates the Phase 8 spill scheduling implementation:

1. **CSV spill schedule parsing** — Rank 0 reads two events from CSV, broadcasts via `MPI_Bcast`
2. **Time-windowed activation** — Events activate/deactivate based on time windows (e.g., `[5, 15]` seconds)
3. **Multi-source pool contribution** — Multiple active events simultaneously add liquid to the pool
4. **Inventory tracking** — Cumulative mass released per event and total across all events
5. **CSV diagnostic output** — Per-timestep spill schedule diagnostics including active event count
6. **Backward compatibility** — Constant `spill_rate_kg_s` still works when no schedule file given
7. **MPI safety** — Broadcast-based CSV loading (POD struct array) works correctly on multi-rank

## Configuration

### ATM Domain (Verbatim from `LNG_Regulatory`)

- Grid: 32 × 32 × 64 cells
- Refined LNG grid: 128 × 128 cells (grid_ratio=4)
- Domain extent: 3000 × 3000 × 1024 m
- Time: 20 timesteps at dt=0.5 s = 10 s total
- Physics: MRF PBL, neutral ABL, geostrophic wind 15 m/s, surface layer BC

### LNG Configuration

- Pool: 500 m² initial area, 0.01 m initial depth
- Spill: 0.0 kg/s constant (disabled, using schedule instead)
- Gravity current: enabled with CFL-based velocity cap (Phase 7 post-merge fix)
- Flammability: LFL/UFL tracking enabled
- Regulatory: NFPA 59A 1h-average (Phase 7 retained)

### Spill Schedule (`spill_schedule.csv`)

```
spill_main       0.0   -1    1500.0  1500.0   10.0   20.0
spill_secondary  5.0   15.0  1600.0  1400.0   5.0    10.0
```

- **Event 1 (`spill_main`)**
  - Active entire simulation: t ∈ [0.0, ∞) (end_time_s = -1 means no end)
  - Location: center (1500, 1500) m
  - Radius: 10.0 m (pool area = π×10² ≈ 314 m²)
  - Rate: 20.0 kg/s

- **Event 2 (`spill_secondary`)**
  - Active time window: t ∈ [5.0, 15.0] s
  - During 20-step run (t ∈ [0, 10] s): active from step 11 to step 20 (t ∈ [5.0, 10.0] s)
  - Location: offset (1600, 1400) m
  - Radius: 5.0 m (pool area = π×5² ≈ 79 m²)
  - Rate: 10.0 kg/s

## MPI Pattern (LNG_MPI_SKILLS.md Rule B1)

1. **Rank 0 reads CSV:**
   - Opens `spill_schedule.csv`, parses 2 events
   - Stores in temporary vector on Rank 0 only

2. **MPI Broadcast (collective, not IOProcessor-gated):**
   - Broadcast event count (2) to all ranks via `MPI_Bcast`
   - All ranks allocate schedule.events vector
   - Broadcast entire LNGSpillEvent struct array (POD) via `MPI_Bcast(..., MPI_BYTE, ...)`
   - All ranks now have identical copy

3. **Per-rank execution:**
   - Each rank independently checks time windows and applies spill source
   - No further communication needed (all have same schedule)

## Pass Criteria (12 Items)

### 1. Exit Code and Completion
- [ ] Exit code 0
- [ ] All 20 timesteps complete
- [ ] No floating-point exceptions or seg faults

### 2. CSV Parsing and MPI Broadcast
- [ ] Stdout contains: `[LNG DEBUG] Phase 8: spill schedule loaded, 2 events from spill_schedule.csv`
- [ ] Per-event debug output appears (all ranks print independently):
  ```
  [LNG DEBUG] Phase 8:   event 0 name=spill_main t=[0,−1] s rate=20 kg/s at (1500,1500) m radius=10 m
  [LNG DEBUG] Phase 8:   event 1 name=spill_secondary t=[5,15] s rate=10 kg/s at (1600,1400) m radius=5 m
  ```

### 3. Time-Window Activation: `spill_main` (entire run)
- [ ] Active during steps 1–20 (t ∈ [0.0, 10.0] s)
- [ ] `[LNG DEBUG] Phase 8: spill event 'spill_main' ACTIVE ...` appears exactly 20 times in stdout

### 4. Time-Window Activation: `spill_secondary` (t ∈ [5, 15] s)
- [ ] Inactive steps 1–10 (t ∈ [0.0, 5.0) s) — no debug print for this event
- [ ] Active steps 11–20 (t ∈ [5.0, 10.0] s) — `[LNG DEBUG] Phase 8: spill event 'spill_secondary' ACTIVE ...` appears exactly 10 times

### 5. Diagnostic CSV Output
- [ ] File `lng_spill_diag.csv` created
- [ ] First line is header: `step,time_s,n_active_events,total_released_mass_kg,event_0_rate_kg_s,event_1_rate_kg_s`
- [ ] Exactly 20 data rows (steps 0–19)

### 6. Diagnostic CSV Content
- [ ] Rows 1–10 (steps 0–9, t ∈ [0.0, 5.0] s):
  - `n_active_events = 1` (only `spill_main`)
  - `event_0_rate_kg_s = 20.0`
  - `event_1_rate_kg_s = 0.0`
- [ ] Rows 11–20 (steps 10–19, t ∈ [5.0, 10.0] s):
  - `n_active_events = 2` (both events)
  - `event_0_rate_kg_s = 20.0`
  - `event_1_rate_kg_s = 10.0`
- [ ] `total_released_mass_kg` increases monotonically (cumulative sum of active masses)

### 7. Pool Mass Accumulation
- [ ] `lng_diag.csv` written and contains 21 lines (1 header + 20 steps)
- [ ] `pool_mass` column (Phase 2 output) shows:
  - Initial: ~4.25 kg (0.01 m × 500 m² × 425 kg/m³)
  - Increases during steps 1–10 (both spill rates: 20+10=30 kg/s)
  - Increases at ~30 kg/s during steps 11–20 (both sources contributing)
  - Rates: step 1: ~38 kg (4.25 + 20×0.5 + evap loss), step 11: ~68 kg (38 + 30×0.5 + evap loss)

### 8. Backward Compatibility Test (Not in Main Run)
- [ ] With `spill_schedule_file=""` and `spill_rate_kg_s=20.0`:
  - Spill schedule is not loaded: `[LNG DEBUG] Phase 8: spill_schedule_file empty`
  - Constant spill activates: `[LNG DEBUG] Phase 2: spill source applied (constant) rate=20 kg/s` ×20
  - `lng_spill_diag.csv` has 1 header + 20 rows (if schedule was dummy-loaded)

### 9. Regulatory Compliance (No Regression from Phase 7)
- [ ] `lng_regulatory.csv` created and written
- [ ] Plotfile written at step 10: `Plotfile: plt_lng_00010` in stdout
- [ ] Receptor CSV files generated (Phase 6): `center_receptor.csv`, `downwind_receptor.csv`

### 10. Flammability Diagnostics (No Regression from Phase 5)
- [ ] LFL/UFL areas computed and included in `lng_diag.csv` (Phase 5)
- [ ] No warnings about invalid concentrations

### 11. NaN Validation
- [ ] Stdout contains: `[LNG DEBUG] NaN check PASSED` exactly 20 times (once per step)
- [ ] All critical MultiFabs pass NaN check: pool_depth, evap_flux, vapor_conc, conc_1h_avg, exceed_flag

### 12. Build System
- [ ] Compiled with `-DERF_USE_LNG=ON` with no linker errors
- [ ] No compile warnings about unused variables or functions
- [ ] Binary size reasonable (~150–200 MB for LNG-enabled build)

## Reference Files

### Dust Module Analogs (Pattern Source)
- `ERF_DustBlastSchedule.H` — CSV read + MPI broadcast pattern for timed emission events
- `ERF_DustRoadSchedule.H` — Time-windowed activation periods for dust road sources

### LNG Module References
- `ERF_LNGSpillSchedule.H/cpp` — Phase 8 schedule parsing and application
- `ERF_LNGSpillScheduleDiag.H` — Phase 8 CSV diagnostics output
- `ERF_LNGPool.H/cpp` — `apply_spill_source()` function (reused by Phase 8)
- `LNG_MPI_SKILLS.md` — Rules B1 (collective broadcast), A2 (build system), A4 (CSV output)

## Execution

### Single-Rank Test (Correctness)
```bash
mpirun -n 1 Exec/ERF inputs_lng_spillschedule
```
Expected: All 20 steps complete, 2 events loaded, CSV files written.

### Multi-Rank Test (MPI Broadcast Safety)
```bash
mpirun -n 2 Exec/ERF inputs_lng_spillschedule
```
Expected: Same results on both ranks; all events broadcast correctly; CSV written (IOProcessor only).

## Known Limitations & Future Work

- **Spatial overlap:** Events with overlapping radii contribute additively to pool depth (no collision logic)
- **End time < start time:** Silently treated as inactive event (no error raised)
- **Large CSV files:** Full array broadcast; no streaming or chunked loading (O(n_events) MPI communication)
- **Irregular event spacing:** Times must be strictly increasing for correct mass computation (no sorting)

---

**Test created:** Phase 8 implementation (2026-07-17)
**Author:** Phase 8 Development
**ERF Branch:** copilot/erf-lng-phase8
**Phase Status:** 🔄 IN PROGRESS
