# MovingBodies at 1000+ turbines: bottlenecks and PR plan

These are working notes for scaling `Source/MovingBodies` (branch `ERF-MovingBodies` of `hgopalan/ERF`)
to farms of 1000+ OpenFAST turbines. The work is split into 18 PRs, one issue each, to be done one at a time.

- **Code analysed:** `ERF-MovingBodies` at `3d427e7` (#528), which is newer than the `repo.MD` snapshot
  (`8402aae`, #517). Written 2026-10-16.
- **Line numbers refer to `3d427e7`.** They drift as PRs land, so look each function up again before editing it.
- **Also reviewed:** Kynema-SGF (`kynema/kynema-sgf`, formerly AMR-Wind), whose OpenFAST coupling lives in
  `src/wind_energy/actuator/`, for ideas to adopt or avoid.
- **The costs are order-of-magnitude estimates.** PR 1 adds the timers that will replace them with measurements.
- **No code was changed** for this analysis.

**How to use these notes:**
- Tick a PR in §2 when it merges.
- Check §3 before opening each PR.
- Update §5 when PR 1's measurements change the picture.

## Contents

1. [Assumptions](#1-assumptions)
2. [Progress tracker](#2-progress-tracker)
3. [Rules for every PR](#3-rules-for-every-pr)
4. [PR specs](#4-pr-specs)
5. [Bottlenecks and evidence](#5-bottlenecks-and-evidence)
6. [Kynema-SGF: adopt and avoid](#6-kynema-sgf-adopt-and-avoid)
7. [Settings that help today](#7-settings-that-help-today)
8. [Open questions and measurements](#8-open-questions-and-measurements)
9. [Reference: the step and the files](#9-reference-the-step-and-the-files)

## 1. Assumptions

- **Farm:** about 1000 IEA 15 MW turbines (R ≈ 120 m), with Δx = 10 m on the anchor level.
- **Input defaults** (`ERF_MovingBodiesInputs.H`):
  - `epsilon` = 2 cells (`:28`). The kernel reaches 3ε = 6 cells, so each point touches a cube of
    13³ ≈ 2.2k faces in each direction.
  - `num_force_points_blade` = 50 (`:40`) and `num_points_t` = 16 (`:55`).
  - `diagnostics_int` = 1 (`:118`), so diagnostics are written every step.
  - `avg_start` = 0, `alm_max_tip_cells` = 1 and `envelope_check_interval` = 1.
  - Wake lines are off unless `erf.moving_bodies.wake.lines_xD` is given. When they are on, each line has 61 points
    and is sampled every `erf.moving_bodies.wake.int` steps (0 means `diagnostics_int`).
- **Points per turbine:**
  - ADM: 1 + 3 blades × 50 nodes × 16 = 2401 (`adm_rings`, `ERF_OpenFASTRotor.cpp:266-311`).
  - ALM: about 150–200 (blade nodes plus tower and nacelle), roughly 15× fewer.
- **No regridding and no dynamic meshing.**
  - The grids, the DistributionMapping, `z_phys_nd` and `detJ_cc` are fixed after start-up, so anything derived
    from the mesh can be built once and cached.
  - The exception is `TerrainType::MovingFittedMesh`, which rewrites `z_phys_nd` every step
    (`ERF_MovingBodies.cpp:515-516`). It must never use those caches.
- **Turbine bases stand still.** `Motion/ERF_BodyMotion.H` has only `FixedMotion`. All other motion (rotor, yaw,
  deflection, and platform motion in floating models) comes from OpenFAST.

## 2. Progress tracker

Suggested order:
1. Measure first (PR 1).
2. The safe start-up and I/O PRs (2–8).
3. The per-step communication and compute PRs (9–13).
4. The opt-in PRs that change answers or the run layout (14–18).

"Bit-identical" means the golds don't change.

- [ ] **PR 1 `mb-profiling`:** timers in every phase. Fixes B16. Bit-identical. Do it first: its timings decide
  whether PRs 12, 13, 17 and 18 are worth doing.
- [ ] **PR 2 `mb-startup-parse-once`:** parse each OpenFAST input file once, not on every rank. Fixes B13.
  Bit-identical.
- [ ] **PR 3 `mb-startup-coverage-once`:** build the coverage BoxArrays once, and check disks and wake lines on
  their owners only. Fixes B14. Bit-identical.
- [ ] **PR 4 `mb-diagnostics-files`:** keep log files open, create directories once, and rewrite statistics files
  rarely. Fixes B8. Bit-identical; file contents are identical at every flush point.
- [ ] **PR 5 `mb-farm-totals-sparse`:** send farm-total rows point-to-point and fuse the source integrals. Fixes B9.
  Bit-identical; logs agree to every printed digit.
- [ ] **PR 6 `mb-restart-trim-parallel`:** owners trim their own logs, reading from the end of the file. Fixes B10.
  Bit-identical.
- [ ] **PR 7 `mb-restart-state-owner-read`:** only owners read their bodies' restart state. Fixes B11. Bit-identical.
- [ ] **PR 8 `mb-checkpoint-layout`:** per-body checkpoint subdirectories, binary node state, no extra barrier.
  Fixes B12. Bit-identical, and old checkpoints still restart.
- [ ] **PR 9 `mb-static-vertical-reach`:** find the exact vertical reach from static height tables instead of using
  the whole column. Fixes B3. Bit-identical.
- [ ] **PR 10 `mb-static-envelopes`:** static envelopes and participants, and tip cells and `check_levels` with no
  per-step collectives. Fixes B6. Bit-identical. Easier after PR 9.
- [ ] **PR 11 `mb-sparse-source-boxes`:** zero, add and integrate the source only on boxes a kernel can reach.
  Fixes B7. Bit-identical. Needs PR 10.
- [ ] **PR 12 `mb-spread-kernel-parallelism`:** more threads per point, fused launches, OpenMP, device binning.
  Fixes B5. Bit-identical if the summation order is kept. After PR 1; better after PR 9.
- [ ] **PR 13 `mb-exchange-bookkeeping`:** precomputed message sections, a static rank lookup, reused buffers.
  Only if PR 1 shows host bookkeeping matters. Bit-identical. After PR 10.
- [ ] **PR 14 `mb-one-round-norm`:** the owner computes the whole normalisation, so spreading takes one message round
  instead of three. Fixes B4. Opt-in; last-bit change. After PRs 9 and 10.
- [ ] **PR 15 `mb-adm-reduced-points`:** ADM spreading with merged rings or a separable projection. Fixes B2.
  Opt-in; changes answers and needs new golds.
- [ ] **PR 16 `mb-owner-cost-model`:** owner choice weighted by cost. Fixes B15. Opt-in; bit-identical.
- [ ] **PR 17 `mb-openfast-ranks`:** extra ranks that hold no boxes and only run OpenFAST. Fixes B1 when turbines
  outnumber field ranks. Bit-identical. After PR 16.
- [ ] **PR 18 `mb-openfast-overlap`:** overlap OpenFAST with the first RK stage. Fixes B1. Only if PR 1 shows that
  OpenFAST dominates. Opt-in; roundoff change. After PR 1, and on CPU after PR 17.

PRs 2–8, 15 and 16 depend on nothing else.

## 3. Rules for every PR

These come from `repo.MD`, plus one rule added for this series (static-mesh caches).

- **Branch and target:** branch `mb-<topic>` off the latest `ERF-MovingBodies`, and target `ERF-MovingBodies`.
- **Registration:**
  - New sources go in both `CMake/BuildERFExe.cmake` and the folder's `Make.package`.
  - New gtests go in `Tests/Unit/CMakeLists.txt` (in the OpenFAST block if needed).
  - `BodyExchange` stays outside `ERF_USE_OPENFAST`, so the disk-only build keeps working.
- **Builds**, all with warnings on and 0 warnings: stub, real OpenFAST 5.0.0, OpenFAST off, moving bodies off,
  MPI off, single precision.
- **Invariants:**
  - Results don't depend on the decomposition. Exchange results equal the global sampler and spreader bit for bit
    on 1–2 ranks and within 1e-12 on 4, and logs agree to every printed digit.
  - Every rank makes the same collective calls, with the same control flow.
  - Restarts work on the same rank count and on a different one (`*_RestartOnOneRank`).
  - Device lambdas capture values only, never `this`.
  - Files are written by the right rank:
    - owner: `_erf`, `_flow`, `_stats`, `_correction`, `_fllc`, `_disk`, `_wake`, `_wake_avg`, `_nodes.dat`,
      `_fllc.dat`;
    - I/O rank: `total_load.csv`, `momentum_source.csv`, `ground.csv` and the checkpoint `state`.
  - Golds change only with a stated reason. A deliberate change keeps the old behaviour selectable.
- **Static-mesh caches (new for this series):**
  - Build them once, after `set_ground`.
  - Keep a cheap check that the grids are unchanged; the code already compares BoxArrays in `advance` and
    `ensure_envelopes`. If the grids change, rebuild the caches or abort.
  - Never use them under `MovingFittedMesh`.
- **New keys** get:
  - validation that names the key;
  - a row in `Docs/sphinx_doc/Inputs.rst`;
  - a description in `theory/MovingBodies.rst` (or `CouplingToOpenFAST.rst`);
  - an abort test.
- **Tests:**
  - unit tests in `erf_unit_tests` and `erf_parallel_tests` on 1, 2 and 4 ranks;
  - planted bugs, reported as "N of M mutants caught";
  - GPU-style CPU checks: SNaN-filled memory, FP traps, tiled vs untiled, one box per rank;
  - box, restart and owner parity where relevant.
- **Before pushing:**
  - `ctest --test-dir build -L moving-bodies`;
  - restart parity (`-L restart-parity -R "OpenFAST|MovingBodies"`);
  - box parity (`-L box-parity -R "OpenFAST|Actuator"`);
  - the parallel gtests (`ActuatorSampling`, `ActuatorSpreading`, `BodyExchange`);
  - `-L openfast-real` when rotor or ALM code changes.
- **Style:** `.github/workflows/style/check_tabs.sh`, `check_trailing_whitespaces.sh`, `codespell`,
  `git diff --check`, and a Sphinx build with `-W`.
- **PR description:** use the `repo.MD` template (Summary, Theory, Changes, Results, Tests, Verification, Not run).
  Performance PRs put before-and-after times from PR 1's timers, and from the scaling deck (§8), in Results.
- **Run log:** log each body's owner and exchange partners.

## 4. PR specs

Each spec gives the problem, the change, how answers change, and the tests, plus a Kynema-SGF cue where one applies.
New keys and changed behaviour also need the documentation listed in §3.

### PR 1: `mb-profiling` (B16)

**Problem.** There are no timers anywhere in `Source/MovingBodies`. Without them the costs in §5 can't be checked,
and the later PRs can't show their gains.

**Change.**
- Add `BL_PROFILE` regions inside `MovingBodies::advance` (`ERF_MovingBodies.cpp:860-969`) for:
  - velocity sampling, and each exchange round;
  - the OpenFAST step, per turbine and in total;
  - both spreading passes;
  - zeroing the source (`:1205-1207`) and adding it (`:1304-1306`);
  - each diagnostics writer, the statistics, and wake sampling.
- Also time the checkpoint write and read, and start-up: the driver constructor, `set_ground`, the audits, the
  coverage check, and wake set-up.
- Optionally print a summary every K steps, with the minimum, mean and maximum over ranks for each phase. Its reduction
  runs only on summary steps, on every rank. A key for it needs the usual key work (§3).
- The regions show up with `-DERF_ENABLE_TINY_PROFILE=ON`.

**Answers.** Bit-identical.

**Tests.**
- The existing suite.
- If the summary is added: an abort test for its key, and a check that nothing extra prints by default.

**Kynema cue.** It wraps every actuator phase in `BL_PROFILE` (`Actuator.cpp`, `turbine/fast/FastIface.cpp`).

### PR 2: `mb-startup-parse-once` (B13)

**Problem.** Every rank parses every turbine's OpenFAST input files at start-up:
- `check_induction` (`ERF_OpenFASTDriver.cpp:47`) runs in the driver constructor (`:87`, call at `:98`);
- `check_tower_shadow_off` (`:74`) is called from the `MovingBodies` constructor (`ERF_MovingBodies.cpp:212`).

That is about 7 file opens per turbine per rank. With 1000 turbines on 4096 ranks it comes to about 29 million opens
of the same files, all on one parallel file system. Separately, the audit's `openfast_value`
(`ERF_OpenFASTAudit.cpp:18-34`) reopens a file for every key it looks up.

**Change.**
- Parse each distinct file once, memoised by path, either on the I/O rank or spread over the owners.
- Broadcast the resulting flags and messages in one message.
- Give the audit one parsed key map per file.

**Answers.** Bit-identical. The induction and tower-shadow aborts still fire, with the same messages.

**Tests.**
- The existing abort tests, on 2 ranks with the turbine owned by rank 1.
- A unit test that the memoised parser returns the same values as today's parser on the stub and IEA 15 MW inputs.
- Planted bug: a skipped broadcast must fail a test.

### PR 3: `mb-startup-coverage-once` (B14)

**Problem.**
- `points_covered_by` (`ERF_ActuatorSampling.cpp:735-809`) rebuilds a converted BoxArray on every call, and also a
  flattened one for footprints (`:742-756`).
- `check_coverage` calls it for every disk and every wake line, on every rank (`ERF_MovingBodies.cpp:1576-1595`).
- `build_wake_lines` calls it again for every wake line (`:1059`), and runs one collective `terrain_heights` per
  rotor (`:1021`).

For 1000 bodies that is about 1000 BoxArray builds per rank and about 2000 small collectives. It is slow, but it
happens only at start-up.

**Change.**
- Build the converted and flattened BoxArrays once per check.
- Check disks and wake lines on their owners only. Combine the results with one integer reduction, and abort with the
  owner's message.
- Make one batched `terrain_heights` call for all rotors.

**Answers.** Bit-identical, with the same abort messages.

**Tests.**
- The existing coverage abort tests, on 2 ranks with the body on rank 1.
- Planted bug: a non-owner that skips its share of the reduction must fail.

### PR 4: `mb-diagnostics-files` (B8)

**Problem.** On every diagnostics step (every step by default), each owner opens, writes and closes its bodies'
files:
- `_erf.csv` and `_flow.csv` (`ERF_OpenFASTDriver.cpp:584,593`);
- `_stats.csv`, rewritten in full (`ERF_RunningStats.H:95-105`, called at `ERF_MovingBodies.cpp:986,994`);
- `_disk.csv` (`ERF_PrescribedCtDisk.cpp:109`);
- `_correction.csv` and `_fllc.csv`, when those features are used;
- with wake lines, `_wake.csv` and a full rewrite of `_wake_avg.csv` (`ERF_WakeLines.cpp:112-134`).

Every file opened through `open_log` (`ERF_DiagnosticsLog.H:24-36`) also runs `UtilCreateDirectory` (one `mkdir`
per path component) and a probe `ifstream`.
- For 1000 turbines that is about 3–7k opens and 10–30k file-system metadata operations per step.
- Wake lines add about 44 KB per body per sample: about 1.5 TB of CSV over 36k steps for 1000 bodies.

**Change.**
- Create the output directories once, at start-up.
- Owners keep their log streams open for the whole run:
  - flush every M rows, at every checkpoint (so that restart trimming sees every row), and at exit;
  - cap the number of open handles when an owner holds many bodies, because about 5 per turbine × N/P can reach
    the default limit of 1024.
- Rewrite `_stats.csv` and `_wake_avg.csv` only every K diagnostics steps, at checkpoints and at the end.
- Optionally write the wake samples per body in a binary or NetCDF format.

**Answers.** The solution is bit-identical, and file contents are byte-identical at every flush point.

**Tests.**
- Restart parity: rows after the checkpoint are trimmed and then appended again, exactly once.
- The owner-file planted-bug tests.
- The files are complete after a normal exit and at a checkpoint.

**Kynema cue.**
- Each turbine's root rank writes one NetCDF file every `output_frequency` steps (default 10).
- The output directory is created once, in `prepare_outputs` (`turbine/fast/turbine_fast_ops.H`, `ProcessOutputsOp`).

### PR 5: `mb-farm-totals-sparse` (B9)

**Problem.** Two costs on each diagnostics step:
- `write_total_load` gathers a dense N × 5 table (`turbine_table(5)`, `ERF_MovingBodies.cpp:1141`) through
  `gather_rows` (`:44-59`). That is a reduction of 5N reals from every rank, almost all of them zeros.
- `write_source_diagnostics` makes three full-level `integrate_source` passes (`:1175-1177`;
  `ERF_ActuatorSpreading.cpp:609-631`), each with its own reduction.

**Change.**
- Owners send their rows point-to-point to the I/O rank, which writes them in body order.
- One fused pass integrates all three components, with one 3-value reduction. After PR 11 it covers only the boxes that
  hold sources.

**Answers.**
- The solution is bit-identical.
- `total_load.csv` is byte-identical.
- `momentum_source.csv` should be byte-identical too; at worst it agrees to every printed digit.

**Tests.** The `fx == -thrust_x` and `-load_x` checks in `RunOpenFASTADM.cmake`, and restart parity.

### PR 6: `mb-restart-trim-parallel` (B10)

**Problem.** On restart, the I/O rank alone trims every log back to the checkpoint (`trim_log_after`,
`ERF_DiagnosticsLog.H:57-80`, called at `ERF_MovingBodies.cpp:784-805`).
- It reads each file into memory in full and then rewrites it.
- Every other rank waits at the barrier (`:804`).
- For 1000 bodies that is about 5000 files, which can total hundreds of GB after a long run.

**Change.**
- Each owner trims its own bodies' logs, and the I/O rank trims the farm files; then one barrier.
- Find the cut by reading backwards from the end of the file, then truncate in place
  (`std::filesystem::resize_file`).
- Keep today's rules: the header is kept, unparsable rows are kept, and end-stamped and start-stamped logs keep their
  tolerance rule.

**Answers.** Bit-identical; the files come out the same.

**Tests.**
- Restart parity, on the same rank count and on a different one.
- A unit test of the trim with rows past the checkpoint, with none past it, with end- and start-stamped rows, with an
  empty file, and with a missing file.
- Planted bug: an owner that trims a file it doesn't own must fail.

### PR 7: `mb-restart-state-owner-read` (B11)

**Problem.** On restart every rank reads the statistics state of every body (`ERF_MovingBodies.cpp:677-678`) and of
every wake line (`:1074`), through `RunningStats::read_state` (`ERF_RunningStats.H:124-177`).
- Each file costs one `file_exists_everywhere` broadcast and one `ReadAndBcastFile` (two broadcasts). That is
  about 3 broadcasts × 2000 files.
- Only the owner keeps the data.
- The owners are already known at this point, because `set_ground` runs before `read_checkpoint`.

**Change.**
- Only owners read their bodies' state files.
- Existence and validity are combined in one integer reduction, so every rank aborts together, with the owner's
  message.
- The owners print the restored-count messages.

**Answers.** Bit-identical.

**Tests.**
- Restart parity, including on a different rank count.
- Abort tests for a missing state file and for a corrupt one, on 2 ranks with the body owned by rank 1
  (`amr.max_grid_size_x = 24`).

### PR 8: `mb-checkpoint-layout` (B12)

**Problem.** `write_checkpoint` (`ERF_MovingBodies.cpp:223-290`) does:
- a `ReduceLongSum` (`:231`) and a dense `turbine_table(1)` (`:232`);
- a global `Barrier` (`:263`);
- about 3–6 file writes per turbine into one flat directory: the statistics state, the wake state, `_nodes.dat`,
  `_fllc.dat`, and OpenFAST's own checkpoint.

That is about 5k file creates in one directory per checkpoint, which parallel file systems handle badly.

**Change.**
- Per-body or per-owner subdirectories under `<chk>/moving_bodies/`.
- Node state in binary.
- No global barrier beyond what ERF's checkpoint already needs.
- Reading falls back to the flat layout, so older checkpoints still restart.

**Answers.** Bit-identical.

**Tests.**
- Restart parity, on the same rank count and on a different one.
- A restart from an old-layout checkpoint.

**Kynema cue.** OpenFAST checkpoints are written per turbine, every `chkpt_interval` steps.

### PR 9: `mb-static-vertical-reach` (B3)

**Problem.** ERF always allocates `z_phys_nd` (`ERF_MakeNewArrays.cpp:55`) and `detJ_cc` (`:124-125`, commented
"regardless of terrain" and set to 1). So in the exchange, `fitted` is true even on flat meshes
(`ERF_BodyExchange.cpp:489,621`). Three things follow:
- **Routing uses the whole column.** `cells_within` (`:73-88`) stretches every box to the domain height. Sampled
  points (`:512`) and kernel points (`:655`) therefore go to every rank with boxes in that column, and each of those
  ranks replies.
- **Kernel loops use the whole box height.** `KernelFaces` (`ERF_ActuatorSpreading.cpp:123`) drops the height bounds
  for a whole rank if any of that rank's boxes doesn't reach the ground (`:147`). `ZBounds::k_range`
  (`ERF_ActuatorSampling.H:131-152`) then returns the full box height, so the loops can do up to about 5× useless
  work.
- **The mesh bounds are swept every step.** `mesh_z_bounds` (`ERF_ActuatorSampling.cpp:416-452`) sweeps the whole
  level on every step (`ERF_BodyExchange.cpp:751`).

**Change.**
- Build static per-k height tables once: the lowest and highest node height of each k level, both globally and per
  box. Also detect flat and 1-D-stretched meshes.
- Use the tables to get exact k ranges for routing, participants, sampling and both kernel passes. They replace the
  per-step `mesh_z_bounds` call.

**Answers.** Bit-identical: only faces and ranks with exactly zero kernel weight are skipped.

**Tests.**
- Box-parity twins on terrain with `amr.max_grid_size_z` < nz (boxes stacked in z), and on a stretched mesh.
- Parallel gtests on 1, 2 and 4 ranks against the global sampler and spreader.
- Planted bug: a k range that is one level too narrow must be caught.

### PR 10: `mb-static-envelopes` (B6)

**Problem.** Every step runs global synchronisations, and each one makes every rank wait for the slowest owner (B1).
- **Envelope checks.**
  - With `envelope_check_interval` = 1, `ensure_envelopes` (`ERF_BodyExchange.cpp:387-479`) does a `ReduceIntMax`
    (`:436`) on every exchange: sampling, spreading, and wake sampling on wake steps.
  - When an envelope grows, it adds a 6N-value `ReduceRealSum` (`:466`) and `find_participants` for every body
    (`:477`).
  - An interval above 1 is a partial fix that exists today. It builds envelopes from the whole sphere the rotor
    sweeps (`:452-460`) and checks them only every K exchanges.
- **Tip cells.** With any ALM turbine, `update_tip_cells` (`ERF_MovingBodies.cpp:488-568`, called at `:1231`) does a
  `ReduceIntMax` every step (`:527`), whatever the envelope interval.
  - If a tip region has moved, it adds an N × 9 `turbine_table` reduction (`:530`) and a collective
    `tip_cell_heights_batched` (`:544`).
- **Finer levels.** When finer levels exist, `check_levels` (`:1432-1497`, called every step from
  `ERF_Advance.cpp:178`) tests every force point of every owned turbine against the finer grids.
  - It is owner-local, with no collective.
  - But it makes about 2400 `BoxArray::intersects` calls per ADM turbine, per finer level, per step.

**Change.**
- **Envelopes.**
  - Build each body's envelope and participant list once, from its geometry:
    - the sphere the rotor can sweep at any yaw angle, that is, the tip radius plus the kernel reach;
    - the tower down to the ground;
    - a margin for platform motion.
  - Keep the owner-local "left its envelope" abort (`:399-405`). There is no per-step reduction.
- **Tip cells.** Each owner gathers a static patch of `z_phys_nd` around its rotor once. When its tip region moves,
  it finds the limiting cell height locally, with no per-step reduction and no N × 9 table.
- **`check_levels`.** Check the static envelopes against the finer levels once.

**Answers.** Bit-identical. Extra participants add exact zeros, and the tip-cell heights come from the same formula
on the same data.

**Tests.**
- Owner tests (`add_test_moving_bodies_owners`) and the exchange-partner lines in the run log.
- Planted bug: a body that leaves its envelope must abort.
- Restart parity, the parallel gtests, and runs with `verify_tip_cells = 1`.

**Kynema cue.**
- Each turbine has a static `bound_box`: the base ± 1.25 R, up to hub height + 1.25 R (`turbine/turbine_utils.cpp`).
- `determine_influenced_procs` (`actuator_utils.cpp`) finds the influenced ranks, and reruns only after a regrid
  (`post_regrid_actions`).
- There is no per-step envelope reduction.

### PR 11: `mb-sparse-source-boxes` (B7)

**Problem.** The momentum source is nonzero only near the rotors, but it is handled over the whole level:
- `spread_sources` zeroes three full-level face MultiFabs every step (`ERF_MovingBodies.cpp:1205-1207`);
- `add_momentum_sources` (`:1299-1307`, called from `ERF_TI_slow_rhs_pre.H:219-224`) adds them to ERF's sources over
  the whole level, in every RK stage.

That makes about 12 full face-array passes per step, plus 3 more on diagnostics steps (`integrate_source`). The three
face MultiFabs also take memory over the whole level.

**Change.**
- With static envelopes (PR 10), the set of local boxes that any kernel can reach is fixed.
- Zero, add and integrate only on those boxes.
- Optionally allocate the source data only there.

**Answers.** Bit-identical.

**Tests.**
- Box parity.
- Tiled vs untiled.
- SNaN-filled memory, so that untouched boxes are never read.

### PR 12: `mb-spread-kernel-parallelism` (B5)

**Problem.** The spreading kernels use little of the hardware.
- **Normalisation pass.** It runs one thread per point per box, and each thread loops serially over every face its
  kernel reaches (`spread_partial_norms`, `ERF_ActuatorSpreading.cpp:337`, loop at `:400`). With 2401 points per ADM
  turbine, a GPU sees a few thousand threads that each do about 2–7k kernel evaluations.
- **Launches.** Both passes launch once per box and direction (`:387`, `:522`). The `MFIter` loops are untiled and
  have no OpenMP.
- **Binning.** Points are binned on the host and uploaded every step (`:368`, `:501`).

**Change.**
- Use several threads per point in the normalisation pass, and combine their partial sums in a fixed order.
- Fuse launches across boxes and directions.
- On CPU, add OpenMP over boxes, and combine the per-box partial sums in box order.
- Bin the points on the device.

**Answers.** Bit-identical if today's summation order is kept. Otherwise a last-bit change, with the reason stated.

**Tests.**
- The parallel gtests, bit for bit on 1–2 ranks.
- Tiled vs untiled.
- Identical results with 1 OpenMP thread and with N.
- The GPU-style CPU checks.

**Kynema cue.** `#pragma omp parallel if (Gpu::notInLaunchRegion())` over `MFIter` in the source-term loop of
`Actuator.cpp`.

### PR 13: `mb-exchange-bookkeeping` (only if PR 1 shows it matters)

**Problem.** Host bookkeeping on every exchange scales with the number of bodies or points:
- `sections` (`ERF_BodyExchange.cpp:62-69`) loops over all bodies, once per partner and per round;
- `ranks_touching` (`:132-141`) builds a `std::set` and calls `BoxArray::intersections` for every point, every step
  (`:512`, `:655`);
- message buffers are reallocated every step.

**Change.**
- Precompute each partner's sections.
- Find each point's rank from a static map. This becomes possible once PR 10 makes the envelopes static.
- Reuse the buffers.

**Answers.** Bit-identical.

**Tests.** The parallel gtests on 1, 2 and 4 ranks, and the owner tests.

### PR 14: `mb-one-round-norm` (B4, opt-in)

**Problem.** Spreading always takes three message rounds: points out, partial normalisations back, scales out
(`ERF_BodyExchange.cpp:760-851`).
- A one-round path exists: the owner can find a point's whole normalisation with `uniform_mesh_norm`
  (`ERF_ActuatorSpreading.cpp:431-473`).
- But that path is used only when both `z_phys_nd` and `detJ_cc` are null (`ERF_BodyExchange.cpp:672-681`), which
  never happens in a run (see PR 9).
- Today it runs only in unit tests, and it is a serial host loop.

**Change.**
- With a static mesh, the owner computes each point's whole normalisation:
  - on a flat or 1-D-stretched mesh, on the device, with the logic of `uniform_mesh_norm`;
  - on terrain, from a static owner-held patch of `z_phys_nd` and `detJ_cc` that is gathered once.
- The owner sends F/S with each point, so there is one round instead of three.
  - The owner's face set must match the participants' faces exactly, including periodic images and domain edges.
  - Points whose reach isn't covered fall back to three rounds, as they do with `uniform_mesh_norm` today.
- Keep the discrete normalisation, so that `fx == -thrust_x` stays exact.
- Add a new key, e.g. `erf.moving_bodies.spread_normalisation = exchange | owner`, with `exchange` as the default.

**Answers.** A last-bit change.
- The normalisation is summed in one order on the owner, instead of from per-rank partial sums added in rank order.
- After the change, results are identical on every rank count. Today 4 ranks agree only within 1e-12.

**Tests.**
- An option-parity test against the three-round path: agreement within 1e-14, and logs to every printed digit.
- Box parity, and bit-identical results on 1, 2 and 4 ranks with the new option.
- Planted bug: a face missing from the owner's set must be caught.

### PR 15: `mb-adm-reduced-points` (B2, opt-in, changes answers)

**Problem.** `adm_rings` (`ERF_OpenFASTRotor.cpp:266-311`) gives every blade force node its own ring of
`num_points_t` points, all at the same angles.
- A 3-bladed rotor with 50 nodes per blade becomes 2401 points.
- The three blades' rings nearly coincide; they differ only by deflection.
- Spreading costs about 3×10⁷ kernel evaluations per turbine per step, or about 3×10¹⁰ for the farm. That is the
  main compute cost of an ADM farm.

**Change.** A new per-body key, e.g. `erf.moving_bodies.<name>.adm_spreading`:
- `rings`: today's behaviour, and the default;
- `merged_rings`: one ring per radial station, carrying the three blades' summed force. That is about 800 points,
  3× cheaper.
- `projection`: Kynema-style separable weights taken straight from the blade nodes, with no ring points. The weight
  is a radial hat × an azimuthal hat × a 1-D Gaussian normal to the disk.

Every mode keeps the exact total-force normalisation (`fx == -thrust_x`).

**Answers.**
- Only the new modes change answers, and the default golds stay.
- The new modes need new cases with their own golds: stub ADM, disk-corrected sampling, and the real-OpenFAST
  Askervein case.

**Tests.**
- The new regression cases.
- Option parity between `rings` and `merged_rings` on a rigid rotor, where they should agree closely.
- Thrust and torque comparisons in the PR's Results table.

**Kynema cue.** `turbine/ActSrcDiskOp_Turbine.H` and `disk/disk_spreading.H` (`linear_basis_spreading`).

### PR 16: `mb-owner-cost-model` (B15, opt-in)

**Problem.** `choose_owners` (`ERF_BodyExchange.cpp:144-236`) balances the number of bodies per rank, not their
cost; it does keep OpenFAST time-step classes apart.
- When turbines outnumber ranks, an ALM turbine with 100 substeps costs far more than a prescribed-Ct disk.
- Every step then waits for the slowest owner.

**Change.**
- Optional per-body weights: the OpenFAST substeps (`erf.fixed_dt / DT`), the node count, and the body type (ADM,
  ALM or disk). Later, measured costs from PR 1.
- Keep the time-step classes apart.
- A new key, with today's count-based choice as the default.

**Answers.** Bit-identical, because the owner adds the normalisation parts in rank order whichever rank it is. The
owner-test expectations change only when the option is on.

**Tests.**
- `add_test_moving_bodies_owners` cases for the weighted choice.
- A restart on a different rank count.

**Kynema cue.** `determine_root_proc` (`actuator_utils.cpp`) picks a free influenced rank first, and otherwise the
least-loaded one. ERF's choice is already close to this.

### PR 17: `mb-openfast-ranks` (B1, when turbines outnumber field ranks)

**Problem.** GPU machines have few ranks (one per GPU) but many idle CPU cores. With 1000 turbines on 64 GPU ranks,
each owner steps about 16 turbines in series, on the critical path.

**Change.**
- Optionally reserve MPI ranks that hold no boxes and only own turbines:
  - build the anchor level's DistributionMapping over the field ranks only, which is feasible because the grids never
    change;
  - `choose_owners` then puts the turbines on the reserved ranks.
- Threads can't replace these ranks. OpenFAST keeps one step counter (`n_t_global`) per process
  (`ERF_OpenFASTDriver.cpp:351`, `check_lockstep` at `:393`).
- This touches ERF's level setup outside `Source/MovingBodies`, so upstream merges need care.

**Answers.** Bit-identical.

**Tests.**
- Owner tests with reserved ranks.
- Box parity.
- Restarts on different rank counts.
- The MPI-off build is unaffected.

### PR 18: `mb-openfast-overlap` (B1, only if PR 1 shows OpenFAST dominates; opt-in)

**Problem.** Each step is a strict chain: sample, then OpenFAST, then spread, then the RK stages.
- The OpenFAST step is `m_driver->step()` (`ERF_MovingBodies.cpp:940`; `ERF_OpenFASTDriver.cpp:346-378`).
- It takes about 20–100 substeps of about 0.2–3 ms each, which is 5–300 ms per turbine.
- All of that time adds directly to every step.

**Change.**
- Optionally overlap OpenFAST with the first RK stage:
  - a host thread steps OpenFAST while the stage-1 slow RHS is computed without the moving-bodies source;
  - the spread source is added to the stage-1 RHS once it is ready;
  - stages 2 and 3 use the source as they do today.
- On GPU builds the host is mostly idle during the RHS kernels. On CPU builds this needs PR 17's OpenFAST ranks.
- First confirm that `erf_slow_rhs_pre` is linear in the momentum source, so that the source can be added afterwards.

**Answers.**
- A roundoff change, because the source enters the sum in a different place.
- A new key, off by default, with an option-parity test against the default.

**Not proposed:** lagged coupling (forces from the previous step), which changes answers at O(dt).

## 5. Bottlenecks and evidence

All references are to `3d427e7`, and the costs assume §1.

### Every step

**B1. OpenFAST is serial and on the critical path.** Addressed by PRs 16, 17 and 18.
- `advance` (`ERF_MovingBodies.cpp:860-969`) samples, steps OpenFAST (`:940`) and spreads. All of it must finish
  before the RK stages call `add_momentum_sources` (`ERF_TI_slow_rhs_pre.H:219-224`).
- The turbines step in lockstep (`ERF_OpenFASTDriver.cpp:346-378`, with `FAST_CFD_Step` at `:366` and
  `check_lockstep` at `:393-405`).
- Cost: dt/DT ≈ 20–100 substeps × about 0.2–3 ms ≈ 5–300 ms per turbine per step. That is multiplied by the number
  of turbines per owner when turbines outnumber ranks.
- `choose_owners` (`ERF_BodyExchange.cpp:144-236`) balances by count only (B15).

**B2. ADM uses 2401 points per turbine.** Addressed by PR 15.
- `adm_rings` (`ERF_OpenFASTRotor.cpp:266-311`) builds one ring per blade node, so the three blades' rings nearly
  coincide.
- About 3×10⁷ kernel evaluations per turbine per step: 2401 points × about 2.2k faces × 3 directions × 2 passes.
- About 3×10¹⁰ for the farm, concentrated on the ranks under the rotors: roughly 0.1–0.5 s per rank per step on CPU.

**B3. The whole column is always used.** Addressed by PR 9.
- `z_phys_nd` and `detJ_cc` always exist (`ERF_MakeNewArrays.cpp:55,124-125`), so `fitted` is always true
  (`ERF_BodyExchange.cpp:489,621`).
- **Routing:** `cells_within` (`:73-88`) spans the whole column, both for sampling (`:512`) and for kernels (`:655`).
- **Kernels:** `KernelFaces` (`ERF_ActuatorSpreading.cpp:123`) drops the bounds for the whole rank if any of its
  local boxes is aloft (`:147`). `mesh_z_bounds` is called with `over_all_ranks = false`, so this is decided per
  rank. `ZBounds::k_range` is at `ERF_ActuatorSampling.H:142-151`.
- `mesh_z_bounds` (`ERF_ActuatorSampling.cpp:416-452`) runs every step (`ERF_BodyExchange.cpp:751`).
- This matters most when columns are split into several boxes in z.

**B4. Normalisation always takes three message rounds.** Addressed by PR 14.
- `uniform` (`ERF_BodyExchange.cpp:672`) is never true in a run, so `uniform_mesh_norm`
  (`ERF_ActuatorSpreading.cpp:431-473`, called at `ERF_BodyExchange.cpp:679`) runs only in unit tests.
- The three rounds are at `ERF_BodyExchange.cpp:760-851`.

**B5. The kernels use little parallelism.** Addressed by PR 12.
- The normalisation pass uses one thread per point per box (`spread_partial_norms`, `:337`, loop at `:400`).
- There is one launch per box and direction.
- `MFIter(..., false)` runs with no OpenMP.
- Points are binned and uploaded from the host every step (`:368`, `:501`; `add_scaled_kernels` at `:476`).

**B6. Global synchronisations every step.** Addressed by PR 10, with PR 13 for the bookkeeping.
- `ensure_envelopes` (`ERF_BodyExchange.cpp:387-479`) does a `ReduceIntMax` (`:436`) per exchange, up to 3 per step:
  - sampling (`ERF_MovingBodies.cpp:1386`);
  - spreading (`:1294`);
  - wake sampling (`:1112`).

  When an envelope grows it also does a 6N `ReduceRealSum` (`:466`) and `find_participants` (`:477`).
- `update_tip_cells` (`ERF_MovingBodies.cpp:488-568`, called at `:1231` when any turbine is ALM) does a
  `ReduceIntMax` (`:527`). When a region is stale it adds `turbine_table(9)` (`:530`) and
  `tip_cell_heights_batched` (`:544`). The tolerance is 0.1 × min dx (`:482-485`).
- `check_levels` (`:1432-1497`, every step from `ERF_Advance.cpp:178`) loops over every point of every owned turbine
  when finer levels exist. It is owner-local, but it runs every step.

**B7. A sparse source is swept over the whole level.** Addressed by PR 11.
- `setVal` three times (`ERF_MovingBodies.cpp:1205-1207`), and `MultiFab::Add` three times per RK stage
  (`:1304-1306`).
- That is about 12 full face-array passes per step, plus three `integrate_source` passes on diagnostics steps.

### Diagnostics steps (every step by default)

**B8. Per-body file opens.** Addressed by PR 4.
- Plain `ofstream` appends: `_erf.csv` and `_flow.csv` (`ERF_OpenFASTDriver.cpp:584,593`), and `_disk.csv`
  (`ERF_PrescribedCtDisk.cpp:109`).
- Through `open_log` (`ERF_DiagnosticsLog.H:24-36`), which adds a `mkdir` per path component and a probe `ifstream`:
  - `_correction.csv` (`ERF_MovingBodies.cpp:1653`);
  - `_fllc.csv` (`:1901`);
  - `_stats.csv`, rewritten in full (`ERF_RunningStats.H:95-105`, from `ERF_MovingBodies.cpp:986,994`);
  - `_wake.csv`, and `_wake_avg.csv` rewritten in full (`ERF_WakeLines.cpp:112-134`).
- For 1000 turbines that totals about 3–7k opens and 10–30k metadata operations per step. Wake lines add about 1.5 TB
  of CSV per 36k steps.

**B9. Farm totals use dense reductions.** Addressed by PR 5.
- `turbine_table(5)` (`ERF_MovingBodies.cpp:1141`) goes through `gather_rows` (`:44-59`, with reductions at `:53`
  and `:55`).
- There are three `integrate_source` passes (`:1175-1177`). The function is defined at
  `ERF_ActuatorSpreading.cpp:609`, with its reduction at `:631`.

### Checkpoint and restart

**B10. The I/O rank trims every log by itself.** Addressed by PR 6.
- `trim_log_after` (`ERF_DiagnosticsLog.H:57-80`) is called at `ERF_MovingBodies.cpp:784-805`, with a barrier at
  `:804`.
- About 5000 files are each read in full and rewritten.

**B11. Every rank reads every body's state.** Addressed by PR 7.
- `read_state` is called at `ERF_MovingBodies.cpp:677-678` and `:1074`.
- Each read goes through `file_exists_everywhere` and `ReadAndBcastFile` (`ERF_RunningStats.H:124-177`), about 3
  broadcasts per file.

**B12. The checkpoint is one flat directory.** Addressed by PR 8.
- `write_checkpoint` (`ERF_MovingBodies.cpp:223-290`) has a `ReduceLongSum` at `:231` and a `Barrier` at `:263`.
- It writes about 3–6 files per turbine.

### Start-up (once)

**B13. Every rank parses every turbine's files.** Addressed by PR 2.
- `check_induction` (`ERF_OpenFASTDriver.cpp:47`) is called from the constructor at `:98`.
- `check_tower_shadow_off` (`:74`) is called from `ERF_MovingBodies.cpp:212`.
- That is about 7 opens × P × N, or about 29M at P = 4096 and N = 1000.
- `openfast_value` (`ERF_OpenFASTAudit.cpp:18-34`) opens a file for every key.

**B14. Coverage checks rebuild BoxArrays.** Addressed by PR 3.
- `points_covered_by` (`ERF_ActuatorSampling.cpp:735-809`) rebuilds them at `:742-756`.
- It is called for disks and wake lines on every rank (`ERF_MovingBodies.cpp:1576-1595`), and again in
  `build_wake_lines` (`:1059`).
- One collective `terrain_heights` runs per rotor (`:1021`).

### Other

**B15. Owner choice ignores cost.** Addressed by PR 16.

**B16. There are no timers anywhere in `Source/MovingBodies`.** Addressed by PR 1.

### Checked, not bottlenecks

- **The sampling kernels** (`sample_velocity_local`): there are few points, and the work per point is cheap.
- **ALM and FLLC spreading:** about 150–200 points per turbine.
- **`report_participants`** (`ERF_MovingBodies.cpp:423-438`): it compares lists every step but prints only when they
  change.
- **`choose_owners`:** it runs once, at start-up.
- **The start-up audits** (`audit_turbines_setup` and `audit_turbines_flow`): each runs once (flags at
  `ERF_MovingBodies.cpp:928,939`). Their O(N²) overlap check is trivial at N = 1000.

## 6. Kynema-SGF: adopt and avoid

Paths are under `src/wind_energy/actuator/` in `kynema/kynema-sgf`.

### Adopt

| Idea | Where in Kynema-SGF | ERF PR |
|---|---|---|
| `BL_PROFILE` in every phase | `Actuator.cpp`, `turbine/fast/FastIface.cpp` | 1 |
| A static `bound_box` per turbine: the base ± 1.25 R, up to hub height + 1.25 R | `turbine/turbine_utils.cpp` | 10 |
| Influenced ranks recomputed only after a regrid | `actuator_utils.cpp` (`determine_influenced_procs`), `post_regrid_actions` | 10 |
| Root rank: a free influenced rank first, otherwise the least-loaded one | `actuator_utils.cpp` (`determine_root_proc`) | 16 |
| One NetCDF file per turbine, written every `output_frequency` steps (default 10), with the directory created once | `turbine/fast/turbine_fast_ops.H` (`ProcessOutputsOp`, `prepare_outputs`) | 4 |
| OpenFAST checkpoints per turbine, every `chkpt_interval` steps | `turbine/fast/` | 8 |
| `#pragma omp parallel if (Gpu::notInLaunchRegion())` over `MFIter` | `Actuator.cpp` (the source-term loop) | 12 |
| One packed message per turbine and rank, 15 floats per node | `turbine/external/turbine_external_utils.H` (`scatter_data`) | Nothing to do: ERF already sends one message per partner |
| A separable ADM projection (radial × azimuthal × normal) | `turbine/ActSrcDiskOp_Turbine.H`, `disk/disk_spreading.H` (`linear_basis_spreading`) | 15 |
| Each turbine steps its own substeps | `turbine/external/ExtTurbIface.H` (`advance_turbine`) | 17 (ERF must keep lockstep within a process) |

### Avoid

| Pattern | Where | Why ERF should not copy it |
|---|---|---|
| Sampling through a particle container: two `Redistribute` calls plus a dense `MPI_Allreduce` of every sample point | `ActuatorContainer.cpp` (`populate_field_buffers`) | At 1000 ADM turbines that is about 2.4M values per rank per step; ERF's point-to-point exchange is already better |
| A `Comm_dup` and two global broadcasts per turbine at initialisation | `turbine/external/turbine_external_ops.H` (`init_data_op`) | N global collectives at start-up |
| Blocking per-turbine `Send`/`Recv` in global turbine order | `turbine/external/turbine_external_utils.H` | It serialises the farm |
| A line kernel with no cutoff, looped over the tile box `bx` instead of the valid box | `ActSrcLineOp.H` | Wasted work; ERF's 3ε cutoff is better |
| A hard assert that the number of OpenFAST turbines is at most the number of ranks | `Actuator.cpp` | ERF already allows more turbines than ranks |
| An analytic normalisation of the kernel | the spreading operators | It would break ERF's exact `fx == -thrust_x` |

## 7. Settings that help today

None of these needs a code change.

| Setting | Effect | Bottleneck |
|---|---|---|
| `erf.moving_bodies.diagnostics_int` = 10 to 100 | Fewer log rows, statistics rewrites, farm-total reductions and source integrals | B8, B9 |
| No `erf.moving_bodies.wake.lines_xD` (wake lines off), or a large `erf.moving_bodies.wake.int` | Removes the wake exchange, its envelope reduction and the largest files | B6, B8 |
| `erf.moving_bodies.avg_start` set after the spin-up | Statistics are accumulated and rewritten only from then on | B8 |
| `erf.moving_bodies.envelope_check_interval` ≫ 1 | Envelopes cover the whole swept sphere, and there is one envelope reduction per K exchanges instead of one per exchange (the tip-cell reduction stays) | B6 |
| `amr.max_grid_size_z` ≥ nz on the anchor level | Each column lies in one box, so whole-column routing and the per-rank height-bound fallback cost nothing extra | B3 |
| At least as many ranks as turbines, and the same OpenFAST `DT` for every turbine | One turbine per owner, and no time-step class needs ranks of its own | B1, B15 |
| `erf.moving_bodies.verify_tip_cells` = 0 (the default) | No extra tip-cell search every step | B6 |
| In OpenFAST: a coarse `DT_Out`, binary `OutFileFmt`, `SumPrint` and `Echo` off, short `OutList`s | Less OpenFAST I/O in every substep | B1 |

## 8. Open questions and measurements

- **OpenFAST cost.** What does one OpenFAST substep of the IEA 15 MW model cost in the planned configuration (which
  modules are on)? PR 1 measures it, and the answer decides PRs 17 and 18.
- **ADM or ALM?**
  - ADM: PRs 9, 12, 14 and 15 matter most.
  - ALM: PR 10 (the per-step tip-cell reduction) and PRs 16–18 matter most.
- **Split columns.** Do the production decks split columns into several boxes in z? If not, B3 costs little beyond
  the per-step `mesh_z_bounds` sweep.
- **Stale tip regions.** How often do the ALM tip regions go stale in a real run, through yaw or blade deflection?
  Each stale step costs an N × 9 reduction and a collective search. Count it with PR 1's timers.
- **Floating turbines.** How large a margin do static envelopes need for platform motion?
- **Real OpenFAST on several ranks.** It is still an open follow-up in `repo.MD`, and a 1000-turbine run depends on
  it. Test it before or alongside PRs 16–18.
- **Scaling deck.** Build one to measure every PR. It is not a regression test.
  - N stub turbines (16 up to 1024) on a flat periodic domain.
  - A fixed number of steps, run at several rank counts.
  - Report PR 1's per-phase times.

## 9. Reference: the step and the files

### One step on the anchor level

1. `ERF_Advance.cpp:174-182` calls `check_levels` (`:178`), then `moving_bodies->advance(...)` (`:180`) with
   `z_phys_nd[lev]` and `detJ_cc[lev]`.
2. `MovingBodies::advance` (`ERF_MovingBodies.cpp:860-969`) runs these in order:
   1. `check_coverage` on the first step, and again only if the grids change (`:898`);
   2. `audit_turbines_setup`, once (`:928`);
   3. `supply_velocities` (`:1340`; the exchange `sample_velocity` is at `:1386`);
   4. `audit_turbines_flow`, once (`:939`);
   5. `m_driver->step()` (`:940`);
   6. on diagnostics steps: `write_diagnostics`, the FLLC and correction logs, and `write_total_load` (`:941-946`);
   7. `spread_sources` (`:952`; defined at `:1194-1296`, with the tip cells at `:1231` and the exchange
      `spread_forces` at `:1294`);
   8. on diagnostics steps: `write_source_diagnostics` (`:959`);
   9. `accumulate_statistics`, once `time + dt >= avg_start` (`:965`);
   10. wake sampling, every `wake.int` steps (`write_wake_diagnostics`, `:966-968`).
3. Every RK stage calls `add_momentum_sources` (`ERF_TI_slow_rhs_pre.H:219-224`, which calls
   `ERF_MovingBodies.cpp:1299-1307`).
4. The checkpoint write is at `ERF_Checkpoint.cpp:184`. On restart, `set_ground` and then `read_checkpoint` run at
   `ERF_Checkpoint.cpp:2198-2202`.

### Key files (line numbers at `3d427e7`)

| File | What is there |
|---|---|
| `Source/MovingBodies/ERF_MovingBodies.cpp` | `gather_rows` 44, `write_checkpoint` 223, `choose_owners` 357, `report_participants` 423, `turbine_table` 475, `update_tip_cells` 488, `read_checkpoint` 654, `advance` 860, `build_wake_lines` 999, `write_wake_diagnostics` 1097, `write_total_load` 1133, `write_source_diagnostics` 1173, `spread_sources` 1194, `add_momentum_sources` 1299, `supply_velocities` 1340, `check_levels` 1432, `check_coverage` 1500 |
| `Source/MovingBodies/Core/ERF_BodyExchange.cpp` | `sections` 62, `cells_within` 73, `ranks_touching` 132, `choose_owners` 144, `find_participants` 373, `ensure_envelopes` 387, `sample_velocity` 482, `spread_forces` 612 |
| `Source/MovingBodies/Core/ERF_ActuatorSpreading.cpp` | `KernelFaces` 123, `spread_partial_norms` 337, `uniform_mesh_norm` 431, `add_scaled_kernels` 476, `integrate_source` 609 |
| `Source/MovingBodies/Core/ERF_ActuatorSampling.H` | `ZBounds` 131 (`k_range` 142) |
| `Source/MovingBodies/Core/ERF_ActuatorSampling.cpp` | `terrain_heights` 354, `mesh_z_bounds` 416, `tip_cell_heights_batched` 633, `min_cell_heights` 712, `points_covered_by` 735 |
| `Source/MovingBodies/Core/ERF_DiagnosticsLog.H` | `open_log` 24, `trim_log_after` 57 |
| `Source/MovingBodies/Core/ERF_RunningStats.H` | `write` 95, `write_state` 108, `read_state` 124 |
| `Source/MovingBodies/Core/ERF_WakeLines.cpp` | `write_instantaneous` 112, `write_average` 127 |
| `Source/MovingBodies/ForceModels/ERF_OpenFASTRotor.cpp` | `turbine_points` 241, `adm_rings` 266 |
| `Source/MovingBodies/ForceModels/ERF_PrescribedCtDisk.cpp` | `open_diagnostics` 96, `write_diagnostics` 106 |
| `Source/MovingBodies/OpenFAST/ERF_OpenFASTDriver.cpp` | `check_induction` 47, `check_tower_shadow_off` 74, constructor 87, `step` 346, `check_lockstep` 393, log writes 564–593 |
| `Source/MovingBodies/OpenFAST/ERF_OpenFASTAudit.cpp` | `openfast_value` 18 |
| `Source/MovingBodies/ERF_MovingBodiesInputs.H` | Input defaults (see §1) |
| `Source/MovingBodies/ERF_MovingBodiesInputs.cpp` | Global keys and wake keys parsed at 76–126 |
| `Source/TimeIntegration/ERF_Advance.cpp` | The `check_levels` and `advance` calls, 174–182 |
| `Source/TimeIntegration/ERF_TI_slow_rhs_pre.H` | The `add_momentum_sources` call, 219–224 |
| `Source/ERF_MakeNewArrays.cpp` | `z_phys_nd` (55) and `detJ_cc` (125), always allocated |
| `Source/IO/ERF_Checkpoint.cpp` | Checkpoint write 184; restart 2198–2202 |
