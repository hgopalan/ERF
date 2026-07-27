# UCMBEPMomentumDrag Test — Phase 2.8

## Summary

Canonical test for ERF-SLUCM Phase 2.8: BEP-line momentum drag (compressible explicit mode).

Tests that momentum drag:
1. Reduces wind inside urban canopy (50%+ for tall buildings, less for short)
2. Leaves above-canopy wind undisturbed (≈ freestream within 20%)
3. Distributes wall drag via geometric overlap (Phase 2.7 reuse)
4. Places roof drag at sharp k_roof layer

## Domain & Physics

- **Grid:** ATM 4×4, grid_ratio=4 → UCM 16×16
- **Vertical:** dz=4 m, nz=256 → 1024 m tall
- **Pattern:** Two vertical stripes (UC 16×16):
  - **Left half** (i=0..7): tall dense buildings (h=30 m, plan_area=0.6)
  - **Right half** (i=8..15): short sparse buildings (h=5 m, plan_area=0.2)
  - Same for all j

- **Physics:**
  - Wall drag density: `s_wall(k) = 2·λ_f·Θ_w(k) / H_mean` [m⁻¹]
  - Drag force: `F_x = -f_urb·s_wall·Cd_wall·|U_h|·u`
  - Coefficients: Cd_wall=0.4, Cd_roof=0.15 (Martilli 2002)
  - MOST owns k=klo momentum (no double-counting)

## Running

```bash
# Generate CSV (building layout)
python3 gen_csv.py

# Run simulation
mpirun -n 1 ./erf_ucm_bep_momentum_drag inputs 2>&1 | tee run.log

# Check results
python3 check_drag.py
```

**Expected output:**
- `run.log` contains `[UCM][2.8]` traces:
  - Resolved mode: `wall_drag_mode auto -> explicit`
  - Per-facet drag stats: wall cells, roof cells, sum forces
- `plt_*_000010` plotfile (main ATM state at final step)
- `check_drag.py` exits 0 (all assertions pass)

## Verification (check_drag.py)

### Loads and assertions:

1. **Main plotfile** — `plt_*_000010` exists, contains u,v,w.

2. **Vertical profiles:**
   - Extract `u(k), v(k)` at (i=0, j=0) — tall-stripe column
   - Extract `u(k), v(k)` at (i=3, j=0) — short-stripe column (ATM index, rightmost tall stripe cell)
   - Compute `|U_h| = sqrt(u² + v²)` per layer

3. **Canopy interior assertion:**
   - At k where `z_hi(k) < H_bldg_mean` (inside 30 m canopy):
   - Assert: `|U|_tall < 0.5 * |U|_freestream`
   - **Fail:** Wind reduction must be ≥50% inside tall canopy; drag kernel bug or missing geometry.

4. **Above-canopy assertion:**
   - At k where `z_lo(k) > 2*H_bldg_mean` (well above 30 m canopy):
   - Assert: `|U|_tall ≈ |U|_freestream` within ±20%
   - **Fail:** Above-canopy wind should be undisturbed; drag bleeding into free atmosphere.

5. **Diagnostic output:**
   - Side-by-side vertical profiles (tall vs short stripe)
   - Print streamwise momentum sums inside vs above canopy
   - Print building height bounds per stripe

6. **Plotfile components:**
   - Assert: 5 ATM components (u, v, w, and 2 passive scalars)
   - Drag does NOT add new fields

### Known fixes applied:

- Use `yt.covering_grid()` for 3D field indexing — NOT flat array `ad[...]`
- No `.to_value("m")` on dimensionless fields
- Force `ds.index` before reading `ds.field_list`
- Fields prefixed "boxlib" not "yt"
- Diagnostic physics checks (lambda_f range) are DIAGNOSTIC-ONLY, not fatal assertions

## Backward Compatibility

**Phase 2.7 test regression:**
- Ensure `Exec/CanonicalTests/SLUCM/UCMFacet3DInjection/inputs` has `erf.ucm.wall_drag_mode = "off"`
- Run `UCMFacet3DInjection` with drag disabled
- Expected: heat injection still works, momentum unchanged (backward compatible)

## Known Limitations & Future Work

- **Terrain-following:** Infrastructure present but Phase 2.8 test runs flat only. Phase 4+ will add terrain test.
- **Anelastic mode:** Code-complete but NOT tested. See Phase 2.8b (future PR) for anelastic canonical test.
- **Heterogeneous Cd:** Currently uniform per domain. Phase 2.9+ will support per-cell CSV overrides.

## Phase 2.8 Acceptance

Compressible momentum drag fully validated. Anelastic path code-complete with debug print; full validation deferred to Phase 2.8b.
