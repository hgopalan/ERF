# LNG_BuildOnly Regression Test

## Purpose

This regression test validates Phase 1 of the ERF-LNG hazardous gas dispersion module:
- **CMake integration**: Verifies that the ERF code compiles cleanly with `-DERF_ENABLE_LNG=ON`
- **GNUmake integration**: Verifies that the code builds with `USE_LNG=TRUE`
- **LNGParams ParmParse**: Confirms that all 32 parameters are read from the input file with correct defaults and units
- **LNGGrid construction**: Tests 2D grid refinement algorithm (extract k=0 ATM slab, refine by `grid_ratio`)
- **MultiFab allocation**: Verifies that all 13 LNG MultiFabs are allocated without out-of-memory errors
- **Initialization debug output**: Ensures initialization summary and per-step debug prints appear in stdout
- **Prerequisite validation**: Tests that all 5 prerequisite checks execute without aborting
- **Zero-physics stub behavior**: Confirms that `advance()` and coupling methods return without error
- **CSV header creation**: Validates that `lng_diag.csv` is created with the correct header line
- **NaN detection**: Ensures no `NaN` or `Inf` appears in any LNG MultiFab after 5 steps

## How to Build

### CMake

```bash
cd /path/to/ERF
mkdir build_lng
cd build_lng
cmake -DERF_ENABLE_LNG=ON -DERF_ENABLE_DUST=ON -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

### GNUmake

```bash
cd /path/to/ERF
make USE_LNG=TRUE -j 8
```

## How to Run

After building (assuming CMake was used):

```bash
cd build_lng
cd Exec/RegTests/LNG_BuildOnly
./ERF3d inputs_lng_buildonly
```

Or with GNUmake:

```bash
cd /path/to/ERF/Exec/RegTests/LNG_BuildOnly
./ERF3d inputs_lng_buildonly
```

## Expected Output

The stdout should contain:

1. **Initialization summary (from ERF_LNGPrerequisites):**
```
[LNG] ============================================================
[LNG] ERF-LNG Phase 1 initialized
[LNG]   Pool area       : 100.0 m^2
[LNG]   Spill rate      : 10.0 kg/s
[LNG]   LNG composition : CH4=0.90  C2H6=0.08  N2=0.02
[LNG]   Mol. weight     : 17.4 g/mol
[LNG]   Boiling point   : 111.7 K
[LNG]   LFL/UFL         : 0.05 / 0.15 (vol/vol)
[LNG]   Grid ratio      : 1
[LNG]   ATM feedback    : 0.0
[LNG]   Debug mode      : on
[LNG]   Verbose level   : 2
[LNG] ============================================================
```

2. **Per-step debug output (5 lines, one per step):**
```
[LNG DEBUG] Step  0  time=0.000e+00  pool_cells=<N>  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] Step  1  time=1.000e+00  pool_cells=<N>  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] Step  2  time=2.000e+00  pool_cells=<N>  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] Step  3  time=3.000e+00  pool_cells=<N>  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
[LNG DEBUG] Step  4  time=4.000e+00  pool_cells=<N>  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3
```

3. **CSV diagnostics output** — File `lng_diag.csv` must exist and begin with:
```
step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2
0,0.0,<N>,100.0,0.0,0.0,0.0,0.0,0.0,0.0
1,1.0,<N>,100.0,0.0,0.0,0.0,0.0,0.0,0.0
2,2.0,<N>,100.0,0.0,0.0,0.0,0.0,0.0,0.0
3,3.0,<N>,100.0,0.0,0.0,0.0,0.0,0.0,0.0
4,4.0,<N>,100.0,0.0,0.0,0.0,0.0,0.0,0.0
```

## Pass Criteria

All of the following must be true for the test to pass:

1. **Build without warnings**: `cmake -DERF_ENABLE_LNG=ON ..` builds without compiler warnings or errors (test at `-Wall -Wextra` level)
2. **GNUmake builds**: `USE_LNG=TRUE make` builds without errors
3. **Executable runs**: `./ERF3d inputs_lng_buildonly` exits with code 0 (success)
4. **Initialization output present**: The line `[LNG] ERF-LNG Phase 1 initialized` appears in stdout
5. **Debug steps counted**: The pattern `[LNG DEBUG] Step` appears exactly 5 times (one per timestep)
6. **CSV header exists**: File `lng_diag.csv` exists in the run directory after completion
7. **CSV header correct**: The first line of `lng_diag.csv` is exactly:
   ```
   step,time_s,pool_cells,pool_area_m2,pool_mass_kg,total_vapor_mass_kg,evap_flux_max_kg_m2_s,vapor_conc_max_kg_m3,lfl_area_m2,ufl_area_m2
   ```
8. **CSV rows written**: 5 data rows appear after the header (one per timestep)
9. **No NaN detected**: The log does not contain `[LNG] NaN detected in`
10. **No fatal errors**: The log contains no `[LNG] Aborting` messages
11. **Zero verbose overhead when disabled**: When `erf.lng.enable = false`, no `[LNG]` output appears (test via separate run)

## Known Limitations

- **No physics**: All evaporation, spreading, and transport are stubbed out (Phase 2+)
- **No ATM coupling**: `apply_to_cc_source()` and `extract_atm_return_fields()` are stubs
- **No plotfile output**: The `write_lng_plotfile()` stub prints only a message, no actual data is written
- **No flammability mapping**: `track_flammability`, `lng_lfl_mask`, and `lng_ufl_mask` are allocated but never updated (Phase 6)
- **Diagnostic CSV**: Only headers and placeholder rows are written (Phase 2+)

## References

- `Source/LNG/LNG_DEVELOPMENT.md` — Complete development roadmap and architecture
- `Source/LNG/ERF_LNGParams.H` — Parameter definitions and defaults
- `Source/LNG/ERF_LNGLayer.H` — Container class interface and MultiFab ownership
- `Source/LNG/ERF_LNGPrerequisites.cpp` — Prerequisite checks and initialization summary
