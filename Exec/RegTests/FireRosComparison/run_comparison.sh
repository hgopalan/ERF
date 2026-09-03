#!/bin/bash
# Run every rate-of-spread variant in this directory and print the comparison.
#
#   ./run_comparison.sh /path/to/erf_exec [extra erf args...]
#
# Each variant is the same wind-driven grass fire; only the rate-of-spread
# formulation differs. The burned-cell count is the discriminator: a
# direction-dependent rate slows the flanks and backing fire, so it burns less
# area than the same model applied isotropically. The hybrid decks add the
# count of cells that take the secondary model; the *_none, *_all and
# rothermel_fuelmap decks are identities and must match their single-model
# counterpart in both cells and max_ROS.

set -u

EXE=${1:?usage: run_comparison.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="rothermel_isotropic rothermel_directional
          balbi2009_isotropic balbi2009_directional
          balbi2020_isotropic balbi2020_directional
          rothermel_fuelmap
          hybrid_none hybrid_all hybrid_region hybrid_fuel
          hybrid_none_directional hybrid_all_directional hybrid_region_directional
          hybrid_wind_off hybrid_wind
          macarthur_isotropic macarthur_directional
          cheney_gould_isotropic cheney_gould_directional
          behave_isotropic behave_directional
          rothermel_nearest balbi2020_reference_wind balbi2020_extinction_wet
          hybrid_behave_cheney hybrid_behave_cheney_directional hybrid_blend_width"

printf "%-24s %8s %10s %14s %10s\n" variant exit cells max_ROS sec_cells
printf "%-24s %8s %10s %14s %10s\n" ------------------------ -------- ---------- -------------- ----------

for v in $VARIANTS; do
    log="run_${v}.log"
    "$EXE" "inputs_${v}" "$@" > "$log" 2>&1
    rc=$?
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ros=$(grep 'Rate-of-spread computed' "$log" | tail -1 | sed 's/.*Max: //; s/,.*//')
    sec=$(grep 'Hybrid ROS:' "$log" | tail -1 | sed 's/.*secondary_cells=//')
    printf "%-24s %8s %10s %14s %10s\n" "$v" "$rc" "${cells:-n/a}" "${ros:-n/a}" "${sec:--}"
done
