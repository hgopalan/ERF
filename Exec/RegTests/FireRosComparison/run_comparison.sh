#!/bin/bash
# Run every rate-of-spread variant in this directory and print the comparison.
#
#   ./run_comparison.sh /path/to/erf_exec [extra erf args...]
#
# Each variant is the same wind-driven grass fire; only the rate-of-spread
# formulation differs. The burned-cell count is the discriminator: a
# direction-dependent rate slows the flanks and backing fire, so it burns less
# area than the same model applied isotropically.

set -u

EXE=${1:?usage: run_comparison.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="rothermel_isotropic rothermel_directional
          balbi2009_isotropic balbi2009_directional
          balbi2020_isotropic balbi2020_directional"

printf "%-24s %8s %10s %14s\n" variant exit cells max_ROS
printf "%-24s %8s %10s %14s\n" ------------------------ -------- ---------- --------------

for v in $VARIANTS; do
    log="run_${v}.log"
    "$EXE" "inputs_${v}" "$@" > "$log" 2>&1
    rc=$?
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ros=$(grep 'Rate-of-spread computed' "$log" | tail -1 | sed 's/.*Max: //; s/,.*//')
    printf "%-24s %8s %10s %14s\n" "$v" "$rc" "${cells:-n/a}" "${ros:-n/a}"
done
