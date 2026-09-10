#!/bin/bash
# Run the FireAccelerationClock decks and compare the head of a point fire under
# each erf.fire.accel.clock with its path's unaccelerated head and the share
# 1 - (1 - exp(-A t)) / (A t) a fire carrying its ignition clock covers.
#
#   [MPIRUN="mpirun -np 2"] [PYTHON=python3] ./run_fireaccelerationclock.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_fireaccelerationclock.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_fireaccelerationclock.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="farsite_off farsite_legacy farsite_front levelset_off levelset_legacy levelset_front"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_fireaccelerationclock.py $VARIANTS
