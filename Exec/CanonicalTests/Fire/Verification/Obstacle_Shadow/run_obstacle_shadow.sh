#!/bin/bash
# Run the Obstacle_Shadow decks and check them against the exact answer.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] ./run_obstacle_shadow.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_obstacle_shadow.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_obstacle_shadow.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="disc30 disc30_wallx"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_obstacle_shadow.py $VARIANTS
