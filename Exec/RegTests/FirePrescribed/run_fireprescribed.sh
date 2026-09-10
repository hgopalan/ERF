#!/bin/bash
# Run both FirePrescribed decks and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_fireprescribed.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_fireprescribed.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_fireprescribed.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="ros_circle heat_patch"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" erf.plot_file_1="plt_atm_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_fireprescribed.py $VARIANTS
