#!/bin/bash
# Run the FireDirectionalShape decks and check the rates against Rothermel.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] ./run_firedirectionalshape.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_firedirectionalshape.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_firedirectionalshape.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="isotropic projection projection_key ellipse ellipse_anderson"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_firedirectionalshape.py $VARIANTS
