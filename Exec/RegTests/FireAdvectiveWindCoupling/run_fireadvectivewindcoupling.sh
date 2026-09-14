#!/bin/bash
# Run the FireAdvectiveWindCoupling decks and check the head rate against Rothermel.
#
#   [MPIRUN="mpirun -np 2"] [PYTHON=python3] ./run_fireadvectivewindcoupling.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_fireadvectivewindcoupling.sh x      # checks only, on existing output

set -u
EXE=${1:?usage: run_fireadvectivewindcoupling.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="default projection advective"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_fireadvectivewindcoupling.py $VARIANTS
