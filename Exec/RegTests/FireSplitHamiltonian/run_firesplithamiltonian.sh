#!/bin/bash
# Run the FireSplitHamiltonian decks and check rotation invariance and the head rate.
#
#   [MPIRUN="mpirun -np 1"] [PYTHON=python3] ./run_firesplithamiltonian.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_firesplithamiltonian.sh x      # checks only, on existing output
#   SERIAL=1 ./run_firesplithamiltonian.sh ...      # one deck at a time instead of all four at once

set -u
EXE=${1:?usage: run_firesplithamiltonian.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="baseline_0 baseline_34 split_0 split_34"

pids=""
for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    if [ "${SERIAL:-0}" = "1" ]; then
        ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
    else
        ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 &
        pids="$pids $!:$v"
    fi
done
for p in $pids; do
    wait "${p%%:*}" || { echo "run ${p##*:} failed (see run_${p##*:}.log)"; exit 1; }
done

${PYTHON:-python3} check_firesplithamiltonian.py
