#!/bin/bash
# Run the three Interacting_Fires decks and report when and where the fires meet.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] ./run_interacting_fires.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_interacting_fires.sh x      # only rerun the report on existing output

set -u
EXE=${1:?usage: run_interacting_fires.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="coalescing junction parallel"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf "plt_fire_${v}_"?????
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 \
        || { echo "run $v failed (see run_$v.log)"; tail -n 30 "run_$v.log"; exit 1; }
done

${PYTHON:-python3} report_interacting_fires.py
