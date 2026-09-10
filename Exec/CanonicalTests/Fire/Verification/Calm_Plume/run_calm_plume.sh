#!/bin/bash
# Run the Calm_Plume decks and check them against plume theory.
#
#   [MPIRUN="mpirun -np 8"] [PYTHON=python3] ./run_calm_plume.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_calm_plume.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_calm_plume.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="neutral stable"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.plot_file_1="plt_atm_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_calm_plume.py $VARIANTS
