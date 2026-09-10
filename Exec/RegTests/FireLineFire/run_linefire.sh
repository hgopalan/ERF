#!/bin/bash
# Run the line-fire variants and check the rates of spread.
#
#   [MPIRUN="mpirun -np 4"] ./run_linefire.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_linefire.sh x      # only rerun the checks on existing output
#
# The one-way variants must spread at the Rothermel rates for the wind the
# fire samples (head R0(1 + phi_w), backing R0, to 10 %); the two-way variant
# is reported next to Coen et al. (2013).

set -u
EXE=${1:?usage: run_linefire.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="nowind wind2p5 wind5 nowind_2way wind2p5_2way wind5_2way"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

python3 check_linefire.py $VARIANTS
