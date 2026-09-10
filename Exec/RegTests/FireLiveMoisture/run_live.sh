#!/bin/bash
# Run the live-moisture variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_live.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_live.sh x     # only rerun the checks on existing output
#
# Checks (check_live.py): the deck with the default key written out
# reproduces the historical deck; both start with the live classes at
# erf.fire.moisture_live = 0.90; "fixed" still holds them there at 60 s, while
# "legacy" has dropped them to the 0.40 dead-fuel clamp; the dead classes do
# not depend on the setting; and the rate of spread does, through BEHAVE's
# live damping and live-to-dead herbaceous transfer.

set -u
EXE=${1:?usage: run_live.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="legacy legacy_key fixed"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -d "plt_fire_$v" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

python3 check_live.py plt_fire_legacy plt_fire_legacy_key plt_fire_fixed 0.90 2>&1 | grep -v "^yt"
exit ${PIPESTATUS[0]}
