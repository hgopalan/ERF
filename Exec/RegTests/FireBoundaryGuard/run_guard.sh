#!/bin/bash
# Run the boundary-guard variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_guard.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_guard.sh x     # only rerun the checks on existing output
#
# warn and far run to 20 s and are checked by check_guard.py; abort must stop
# on its first step with the guard message and a nonzero exit status.

set -u
EXE=${1:?usage: run_guard.sh /path/to/erf_exec [extra args]}
shift || true

for v in warn far; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -f fire_stats_$v.csv
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

status=0
python3 check_guard.py warn far || status=1

if [ "${SKIP_RUN:-0}" != "1" ] || [ ! -f run_abort.log ]; then
    rm -f fire_stats_abort.csv
    ${MPIRUN:-} "$EXE" inputs_abort "$@" > run_abort.log 2>&1 && { echo "  abort: the run did not stop: FAIL"; status=1; }
fi
if grep -q "reached the boundary guard band" run_abort.log; then
    echo "  abort: stopped on the first step with the guard message: PASS"
else
    echo "  abort: no guard message in run_abort.log: FAIL"; status=1
fi

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
