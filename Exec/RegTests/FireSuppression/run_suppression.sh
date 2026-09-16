#!/bin/bash
# Run every FireSuppression variant on both propagation paths and check it.
#
#   [MPIRUN="mpirun -np 2"] ./run_suppression.sh /path/to/erf_exec [extra erf args...]
#   VARIANTS="line_early hold" METHODS=farsite ./run_suppression.sh /path/to/erf_exec
#   SKIP_RUN=1 ./run_suppression.sh x        # only rerun the checks on existing output
#
# The deterministic variants (line_early, line_late, drop_hold, drop_slow,
# hold, burnout) are checked from their last fire plotfile and suppression
# log by check_suppression.py; poll runs through run_poll.sh (a writer
# process appends the line while the run polls) and restart through
# run_restart.sh (straight, checkpoint, restart, plotfiles compared).

set -u
EXE=${1:?usage: run_suppression.sh /path/to/erf_exec [extra args]}
shift || true
PYTHON=${PYTHON:-python3}
here=$(cd "$(dirname "$0")" && pwd)
[ -f erf_plotfile.py ] || cp "$here/../../CanonicalTests/Canonical_RANS/erf_plotfile.py" .

status=0
for v in ${VARIANTS:-line_early line_late drop_hold drop_slow hold burnout}; do
    for m in ${METHODS:-levelset farsite}; do
        plt="plt_${v}_${m}_"
        log="suppression_${v}_${m}.csv"
        if [ "${SKIP_RUN:-0}" != "1" ] || [ ! -d "${plt}00040" ]; then
            rm -rf "${plt}"* "$log"
            ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire.propagation_method="$m" erf.fire_plot_file="$plt" \
                erf.fire.suppression.log="$log" "$@" > "run_${v}_${m}.log" 2>&1 \
                || { echo "run $v/$m failed (see run_${v}_${m}.log)"; status=1; continue; }
        fi
        $PYTHON check_suppression.py "$v" "${plt}00040" "$log" || status=1
    done
done
for v in ${VARIANTS:-poll restart}; do
    case $v in
        poll)    for m in ${METHODS:-levelset farsite}; do METHOD=$m sh run_poll.sh "$EXE" "$@" || status=1; done ;;
        restart) for m in ${METHODS:-levelset farsite}; do METHOD=$m sh run_restart.sh "$EXE" "$@" || status=1; done ;;
    esac
done
[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
