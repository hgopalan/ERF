#!/bin/sh
# Restart equality with a line under construction across the checkpoint.
#
#   [MPIRUN="mpirun -np 2"] [METHOD=farsite] sh run_restart.sh /path/to/erf_exec [extra erf args...]
#
# Three runs: straight to step 40, to step 23 with a checkpoint, and a
# restart from it to step 40. The two fire plotfiles at step 40 must agree in
# every field, the suppression mask, factor and line progress included.

set -u
EXE=${1:?usage: run_restart.sh /path/to/erf_exec [extra args]}
shift || true
PYTHON=${PYTHON:-python3}
m=${METHOD:-levelset}
here=$(cd "$(dirname "$0")" && pwd)
[ -f erf_plotfile.py ] || cp "$here/../../CanonicalTests/Canonical_RANS/erf_plotfile.py" .

rm -rf chk00023 plt_restart_*_${m}_* suppression_restart_*_${m}.csv
for leg in straight chk restart; do
    ${MPIRUN:-} "$EXE" "inputs_restart_$leg" erf.fire.propagation_method="$m" \
        erf.fire_plot_file="plt_restart_${leg}_${m}_" erf.fire.suppression.log="suppression_restart_${leg}_${m}.csv" \
        "$@" > "run_restart_${leg}_${m}.log" 2>&1 || { echo "  restart/$m: the $leg run failed"; tail -20 "run_restart_${leg}_${m}.log"; exit 1; }
done
grep -q "Suppression: restored 1 action" "run_restart_restart_${m}.log" \
    && echo "  restart/$m: the checkpoint's action state was restored: PASS" \
    || { echo "  restart/$m: no restored-action line in the restart log: FAIL"; exit 1; }
# The restarted leg reapplies nothing: its log holds no applied event.
if grep -q ",applied," "suppression_restart_restart_${m}.csv"; then
    echo "  restart/$m: the restarted run applied an action again: FAIL"; exit 1
else
    echo "  restart/$m: no action reapplied after the restart: PASS"
fi
$PYTHON check_suppression.py restart "plt_restart_straight_${m}_00040" "plt_restart_restart_${m}_00040"
