#!/bin/bash
# Run the stick-moisture variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_stick.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_stick.sh x     # only rerun the checks on existing output
#
# Checks: the deck with the key written out reproduces the historical deck
# (level set and dead classes identical at 60 s); the stick deck's classes
# stay within bounds and move from the deck values in lag order (1 h most,
# 100 h least); the stick deck differs from the time-lag deck (it is a
# different model); a restart from a checkpoint at 30 s reproduces the
# straight stick run at 60 s exactly, which needs the shells to come back
# from the checkpoint.

set -u
EXE=${1:?usage: run_stick.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="timelag timelag_key stick stick_chk stick_restart"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    [ "$v" = "stick_chk" ] && rm -rf chk00240
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }
printf "%-14s %7s  %s\n" variant cells "stick line"
for v in $VARIANTS; do printf "%-14s %7s  %s\n" "$v" "$(cells $v)" "$(grep -m1 'stick model' run_$v.log | cut -c1-90)"; done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }
python3 check_stick.py same plt_fire_timelag/plt_fire_00480 plt_fire_timelag_key/plt_fire_00480 2>&1 | grep -v "^yt" || status=1
python3 check_stick.py order plt_fire_stick/plt_fire_00480 0.08 0.08 0.08 2>&1 | grep -v "^yt" || status=1
if python3 check_stick.py same plt_fire_timelag/plt_fire_00480 plt_fire_stick/plt_fire_00480 > /dev/null 2>&1; then
    check "stick deck differs from the time-lag deck" 0
else
    check "stick deck differs from the time-lag deck" 1
fi
python3 check_stick.py same plt_fire_stick/plt_fire_00480 plt_fire_stick_restart/plt_fire_00480 2>&1 | grep -v "^yt" | sed 's/identical/identical after the restart/' || status=1
r=$(grep -q "moisture stick shells to checkpoint" run_stick_chk.log && [ -f chk00240/Level_0/FireStickMC_H ] && echo 1 || echo 0)
check "stick shells written to the checkpoint (chk00240/Level_0/FireStickMC)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
