#!/bin/bash
# Run the temperature-threshold ignition variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_threshold.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_threshold.sh x     # only rerun the checks on existing output
#
# Checks: the deck with the default keys written out reproduces the untouched
# deck line for line (the option is off by default and changes nothing); with
# the threshold on, cells ignite by threshold and the fire ends larger than
# without it; with a 4 s start time nothing ignites by threshold before the
# step whose window contains 4 s; a 5 m disc ignites at least as many cells
# as a one-cell stamp; the FARSITE path ignites by threshold too and ends at
# least as large as without it.

set -u
EXE=${1:?usage: run_threshold.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="off off_key on on_spinup on_r5 off_farsite on_farsite"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

cells() { grep 'active fire cells' "run_$1.log" | awk '{print $NF}'; }
thr()   { grep 'Threshold ignition:' "run_$1.log" | awk '{print $5}'; }      # cells ignited by threshold, per step
thr_total() { thr $1 | awk '{s += $1} END {print s + 0}'; }
# The threshold line is printed only on the steps the threshold is active, so
# the global step of a line is its number plus the steps before the start time.
thr_first() { thr $1 | awk -v off="$(( $(cells $1 | wc -l) - $(thr $1 | wc -l) ))" '$1 > 0 {print NR + off; exit}'; }
tmax()  { grep 'Threshold ignition:' "run_$1.log" | sed -n 's/.*max surface temp \([0-9.e+-]*\) K.*/\1/p' | sort -g | tail -1; }

printf "%-12s %6s %10s %10s %10s %10s %10s\n" variant steps cells_end thr_steps thr_total thr_first Tmax_K
printf "%-12s %6s %10s %10s %10s %10s %10s\n" ------------ ------ ---------- ---------- ---------- ---------- ----------
for v in $VARIANTS; do
    f=$(thr_first $v); [ -z "$f" ] && f=-
    printf "%-12s %6s %10s %10s %10s %10s %10s\n" "$v" "$(cells $v | wc -l | tr -d ' ')" "$(cells $v | tail -1)" \
        "$(thr $v | wc -l | tr -d ' ')" "$(thr_total $v)" "$f" "$(tmax $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(cells off) <(cells off_key) > /dev/null && echo 1 || echo 0)
check "default keys written out reproduce the untouched deck ($(cells off | wc -l | tr -d ' ') steps)" "$same"
r=$(awk -v n="$(thr_total on)" 'BEGIN { print (n > 0) ? 1 : 0 }')
check "threshold at 315 K ignites cells ($(thr_total on) over the run, first at step $(thr_first on))" "$r"
r=$(awk -v a="$(cells off | tail -1)" -v b="$(cells on | tail -1)" 'BEGIN { print (b > a) ? 1 : 0 }')
check "the fire ends larger with the threshold on ($(cells off | tail -1) -> $(cells on | tail -1) cells)" "$r"
# The threshold is held off for the steps before 4 s (16 steps of 0.25 s, or
# 17 when advance() is handed the start of the step), so the threshold line
# is absent for exactly those steps and the first ignition comes after them.
fs=$(thr_first on_spinup); [ -z "$fs" ] && fs=0
gated=$(( $(cells on_spinup | wc -l) - $(thr on_spinup | wc -l) ))
r=$(awk -v f="$fs" -v g="$gated" -v n="$(thr_total on_spinup)" 'BEGIN { print ((g == 16 || g == 17) && n > 0 && f > g) ? 1 : 0 }')
check "4 s start time: threshold held off for $gated steps, first ignition at step $fs ($(thr_total on_spinup) cells over the run)" "$r"
r=$(awk -v a="$(cells on | tail -1)" -v b="$(cells on_r5 | tail -1)" 'BEGIN { print (b >= a) ? 1 : 0 }')
check "a 5 m disc ends at least as large as a one-cell stamp ($(cells on | tail -1) vs $(cells on_r5 | tail -1) cells)" "$r"
r=$(awk -v n="$(thr_total on_farsite)" -v a="$(cells off_farsite | tail -1)" -v b="$(cells on_farsite | tail -1)" 'BEGIN { print (n > 0 && b >= a) ? 1 : 0 }')
check "FARSITE path: ignites by threshold ($(thr_total on_farsite) cells) and ends at least as large ($(cells off_farsite | tail -1) vs $(cells on_farsite | tail -1) cells)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
