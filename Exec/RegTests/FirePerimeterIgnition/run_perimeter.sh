#!/bin/bash
# Run the perimeter-ignition variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_perimeter.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_perimeter.sh x     # only rerun the checks on existing output
#
# Checks: the deck with the keys written out reproduces the historical deck
# line for line; the spin-up deck has no fire before 30 s, its first burning
# step is the one whose window contains 30 s, and that step's burned cells
# match the historical deck's first step (to 2 %); the interior deck's step-0
# plotfile has fuel = w0 exp(-d/(R tau)) and arrival = -d/R in every burned
# cell; the spin-up interior deck's stamped cells obey arrival = 30 - d0/R at
# 60 s with d0 the geometric distance inside the square (arrival times are
# clamped at the simulation start); the probes report those arrival times.

set -u
EXE=${1:?usage: run_perimeter.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="t0 t0_key spinup interior spinup_interior"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

cells() { grep 'active fire cells' "run_$1.log" | awk '{print $NF}'; }
probe() { grep "\[FIRE PROBE\] $2 " "run_$1.log" | sed -n 's/.*arrival_time_s=//p' | head -1; }

first() { cells $1 | awk '$1 > 0 { print NR; exit }'; }
printf "%-16s %8s %10s %10s %10s %10s\n" variant steps first_burn cells_1st cells_end probe_c
printf "%-16s %8s %10s %10s %10s %10s\n" ---------------- -------- ---------- ---------- ---------- ----------
for v in $VARIANTS; do
    f=$(first $v)
    printf "%-16s %8s %10s %10s %10s %10s\n" "$v" "$(cells $v | wc -l | tr -d ' ')" "$f" "$(cells $v | sed -n ${f}p)" "$(cells $v | tail -1)" "$(probe $v 1)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(cells t0) <(cells t0_key) > /dev/null && echo 1 || echo 0)
check "keys written out reproduce the historical deck ($(cells t0 | wc -l | tr -d ' ') steps)" "$same"

# The step whose window (t - dt, t] contains 30 s: step 240 when the window is
# closed at its end, 241 when advance() is handed the start time; accept both.
fs=$(first spinup)
r=$(awk -v f="$fs" 'BEGIN { print (f == 240 || f == 241) ? 1 : 0 }')
check "spin-up deck: no fire before 30 s, first burning step $fs (window containing 30 s)" "$r"
r=$(awk -v a="$(cells t0 | sed -n 1p)" -v b="$(cells spinup | sed -n ${fs}p)" 'BEGIN { d = (b - a) / a; if (d < 0) d = -d; print (d < 0.02) ? 1 : 0 }')
check "spin-up deck: burned cells at the stamp step match the historical deck's first step to 2 % ($(cells t0 | sed -n 1p) vs $(cells spinup | sed -n ${fs}p))" "$r"

python3 check_perimeter.py interior plt_fire_interior/plt_fire_00000 0.5 60.0 0.0 2>&1 | grep -v "^yt" || status=1
python3 check_perimeter.py arrival plt_fire_spinup_interior/plt_fire_00480 0.5 30.0 2>&1 | grep -v "^yt" || status=1

# Probes 1 and 2 sit at cell centres 110.625 m and 118.125 m, 9.375 m and 1.875 m
# inside the east edge: arrival 30 - d/0.5 = 11.25 s and 26.25 s in the spin-up
# interior deck, and 0 (clamped from -18.75 and -3.75 s) in the interior deck.
r=$(awk -v a1="$(probe interior 1)" -v a2="$(probe interior 2)" -v b1="$(probe spinup_interior 1)" -v b2="$(probe spinup_interior 2)" '
    function ne(x, y) { return (x - y > 1e-6 || y - x > 1e-6) }
    BEGIN { print (ne(a1, 0) || ne(a2, 0) || ne(b1, 11.25) || ne(b2, 26.25)) ? 0 : 1 }')
check "probes report the stamped arrival times (interior $(probe interior 1), $(probe interior 2) s; spin-up interior $(probe spinup_interior 1), $(probe spinup_interior 2) s)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
