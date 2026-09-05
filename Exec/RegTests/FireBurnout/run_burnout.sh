#!/bin/bash
# Run the burnout variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_burnout.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_burnout.sh x     # only rerun the checks on existing logs
#
# Every step with fire_debug prints the largest heat flux on the fire grid,
# the total power and the fuel left. Checks: the deck with the key written
# out reproduces the historical deck line for line; in each sfire deck the
# largest heat flux over the run equals the fresh-cell value w0 h / tau the
# code printed (a cell just ignited, fuel intact) and never exceeds it; the
# energy released over the run equals h times the fuel consumed to 2 % on
# grass (the flux is held over a step, an O(dt/tau) bias shared with the
# historical form) and 0.1 % on litter; on litter the sfire deck leaves more
# fuel at 60 s than the crossing-time deck.

set -u
EXE=${1:?usage: run_burnout.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="residence residence_key sfire_grass sfire_override residence_litter sfire_litter"
DT=0.125

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

qmax()  { grep 'Current max heat flux' "run_$1.log" | sed 's/.*Current max heat flux: \([^ ]*\) W.*/\1/'; }
power() { grep 'Current max heat flux' "run_$1.log" | sed 's/.*total_power_W=\([^ ]*\) .*/\1/'; }
fuel()  { grep 'Current max heat flux' "run_$1.log" | sed 's/.*fuel_kg=\([^ ]*\).*/\1/'; }
fresh() { grep 'fresh-cell heat flux' "run_$1.log" | tail -1 | sed 's/.*tau=\([^ ]*\) W.*/\1/'; }
tau()   { grep 'fresh-cell heat flux' "run_$1.log" | tail -1 | sed 's/.* tau=\([^ ]*\) s, w0.*/\1/'; }
cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }
heat()  { grep 'fresh-cell heat flux' "run_$1.log" | tail -1 | sed 's/.*w0=\([^ ]*\) kg.*/\1/'; }

printf "%-18s %8s %12s %12s %12s %12s %7s\n" variant tau_s Qmax_run Q_fresh P_peak_MW fuel_left_kg cells
printf "%-18s %8s %12s %12s %12s %12s %7s\n" ------------------ -------- ------------ ------------ ------------ ------------ -------
for v in $VARIANTS; do
    printf "%-18s %8s %12s %12s %12s %12s %7s\n" "$v" "$(tau $v)" "$(qmax $v | sort -g | tail -1 | cut -c1-12)" "$(fresh $v | cut -c1-12)" \
        "$(power $v | sort -g | tail -1 | awk '{printf "%.4f", $1 / 1e6}')" "$(fuel $v | tail -1 | cut -c1-12)" "$(cells $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(qmax residence; power residence; fuel residence) <(qmax residence_key; power residence_key; fuel residence_key) > /dev/null && echo 1 || echo 0)
check "key written out reproduces the historical deck ($(qmax residence | wc -l | tr -d ' ') steps)" "$same"

for v in sfire_grass sfire_override sfire_litter; do
    r=$(qmax $v | awk -v f="$(fresh $v)" 'BEGIN { ok = 1; mx = 0 } { if ($1 > mx) mx = $1; if ($1 > f * (1 + 1e-9)) ok = 0 }
        END { d = (mx - f) / f; if (d < 0) d = -d; if (d > 1e-9) ok = 0; printf "%d %.2e", ok, d }')
    set -- $r
    check "$v: largest heat flux over the run = fresh-cell w0 h / tau and never above it (rel. diff $2)" "$1"
done

# energy released = h x fuel consumed (fuel_kg is the whole fire grid; h from w0 line: Q_fresh tau / w0).
# Each log line prints the power before that step's depletion and the fuel after it, so the
# power is integrated from the second line to match the fuel difference from the first.
for v in sfire_grass sfire_litter; do
    case $v in sfire_grass) tol=0.02;; *) tol=0.001;; esac
    r=$(paste <(power $v) <(fuel $v) | awk -v dt=$DT -v qf="$(fresh $v)" -v ta="$(tau $v)" -v w0="$(heat $v)" -v tol=$tol '
        NR == 1 { f0 = $2; f1 = $2; next } { e += $1 * dt; f1 = $2 }
        END { h = qf * ta / w0; c = h * (f0 - f1); d = (e - c) / c; if (d < 0) d = -d; printf "%d %.4f %.4e %.4e", (d <= tol) ? 1 : 0, d, e, c }')
    set -- $r
    check "$v: energy released over the run = h x fuel consumed to $tol (rel. diff $2; $3 vs $4 J)" "$1"
done

r=$(awk -v a="$(fuel residence_litter | tail -1)" -v b="$(fuel sfire_litter | tail -1)" 'BEGIN { print (b > a) ? 1 : 0 }')
check "litter: the sfire deck leaves more fuel at 60 s than the crossing-time deck ($(fuel sfire_litter | tail -1 | cut -c1-8) vs $(fuel residence_litter | tail -1 | cut -c1-8) kg)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
