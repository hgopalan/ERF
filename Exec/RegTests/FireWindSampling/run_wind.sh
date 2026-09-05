#!/bin/bash
# Run the wind-sampling variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_wind.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_wind.sh x     # only rerun the checks on existing logs
#
# Every step with fire_debug prints the largest reference wind on the fire
# grid before the wind adjustment factor. Checks: the deck with the key
# written out reproduces the historical deck line for line; the sampled
# deck's reference wind is ln(6.1/0.1)/ln(20/0.1) times the deck that takes
# the wind at 20 m directly, at every step; the factor the code prints is
# that value. The burned cells of the historical and sampled decks are
# tabulated for comparison.

set -u
EXE=${1:?usage: run_wind.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="off off_key sample20 ref20"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

wind()  { grep 'Max reference wind' "run_$1.log" | sed 's/.*Max reference wind: \([^ ]*\) m.*/\1/'; }
fac()   { grep 'log-law factor' "run_$1.log" | tail -1 | sed 's/.*: //'; }
cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }

printf "%-10s %6s %12s %12s %10s %7s\n" variant steps wind_s1 wind_end factor cells
printf "%-10s %6s %12s %12s %10s %7s\n" ---------- ------ ------------ ------------ ---------- -------
for v in $VARIANTS; do
    printf "%-10s %6s %12s %12s %10s %7s\n" "$v" "$(wind $v | wc -l | tr -d ' ')" "$(wind $v | sed -n 1p | cut -c1-12)" "$(wind $v | tail -1 | cut -c1-12)" "$(fac $v | cut -c1-10)" "$(cells $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(wind off) <(wind off_key) > /dev/null && echo 1 || echo 0)
check "key written out reproduces the historical deck ($(wind off | wc -l | tr -d ' ') steps)" "$same"

r=$(paste <(wind ref20) <(wind sample20) | awk 'BEGIN { f = log(6.1 / 0.1) / log(20.0 / 0.1); ok = 1; worst = 0 }
    { if ($1 > 0) { d = ($2 - f * $1) / $1; if (d < 0) d = -d; if (d > worst) worst = d; if (d > 1e-6) ok = 0 } }
    END { printf "%d %.2e %.6f", ok, worst, f }')
set -- $r
check "sampled reference wind = ln(6.1/0.1)/ln(20/0.1) x the wind taken at 20 m, every step (worst rel. diff $2, factor $3)" "$1"

r=$(awk -v f="$(fac sample20)" 'BEGIN { e = f - log(6.1 / 0.1) / log(20.0 / 0.1); if (e < 0) e = -e; print (e < 1e-6) ? 1 : 0 }')
check "printed log-law factor is that value ($(fac sample20))" "$r"

echo "  burned cells at 60 s: historical $(cells off), sampled at 20 m $(cells sample20) (the profile above the surface layer is faster than the log law assumes)"
[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
