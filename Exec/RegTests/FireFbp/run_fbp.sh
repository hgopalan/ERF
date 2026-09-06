#!/bin/bash
# Run the FBP variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_fbp.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_fbp.sh x     # only rerun the checks on existing logs
#
# The wind is uniform (8 m/s at every height) at the first step, so the
# largest rate of spread the code prints at step 1 is the FBP head rate at
# 8 m/s on flat ground. Checks: for C-2, O-1b (80 % cured) and M-1 (60 %
# conifer) that rate equals the independent fbp_reference.py to 1e-8; the
# directional C-2 deck's head rate at step 1 equals the isotropic one; and
# the directional deck burns less area (its flanks run slower).

set -u
EXE=${1:?usage: run_fbp.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="rothermel fbp_c2 fbp_o1b fbp_m1 fbp_c2_directional"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

ros1()  { grep 'max_ROS=' "run_$1.log" | head -1 | sed 's/.*max_ROS=\([^ ]*\) .*/\1/'; }
wind1() { grep 'Max reference wind' "run_$1.log" | head -1 | sed 's/.*Max reference wind: \([^ ]*\) m.*/\1/'; }
cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }

printf "%-20s %10s %14s %14s %7s\n" variant wind_s1 ROS_s1 ROS_reference cells
printf "%-20s %10s %14s %14s %7s\n" -------------------- ---------- -------------- -------------- -------
ref_c2=$(python3 fbp_reference.py C2 90 60 "$(wind1 fbp_c2)")
ref_o1b=$(python3 fbp_reference.py O1B 92 40 "$(wind1 fbp_o1b)" 80)
ref_m1=$(python3 fbp_reference.py M1 90 60 "$(wind1 fbp_m1)" 60 60)
for v in $VARIANTS; do
    case $v in fbp_c2|fbp_c2_directional) r=$ref_c2;; fbp_o1b) r=$ref_o1b;; fbp_m1) r=$ref_m1;; *) r=-;; esac
    printf "%-20s %10s %14s %14s %7s\n" "$v" "$(wind1 $v)" "$(ros1 $v | cut -c1-14)" "$(echo $r | cut -c1-14)" "$(cells $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }
rel() { awk -v a="$1" -v b="$2" 'BEGIN { d = (a - b) / b; if (d < 0) d = -d; printf "%.2e", d }'; }
ok()  { awk -v d="$1" -v t="$2" 'BEGIN { print (d <= t) ? 1 : 0 }'; }

for pair in "fbp_c2 $ref_c2 C-2" "fbp_o1b $ref_o1b O-1b" "fbp_m1 $ref_m1 M-1"; do
    set -- $pair
    d=$(rel "$(ros1 $1)" "$2")
    check "$3 head rate at step 1 equals the independent FBP ($(ros1 $1) vs $2 m/s, rel. diff $d)" "$(ok $d 1e-8)"
done
d=$(rel "$(ros1 fbp_c2_directional)" "$(ros1 fbp_c2)")
check "directional C-2 head rate at step 1 equals the isotropic one (rel. diff $d)" "$(ok $d 1e-8)"
r=$(awk -v a="$(cells fbp_c2)" -v b="$(cells fbp_c2_directional)" 'BEGIN { print (b < a) ? 1 : 0 }')
check "directional C-2 burns less area than isotropic ($(cells fbp_c2_directional) vs $(cells fbp_c2) cells)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
