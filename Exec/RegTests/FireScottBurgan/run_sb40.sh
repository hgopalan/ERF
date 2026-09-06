#!/bin/bash
# Run the Scott-Burgan variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_sb40.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_sb40.sh x     # only rerun the checks on existing output
#
# Checks: the deck with the fuel set written out reproduces the historical
# deck line for line; the uniform GR2 deck's fuel parameters equal the
# Scott-Burgan table (with the herbaceous curing at the deck's live
# moisture); the map deck's initial fuel equals the sum of the cells' model
# loads, its non-burnable cells never burn, and its burned area is finite;
# the crosswalk deck reproduces the hand-crosswalked Anderson map line for
# line.

set -u
EXE=${1:?usage: run_sb40.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="anderson anderson_key sb_gr2 sb_map sb_map_crosswalk anderson_map"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

cells() { grep 'active fire cells' "run_$1.log" | awk '{print $NF}'; }
qmax()  { grep 'Current max heat flux' "run_$1.log" | sed 's/.*Current max heat flux: \([^ ]*\) W.*/\1/'; }
fuel()  { grep 'Current max heat flux' "run_$1.log" | sed 's/.*fuel_kg=\([^ ]*\).*/\1/'; }
ros()   { grep 'max_ROS=' "run_$1.log" | tail -1 | sed 's/.*max_ROS=\([^ ]*\) .*/\1/'; }

printf "%-18s %8s %12s %12s %12s\n" variant cells max_ROS fuel_kg_0 fuel_kg_end
printf "%-18s %8s %12s %12s %12s\n" ------------------ -------- ------------ ------------ ------------
for v in $VARIANTS; do
    printf "%-18s %8s %12s %12s %12s\n" "$v" "$(cells $v | tail -1)" "$(ros $v | cut -c1-12)" "$(fuel $v | head -1 | cut -c1-12)" "$(fuel $v | tail -1 | cut -c1-12)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(cells anderson; qmax anderson) <(cells anderson_key; qmax anderson_key) > /dev/null && echo 1 || echo 0)
check "fuel set written out reproduces the historical deck ($(cells anderson | wc -l | tr -d ' ') steps)" "$same"

python3 check_sb40.py params run_sb_gr2.log 102 || status=1
python3 check_sb40.py fuel run_sb_map.log fuel_map_sb40.asc 1.25 || status=1
python3 check_sb40.py mask plt_fire_sb_map/plt_fire_00480 fuel_map_sb40.asc 2>&1 | grep -v "^yt" || status=1

same=$(diff <(cells sb_map_crosswalk; qmax sb_map_crosswalk; fuel sb_map_crosswalk) <(cells anderson_map; qmax anderson_map; fuel anderson_map) > /dev/null && echo 1 || echo 0)
check "crosswalk deck reproduces the hand-crosswalked Anderson map line for line" "$same"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
