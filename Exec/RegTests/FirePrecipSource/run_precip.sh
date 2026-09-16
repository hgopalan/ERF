#!/bin/bash
# Run the rain-source variants and check them.
#
#   [MPIRUN="mpirun -np 1"] ./run_precip.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_precip.sh x     # only rerun the checks on existing output
#
# Five 40-step runs of the same passive grass fire under Kessler rain: the
# historical deck with no rain key, the uniform 2 mm/hr rate, the atmosphere's
# rain per column, the latter to a checkpoint at step 27, and the restart from
# it. check_precip.py then compares the fire plotfiles at step 40 (see its
# docstring for the checks). The 20-cell grid runs on one rank.

set -u
EXE=${1:?usage: run_precip.sh /path/to/erf_exec [extra args]}
shift || true
PY=${PYTHON:-python3}
VARIANTS="legacy uniform atmosphere atmosphere_chk atmosphere_restart"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    [ "$v" = "atmosphere_chk" ] && rm -rf chk00027
    rm -rf plt_fire_$v plt_$v fire_stats_$v.csv; mkdir -p plt_fire_$v plt_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ erf.plot_file_1=plt_$v/plt "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; tail -n 30 "run_$v.log"; exit 1; }
done

rain() { grep 'Rain rate' "run_$1.log" | tail -1 | sed 's/.*max=//; s/ mm.*//'; }
cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }
mc() { grep 'Max 1-hour moisture' "run_$1.log" | tail -1 | awk '{print $NF}'; }
printf "%-20s %12s %14s %8s\n" variant "max rain mm/hr" "max 1-h moist" cells
for v in $VARIANTS; do printf "%-20s %12s %14s %8s\n" "$v" "$(rain $v)" "$(mc $v)" "$(cells $v)"; done
echo
# the last row of the statistics CSVs: precip_max_mm_hr is the last column
for v in legacy uniform atmosphere; do
    printf "%-20s last CSV row: %s\n" "$v" "$(tail -1 fire_stats_$v.csv)"
done
echo

$PY check_precip.py
