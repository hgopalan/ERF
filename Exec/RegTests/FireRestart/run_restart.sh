#!/bin/bash
# Check that the fire state survives a checkpoint and restart.
#
#   ./run_restart.sh /path/to/erf_exec [extra erf args...]
#
# For each row (FARSITE path, level-set path, lagged coupling, exposure
# diagnostics) three runs are made: straight to
# 200 s, to 100 s with a checkpoint at step 200, and a restart from that
# checkpoint to 200 s. The burned-cell count and head rate of spread at 200 s
# of the restarted run must equal those of the straight run; the exposure row
# also compares the last line of the two exposure CSVs.

set -u

EXE=${1:?usage: [MPIRUN="mpirun -np 8"] run_restart.sh /path/to/erf_exec [extra args]}
shift || true

run() {
    local name=$1
    shift
    ${MPIRUN:-} "$EXE" "inputs_${name}" "$@" > "run_${name}.log" 2>&1
    echo $?
}
stats() {
    local log=$1
    local cells ros
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ros=$(grep 'Rate-of-spread computed' "$log" | tail -1 | sed 's/.*Max: //; s/,.*//')
    echo "${cells:-n/a} ${ros:-n/a}"
}

printf "%-10s %8s %10s %14s %10s %14s %8s\n" path straight_rc s_cells s_ROS r_cells r_ROS match
printf "%-10s %8s %10s %14s %10s %14s %8s\n" ---------- -------- ---------- -------------- ---------- -------------- --------

for p in farsite levelset coupled exposure; do
    rm -rf chk00200
    rc_s=$(run ${p}_straight "$@")
    rc_c=$(run ${p}_chk "$@")
    rc_r=$(run ${p}_restart "$@")
    read s_cells s_ros <<< "$(stats run_${p}_straight.log)"
    read r_cells r_ros <<< "$(stats run_${p}_restart.log)"
    ok=no
    [ "$rc_s$rc_c$rc_r" = "000" ] && [ "$s_cells" = "$r_cells" ] && [ "$s_ros" = "$r_ros" ] && ok=yes
    if [ "$p" = "exposure" ]; then
        # The exposure row must also reproduce the last CSV line of every structure.
        rm -f exposure_chk.csv
        if [ "$(tail -1 exposure_straight.csv 2>/dev/null)" != "$(tail -1 exposure_restart.csv 2>/dev/null)" ]; then ok=no; fi
    fi
    printf "%-10s %8s %10s %14s %10s %14s %8s\n" "$p" "$rc_s$rc_c$rc_r" "$s_cells" "$s_ros" "$r_cells" "$r_ros" "$ok"
done
