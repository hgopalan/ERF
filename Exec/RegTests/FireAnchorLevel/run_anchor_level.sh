#!/bin/bash
# Run the FireAnchorLevel variants and check them.
#
#   [MPIRUN="mpirun -np 2"] ./run_anchor_level.sh /path/to/erf_exec
#   SKIP_RUN=1 ./run_anchor_level.sh x     # only rerun the checks on existing output
#
# single        inputs_single: the reference, one level at level 1's resolution
# anchor        inputs_base: the fire on level 1, with checkpoints every 20 steps
# restart       inputs_base restarted from step 20 of anchor's checkpoint
# heat          inputs_heat: a prescribed heat disc on level 1, two-way coupling
# heat_anchor0  inputs_heat with erf.fire.anchor_level = 0: warns, loses the heat
# moved         heat's checkpoint restarted with erf.fire.anchor_level = 0: must stop
#               (the heat deck has no fuel map, whose size would stop the run first)
# heat_mrf      inputs_heat with MRF and its fire thermal excess, which read the fire
#               flux on level 1, checkpoints every 10 steps
# heat_mrf_restart  heat_mrf restarted from its step-10 checkpoint

set -u
MRF="erf.pbl_type=MRF erf.pbl_mrf_fire_thermal_excess=true"
EXE=${1:?usage: run_anchor_level.sh /path/to/erf_exec}
PY=${PYTHON:-python3}

run () {   # variant deck [extra erf args...]
    v=$1; deck=$2; shift 2
    ${MPIRUN:-} "$EXE" "$deck" erf.fire_plot_file="plt_fire_${v}_" erf.plot_file_1="plt_atm_${v}_" \
        erf.fire.fire_stats_csv_file="fire_stats_${v}.csv" "$@" > "run_${v}.log" 2>&1
}

must_run () {
    if ! run "$@"; then
        echo "run $1 failed; the end of run_$1.log:"
        tail -n 30 "run_$1.log"
        exit 1
    fi
}

if [ "${SKIP_RUN:-0}" != "1" ]; then
    rm -rf plt_fire_* plt_atm_* chk_anchor* chk_heat* fire_stats_*.csv run_*.log Backtrace.*
    must_run single       inputs_single
    must_run anchor       inputs_base erf.check_int=20 erf.check_file=chk_anchor
    must_run restart      inputs_base erf.restart=chk_anchor00020
    must_run heat         inputs_heat erf.check_int=10 erf.check_file=chk_heat
    must_run heat_anchor0 inputs_heat erf.fire.anchor_level=0
    run moved inputs_heat erf.restart=chk_heat00010 erf.fire.anchor_level=0 && \
        { echo "  restart with the fire grid moved to level 0 did not stop: FAIL"; exit 1; }
    must_run heat_mrf         inputs_heat $MRF erf.check_int=10 erf.check_file=chk_heat_mrf
    must_run heat_mrf_restart inputs_heat $MRF erf.restart=chk_heat_mrf00010
fi

status=0
$PY check_anchor_level.py parity heat restart mrf || status=1
if grep -q "holds the fire state on level 1, but this run puts the fire grid on level 0" run_moved.log; then
    echo "  restart with the fire grid moved to level 0 stops at start-up: PASS"
else
    echo "  restart with the fire grid moved to level 0 stops at start-up: FAIL"
    status=1
fi

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
