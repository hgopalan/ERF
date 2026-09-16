#!/bin/bash
# Run the FireStructureIgnition variants and check them.
#
#   [MPIRUN="mpirun -np 1"] ./run_structure_ignition.sh /path/to/erf_exec
#   SKIP_RUN=1 ./run_structure_ignition.sh x     # only rerun the checks
#
# off        inputs_off: the old path, exposure without ignition
# on         inputs_on: ignition on, heat and ember criteria, with a checkpoint at step 20
# radiation  inputs_radiation: no spotting, ember criterion off; B must ignite from A's radiation alone
# restart    inputs_on restarted from the step-20 checkpoint, into its own outputs
#
# The checker is then run on the three ignition cases, and once on the old
# path, where it must fail (the old path has no ignition columns).

set -u
EXE=${1:?usage: run_structure_ignition.sh /path/to/erf_exec}
PY=${PYTHON:-python3}

run () {   # variant deck [extra erf args...]
    v=$1; deck=$2; shift 2
    ${MPIRUN:-} "$EXE" "$deck" "$@" > "run_${v}.log" 2>&1
}

must_run () {
    if ! run "$@"; then
        echo "run $1 failed; the end of run_$1.log:"
        tail -n 30 "run_$1.log"
        exit 1
    fi
}

if [ "${SKIP_RUN:-0}" != "1" ]; then
    rm -rf plt_fire_* chk_on* exposure_*.csv fire_stats_*.csv run_*.log Backtrace.*
    must_run off       inputs_off
    must_run on        inputs_on erf.check_int=20 erf.check_file=chk_on
    must_run radiation inputs_radiation
    must_run restart   inputs_on erf.restart=chk_on00020 erf.fire_plot_file=plt_fire_restart_ \
                       erf.fire.exposure.file=exposure_restart.csv erf.fire.fire_stats_csv_file=fire_stats_restart.csv
fi

status=0
$PY check_structure_ignition.py on radiation restart || status=1
echo "== old path (ignition off): the ignition checks must fail"
if $PY check_structure_ignition.py old > check_old.log 2>&1; then
    echo "  the checker passed on the old path: FAIL"
    status=1
else
    echo "  the checker fails on the old path: PASS"
fi
grep -q "FIRE STRUCTURE" run_off.log && { echo "  the old path printed a structure ignition line: FAIL"; status=1; }

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
