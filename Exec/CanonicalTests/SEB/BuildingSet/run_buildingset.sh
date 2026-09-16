#!/bin/bash
# Canonical building-set morning: run, check, plot.
#
#   ./run_buildingset.sh /path/to/erf_exec        # NP=4 by default, about two hours
set -u
EXE=${1:?usage: run_buildingset.sh /path/to/erf_exec}
NP=${NP:-4}
rm -f ibseb_set.csv surf_hist mean_profiles; rm -rf faces plt* chk*; mkdir -p faces
echo "== building set, 6 h from 05:00 ($NP ranks)"
# Open MPI sometimes reports an improper exit after a run that finished and
# finalised; trust the log and the last plotfile instead of the exit code then.
if ! mpirun -np $NP "$EXE" inputs > run_set.log 2>&1; then
    if awk '/AMReX .* finalized/ {f = 1} END {exit !f}' run_set.log && [ -f plt43200/Header ]; then
        echo "mpirun exited non-zero after ERF finalised; checking the output"
    else
        echo "run failed (see run_set.log)"; exit 1
    fi
fi
python3 check_buildingset.py ibseb_set.csv faces/set run_set.log mean_profiles && echo "ALL PASS" || { echo "SOME CHECKS FAILED"; exit 1; }
