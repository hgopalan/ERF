#!/bin/bash
# Run the five variants of the WUI_Subdivision case and check them.
#
#   [MPIRUN="mpirun -np 2"] ./run_wui.sh /path/to/erf_exec [extra erf args...]
#
# The domain is two boxes, so at most two ranks. SKIP_RUN=1 only re-checks.
set -u
EXE=${1:?usage: [MPIRUN="mpirun -np 2"] run_wui.sh /path/to/erf_exec [extra args]}
shift || true
if [ -z "${SKIP_RUN:-}" ]; then
    [ -f houses_10m_96x48.txt ] || python3 gen_wui.py
    for v in wildland wildland_spotting subdivision defensible coupled; do
        rm -f exposure_${v}.csv fire_stats_${v}.csv
        ${MPIRUN:-} "$EXE" inputs_${v} "$@" > run_${v}.log 2>&1
        echo "$v: exit $?"
    done
fi
python3 check_wui.py
