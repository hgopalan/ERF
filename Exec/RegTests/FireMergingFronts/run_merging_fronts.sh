#!/bin/bash
# Run the FireMergingFronts decks and check them against the geometry of their ignitions.
#
#   [MPIRUN="mpirun -np 2"] [PYTHON=python3] ./run_merging_fronts.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_merging_fronts.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_merging_fronts.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="coalescing junction parallel"
here=$(cd "$(dirname "$0")" && pwd)
[ -f erf_plotfile.py ] || cp "$here/../../CanonicalTests/Canonical_RANS/erf_plotfile.py" .

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf "plt_fire_${v}_"?????
    ${MPIRUN:-} "$EXE" "inputs_$v" "$@" > "run_$v.log" 2>&1 \
        || { echo "run $v failed (see run_$v.log)"; tail -n 30 "run_$v.log"; exit 1; }
done

${PYTHON:-python3} check_merging_fronts.py
