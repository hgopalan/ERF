#!/bin/bash
# Run the Slope_No_Wind decks and check the rates against Rothermel's slope factor.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] ./run_slope_no_wind.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_slope_no_wind.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_slope_no_wind.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="iso_s30 iso_s60 dir_s30 dir_s60 ell_s30 ell_s60 and_s30 and_s60"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_slope_no_wind.py $VARIANTS
