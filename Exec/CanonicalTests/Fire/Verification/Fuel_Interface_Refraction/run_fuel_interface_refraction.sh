#!/bin/bash
# Run the Fuel_Interface_Refraction decks and check them against the exact answer.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] ./run_fuel_interface_refraction.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_fuel_interface_refraction.sh x      # only rerun the checks on existing output

set -u
EXE=${1:?usage: run_fuel_interface_refraction.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="fast_to_slow slow_to_fast"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file="plt_fire_${v}_" "$@" \
        > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

${PYTHON:-python3} check_fuel_interface_refraction.py $VARIANTS
