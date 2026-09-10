#!/bin/bash
# Run the equilibrium-moisture-curve variants and compare them.
#
#   [MPIRUN="mpirun -np 4"] ./run_emc.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_emc.sh x     # only rerun the checks on existing output
#
# Checks: the deck with erf.fire.emc_model = legacy written out reproduces the
# historical deck bit for bit; van_wagner differs from it; dry air reaches the
# fuel as zero RH and the humid sounding as about 40 %; every variant moves the
# dead classes in lag order; van_wagner leaves the fuel drier than legacy in
# dry air (its curves go to zero there, legacy's clamp at 0.035-0.060) and
# less wet in humid air (0.09-0.11 against 0.17-0.19 at 40 % RH), and the
# no-wind Rothermel rate follows.

set -u
EXE=${1:?usage: run_emc.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="legacy legacy_key van_wagner humid_legacy humid_van_wagner"

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

python3 check_emc.py table
echo

status=0
python3 check_emc.py same legacy legacy_key || status=1
if python3 check_emc.py same legacy van_wagner > /dev/null 2>&1; then
    echo "  van_wagner differs from legacy: FAIL"; status=1
else
    echo "  van_wagner differs from legacy: PASS"
fi
python3 check_emc.py order || status=1

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
