#!/bin/bash
# Run every deck calm and at 4.005 m/s, then check the head rates.
#
#   [MPIRUN="mpirun -np 1"] [PYTHON=python3] ./run_regtest.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_regtest.sh x     # only rerun the check on existing output
#
# Ten five-step runs: the four combinations of reaction_velocity_formula and
# wrf_bmst_compat, plus the rothermel_bmst deck through a fuel map
# (rothermel_per_fuel), each without wind and with it. Without wind every pair
# of combinations differs by more than 1 %; at 4.005 m/s the albini pair differs
# by only 0.006 %, because the lower load slows the no-wind rate and raises the
# wind factor by almost the same amount. check_regtest.py compares each head
# rate with the closed-form Rothermel rate of that combination.
#
# If MPI_Init fails with MPICH's default OFI provider, export FI_PROVIDER=tcp.

set -u
EXE=${1:?usage: run_regtest.sh /path/to/erf_exec [extra args]}
shift || true
PY=${PYTHON:-python3}
DECKS="albini albini_bmst rothermel rothermel_bmst rothermel_bmst_map"

if [ "${SKIP_RUN:-0}" != "1" ]; then
    for d in $DECKS; do
        for w in calm wind; do
            if [ "$w" = "calm" ]; then u=0.0; else u=4.005; fi
            tag="${d}_${w}"
            rm -f "fire_stats_${tag}.csv"
            ${MPIRUN:-} "$EXE" "inputs_$d" erf.fire.prescribed_wind_x=$u \
                erf.fire.fire_stats_csv_file="fire_stats_${tag}.csv" "$@" < /dev/null \
                > "run_${tag}.log" 2>&1 || { echo "run $tag failed (see run_${tag}.log)"; tail -n 30 "run_${tag}.log"; exit 1; }
        done
    done
fi

$PY check_regtest.py
