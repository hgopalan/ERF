#!/bin/bash
# Run the UNR WRF-Fire case-study decks and print their heads next to WRF's.
#
#   [MPIRUN="mpirun -np 4"] [PYTHON=python3] [DECKS="case1_compressible ..."] ./run_unr_wrf_fire.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_unr_wrf_fire.sh x      # run nothing; check the existing output only
#
# With SKIP_RUN=1 no deck is run, whether or not it has a run_<deck>.log (a deck run
# by hand has none); the check fails on a deck without output.
#
# The anelastic decks need an ERF built with FFT (erf.use_fft = 1). All eight decks
# take several hours on four ranks; DECKS picks a subset (README.md lists the costs).

set -u
EXE=${1:?usage: run_unr_wrf_fire.sh /path/to/erf_exec [extra args]}
shift || true
DECKS=${DECKS:-"case1_compressible case1_anelastic case2_compressible case2_anelastic
                case3_wind2p5_anelastic case3_wind2p5_compressible case3_nowind_anelastic case3_nowind_compressible"}

if [ "${SKIP_RUN:-0}" != "1" ]; then
    for v in $DECKS; do
        ${MPIRUN:-} "$EXE" "inputs_$v" "$@" \
            > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
    done
fi

${PYTHON:-python3} check_unr_wrf_fire.py $DECKS
