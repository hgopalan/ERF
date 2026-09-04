#!/bin/bash
# Run every variant of the exposure deck and print, per variant and per
# structure, the last line of its exposure CSV.
#
#   ./run_exposure.sh /path/to/erf_exec [extra erf args...]
#   MPIRUN="mpirun -np 4" ./run_exposure.sh /path/to/erf_exec
#   SKIP_RUN=1 ./run_exposure.sh x     # only rebuild the table from existing CSVs
#
# Columns: structure id and centre, the fraction of its wall band burned, the
# first and last arrival of the front there and their difference (residence),
# the peak fireline intensity and the mean and largest accumulated heat load
# in the band, and the number of embers that landed on the footprint. The
# last column is the number of brands that landed anywhere over the run.

set -u

EXE=${1:?usage: [MPIRUN="mpirun -np 4"] run_exposure.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="noib ib noib_spotting noib_spotting_front"

printf "%-19s %3s %6s %6s %7s %7s %7s %6s %9s %9s %9s %7s %7s\n" variant id x y burned t_first t_last resid peak_kWm HL_mean HL_max embers landed
printf "%-19s %3s %6s %6s %7s %7s %7s %6s %9s %9s %9s %7s %7s\n" ------------------- --- ------ ------ ------- ------- ------- ------ --------- --------- --------- ------- -------

for v in $VARIANTS; do
    log="run_${v}.log"
    csv="exposure_${v}.csv"
    if [ "${SKIP_RUN:-0}" != "1" ] || [ ! -f "$csv" ]; then
        rm -f "$csv"
        ${MPIRUN:-} "$EXE" "inputs_${v}" "$@" > "$log" 2>&1 || echo "$v exited with $?"
    fi
    landed=$(grep -o 'launched=[0-9]*' "$log" 2>/dev/null | awk -F= '{s+=$2} END {print s+0}')
    # Last row per structure id (column 2).
    tail -n +2 "$csv" | awk -F, '{last[$2]=$0} END {for (k in last) print last[k]}' | sort -t, -k2,2n |
    while IFS=, read -r t id x y h foot wall wf t0 t1 res pk hlm hlx emb; do
        printf "%-19s %3s %6.0f %6.0f %7.2f %7.0f %7.0f %6.0f %9.1f %9.3f %9.3f %7s %7s\n" "$v" "$id" "$x" "$y" "$wf" "$t0" "$t1" "$res" "$pk" "$hlm" "$hlx" "$emb" "$landed"
    done
done
