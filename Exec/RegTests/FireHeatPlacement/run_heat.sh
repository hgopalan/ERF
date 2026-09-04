#!/bin/bash
# Run the heat-placement variants and print the comparison.
#
#   [MPIRUN="mpirun -np 8"] ./run_heat.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_heat.sh x     # only rebuild the table from existing logs
#
# Columns: burned fire cells at 120 s; the injected-to-supplied energy ratio
# of the last coupling call (1 - exp(-z_top/alfg) with the default tendency,
# about rho times that with the historical form of the _legacy rows); the largest share of a
# partial column's heating that lands below the roof (1 - exp(-H/alfg) for the
# plain profile, reduced by the open fraction with heat_open_fraction); the
# largest potential temperature of the state in cells below a roof; and the
# arrival time [s] at the middle box's upwind face and gap probes.

set -u

EXE=${1:?usage: run_heat.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="overwrite_noib add_noib add_noib_open add_noib_open_legacy
          overwrite_ib add_ib add_ib_open add_ib_open_legacy
          overwrite_ib_slow add_ib_slow"

printf "%-22s %5s %7s %10s %12s %10s  %s\n" variant exit cells E_ratio below_roof theta_blk "u2 g1 g2"
printf "%-22s %5s %7s %10s %12s %10s  %s\n" ---------------------- ----- ------- ---------- ------------ ---------- ----------

for v in $VARIANTS; do
    log="run_${v}.log"
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "$log" ]; then
        rc=$(grep -q 'MPI_ABORT\|SIGABRT\|Segfault' "$log" && echo 6 || echo 0)
    else
        ${MPIRUN:-} "$EXE" "inputs_${v}" "$@" > "$log" 2>&1
        rc=$?
    fi
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ratio=$(grep 'energy_in=' "$log" | tail -1 | sed 's/.*ratio=//; s/ .*//' | cut -c1-8)
    blocked=$(grep 'energy_in=' "$log" | tail -1 | sed 's/.*below_roof_share_max=//; s/ .*//' | cut -c1-10)
    thblk=$(grep 'energy_in=' "$log" | tail -1 | sed 's/.*theta_blocked_max=//' | cut -c1-8)
    probes=""
    for n in 1 3 4; do
        t=$(grep "\[FIRE PROBE\] $n " "$log" | sed -n 's/.*arrival_time_s=//p' | head -1)
        if [ -n "$t" ]; then probes="$probes $(printf '%4.0f' "$t")"; else probes="$probes    -"; fi
    done
    printf "%-22s %5s %7s %10s %12s %10s  %s\n" "$v" "$rc" "${cells:-n/a}" "${ratio:--}" "${blocked:--}" "${thblk:--}" "$probes"
done
