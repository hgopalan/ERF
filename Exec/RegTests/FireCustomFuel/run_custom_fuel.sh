#!/bin/bash
# Run the deck-defined fuel variants and check them.
#
#   [MPIRUN="mpirun -np 2"] ./run_custom_fuel.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_custom_fuel.sh x     # only rerun the checks on existing output
#   FCOMPARE=/path/to/amrex_fcompare ./run_custom_fuel.sh ...   # adds the box-parity check
#
# Checks: a deck-defined model written out in SI reproduces the compiled
# Anderson model it copies; the start-up summary reports the deck's own SI
# back; a raster mixing a published set with a deck-defined code starts with
# each cell's own load; a coarse deck-defined bed spreads slower than grass;
# the same map run on one box and on split boxes agrees; and every range check
# aborts naming its input.

set -u
EXE=${1:?usage: run_custom_fuel.sh /path/to/erf_exec [extra args]}
shift || true

RUN_VARIANTS="anderson1 custom_grass custom_heavy custom_map"
BAD_VARIANTS="bad_code bad_missing bad_depth bad_heat bad_burnout bad_undeclared"

for v in $RUN_VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 \
        || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

# One box on one rank against the default decomposition: the fuel slot table is
# read per cell and the raster check reduces across ranks, so both have to give
# the same answer.
if [ "${SKIP_RUN:-0}" != "1" ] || [ ! -f run_custom_map_onebox.log ]; then
    rm -rf plt_fire_custom_map_onebox; mkdir -p plt_fire_custom_map_onebox
    "$EXE" inputs_custom_map amr.max_grid_size=64 amr.max_grid_size_z=32 \
        erf.fire_plot_file=plt_fire_custom_map_onebox/plt_fire_ "$@" \
        > run_custom_map_onebox.log 2>&1 \
        || { echo "run custom_map_onebox failed (see run_custom_map_onebox.log)"; exit 1; }
fi

# The range checks: each of these decks must stop the run.
for v in $BAD_VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    ${MPIRUN:-} "$EXE" "inputs_$v" max_step=1 "$@" > "run_$v.log" 2>&1
done

# Every rate-of-spread model reads its fuel through FuelModelParams, so each
# has to respond to the deck-defined properties. Ten short runs, grass against
# the coarse deck fuel, at 10 s.
MODELS="rothermel balbi behave macarthur cheney_gould"
for m in $MODELS; do
    for v in anderson1 custom_heavy; do
        if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_model_${m}_$v.log" ]; then continue; fi
        ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire.ros_model=$m stop_time=10.0 \
            erf.fire_plot_int=-1 "$@" > "run_model_${m}_$v.log" 2>&1 \
            || { echo "run $m/$v failed (see run_model_${m}_$v.log)"; exit 1; }
    done
done

ros() { grep 'max_ROS=' "run_$1.log" | tail -1 | sed 's/.*max_ROS=\([^ ]*\) .*/\1/'; }
fuel() { grep 'Current max heat flux' "run_$1.log" | sed 's/.*fuel_kg=\([^ ]*\).*/\1/'; }
cells() { grep 'active fire cells' "run_$1.log" | tail -1 | awk '{print $NF}'; }

printf "%-20s %8s %14s %14s %14s\n" variant cells max_ROS fuel_kg_0 fuel_kg_end
printf "%-20s %8s %14s %14s %14s\n" -------------------- -------- -------------- -------------- --------------
for v in $RUN_VARIANTS; do
    printf "%-20s %8s %14s %14s %14s\n" "$v" "$(cells $v)" "$(ros $v | cut -c1-14)" \
        "$(fuel $v | head -1 | cut -c1-14)" "$(fuel $v | tail -1 | cut -c1-14)"
done
echo

printf "%-14s %16s %16s\n" ros_model grass_ROS deck_fuel_ROS
printf "%-14s %16s %16s\n" -------------- ---------------- ----------------
for m in $MODELS; do
    printf "%-14s %16s %16s\n" "$m" "$(ros model_${m}_anderson1 | cut -c1-16)" \
        "$(ros model_${m}_custom_heavy | cut -c1-16)"
done
echo

status=0
for m in $MODELS; do
    a=$(ros model_${m}_anderson1); b=$(ros model_${m}_custom_heavy)
    if [ -z "$a" ] || [ -z "$b" ]; then
        echo "  FAIL $m: one of the runs printed no max_ROS"; status=1
    elif python3 -c "import sys; sys.exit(0 if abs($a-$b) > 1e-6*max(abs($a),1.0) else 1)"; then
        echo "  PASS $m responds to the deck-defined fuel ($a -> $b m/s)"
    else
        echo "  FAIL $m gives the same rate for grass and the deck fuel ($a); it is not reading the entry"
        status=1
    fi
done

python3 check_custom_fuel.py --selftest                                          || status=1
python3 check_custom_fuel.py identity run_anderson1.log run_custom_grass.log     || status=1
python3 check_custom_fuel.py summary  run_custom_grass.log 1000                  || status=1
python3 check_custom_fuel.py fuel     run_custom_map.log fuel_map_mixed.asc 1.25 || status=1
python3 check_custom_fuel.py slower   run_anderson1.log run_custom_heavy.log 2.0 || status=1

# Needs yt; skipped rather than failed where it is not installed.
if python3 -c "import yt" 2>/dev/null; then
    python3 check_custom_fuel.py crossed plt_fire_custom_map/plt_fire_00480 1000 102 2.0 || status=1
else
    echo "  SKIP crossed (yt not installed)"
fi

python3 check_custom_fuel.py identity run_custom_map.log run_custom_map_onebox.log || status=1

if [ -n "${FCOMPARE:-}" ]; then
    # -a allows the two BoxArrays; the tolerances are the round-off a different
    # decomposition leaves, not a physics tolerance.
    out=$("$FCOMPARE" -a --abs_tol 1.0e-12 --rel_tol 1.0e-10 \
          plt_fire_custom_map/plt_fire_00480 plt_fire_custom_map_onebox/plt_fire_00480 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  PASS box parity: the split and one-box fire plotfiles agree"
    else
        echo "  FAIL box parity (fcompare exit $rc)"; echo "$out" | tail -8; status=1
    fi
fi

python3 check_custom_fuel.py abort run_bad_code.log       "outside the custom range" || status=1
python3 check_custom_fuel.py abort run_bad_missing.log    sav_1h_1_m                 || status=1
python3 check_custom_fuel.py abort run_bad_depth.log      depth_m                    || status=1
python3 check_custom_fuel.py abort run_bad_heat.log       heat_content_J_kg          || status=1
python3 check_custom_fuel.py abort run_bad_burnout.log    burnout_time_s             || status=1
python3 check_custom_fuel.py abort run_bad_undeclared.log 1007                       || status=1

exit $status
