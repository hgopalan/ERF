#!/bin/sh
# Coupled restart: the atmosphere and the fire after a restart equal a straight run.
#
#   [MPIRUN="mpirun -np 2"] sh run_coupled_restart.sh /path/to/erf_exec [extra erf args...]
#
# The lagged coupling injects, in each step, the fire heat flux of the step
# before. That flux is held by the coupling layer and saved in the checkpoint
# (FireQAtmPrev, FireQLatAtmPrev); if a restart did not read it back, the first
# restarted step would inject no fire heat and the atmosphere would drift from
# the straight run. run_restart.sh compares only the burned-cell count and the
# head rate of spread, which that first step barely changes.
#
# Four runs of inputs_coupled_straight on nine boxes (MPIRUN, two ranks in CTest):
#   straight  steps 0-40
#   chk       steps 0-23, checkpoint chk00023 (23 is not a multiple of the
#             level-set reinitialisation interval)
#   restart   from chk00023 to step 40
#   nofeed    steps 0-40 with erf.fire.fire_atm_feedback = 0
# The atmosphere and fire plotfiles of step 40 of the straight and restarted
# runs must agree with amrex_fcompare at zero tolerance. The nofeed run must
# differ from the straight one in theta, so the fire heat really reaches the
# air in this window and the comparison above can see a missing flux.

set -u
EXE=${1:?usage: run_coupled_restart.sh /path/to/erf_exec [extra args]}
shift || true

# FCOMPARE comes from CTest (the build's FCOMPARE_EXE); the submodule tool is the fallback for a hand run
FCOMPARE=${FCOMPARE:-$(dirname "$EXE")/../Submodules/AMReX/Tools/Plotfile/amrex_fcompare}
[ -x "$FCOMPARE" ] || FCOMPARE=$(dirname "$EXE")/../Submodules/AMReX/Tools/Plotfile/amrex_fcompare
[ -x "$FCOMPARE" ] || FCOMPARE=$(dirname "$EXE")/../Submodules/AMReX/Tools/Plotfile/fcompare
if [ ! -x "$FCOMPARE" ]; then
    echo "  coupled restart: amrex_fcompare not found next to $EXE: FAIL"; exit 1
fi

rm -rf chk_cpl* plt_cpl_* plt_fire_cpl_* fire_stats*
common="amr.blocking_factor_x=4 amr.blocking_factor_y=4 amr.max_grid_size_x=8 amr.max_grid_size_y=8 stop_time=1000.0 erf.plot_int_1=40 erf.fire_plot_int=40 amrex.call_addr2line=0"
plotvars="erf.plot_vars_1=density x_velocity y_velocity z_velocity theta pressure"
leg() { # name, then extra inputs
    name=$1; shift
    ${MPIRUN:-} "$EXE" inputs_coupled_straight $common "$plotvars" \
        erf.plot_file_1="plt_cpl_${name}_" erf.fire_plot_file="plt_fire_cpl_${name}_" "$@" \
        > "run_cpl_${name}.log" 2>&1 \
        || { echo "  coupled restart: the $name run failed"; tail -20 "run_cpl_${name}.log"; exit 1; }
}
leg straight max_step=40 erf.check_int=-1 "$@"
leg chk      max_step=23 erf.check_int=23 erf.check_file=chk_cpl "$@"
leg restart  max_step=40 erf.check_int=-1 erf.restart=chk_cpl00023 "$@"
leg nofeed   max_step=40 erf.check_int=-1 erf.fire.fire_atm_feedback=0 "$@"

ok=yes
# 1. the run is decomposed: the lagged flux crosses box edges
nbox=$(grep -c '^((' plt_cpl_straight_00040/Level_0/Cell_H)
if [ "$nbox" -gt 1 ]; then
    echo "  coupled restart: $nbox boxes: PASS"
else
    echo "  coupled restart: $nbox box, the run is not decomposed: FAIL"; ok=no
fi
# 2. the restart found the lagged fluxes in the checkpoint
if grep -q "no lagged fire fluxes" run_cpl_restart.log; then
    echo "  coupled restart: the checkpoint held no lagged fluxes: FAIL"; ok=no
else
    echo "  coupled restart: the lagged fluxes were in the checkpoint: PASS"
fi
# 3. the fire heats the air in this window
"$FCOMPARE" plt_cpl_straight_00040 plt_cpl_nofeed_00040 > cmp_nofeed.log 2>&1
dtheta=$(awk '$1 == "theta" { print $2 }' cmp_nofeed.log)
if awk -v d="${dtheta:-0}" 'BEGIN { exit !(d + 0 > 0) }'; then
    echo "  coupled restart: the fire feedback changes theta by up to $dtheta K: PASS"
else
    echo "  coupled restart: theta with and without fire feedback is the same, the test cannot see the flux: FAIL"; ok=no
fi
# 4. straight and restarted runs agree at zero tolerance
for p in plt_cpl plt_fire_cpl; do
    if "$FCOMPARE" ${p}_straight_00040 ${p}_restart_00040 > cmp_${p}.log 2>&1; then
        echo "  coupled restart: ${p} at step 40, straight vs restarted: PASS"
    else
        echo "  coupled restart: ${p} at step 40, straight vs restarted differ: FAIL"; cat cmp_${p}.log; ok=no
    fi
done
[ "$ok" = yes ] || exit 1
