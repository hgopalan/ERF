#!/bin/sh
# Coupled restart: the atmosphere and the fire after a restart equal a straight run.
#
#   [MPIRUN="mpirun -np 2"] sh run_coupled_restart.sh /path/to/erf_exec [extra erf args...]
#
# The lagged coupling injects, in each step, the fire heat fluxes of the step
# before: the sensible flux into theta and the latent flux into the water
# vapour. They are held by the coupling layer and saved in the checkpoint
# (FireQAtmPrev, FireQLatAtmPrev); if a restart did not read them back, the
# first restarted step would inject no fire heat or vapour and the atmosphere
# would drift from the straight run. run_restart.sh compares only the
# burned-cell count and the head rate of spread, which that first step barely
# changes.
#
# Four runs of inputs_coupled_straight with Kessler moisture (the latent flux is
# injected only with a moisture model) on nine boxes (MPIRUN, two ranks in CTest):
#   straight  steps 0-40
#   chk       steps 0-23, checkpoint chk00023 (23 is not a multiple of the
#             level-set reinitialisation interval)
#   restart   from chk00023 to step 40
#   nofeed    steps 0-40 with erf.fire.fire_atm_feedback = 0
# The atmosphere (theta and qv included) and fire plotfiles of step 40 of the
# straight and restarted runs must agree with amrex_fcompare at zero tolerance.
# The first restarted step must inject the same nonzero sensible flux as the
# straight run (the fire_debug line), so the checkpoint holds a burning fire.
# The nofeed run must differ from the straight one in theta and qv, so the fire
# heat and vapour reach the air in this window and the comparison can see a
# missing flux.

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
common="erf.moisture_model=Kessler amr.blocking_factor_x=4 amr.blocking_factor_y=4 amr.max_grid_size_x=8 amr.max_grid_size_y=8 stop_time=1000.0 erf.plot_int_1=40 erf.fire_plot_int=40 amrex.call_addr2line=0"
plotvars="erf.plot_vars_1=density x_velocity y_velocity z_velocity theta qv pressure"
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
# 1. the run is decomposed as the comment above says (8 + 8 + 4 cells in x and
#    in y): the lagged fluxes cross box edges in both directions
nbox=$(grep -c '^((' plt_cpl_straight_00040/Level_0/Cell_H)
if [ "$nbox" = 9 ]; then
    echo "  coupled restart: $nbox boxes: PASS"
else
    echo "  coupled restart: $nbox boxes, not the 9 the test is built for: FAIL"; ok=no
fi
# 2. the restart found the lagged fluxes in the checkpoint
if grep -q "no lagged fire fluxes" run_cpl_restart.log; then
    echo "  coupled restart: the checkpoint held no lagged fluxes: FAIL"; ok=no
else
    echo "  coupled restart: the lagged fluxes were in the checkpoint: PASS"
fi
# 3. the first restarted step injects the straight run's nonzero sensible flux
#    (the first max_Q_prev of step 24; a missing line, fire_debug off, fails)
q_first() { awk '/Coarse STEP 24 starts/ { f = 1 } f && /max_Q_prev=/ { sub(/.*max_Q_prev=/, ""); print $1; exit }' "$1"; }
q_str=$(q_first run_cpl_straight.log); q_rst=$(q_first run_cpl_restart.log)
if [ -n "$q_str" ] && [ "$q_str" = "$q_rst" ] && awk -v q="$q_str" 'BEGIN { exit !(q + 0 > 0) }'; then
    echo "  coupled restart: step 24 injects ${q_str} W/m2, straight and restarted: PASS"
else
    echo "  coupled restart: step 24 injects '${q_str}' W/m2 straight and '${q_rst}' restarted: FAIL"; ok=no
fi
# 4. the fire heat and vapour reach the air in this window
"$FCOMPARE" plt_cpl_straight_00040 plt_cpl_nofeed_00040 > cmp_nofeed.log 2>&1
for v in theta qv; do
    d=$(awk -v v=$v '$1 == v { print $2 }' cmp_nofeed.log)
    if awk -v d="${d:-0}" 'BEGIN { exit !(d + 0 > 0) }'; then
        echo "  coupled restart: the fire feedback changes $v by up to $d: PASS"
    else
        echo "  coupled restart: $v with and without fire feedback is the same, the test cannot see the flux: FAIL"; ok=no
    fi
done
# 5. straight and restarted runs agree at zero tolerance
for p in plt_cpl plt_fire_cpl; do
    if "$FCOMPARE" ${p}_straight_00040 ${p}_restart_00040 > cmp_${p}.log 2>&1; then
        echo "  coupled restart: ${p} at step 40, straight vs restarted: PASS"
    else
        echo "  coupled restart: ${p} at step 40, straight vs restarted differ: FAIL"; cat cmp_${p}.log; ok=no
    fi
done
[ "$ok" = yes ] || exit 1
