#!/bin/bash
# erf.fire.levelset.reinit_clamp, old against new: run the deck with each clamp,
# and with reinit_iters = 0 under each clamp, then compare the level sets.
#
#   [MPIRUN="mpirun -np 1"] [PYTHON=python3] ./run_reinit_clamp.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_reinit_clamp.sh x     # run nothing; check the existing output
#
# Four 24-step runs (5 s steps, 4 fire subcycles each), 1 to 2 s each on one
# rank. check_reinit_clamp.py prints the
# burned area and head position of each and the largest |phi| difference.
#
# If MPI_Init fails with MPICH's default OFI provider, export FI_PROVIDER=tcp.

set -u
EXE=${1:?usage: run_reinit_clamp.sh /path/to/erf_exec [extra args]}
shift || true
PY=${PYTHON:-python3}

run () {
    tag=$1; shift
    rm -rf "plt_fire_${tag}_"*
    ${MPIRUN:-} "$EXE" inputs erf.fire_plot_file="plt_fire_${tag}_" "$@" < /dev/null \
        > "run_${tag}.log" 2>&1 || { echo "run $tag failed (see run_${tag}.log)"; tail -n 30 "run_${tag}.log"; exit 1; }
}

if [ "${SKIP_RUN:-0}" != "1" ]; then
    run step_start           erf.fire.levelset.reinit_clamp=step_start "$@"
    run post_advection       erf.fire.levelset.reinit_clamp=post_advection "$@"
    run iters0_step_start    erf.fire.levelset.reinit_clamp=step_start     erf.fire.levelset.reinit_iters=0 "$@"
    run iters0_post_advection erf.fire.levelset.reinit_clamp=post_advection erf.fire.levelset.reinit_iters=0 "$@"
fi

$PY check_reinit_clamp.py
