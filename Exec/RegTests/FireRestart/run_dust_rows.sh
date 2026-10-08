#!/bin/sh
# Dust output across checkpoints: every step's dust_diag.dat row is written once.
#
#   [MPIRUN="mpirun -np 1"] sh run_dust_rows.sh /path/to/erf_exec [extra erf args...]
#
# Three runs of the `dust` decks with dust plotfiles every 10 steps:
#   chk      steps 0-20, checkpoints chk00000 and chk00020, rows into dust_rows.dat
#   restart  from chk00020 to step 40, appending to dust_rows.dat
#   from0    from chk00000 to step 10, rows into dust_rows0.dat
# dust_rows.dat must hold steps 0..40 once each, and the dust plotfiles of
# steps 20 and 40 must be written once. The last step of a run (time loop, then
# WriteAtFinalTime) and the step a restart starts on (the original run, then
# InitData) used to be written twice. dust_rows0.dat must start at step 1:
# chk00000 is written after the step-0 dust output, so it already counts step
# 0 as written.

set -u
EXE=${1:?usage: run_dust_rows.sh /path/to/erf_exec [extra args]}
shift || true

rm -rf chk00000 chk00020 chk00040 chk00010 plt_dust_rows_* dust_rows.dat dust_rows0.dat
common="erf.dust.dust_plot_int=10 erf.dust.dust_plot_prefix=plt_dust_rows_ erf.fire_plot_int=-1 erf.plot_int_1=-1"
run () {
    leg=$1; shift
    ${MPIRUN:-} "$EXE" "$@" $common > "run_dust_rows_$leg.log" 2>&1 \
        || { echo "  dust rows: the $leg run failed"; tail -20 "run_dust_rows_$leg.log"; exit 1; }
}
run chk     inputs_dust_chk     max_step=20 erf.check_int=20 erf.dust.dust_diag_file=dust_rows.dat
run restart inputs_dust_restart erf.restart=chk00020 max_step=40 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows.dat
run from0   inputs_dust_restart erf.restart=chk00000 max_step=10 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows0.dat

ok=yes
steps=$(awk -F, '/^[0-9]/ {print $1}' dust_rows.dat | tr '\n' ' ')
expect=$(awk 'BEGIN { for (n = 0; n <= 40; n++) printf "%d ", n }')
if [ "$steps" = "$expect" ]; then
    echo "  dust rows: dust_rows.dat holds steps 0..40 once each across the restart: PASS"
else
    echo "  dust rows: dust_rows.dat steps are [$steps], expected 0..40 once each: FAIL"; ok=no
fi
for s in 00020 00040; do
    n=$(cat run_dust_rows_chk.log run_dust_rows_restart.log | grep -c "Writing dust plotfile plt_dust_rows_$s")
    if [ "$n" = "1" ]; then
        echo "  dust rows: plt_dust_rows_$s written once: PASS"
    else
        echo "  dust rows: plt_dust_rows_$s written $n times, expected once: FAIL"; ok=no
    fi
done
first=$(awk -F, '/^[0-9]/ {print $1; exit}' dust_rows0.dat)
if [ "$first" = "1" ]; then
    echo "  dust rows: the restart from chk00000 starts its rows at step 1: PASS"
else
    echo "  dust rows: the restart from chk00000 starts its rows at step '$first', expected 1: FAIL"; ok=no
fi
[ "$ok" = yes ]
