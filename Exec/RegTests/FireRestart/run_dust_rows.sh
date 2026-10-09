#!/bin/sh
# Dust output across checkpoints: every step's dust_diag.dat row is written once.
#
#   [MPIRUN="mpirun -np 1"] sh run_dust_rows.sh /path/to/erf_exec [extra erf args...]
#
# Three runs of the `dust` decks with dust plotfiles every 10 steps:
#   chk      steps 0-20, checkpoints chk00000 and chk00020, rows into dust_rows.dat
#   restart  from chk00020 to step 40, appending to dust_rows.dat
#   from0    from chk00000 to step 10, rows into dust_rows0.dat
#   straight steps 0-40 with no checkpoint, rows into dust_rows_straight.dat
# dust_rows.dat must hold steps 0..40 once each, and the dust plotfiles of
# steps 20 and 40 must be written once. Rows 21..40 of the restarted run must
# equal the straight run's as strings: the emission flux and friction
# velocity coarsened to the atmosphere columns are rebuilt from the restored
# fields after the checkpoint read, so the first restarted dycore injects and
# deposits what the uninterrupted one did (until October 2026 it injected
# nothing and deposited at the u* floor: deposition_total 4.533e-4 against
# 4.556e-4 at step 21, 1.671e-3 against 1.716e-3 at step 40). The last step of a run (time loop, then
# WriteAtFinalTime) and the step a restart starts on (the original run, then
# InitData) used to be written twice. dust_rows0.dat must start at step 1:
# chk00000 is written after the step-0 dust output, so it already counts step
# 0 as written.

set -u
EXE=${1:?usage: run_dust_rows.sh /path/to/erf_exec [extra args]}
shift || true

rm -rf chk00000 chk00020 chk00040 chk00010 chk00030 plt_dust_rows_* dust_rows.dat dust_rows0.dat dust_rows_straight.dat dust_rows_expo.dat \
       dust_naaqs_expo.csv dust_naaqs_straight.csv dust_rows_bins.dat run_dust_rows_*.log *_chk.csv *_chk.dat *_restart.csv *_restart.dat
common="erf.dust.dust_plot_int=10 erf.dust.dust_plot_prefix=plt_dust_rows_ erf.fire_plot_int=-1 erf.plot_int_1=-1"
run () {
    leg=$1; shift
    ${MPIRUN:-} "$EXE" "$@" $common > "run_dust_rows_$leg.log" 2>&1 \
        || { echo "  dust rows: the $leg run failed"; tail -20 "run_dust_rows_$leg.log"; exit 1; }
}
# a stale CSV from an earlier run is not continued by a fresh start (every
# writer appends, so its rows sat ahead of the new ones until October 2026)
printf '999,stale,row\n' > dust_naaqs_chk.csv
run chk     inputs_dust_chk     max_step=20 erf.check_int=20 erf.dust.dust_diag_file=dust_rows.dat
run restart inputs_dust_restart erf.restart=chk00020 max_step=40 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows.dat
run from0   inputs_dust_restart erf.restart=chk00000 max_step=10 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows0.dat
run straight inputs_dust_chk    max_step=40 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows_straight.dat erf.dust.dust_naaqs_file=dust_naaqs_straight.csv
# the emitted-mass shares do not depend on the averaging: the instantaneous
# PM columns of the exponential run equal the window run's (with the shares
# initialised only under window averaging, the exponential run put a third
# of the road dust into PM2.5: 6.73 against 0 ug/m3 at step 20)
run expo    inputs_dust_chk    max_step=20 erf.check_int=-1 erf.dust.averaging=exponential erf.dust.dust_diag_file=dust_rows_expo.dat erf.dust.dust_naaqs_file=dust_naaqs_expo.csv

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
awk -F, '/^[0-9]/ && $1 >= 21' dust_rows.dat          > rows_restart.txt
awk -F, '/^[0-9]/ && $1 >= 21' dust_rows_straight.dat > rows_straight.txt
if [ -s rows_restart.txt ] && cmp -s rows_restart.txt rows_straight.txt; then
    echo "  dust rows: rows 21..40 after the restart equal the straight run's: PASS"
else
    echo "  dust rows: rows 21..40 after the restart differ from the straight run's: FAIL"; ok=no
    diff rows_restart.txt rows_straight.txt | head -6
fi
rm -f rows_restart.txt rows_straight.txt
if grep -q "999,stale,row" dust_naaqs_chk.csv; then
    echo "  dust rows: the fresh start kept a stale dust_naaqs_chk.csv row from an earlier run: FAIL"; ok=no
else
    echo "  dust rows: the fresh start removed the stale dust_naaqs_chk.csv: PASS"
fi
pm_w=$(awk -F, '/^20,/ {print $3","$5}' dust_naaqs_chk.csv)
pm_e=$(awk -F, '/^20,/ {print $3","$5}' dust_naaqs_expo.csv)
if [ -n "$pm_w" ] && [ "$pm_w" = "$pm_e" ]; then
    echo "  dust rows: PM2.5/PM10 at step 20 equal under window and exponential averaging ($pm_w): PASS"
else
    echo "  dust rows: PM2.5/PM10 at step 20 differ between window ($pm_w) and exponential ($pm_e) averaging: FAIL"; ok=no
fi
# a restart whose deck changes the bin count aborts naming n_size_bins before
# VisMF::Read dies on the component count
${MPIRUN:-} "$EXE" inputs_dust_restart erf.restart=chk00020 max_step=21 erf.check_int=-1 \
    erf.dust.n_size_bins=2 erf.dust.bin_diameters="7.0e-6 2.5e-6" erf.dust.dust_diag_file=dust_rows_bins.dat $common > run_dust_rows_bins.log 2>&1
if grep -q "checkpoint was written with erf.dust.n_size_bins = 3" run_dust_rows_bins.log; then
    echo "  dust rows: a restart with another n_size_bins aborts naming the key: PASS"
else
    echo "  dust rows: a restart with another n_size_bins did not abort naming the key: FAIL"; ok=no; tail -5 run_dust_rows_bins.log
fi
rm -rf dust_rows_bins.dat
# a restart from a checkpoint that is not the last one drops the rows written
# after it: restarting from chk00020 to step 30 into the file that already
# holds steps 0..40 must leave steps 0..30 once each (0..40 then 21..30 before)
${MPIRUN:-} "$EXE" inputs_dust_restart erf.restart=chk00020 max_step=30 erf.check_int=-1 erf.dust.dust_diag_file=dust_rows.dat $common > run_dust_rows_trim.log 2>&1 \
    || { echo "  dust rows: the trim run failed"; tail -20 run_dust_rows_trim.log; exit 1; }
steps=$(awk -F, '/^[0-9]/ {print $1}' dust_rows.dat | tr '\n' ' ')
expect=$(awk 'BEGIN { for (n = 0; n <= 30; n++) printf "%d ", n }')
if [ "$steps" = "$expect" ]; then
    echo "  dust rows: a restart from an earlier checkpoint drops the rows past it (steps 0..30 once each): PASS"
else
    echo "  dust rows: after the restart from chk00020 to step 30 dust_rows.dat holds [$steps], expected 0..30 once each: FAIL"; ok=no
fi
[ "$ok" = yes ]
