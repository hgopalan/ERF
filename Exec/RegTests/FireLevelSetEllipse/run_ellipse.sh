#!/bin/bash
# Run the spread-shape variants and check them.
#
#   [MPIRUN="mpirun -np 4"] ./run_ellipse.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_ellipse.sh x     # only rerun the checks on existing output
#
# Checks: the deck with the key written out reproduces the historical deck
# line for line; the burned region of each ellipse deck is the Huygens
# envelope of the 6 m ignition disc, its back and half-width travel predicted
# from the measured head travel with the ellipse's ratios (fixed 3, or the
# time-mean Anderson ratio the code printed) to one fire cell (1.25 m); the
# head extents of the disc and the ellipse decks agree to 10 % (the ellipse
# leaves the head rate alone); the ellipse's back extent is below the disc's.
# The directional deck is tabulated for comparison. The bounding-box ratio
# stays well below L/W in 60 s because the ignition disc dominates a 13 m
# head run, which is why the envelope, not the ratio, is checked.

set -u
EXE=${1:?usage: run_ellipse.sh /path/to/erf_exec [extra args]}
shift || true
VARIANTS="off off_key directional ellipse ellipse_lw3"
X0=60.0; Y0=80.0; R0=6.0; TOL_M=1.25

for v in $VARIANTS; do
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "run_$v.log" ]; then continue; fi
    rm -rf plt_fire_$v; mkdir -p plt_fire_$v
    ${MPIRUN:-} "$EXE" "inputs_$v" erf.fire_plot_file=plt_fire_$v/plt_fire_ "$@" > "run_$v.log" 2>&1 || { echo "run $v failed (see run_$v.log)"; exit 1; }
done

cells() { grep 'active fire cells' "run_$1.log" | awk '{print $NF}'; }
lbmean() { grep 'Spread ellipse' "run_$1.log" | sed 's/.*LB=\([^ ]*\) .*/\1/' | awk '{ s += $1; n++ } END { if (n) printf "%.4f", s / n; else print "-" }'; }
shape() { python3 check_shape.py plt_fire_$1/plt_fire_00480 $X0 $Y0 2>&1 | grep -v "^yt" | head -1; }

printf "%-14s %7s %8s  %s\n" variant cells LB_mean "shape at 60 s"
printf "%-14s %7s %8s  %s\n" -------------- ------- --------  ------------------------------------------------------------
for v in $VARIANTS; do
    printf "%-14s %7s %8s  %s\n" "$v" "$(cells $v | tail -1)" "$(lbmean $v)" "$(shape $v)"
done
echo

status=0
check() { if [ "$2" = "1" ]; then echo "  $1: PASS"; else echo "  $1: FAIL"; status=1; fi; }

same=$(diff <(cells off) <(cells off_key) > /dev/null && echo 1 || echo 0)
check "key written out reproduces the historical deck ($(cells off | wc -l | tr -d ' ') steps)" "$same"

python3 check_shape.py plt_fire_ellipse_lw3/plt_fire_00480 $X0 $Y0 $R0 3.0 $TOL_M 2>&1 | grep -v "^yt" | tail -1 | sed 's/^/  fixed ratio 3:/' ; [ ${PIPESTATUS[0]} -eq 0 ] || status=1
python3 check_shape.py plt_fire_ellipse/plt_fire_00480 $X0 $Y0 $R0 $(lbmean ellipse) $TOL_M 2>&1 | grep -v "^yt" | tail -1 | sed 's/^/  Anderson:/' ; [ ${PIPESTATUS[0]} -eq 0 ] || status=1

head_of() { shape $1 | sed 's/.*head \([^ ]*\) m.*/\1/'; }
back_of() { shape $1 | sed 's/.*back \([^ ]*\) m.*/\1/'; }
r=$(awk -v a="$(head_of off)" -v b="$(head_of ellipse)" 'BEGIN { d = (b - a) / a; if (d < 0) d = -d; print (d <= 0.10) ? 1 : 0 }')
check "head extent of the ellipse deck matches the disc deck to 10 % ($(head_of ellipse) vs $(head_of off) m)" "$r"
r=$(awk -v a="$(back_of off)" -v b="$(back_of ellipse)" 'BEGIN { print (b < a) ? 1 : 0 }')
check "back extent of the ellipse deck below the disc deck ($(back_of ellipse) vs $(back_of off) m)" "$r"

[ $status -eq 0 ] && echo "ALL PASS" || echo "SOME CHECKS FAILED"
exit $status
