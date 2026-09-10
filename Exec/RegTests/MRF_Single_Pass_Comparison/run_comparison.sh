#!/bin/bash
# Old vs new MRF: run the MRF canonical decks with two erf_exec binaries and
# tabulate PBLH, Kmv and theta.
#
#   [MPIRUN="mpirun -np 1"] [OUT=dir] ./run_comparison.sh OLD_EXE NEW_EXE [case ...]
#   SKIP_RUN=1 ./run_comparison.sh OLD_EXE NEW_EXE      # only print the tables
#
# OLD_EXE is a build of ERF-Hazard before the MRF passes were deduplicated
# (5764e850b or earlier), NEW_EXE a build with the single set of passes. The
# decks are read from Exec/CanonicalTests, not copied; every case overrides only
# the time step (fixed, so both binaries write at the same times) and the output.
#
# Cases:
#   mrf_unstable             ABL/mrf_unstable, 9 h
#   mrf_unstable_enhanced    the same with countergradient, VH96 and PBLH smoothing on
#   fire_mrf_unstable        Fire/.../ABL_with_MRF/inputs_fire_abl_mrf_unstable, 512 s
#   fire_mrf_unstable_boost  the same with erf.pbl_mrf_fire_thermal_excess = true
#
# The old and new run of each case go in parallel; the cases run one after the other.

set -u

OLD=${1:?usage: [MPIRUN=...] run_comparison.sh OLD_EXE NEW_EXE [case ...]}
NEW=${2:?usage: [MPIRUN=...] run_comparison.sh OLD_EXE NEW_EXE [case ...]}
shift 2
CASES=${*:-mrf_unstable mrf_unstable_enhanced fire_mrf_unstable fire_mrf_unstable_boost}

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
ABL=$ROOT/Exec/CanonicalTests/ABL
FIRE=$ROOT/Exec/CanonicalTests/Fire/Atmospheric_Boundary_Layer/ABL_with_MRF
OUT=${OUT:-$PWD/mrf_single_pass_runs}
MPIRUN=${MPIRUN:-mpirun -np 1}

# erf.most.pblh_calc=MRF only makes the 2D plotfile report the PBLH that MRF
# writes to SurfaceLayer (otherwise -999); nothing else reads it in these decks.
PLOT="erf.check_int=-1 erf.plot_file_1=plt erf.plot2d_file_1=plt2d erf.most.pblh_calc=MRF"
VARS3D="density theta Kmv Khv Lturb"
VARS2D="pblh u_star"

run_one () {
    local label=$1 exe=$2 case=$3 dir deck sounding every
    local -a extra
    case $case in
        mrf_unstable|mrf_unstable_enhanced)
            deck=$ABL/mrf_unstable; sounding=$ABL/mrf_sounding_unstable; every=1200
            extra=(stop_time=9000 erf.fixed_dt=1.5)
            if [ "$case" = mrf_unstable_enhanced ]; then
                extra+=(erf.enable_mrf_countergradient=true
                        erf.enable_vh96_shear_correction=true
                        erf.enable_pblh_smoothing=true)
            fi ;;
        fire_mrf_unstable|fire_mrf_unstable_boost)
            deck=$FIRE/inputs_fire_abl_mrf_unstable; sounding=$FIRE/mrf_sounding_unstable; every=400
            extra=(stop_time=512 erf.fixed_dt=0.32 erf.fire_plot_int=-1)
            if [ "$case" = fire_mrf_unstable_boost ]; then
                extra+=(erf.pbl_mrf_fire_thermal_excess=true)
            fi ;;
        *) echo "unknown case $case" >&2; return 1 ;;
    esac
    dir=$OUT/$case/$label
    rm -rf "$dir" && mkdir -p "$dir" && cp "$sounding" "$dir/"
    # shellcheck disable=SC2086
    (cd "$dir" && $MPIRUN "$exe" "$deck" "${extra[@]}" $PLOT \
        erf.plot_int_1=$every erf.plot2d_int_1=$every erf.profile_int=$every \
        "erf.plot_vars_1=$VARS3D" "erf.plot2d_vars_1=$VARS2D" > run.log 2>&1)
    echo "$case/$label exit $?"
}

if [ -z "${SKIP_RUN:-}" ]; then
    for c in $CASES; do
        for label in ${LABELS:-old new}; do
            if [ "$label" = old ]; then exe=$OLD; else exe=$NEW; fi
            run_one "$label" "$exe" "$c" &
        done
        wait
    done
fi

for c in $CASES; do
    python3 "$HERE/compare_mrf.py" "$OUT/$c/old" "$OUT/$c/new" --label "$c"
done
