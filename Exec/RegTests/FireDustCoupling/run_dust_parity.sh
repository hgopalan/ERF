#!/bin/sh
# Dust decomposition parity: the fire-dust coupling deck on one rank and one box
# against two ranks and four boxes must give the same dust fields and the same
# dust_diag.dat.
#
#   [MPIRUN="mpirun -np 2"] sh run_dust_parity.sh /path/to/erf_exec [extra erf args...]
#
# MPIRUN is the two-rank launcher; the one-rank leg strips the rank count by
# running the executable directly (a single rank needs no launcher). Both legs
# use a 24 x 24 x 30 domain (every box edge divides by the grid ratio 4): the
# "one" leg with amr.max_grid_size = 24 (one box), the "four" leg with
# amr.max_grid_size_x/_y = 12 (four boxes on two ranks). The runs cover the
# paths that only exist across box edges and ranks: the ParallelCopy of the
# fire wind, heat and level set onto the dust BoxArray with the dust
# periodicity, the average_down of the emission flux and friction velocity,
# the wind extraction at a box's high face, the per-site and receptor
# reductions, and dust_fill_boundary at interior box edges. Until October 2026
# every dust CTest ran on one rank and one box.
#
# The check asserts that the two legs really differ in their decomposition
# (Level_0/Cell_H box counts of the dust plotfile), then compares the last dust
# plotfile with amrex_fcompare at a tight tolerance and every numeric column of
# dust_diag.dat row by row.

set -u
EXE=${1:?usage: run_dust_parity.sh /path/to/erf_exec [extra args]}
shift || true
PY=${PYTHON:-python3}

# amrex_fcompare sits next to the ERF build's AMReX tools
FCOMPARE=$(dirname "$EXE")/../Submodules/AMReX/Tools/Plotfile/amrex_fcompare
[ -x "$FCOMPARE" ] || FCOMPARE=$(dirname "$EXE")/../Submodules/AMReX/Tools/Plotfile/fcompare
if [ ! -x "$FCOMPARE" ]; then
    echo "  dust parity: amrex_fcompare not found next to $EXE: FAIL"; exit 1
fi

rm -rf parity_one parity_four
mkdir -p parity_one parity_four
for f in inputs input_sounding; do cp "$f" parity_one/; cp "$f" parity_four/; done

common="amr.n_cell=24 24 30 max_step=20 erf.dust.dust_plot_int=20 erf.dust.dust_plot_prefix=plt_par_ erf.fire_plot_int=-1 erf.plot_int_1=-1 erf.check_int=-1 amrex.call_addr2line=0"
# the receptor and site reductions are among the paths under test
extras="erf.dust.msha_receptor_names=r1 erf.dust.msha_receptor_x=900.0 erf.dust.msha_receptor_y=1100.0 erf.dust.site_names=pit erf.dust.site_x_lo=400.0 erf.dust.site_y_lo=600.0 erf.dust.site_x_hi=1400.0 erf.dust.site_y_hi=1600.0 erf.dust.cm_fractions=0.002"

( cd parity_one  && "$EXE" inputs $common $extras amr.max_grid_size=24 "$@" > run.log 2>&1 ) \
    || { echo "  dust parity: the one-box run failed"; tail -20 parity_one/run.log; exit 1; }
( cd parity_four && ${MPIRUN:-mpirun -np 2} "$EXE" inputs $common $extras amr.max_grid_size_x=12 amr.max_grid_size_y=12 "$@" > run.log 2>&1 ) \
    || { echo "  dust parity: the four-box run failed"; tail -20 parity_four/run.log; exit 1; }

ok=yes
# 1. the decompositions differ (a parity test that compares a run with itself proves nothing)
n_one=$(grep -c '^((' parity_one/plt_par_00020/Level_0/Cell_H)
n_four=$(grep -c '^((' parity_four/plt_par_00020/Level_0/Cell_H)
if [ "$n_one" = "1" ] && [ "$n_four" = "4" ]; then
    echo "  dust parity: dust plotfile boxes one=$n_one four=$n_four: PASS"
else
    echo "  dust parity: dust plotfile boxes one=$n_one four=$n_four, expected 1 and 4: FAIL"; ok=no
fi

# 2. the dust plotfiles agree (fcompare -a: allow the different BoxArrays)
if "$FCOMPARE" -a -r 1.0e-10 --abs_tol 1.0e-14 parity_one/plt_par_00020 parity_four/plt_par_00020 > fcompare_dust.log 2>&1; then
    echo "  dust parity: dust plotfile step 20 agrees to 1e-10: PASS"
else
    echo "  dust parity: dust plotfile step 20 differs: FAIL"; tail -30 fcompare_dust.log; ok=no
fi

# 3. dust_diag.dat agrees column by column
if $PY - parity_one/dust_diag.dat parity_four/dust_diag.dat <<'EOF'
import sys
def rows(p):
    out = []
    for l in open(p):
        s = l.strip()
        if not s or s.startswith("#") or s.startswith("step"):
            continue
        out.append([float(x) for x in s.split(",")])
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
if len(a) != len(b) or not a:
    print(f"  dust parity: dust_diag.dat rows one={len(a)} four={len(b)}: FAIL"); sys.exit(1)
worst = 0.0
for ra, rb in zip(a, b):
    for x, y in zip(ra, rb):
        scale = max(abs(x), abs(y), 1e-300)
        worst = max(worst, abs(x - y) / scale)
print(f"  dust parity: dust_diag.dat {len(a)} rows, worst relative difference {worst:.2e}: " + ("PASS" if worst < 1e-10 else "FAIL"))
sys.exit(0 if worst < 1e-10 else 1)
EOF
then :; else ok=no; fi

# 4. the per-site CM budget and the receptor sample agree
for f in dust_cm_budget.csv msha_receptor_r1.csv; do
    if $PY - parity_one/$f parity_four/$f <<'EOF'
import sys
def rows(p):
    out = []
    for l in open(p):
        s = l.strip()
        if not s or s.startswith("#") or s[0].isalpha():
            continue
        out.append([c for c in s.split(",")])
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
name = sys.argv[1].split("/")[-1]
if len(a) != len(b) or not a:
    print(f"  dust parity: {name} rows one={len(a)} four={len(b)}: FAIL"); sys.exit(1)
worst = 0.0
for ra, rb in zip(a, b):
    for x, y in zip(ra, rb):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            if x != y:
                print(f"  dust parity: {name} label {x} vs {y}: FAIL"); sys.exit(1)
            continue
        worst = max(worst, abs(fx - fy) / max(abs(fx), abs(fy), 1e-300))
print(f"  dust parity: {name} {len(a)} rows, worst relative difference {worst:.2e}: " + ("PASS" if worst < 1e-10 else "FAIL"))
sys.exit(0 if worst < 1e-10 else 1)
EOF
    then :; else ok=no; fi
done

[ "$ok" = yes ] && { echo "  dust parity: all checks passed"; exit 0; }
echo "  dust parity: FAILED"; exit 1
