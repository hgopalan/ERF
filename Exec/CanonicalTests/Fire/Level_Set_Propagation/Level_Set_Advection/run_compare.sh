#!/bin/bash
# Run the first-order, WENO5-Z and hybrid WENO5-Z/first-order level-set decks and print the burned-cell
# counts at 150 / 300 / 450 / 600 s next to the analytic front r = r0 + R t.
#
#   ./run_compare.sh /path/to/erf_exec [extra erf args...]
set -u
EXE=${1:?usage: run_compare.sh /path/to/erf_exec [extra args]}
shift || true
for d in baseline weno5z weno5z_front; do
    "$EXE" inputs_fire_levelset_$d erf.fire_plot_int=50 "$@" > run_$d.log 2>&1
    echo "$d: exit $?"
done
python3 - <<'PY'
import glob, numpy as np, yt
yt.set_log_level(50)
def cells(pf):
    ds = yt.load(pf); g = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return int((np.array(g[("boxlib", "fire_phi")]) < 0).sum()), float(ds.current_time)
print(f"{'':13s} " + " ".join(f"{t:6d} s" for t in (150, 300, 450, 600)))
print(f"{'analytic':13s} " + " ".join(f"{c:8d}" for c in (34, 41, 48, 56)))
for d in ("baseline", "weno5z", "weno5z_front"):
    row = []
    for step in (150, 300, 450, 600):
        pfs = glob.glob(f"plt_fire_levelset_{d}_{step:05d}")
        row.append(cells(pfs[0])[0] if pfs else -1)
    print(f"{d:13s} " + " ".join(f"{c:8d}" for c in row))
PY
