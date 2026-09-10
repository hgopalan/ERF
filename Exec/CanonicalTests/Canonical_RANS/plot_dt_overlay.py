#!/usr/bin/env python3
"""Overlay planar-averaged profiles from several runs of the same case.

Unlike the ``check_*.py`` scripts, which are standard library only because
they gate the CTest entries, this is an optional figure tool and needs
matplotlib. It reads the plotfiles with ``erf_plotfile.py`` and draws one
column per field: the profiles on top, the difference from the first run
below, so that curves which lie on top of each other can still be told
apart at the 1e-3 level.

Usage:

    python3 plot_dt_overlay.py --out overlay.png \
        "anelastic dt 5 s=runA/plt08640" \
        "anelastic dt 60 s=runB/plt00720" \
        "compressible dt 60 s=runC/plt00720"

A label containing "compressible" is drawn dashed, everything else solid;
the colour follows the order the runs are given.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erf_plotfile  # noqa: E402

FIELDS = [
    ("x_velocity", "u [m/s]"),
    ("y_velocity", "v [m/s]"),
    ("theta", r"$\theta$ [K]"),
    ("KE", r"k [m$^2$/s$^2$]"),
]


def parse_runs(specs):
    runs = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit("run spec needs the form 'label=path/to/plotfile': %s" % spec)
        label, path = spec.split("=", 1)
        runs.append((label.strip(), path.strip()))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="LABEL=PLOTFILE")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--zmax", type=float, default=None,
                    help="top of the plotted range [m] (default: whole column)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = parse_runs(args.runs)
    names = [f for f, _ in FIELDS]
    data = []
    for label, path in runs:
        z, avg, _ = erf_plotfile.planar_averages(path, names)
        data.append((label, z, avg))

    zref = data[0][1]
    for label, z, _ in data[1:]:
        if len(z) != len(zref):
            raise SystemExit("runs have different vertical grids; cannot difference them")

    kmax = len(zref)
    if args.zmax is not None:
        kmax = sum(1 for zz in zref if zz <= args.zmax)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, len(FIELDS), figsize=(4.0 * len(FIELDS), 8.0),
                             sharey=True)

    for col, (field, xlabel) in enumerate(FIELDS):
        top, bot = axes[0][col], axes[1][col]
        base = data[0][2][field]
        for i, (label, z, avg) in enumerate(data):
            style = "--" if "compressible" in label.lower() else "-"
            kw = dict(color=colors[i % len(colors)], linestyle=style,
                      linewidth=2.4 if i == 0 else 1.3, alpha=0.9)
            top.plot(avg[field][:kmax], z[:kmax], label=label, **kw)
            if i:
                bot.plot([a - b for a, b in zip(avg[field][:kmax], base[:kmax])],
                         z[:kmax], label=label, **kw)
        top.set_xlabel(xlabel)
        bot.set_xlabel("difference from %s" % data[0][0], fontsize=9)
        bot.axvline(0.0, color="0.6", linewidth=0.8)
        top.grid(alpha=0.3)
        bot.grid(alpha=0.3)
        for ax in (top, bot):
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        bot.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        bot.tick_params(axis="x", labelsize=8)
        if col == 0:
            top.set_ylabel("z [m]")
            bot.set_ylabel("z [m]")

    axes[0][0].legend(fontsize=8, loc="best")
    if args.title:
        fig.suptitle(args.title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("wrote %s" % args.out)

    print("max |difference| from %s over the plotted range:" % data[0][0])
    for label, _, avg in data[1:]:
        cells = []
        for field, _lbl in FIELDS:
            base = data[0][2][field]
            cells.append("%s %.2e" % (field, max(abs(a - b) for a, b in
                                                 zip(avg[field][:kmax], base[:kmax]))))
        print("  %-24s %s" % (label, "  ".join(cells)))


if __name__ == "__main__":
    main()
