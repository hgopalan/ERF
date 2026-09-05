#!/usr/bin/env python3
"""Reference wind on the fire grid under the sampling options.

    python3 plot_wind.py [--out wind.png] [--dt 0.125]

Reads run_off.log, run_sample20.log and run_ref20.log written by run_wind.sh
and draws the largest reference wind against time, with the sampled-to-direct
ratio and the log-law factor in the lower panel.
"""
import argparse, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def series(log):
    out = []
    with open(log) as f:
        for line in f:
            m = re.search(r"Max reference wind: ([^ ]+) m", line)
            if m: out.append(float(m.group(1)))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="wind.png"); ap.add_argument("--dt", type=float, default=0.125)
    a = ap.parse_args()
    S = {k: series(f"run_{k}.log") for k in ("off", "sample20", "ref20")}
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for k, lab in (("off", "at 6.1 m (historical)"), ("ref20", "at 20 m directly"), ("sample20", "sampled at 20 m, log law to 6.1 m")):
        axs[0].plot(np.arange(len(S[k])) * a.dt, S[k], label=lab)
    axs[0].set_ylabel("max reference wind [m/s]"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    n = min(len(S["ref20"]), len(S["sample20"])); ok = S["ref20"][:n] > 0
    axs[1].plot(np.arange(n)[ok] * a.dt, S["sample20"][:n][ok] / S["ref20"][:n][ok], label="sampled / direct")
    f = np.log(61.0) / np.log(200.0)
    axs[1].axhline(f, color="k", ls=":", label="ln(6.1/0.1)/ln(20/0.1)")
    axs[1].set_ylim(f - 0.02, f + 0.02)   # the ratio holds to 1e-10; keep the axis readable
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("ratio"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
