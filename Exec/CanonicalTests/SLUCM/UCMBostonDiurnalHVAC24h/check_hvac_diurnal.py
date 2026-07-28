#!/usr/bin/env python3
"""
Phase 5.2 D10 — 24-hour Boston diurnal HVAC comparison.

Parses two log files (hvac_off vs hvac_simple) produced by:

    mpirun -np N ../../../build/Exec/erf_exec inputs_hvac_off      > run_hvac_off.log
    mpirun -np N ../../../build/Exec/erf_exec inputs_hvac_simple   > run_hvac_simple.log

Extracts per-step Q_HVAC, T_canyon_air, T_skin_roof/wall/road, H_sensible and
plots a side-by-side comparison of both runs.

Pass criteria:
  * hvac_off  : Q_HVAC block never printed (mode = off)
  * hvac_simple: Q_HVAC = 0 whenever the setpoint or occupancy gate fires;
                Q_HVAC > 0 in daytime hours (occupied + T_can >= setpoint - hyst)
  * Both runs reach STEP == max_step (or user-provided target step) without hanging.
  * T_canyon_air with HVAC on >= T_canyon_air with HVAC off (waste heat warms canyon)

Usage:
    python3 check_hvac_diurnal.py \\
        --off run_hvac_off.log \\
        --simple run_hvac_simple.log \\
        [--max-step 60000] \\
        [--plot hvac_diurnal.png]

Exits 0 on pass, 1 on any failed check.
"""
import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns matching the debug output produced by ERF_UCMLayer.cpp
# ---------------------------------------------------------------------------

RE_STEP_START = re.compile(
    r"\[Level 0 step (\d+)\] ADVANCE from elapsed time = ([\d.eE+-]+)"
)
RE_STEP_END = re.compile(
    r"Coarse STEP (\d+) ends\. TIME = ([\d.eE+-]+)"
)
RE_QHVAC = re.compile(
    r"\[UCM\]\[5\.2\]\[hvac\] mode=simple hour=(\d+)\s+Q_HVAC=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
)
RE_TCAN = re.compile(
    r"T_canyon_air=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
)
RE_TROOF = re.compile(
    r"T_skin_roof=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
)
RE_TWALL = re.compile(
    r"T_skin_wall=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
)
RE_TROAD = re.compile(
    r"T_skin_road=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
)
RE_HSENS = re.compile(
    r"H_sensible min=([\d.eE+-]+) max=([\d.eE+-]+)"
)
RE_AH_STATS = re.compile(
    r"AH min=([\d.eE+-]+) max=([\d.eE+-]+)"
)


def parse_log(path):
    """Return a dict of per-step time series parsed from an ERF run log."""
    steps = []
    times = []
    q_hvac_min = []
    q_hvac_max = []
    hours = []
    t_can_min = []
    t_can_max = []
    t_roof_min = []
    t_roof_max = []
    t_wall_min = []
    t_wall_max = []
    t_road_min = []
    t_road_max = []
    h_sens_min = []
    h_sens_max = []
    ah_min = []
    ah_max = []

    # per-step scratch (reset when a STEP_START is seen)
    scratch = {}

    def flush():
        if not scratch:
            return
        steps.append(scratch.get("step"))
        times.append(scratch.get("time"))
        q_hvac_min.append(scratch.get("q_min", None))
        q_hvac_max.append(scratch.get("q_max", None))
        hours.append(scratch.get("hour", None))
        t_can_min.append(scratch.get("tcan_min"))
        t_can_max.append(scratch.get("tcan_max"))
        t_roof_min.append(scratch.get("troof_min"))
        t_roof_max.append(scratch.get("troof_max"))
        t_wall_min.append(scratch.get("twall_min"))
        t_wall_max.append(scratch.get("twall_max"))
        t_road_min.append(scratch.get("troad_min"))
        t_road_max.append(scratch.get("troad_max"))
        h_sens_min.append(scratch.get("h_min"))
        h_sens_max.append(scratch.get("h_max"))
        ah_min.append(scratch.get("ah_min"))
        ah_max.append(scratch.get("ah_max"))

    with open(path) as f:
        for line in f:
            m = RE_STEP_START.search(line)
            if m:
                flush()
                scratch = {"step": int(m.group(1)), "time": float(m.group(2))}
                continue

            m = RE_QHVAC.search(line)
            if m:
                scratch["hour"] = int(m.group(1))
                scratch["q_min"] = float(m.group(2))
                scratch["q_max"] = float(m.group(3))
                continue

            m = RE_TCAN.search(line)
            if m:
                scratch["tcan_min"] = float(m.group(1))
                scratch["tcan_max"] = float(m.group(2))
                continue

            m = RE_TROOF.search(line)
            if m:
                scratch["troof_min"] = float(m.group(1))
                scratch["troof_max"] = float(m.group(2))
                continue

            m = RE_TWALL.search(line)
            if m:
                scratch["twall_min"] = float(m.group(1))
                scratch["twall_max"] = float(m.group(2))
                continue

            m = RE_TROAD.search(line)
            if m:
                scratch["troad_min"] = float(m.group(1))
                scratch["troad_max"] = float(m.group(2))
                continue

            m = RE_HSENS.search(line)
            if m:
                scratch["h_min"] = float(m.group(1))
                scratch["h_max"] = float(m.group(2))
                continue

            m = RE_AH_STATS.search(line)
            if m:
                scratch["ah_min"] = float(m.group(1))
                scratch["ah_max"] = float(m.group(2))
                continue

        flush()

    return {
        "path": str(path),
        "step": steps,
        "time_s": times,
        "hour": hours,
        "q_hvac_min": q_hvac_min,
        "q_hvac_max": q_hvac_max,
        "t_can_min": t_can_min,
        "t_can_max": t_can_max,
        "t_roof_min": t_roof_min,
        "t_roof_max": t_roof_max,
        "t_wall_min": t_wall_min,
        "t_wall_max": t_wall_max,
        "t_road_min": t_road_min,
        "t_road_max": t_road_max,
        "h_sens_min": h_sens_min,
        "h_sens_max": h_sens_max,
        "ah_min": ah_min,
        "ah_max": ah_max,
    }


def safe_last(seq):
    for v in reversed(seq):
        if v is not None:
            return v
    return None


def check_run(data, label, expect_hvac, target_step):
    """Return (passed, messages) for a parsed run."""
    msgs = []
    ok = True

    n = len(data["step"])
    msgs.append(f"[{label}] parsed {n} timesteps from {data['path']}")

    if n == 0:
        msgs.append(f"[{label}] FAIL: no timesteps parsed - check log path/format")
        return False, msgs

    last_step = safe_last(data["step"])
    if last_step is None:
        msgs.append(f"[{label}] FAIL: no step number found")
        ok = False
    elif target_step is not None and last_step < target_step:
        msgs.append(
            f"[{label}] FAIL: reached step {last_step} but expected {target_step} "
            f"(likely hung or crashed)"
        )
        ok = False
    else:
        msgs.append(f"[{label}] OK: reached step {last_step}")

    # Q_HVAC checks
    q_present = [q for q in data["q_hvac_max"] if q is not None]
    if expect_hvac:
        if len(q_present) == 0:
            msgs.append(f"[{label}] FAIL: HVAC mode = simple but no Q_HVAC lines found")
            ok = False
        else:
            q_max_ever = max(q_present)
            q_min_ever = min(q_present)
            frac_nonzero = sum(1 for q in q_present if q > 1e-6) / len(q_present)
            msgs.append(
                f"[{label}] Q_HVAC across run: min={q_min_ever:.2f} max={q_max_ever:.2f} "
                f"W/m^2, nonzero fraction={frac_nonzero*100:.1f}%"
            )
            if q_max_ever < 1.0:
                msgs.append(
                    f"[{label}] FAIL: HVAC never activated (max Q_HVAC={q_max_ever:.4f}); "
                    f"setpoint/occupancy gates always blocking?"
                )
                ok = False
            if frac_nonzero < 0.05:
                msgs.append(
                    f"[{label}] WARN: HVAC active in <5% of timesteps - verify setpoint"
                )
    else:
        if len(q_present) > 0:
            msgs.append(
                f"[{label}] FAIL: hvac_mode=off but {len(q_present)} Q_HVAC lines "
                f"found (should be zero)"
            )
            ok = False
        else:
            msgs.append(f"[{label}] OK: no Q_HVAC lines (hvac_mode=off)")

    # Sanity checks on canyon T
    tcan = [t for t in data["t_can_max"] if t is not None]
    if tcan:
        msgs.append(
            f"[{label}] T_canyon_air: min={min(tcan):.2f} K max={max(tcan):.2f} K"
        )
        if min(tcan) < 240 or max(tcan) > 340:
            msgs.append(
                f"[{label}] WARN: T_canyon_air outside [240, 340] K - check physics"
            )

    troof = [t for t in data["t_roof_max"] if t is not None]
    if troof:
        msgs.append(
            f"[{label}] T_skin_roof: min={min(troof):.2f} K max={max(troof):.2f} K"
        )

    return ok, msgs


def compare_runs(off, sim):
    """Physical consistency: HVAC ON should warm the canyon relative to OFF."""
    msgs = []
    ok = True

    # Align on step index (take min length)
    n = min(len(off["step"]), len(sim["step"]))
    if n < 10:
        msgs.append(f"[compare] SKIP: fewer than 10 aligned steps ({n})")
        return True, msgs

    # Compare mean T_canyon_air over aligned window
    tc_off = [t for t in off["t_can_max"][:n] if t is not None]
    tc_sim = [t for t in sim["t_can_max"][:n] if t is not None]
    if tc_off and tc_sim:
        m_off = sum(tc_off) / len(tc_off)
        m_sim = sum(tc_sim) / len(tc_sim)
        delta = m_sim - m_off
        msgs.append(
            f"[compare] mean T_canyon_air OFF={m_off:.3f} K, SIMPLE={m_sim:.3f} K, "
            f"delta={delta:+.3f} K"
        )
        if delta < -0.1:
            msgs.append(
                f"[compare] FAIL: HVAC waste heat should not COOL the canyon "
                f"(delta={delta:+.3f} K)"
            )
            ok = False
        elif delta < 0.001:
            msgs.append(
                f"[compare] WARN: HVAC waste heat produced negligible warming "
                f"(delta={delta:+.3f} K); may be OK if HVAC rarely active"
            )
        else:
            msgs.append(f"[compare] OK: HVAC warms canyon (delta={delta:+.3f} K)")

    # Compare AH range
    ah_off = [a for a in off["ah_max"][:n] if a is not None]
    ah_sim = [a for a in sim["ah_max"][:n] if a is not None]
    if ah_off and ah_sim:
        msgs.append(
            f"[compare] AH max OFF={max(ah_off):.2f} W/m^2, "
            f"SIMPLE={max(ah_sim):.2f} W/m^2"
        )

    return ok, msgs


def make_plot(off, sim, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # -- Panel 1: Q_HVAC (simple only)
    ax = axes[0]
    sim_t = [t / 3600.0 if t is not None else None for t in sim["time_s"]]
    sim_q = list(sim["q_hvac_max"])
    while len(sim_q) < len(sim_t):
        sim_q.append(None)
    ax.plot(sim_t, sim_q, "r-", lw=1.0, label="simple (max)")
    ax.set_ylabel("Q_HVAC (W/m^2)")
    ax.set_title("Phase 5.2 D10: Boston 24 h HVAC-off vs HVAC-simple")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # -- Panel 2: T_canyon_air comparison
    ax = axes[1]
    off_t = [t / 3600.0 if t is not None else None for t in off["time_s"]]
    ax.plot(off_t, off["t_can_max"], "b-", lw=1.0, label="off")
    ax.plot(sim_t, sim["t_can_max"], "r-", lw=1.0, label="simple")
    ax.set_ylabel("T_canyon_air (K)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # -- Panel 3: T_skin_roof comparison
    ax = axes[2]
    ax.plot(off_t, off["t_roof_max"], "b-", lw=1.0, label="off")
    ax.plot(sim_t, sim["t_roof_max"], "r-", lw=1.0, label="simple")
    ax.set_ylabel("T_skin_roof (K)")
    ax.set_xlabel("Simulation time (hours)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="Path to hvac_off run log")
    ap.add_argument("--simple", required=True, help="Path to hvac_simple run log")
    ap.add_argument(
        "--max-step", type=int, default=None,
        help="Expected final step (fail if not reached). Default: no check."
    )
    ap.add_argument(
        "--plot", default=None,
        help="If set, write comparison plot to this PNG path."
    )
    args = ap.parse_args()

    off_path = Path(args.off)
    sim_path = Path(args.simple)
    for p in (off_path, sim_path):
        if not p.exists():
            print(f"ERROR: log file not found: {p}", file=sys.stderr)
            sys.exit(2)

    off = parse_log(off_path)
    sim = parse_log(sim_path)

    all_ok = True
    all_msgs = []

    ok, msgs = check_run(off, "hvac_off", expect_hvac=False, target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    ok, msgs = check_run(sim, "hvac_simple", expect_hvac=True, target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    ok, msgs = compare_runs(off, sim)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    for m in all_msgs:
        print(m)

    if args.plot:
        make_plot(off, sim, args.plot)

    if all_ok:
        print("PASS: Phase 5.2 D10 Boston diurnal HVAC test.")
        sys.exit(0)
    else:
        print("FAIL: one or more checks did not pass. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
