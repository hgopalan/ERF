#!/usr/bin/env python3
"""Checks for the WUI_Subdivision case. Reads the last fire plotfile of each
variant (yt), the exposure CSVs and the run logs, prints a table and exits
non-zero when a check fails. See README.md for what each check means.
"""
import csv, glob, math, os, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("check_wui.py needs yt (pip install yt)")

# Rothermel (1972) for fuel model 1, written out here so that the reference is
# independent of the code under test. (The Python in Unit_Tests/
# test_rothermel_unit.py is not usable for this: its wind coefficient C uses
# exp(-0.8711 sigma^-0.55) instead of exp(-0.133 sigma^0.55) and caps the
# wind factor at 0.9 I_R, which gives 17 m/s for short grass.)
def rothermel_fm1(mf=0.06, U_ftmin=0.0):
    # Anderson FBFM13 fuel model 1 (short grass), Rothermel (1972) with the
    # Albini (1976) form of the parameters. English units throughout.
    w0 = 0.74 * 2000.0 / 43560.0     # lb/ft2 (0.74 ton/acre)
    sigma = 3500.0                   # 1/ft
    delta = 1.0                      # ft
    Mx = 0.12
    h = 8000.0                       # BTU/lb
    rho_p, S_T, S_e = 32.0, 0.0555, 0.010
    rho_b = w0 / delta
    beta = rho_b / rho_p
    beta_op = 3.348 * sigma ** -0.8189
    A = 133.0 * sigma ** -0.7913
    gmax = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    gamma = gmax * (beta / beta_op) ** A * math.exp(A * (1.0 - beta / beta_op))
    rm = min(mf / Mx, 1.0)
    eta_M = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3
    eta_s = 0.174 * S_e ** -0.19
    w_n = w0 * (1.0 - S_T)
    I_R = gamma * w_n * h * eta_M * eta_s                     # BTU/ft2/min
    xi = math.exp((0.792 + 0.681 * math.sqrt(sigma)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    eps = math.exp(-138.0 / sigma)
    Q_ig = 250.0 + 1116.0 * mf
    R0 = I_R * xi / (rho_b * eps * Q_ig)                      # ft/min
    C = 7.47 * math.exp(-0.133 * sigma ** 0.55)
    B = 0.02526 * sigma ** 0.54
    E = 0.715 * math.exp(-3.59e-4 * sigma)
    phi_w = C * U_ftmin ** B * (beta / beta_op) ** -E
    return dict(R0_ftmin=R0, phi_w=phi_w, ROS_ftmin=R0 * (1 + phi_w), ROS_ms=R0 * (1 + phi_w) * 0.00508, I_R=I_R, C=C, B=B, E=E)

MEWS_FTMIN = 300.0   # the model's maximum effective wind for fine fuels (SAV > 1000/ft)

DX_FIRE = 5.0
FM1_LOAD_KG_M2 = 0.166     # 0.74 ton/acre
VARIANTS = ["wildland", "wildland_spotting", "subdivision", "defensible", "coupled"]
ok_all = True

def fail(msg):
    global ok_all
    ok_all = False
    print("  FAIL:", msg)

def last_plotfile(v):
    files = sorted(glob.glob(f"plt_fire_{v}_?????"))
    return files[-1] if files else None

def fields(pf, names):
    ds = yt.load(pf)
    g = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    return {n: np.array(g[("boxlib", n)])[:, :, 0] for n in names}, float(ds.current_time)

def cell(x, y):
    return int(x / DX_FIRE), int(y / DX_FIRE)

def arrival(at, x, y):
    i, j = cell(x, y)
    return at[i, j]

def waf_andrews(delta_ft=1.0):
    return 1.83 / math.log((20.0 + 0.36 * delta_ft) / (0.13 * delta_ft))

# ---------------------------------------------------------------- wildland
print("== wildland: spread rate against Rothermel FM1")
pf = last_plotfile("wildland")
if pf is None:
    fail("no wildland plotfile"); sys.exit(1)
F, t_end = fields(pf, ["fire_arrival_time", "fire_phi", "fire_fuel_load"])
at = F["fire_arrival_time"]
t1, t2 = arrival(at, 400.0, 240.0), arrival(at, 470.0, 240.0)
ros_model = 70.0 / (t2 - t1) if (t1 > 0 and t2 > 0 and t2 > t1) else float("nan")
u_mid = waf_andrews() * 10.0                     # m/s at midflame from the 6.1 m wind
u_eff = min(u_mid * 196.85, MEWS_FTMIN)          # ft/min, capped as the model caps it
ref = rothermel_fm1(mf=0.06, U_ftmin=u_eff)["ROS_ms"]
print(f"  arrival at x=400: {t1:.1f} s, x=470: {t2:.1f} s -> head ROS {ros_model:.3f} m/s")
print(f"  Rothermel FM1, 6% moisture, midflame {u_mid:.2f} m/s (WAF {waf_andrews():.3f}), capped at {MEWS_FTMIN:.0f} ft/min: {ref:.4f} m/s")
if not (0.85 * ref <= ros_model <= 1.15 * ref):
    fail(f"wildland head ROS {ros_model:.3f} m/s outside 15% of Rothermel {ref:.3f} m/s")
t_wild_850 = arrival(at, 780.0, 240.0)
print(f"  arrival at x=780 (beyond the last street): {t_wild_850:.1f} s")
if t_wild_850 <= 0:
    fail("the wildland fire did not reach x = 780 m before the end of the run")

# fuel conservation: the fuel consumed equals the initial load over the burned
# area; cells the front reached in the last minute are still burning, so 5%
first = sorted(glob.glob("plt_fire_wildland_?????"))[0]
F0, _ = fields(first, ["fire_fuel_load"])
burned = (F["fire_phi"] < 0)
consumed = (F0["fire_fuel_load"] - F["fire_fuel_load"])[burned].sum() * DX_FIRE**2
expected = F0["fire_fuel_load"][burned].sum() * DX_FIRE**2
print(f"  burned cells {burned.sum()}, fuel consumed {consumed:.1f} kg vs initial load over burned area {expected:.1f} kg")
if expected > 0 and abs(consumed - expected) > 0.05 * expected:
    fail("fuel consumed differs from the initial load over the burned area by more than 5%")

# ---------------------------------------------------------------- structures
def exposure_rows(v):
    f = f"exposure_{v}.csv"
    if not os.path.exists(f):
        return {}
    last = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            last[int(row["structure_id"])] = row
    return last

results = {"wildland": {"t850": t_wild_850, "burned": int(burned.sum())}}

# The reference for the delay through the subdivision is grass with the same
# seeded spotting, so that the houses are the only difference.
print("== wildland_spotting: the reference for the subdivision")
pf = last_plotfile("wildland_spotting")
if pf is None:
    fail("no wildland_spotting plotfile"); t_ref_850 = float("nan")
else:
    F, _ = fields(pf, ["fire_arrival_time", "fire_phi"])
    t_ref_850 = arrival(F["fire_arrival_time"], 780.0, 240.0)
    print(f"  arrival at x=780: {t_ref_850:.1f} s; burned cells {(F['fire_phi'] < 0).sum()}")
    results["wildland_spotting"] = {"t850": t_ref_850, "burned": int((F["fire_phi"] < 0).sum())}
    if t_ref_850 <= 0:
        fail("wildland_spotting: the fire did not reach x = 780 m")

for v in ["subdivision", "defensible", "coupled"]:
    print(f"== {v}")
    pf = last_plotfile(v)
    if pf is None:
        fail(f"no {v} plotfile"); continue
    F, t_end = fields(pf, ["fire_arrival_time", "fire_phi", "fire_fuel_load", "fire_structure_height"])
    house = F["fire_structure_height"] > 0.5
    n_house_cells = int(house.sum())
    burned_in_house = int(((F["fire_phi"] < 0) & house).sum())
    F0, _ = fields(sorted(glob.glob(f"plt_fire_{v}_?????"))[0], ["fire_fuel_load"])
    fuel_lost_in_house = float((F0["fire_fuel_load"] - F["fire_fuel_load"])[house].max()) if n_house_cells else 0.0
    print(f"  structure cells {n_house_cells}; burned inside footprints {burned_in_house}; max fuel lost inside {fuel_lost_in_house:.3e} kg/m2")
    if fuel_lost_in_house > 1e-12:
        fail(f"{v}: fuel was consumed inside a footprint")
    if n_house_cells == 0:
        fail("no structure cells in the fire plotfile")
    if burned_in_house > 0:
        fail("the level set went negative inside a footprint")
    at = F["fire_arrival_time"]
    t850 = arrival(at, 780.0, 240.0)
    t_rows = [arrival(at, x, 240.0) for x in (560.0, 640.0, 720.0)]
    burned_v = (F["fire_phi"] < 0) & ~house
    print(f"  arrival at the three lanes behind the rows: {t_rows[0]:.1f}, {t_rows[1]:.1f}, {t_rows[2]:.1f} s; at x=780: {t850:.1f} s; burned cells {burned_v.sum()}")
    ex = exposure_rows(v)
    reached = sum(1 for r in ex.values() if float(r["t_first_s"]) >= 0)
    hl_max = max((float(r["heat_load_max_MJm2"]) for r in ex.values()), default=0.0)
    pk_max = max((float(r["peak_intensity_kWm"]) for r in ex.values()), default=0.0)
    embers = sum(int(float(r["embers"])) for r in ex.values())
    print(f"  exposure rows for {len(ex)} houses: {reached} reached by the front, peak intensity {pk_max:.0f} kW/m, heat load max {hl_max:.2f} MJ/m2, embers landed {embers}")
    # first-row houses see the front before the back rows
    by_row = {}
    for r in ex.values():
        row = min(range(3), key=lambda k: abs(float(r["x_m"]) - (530.0 + 80.0 * k)))
        if float(r["t_first_s"]) >= 0:
            by_row.setdefault(row, []).append(float(r["t_first_s"]))
    firsts = [min(by_row[k]) if k in by_row else float("nan") for k in range(3)]
    print(f"  first contact per row: {firsts[0]:.1f}, {firsts[1]:.1f}, {firsts[2]:.1f} s")
    results[v] = dict(t850=t850, burned=int(burned_v.sum()), reached=reached, hl_max=hl_max, pk_max=pk_max, embers=embers, rows=firsts)
    if len(ex) != 24:
        fail(f"{v}: expected 24 structures in the exposure CSV, found {len(ex)}")
    if v == "subdivision":
        if t850 <= 0:
            fail("subdivision: the fire did not reach x = 780 m")
        elif t850 <= t_ref_850:
            fail("subdivision: the front reached x = 780 m no later than the grass run with the same spotting")
        if reached == 0:
            fail("subdivision: no house was reached by the front")
        if embers == 0:
            fail("subdivision: no ember landed on a footprint")
    if v == "defensible":
        s = results.get("subdivision", {})
        if s and not (hl_max < s["hl_max"]):
            fail("defensible: heat load at the houses is not below the subdivision's")
        if s and not (reached < s["reached"]):
            fail("defensible: as many houses reached by the front as without defensible space")
    if v == "coupled":
        log = open("run_coupled.log").read()
        if re.search(r"\bnan\b", log, re.I):
            fail("coupled: NaN in the log")
        atm = sorted(glob.glob("plt_atm_coupled_?????"))
        if atm:
            ds = yt.load(atm[-1])
            g = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
            wmax = float(np.array(g[("boxlib", "z_velocity")]).max())
            print(f"  max vertical velocity in the last atmosphere plotfile: {wmax:.2f} m/s")
            if wmax < 0.5:
                fail("coupled: no plume (max w below 0.5 m/s)")
        if t850 <= 0:
            fail("coupled: the fire did not reach x = 780 m")

print("== summary")
print(f"  {'variant':12s} {'t(850 m) s':>11s} {'burned':>7s} {'reached':>8s} {'peak kW/m':>10s} {'HL MJ/m2':>9s} {'embers':>7s}")
for v in VARIANTS:
    r = results.get(v)
    if not r: continue
    print(f"  {v:12s} {r['t850']:11.1f} {r['burned']:7d} {r.get('reached', 0):8d} {r.get('pk_max', 0.0):10.0f} {r.get('hl_max', 0.0):9.2f} {r.get('embers', 0):7d}")
print("RESULT:", "pass" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
