#!/usr/bin/env python3
"""Regenerate the dust-model input reference from one parameter list.

Writes the "Dust Model" section of Docs/sphinx_doc/Inputs.rst (between the
labels ``.. _sec:DustInputs:`` and ``.. _sec:EnsembleInitialization:``) and
the deck Exec/CanonicalTests/Dust/inputs_dust_master_reference, so the two
cannot drift apart. Run from the repository root after changing a dust input:

    python3 Docs/sphinx_doc/tools/gen_dust_inputs.py

Defaults and descriptions follow Source/Dust/ERF_DustParams.H,
Source/FireDust/ERF_FireDustCoupling.H (read in Source/ERF.cpp) and
Source/DataStructs/ERF_TurbStruct.H; edit the list below when those change.
"""
import os, sys, textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INPUTS_RST = os.path.join(ROOT, "Docs", "sphinx_doc", "Inputs.rst")
DECK = os.path.join(ROOT, "Exec", "CanonicalTests", "Dust", "inputs_dust_master_reference")

# (key, definition, acceptable, default, deck_value_or_None, deck_comment)
# prefix "erf.dust." unless the key starts with "erf."
G = []
def grp(title, intro=""):
    G.append((title, intro, []))
def p(key, definition, acceptable, default, deck=None, comment=None):
    G[-1][2].append((key, definition, acceptable, default, deck, comment))

grp("Master control",
    "``erf.dust.enable`` switches the model on. The prerequisites in "
    ":ref:`sec:Dust` (a surface layer, no z decomposition, scalar transport) "
    "are checked once at startup and abort with a message naming the input "
    "to fix.")
p("enable", "Enable the dust model", "Boolean", "false")
p("dust_debug", "Print per-step dust diagnostics to stdout", "Boolean", "false")
p("grid_ratio", "Dust grid refinement factor in x and y; every atmosphere box "
  "length must divide by it, and it must equal erf.fire.grid_ratio when the "
  "fire coupling is on", "Integer > 0", "1")
p("n_size_bins", "Number of particle size bins; each bin is one component of "
  "the emission flux", "Integer > 0", "3")
p("bin_diameter_um", "Representative aerodynamic diameter per bin [µm]; bin 0 "
  "sets the Bagnold base threshold", "Reals", "7.0 3.5 0.7")
p("bin_diameters", "Per-bin diameter [m] for settling, deposition and PM "
  "classification; the last value repeats when shorter than n_size_bins",
  "Reals", "7.0e-6 2.5e-6 50.0e-6")
p("particle_density", "Bulk particle density [kg/m³]", "Real > 0", "2650.0")
p("rho_air", "Air density used in the threshold and saltation flux [kg/m³]",
  "Real > 0", "1.225")
p("z0_dust", "Roughness length of the emitting surface [m], used by the "
  "log-law friction velocity of the terrain-corrected wind", "Real > 0", "0.01")
p("zref", "Height at which the wind is taken from the atmosphere [m]; set "
  "equal to erf.most.zref", "Real > 0", "10.0")

grp("Surface state",
    "Uniform values apply wherever no raster is given. Rasters are ESRI "
    "ASCII (``.asc``); a ``.nc`` path aborts because the NetCDF readers are "
    "not implemented.")
p("silt_fraction", "Surface silt mass fraction [-]", "Real 0-1", "0.10")
p("crust_index", "Surface crust strength index; 0 loose, 1 fully crusted",
  "Real 0-1", "0.0")
p("threshold_A_coeff", "Bagnold threshold coefficient A [-]", "Real > 0", "0.0123")
p("ustar_t_base", "Base threshold friction velocity before the modifiers "
  "[m/s]; negative computes the Bagnold value from bin 0 at startup",
  "Real", "-1.0")
p("alpha_crust", "Crust factor on the threshold: f_chem carries (1 + "
  "alpha_crust * crust_index)", "Real >= 0", "0.5")
p("alpha_efflor", "Efflorescence factor on the threshold: (1 + alpha_efflor "
  "* efflorescence)", "Real >= 0", "0.3")
p("soil_type_file", "Soil type raster; codes 1-16 STATSGO, 100-104 mine "
  "surfaces", "String", '""')
p("silt_fraction_file", "Silt fraction raster; empty keeps silt_fraction",
  "String", '""')
p("crust_index_file", "Crust index raster; empty keeps crust_index", "String", '""')
p("moisture_flag_file", "Surface moisture inhibition raster in [0,1]; empty "
  "means dry", "String", '""')
p("suppression_file", "Suppression agent coverage raster in [0,1]; empty "
  "means none", "String", '""')
p("surface_map_file", "Read but not consumed; the five rasters above carry "
  "the surface state", "String", '""')
p("terrain_file", "Fine-grid terrain raster for the dust slopes; empty "
  "differences the atmosphere's nodal terrain", "String", '""')

grp("Wind and terrain",
    "With an atmosphere present the friction velocity comes from the "
    "surface layer; the ``test_*`` values are the placeholders used when "
    "no atmosphere is coupled.")
p("use_terrain_wind", "Apply the FARSITE terrain correction to the wind at "
  "zref and recompute u* from it by the log law", "Boolean", "false")
p("k_ridge", "Ridge speed-up factor of the terrain correction", "Real", "1.5")
p("k_shelter", "Lee-side shelter factor", "Real", "0.6")
p("k_valley", "Valley channelling factor", "Real", "0.8")
p("k_deflect", "Deflection of the wind vector toward the slope", "Real", "0.3")
p("test_ustar", "Uniform friction velocity when no atmosphere is coupled "
  "[m/s]; 0 gives no emission", "Real >= 0", "0.0")
p("test_surf_temp_K", "Surface temperature placeholder [K]", "Real", "293.15")
p("test_wind_speed", "Wind speed placeholder at zref [m/s]", "Real", "5.0")

grp("PHREEQC geochemistry",
    "PHREEQC runs offline; its output table is re-read at the interval "
    "given here and its columns update the crust, silt, efflorescence and "
    "suppression fields. Sites give each mine its own table.")
p("phreeqc_output_file", "PHREEQC output table (.csv with a header row; a "
  ".nc path aborts); empty disables the reader", "String", '""')
p("phreeqc_update_interval_s", "Interval between re-reads [s]", "Real > 0", "86400.0")
p("phreeqc_crust_var", "Column holding the crust index", "String", '"crust_index"')
p("phreeqc_silt_var", "Column holding the silt fraction", "String", '"silt_fraction"')
p("phreeqc_efflor_var", "Column holding the efflorescence fraction", "String",
  '"efflorescence"')
p("phreeqc_supp_var", "Column holding the suppression modifier", "String",
  '"suppression_mod"')
p("phreeqc_metal_var", "Column holding the toxic-metal mass fraction of bin 0",
  "String", '"metal_as_bin0"')
p("site_names", "Names of the mine sites; empty means a single global table",
  "Strings", "none")
p("site_phreeqc_files", "Per-site PHREEQC table; an empty entry uses the global "
  "table", "Strings", "none")
p("site_x_lo", "Site bounding-box lower x [m], one per site", "Reals", "none")
p("site_y_lo", "Site bounding-box lower y [m]", "Reals", "none")
p("site_x_hi", "Site bounding-box upper x [m]", "Reals", "none")
p("site_y_hi", "Site bounding-box upper y [m]; the last site listed wins "
  "where boxes overlap", "Reals", "none")
p("phreeqc_feedback_interval_s", "Interval for writing the deposition field "
  "back for PHREEQC [s]; 0 writes only at the end of the run", "Real >= 0", "0.0")
p("phreeqc_feedback_file", "Deposition grid file, overwritten at each write",
  "String", '"dust_dep_feedback.dat"')
p("phreeqc_site_summary_file", "Per-site deposition CSV, appended at each "
  "write", "String", '"dust_dep_site_summary.csv"')

grp("Scheduled sources and suppression",
    "Blasting and haul-road traffic add to the wind-driven flux. The CSV "
    "layouts are in :ref:`sec:DustSources`.")
p("blast_schedule_file", "Blast schedule CSV; empty means no blasts", "String", '""')
p("blast_reactivity", "Multiplier on the injected blast mass for fresh "
  "surfaces [-]", "Real >= 1", "2.0")
p("road_schedule_file", "Haul road schedule CSV; empty means no road "
  "emission", "String", '""')
p("road_diag_file", "Per-road emission CSV", "String", '"dust_road_diag.csv"')
p("supp_tau_base_s", "Base decay time of the suppression agent coverage [s]; "
  "about 3600 for water, 43200 for MgCl2", "Real > 0", "3600.0")

grp("Atmosphere coupling",
    "Emission enters the dust scalar at the lowest cell one step after it is "
    "computed; settling, deposition and the return of the surface "
    "concentration act on the same scalar.")
p("atm_feedback", "Scale on the injected flux; 0 disables injection for "
  "surface-only diagnostics", "Real 0-1", "1.0")
p("transport_bins_separately", "One 3D scalar per bin instead of a single "
  "total; only bin 0 is returned to the surface at present", "Boolean", "false")
p("deposition_E0", "Surface collection efficiency of the dry-deposition "
  "resistance [-]; 3e-3 bare mine surface, 1e-4 paved road, 1e-2 vegetation",
  "Real > 0", "3.0e-3")
p("loading_feedback_coeff", "Shao (2001) loading feedback on the threshold "
  "[m³/kg]; 0 disables", "Real >= 0", "0.0")
p("use_dynamic_moisture", "Derive the moisture inhibition from the surface "
  "moisture flux (needs a moisture scheme; harmless without one)", "Boolean",
  "false")
p("erf.dust_mrf_Sc_t", "Turbulent Schmidt number of the dust scalar in the "
  "MRF scheme, read from the erf prefix; 0 or negative uses the Prandtl "
  "number so dust diffuses like heat", "Real", "0 (Pr_t)")

grp("Fire coupling (ERF-Hazard)",
    "Read from the ``erf`` prefix in ``Source/ERF.cpp`` when both the fire "
    "and dust models are enabled; see :ref:`sec:DustFire`.")
p("erf.fire_dust_coupling", "Enable the fire-dust coupling; requires "
  "erf.dust.grid_ratio = erf.fire.grid_ratio", "Boolean", "false")
p("erf.fire_dust_crust_reduction", "Fraction of the crust index removed in "
  "burned cells each step", "Real 0-1", "0.8")
p("erf.fire_dust_wind_to_dust", "Raise the dust u* to the log-law value of "
  "the fire's effective wind where that is larger", "Boolean", "true")
p("erf.fire_dust_wind_z0", "Roughness length of that log law [m]", "Real > 0", "0.1")
p("erf.fire_dust_wind_zref", "Reference height of that log law [m]; match "
  "erf.fire.wind_ref_ht", "Real > 0", "6.1")
p("erf.fire_dust_lofting_enabled", "Multiply the emission flux by the "
  "convective lofting factor of the fire heat flux", "Boolean", "false")
p("erf.fire_dust_lofting_k_loft", "Maximum lofting enhancement [-]", "Real >= 0", "2.0")
p("erf.fire_dust_lofting_Q_threshold", "Fire heat flux below which there is "
  "no lofting [W/m²]", "Real >= 0", "50.0")
p("erf.fire_dust_lofting_Q_ref", "Heat flux scale of the lofting factor "
  "[W/m²]; 0 or negative disables it", "Real", "500.0")

grp("Output and diagnostics",
    "Every CSV is written by rank 0 and appended each step; paths are "
    "relative to the run directory. Formats are in :ref:`sec:DustOutput`.")
p("dust_plot_int", "Steps between dust plotfiles; -1 disables, 0 writes only "
  "at the final step", "Integer", "-1")
p("dust_plot_prefix", "Dust plotfile prefix", "String", '"plt_dust_"')
p("dust_diag_file", "Per-step domain statistics CSV", "String", '"dust_diag.dat"')
p("dust_naaqs_file", "EPA NAAQS PM2.5 and PM10 CSV", "String", '"dust_naaqs.csv"')
p("msha_pel_mg_m3", "MSHA permissible exposure limit on the 8-hour TWA "
  "[mg/m³]", "Real > 0", "5.0")
p("msha_shift_duration_s", "Shift length after which the dose resets [s]",
  "Real > 0", "28800.0")
p("msha_exposure_file", "Per-step exposure CSV", "String", '"msha_exposure.csv"')
p("msha_shift_file", "End-of-shift summary CSV", "String", '"msha_shift_summary.csv"')
p("msha_receptor_names", "Receptor point names; one CSV per receptor",
  "Strings", "none")
p("msha_receptor_x", "Receptor x [m], one per name", "Reals", "none")
p("msha_receptor_y", "Receptor y [m], one per name", "Reals", "none")
p("cm_fractions", "Critical-material mass fraction per bin [kg/kg]; empty "
  "disables the budget, the last value repeats", "Reals", "none")
p("cm_budget_file", "Critical-material budget CSV", "String", '"dust_cm_budget.csv"')
p("visibility_enable", "Koschmieder visibility from PM10", "Boolean", "false")
p("visibility_k_ext", "Mass extinction coefficient [m²/kg]", "Real > 0", "4.0e3")
p("visibility_road_closure_m", "Haul-road closure threshold [m]", "Real > 0", "300.0")
p("visibility_warning_m", "Reduced-visibility warning threshold [m]", "Real > 0",
  "1000.0")
p("visibility_diag_file", "Visibility CSV", "String", '"visibility_diag.csv"')
p("silica_enable", "Respirable crystalline silica from PM10", "Boolean", "false")
p("silica_fraction_rcs", "Silica mass fraction of the dust [-]", "Real 0-1", "0.04")
p("silica_osha_pel_mg_m3", "OSHA PEL for quartz [mg/m³]", "Real > 0", "0.05")
p("silica_diag_file", "Silica CSV", "String", '"silica_diag.csv"')
p("stel_enable", "Short-term exposure limit on the running PM10 average",
  "Boolean", "false")
p("stel_threshold_mg_m3", "STEL threshold [mg/m³]", "Real > 0", "10.0")
p("stel_averaging_s", "STEL averaging period [s]", "Real > 0", "900.0")
p("stel_diag_file", "STEL CSV", "String", '"stel_diag.csv"')
p("enable_particles", "Release Lagrangian super-particles from emitting "
  "cells; needs ERF_ENABLE_PARTICLES", "Boolean", "false")
p("particle_release_interval", "Steps between particle releases", "Integer > 0", "1")

def full(key):
    return key if key.startswith("erf.") else "erf.dust." + key

# ---------------------------------------------------------------- RST
W1, W2, W3, W4 = 46, 60, 26, 36
def cell(text, w):
    return textwrap.wrap(text, w - 2, break_long_words=False, break_on_hyphens=False) or [""]
def row(cols, widths, bold=False):
    parts = [cell(c, w) for c, w in zip(cols, widths)]
    n = max(len(x) for x in parts)
    out = []
    for i in range(n):
        line = "|"
        for part, w in zip(parts, widths):
            s = part[i] if i < len(part) else ""
            line += " " + s.ljust(w - 2) + " |"
        out.append(line)
    return out
def sep(widths, ch="-"):
    return "+" + "+".join(ch * w for w in widths) + "+"

rst = []
rst.append(".. _sec:DustInputs:\n\n.. _inputs-dust-model:\n")
rst.append("Dust Model\n==========\n")
rst.append(textwrap.fill(
    "These inputs configure the dust model described in :ref:`sec:Dust`. "
    "Every key is read once at startup from the ``erf.dust`` prefix, except "
    "the fire-coupling keys and the MRF Schmidt number, which sit directly "
    "under ``erf``. Defaults are those of the parameter struct in "
    "``Source/Dust/ERF_DustParams.H``; the reference deck "
    "``Exec/CanonicalTests/Dust/inputs_dust_master_reference`` lists every "
    "key at its default with a comment; both come from "
    "``Docs/sphinx_doc/tools/gen_dust_inputs.py``, so edit that list and rerun "
    "it when a key changes. Keys marked \"read but not consumed\" are accepted and "
    "have no effect at present.", 78))
rst.append("")
widths = (W1, W2, W3, W4)
for title, intro, params in G:
    rst.append("\n" + title + "\n" + "-" * len(title) + "\n")
    if intro:
        rst.append(textwrap.fill(intro, 78) + "\n")
    rst.append(sep(widths))
    rst += row(("Parameter", "Definition", "Acceptable Values", "Default"), widths)
    rst.append(sep(widths, "="))
    for key, d, a, dflt, *_ in params:
        shown = ("``" + dflt + "``") if dflt.startswith('"') else dflt
        rst += row(("**" + full(key) + "**", d, a, shown), widths)
        rst.append(sep(widths))
    rst.append("")
section = "\n".join(rst) + "\n"
doc = open(INPUTS_RST).read()
a = doc.index(".. _sec:DustInputs:")
b = doc.index(".. _sec:EnsembleInitialization:")
open(INPUTS_RST, "w").write(doc[:a] + section + "\n" + doc[b:])

# ---------------------------------------------------------------- deck
deck = []
deck.append("""# ============================================================================
# ERF dust model: master input reference
# ============================================================================
# Every dust input with its default value and a one-line description, in the
# groups of the Sphinx page Docs/sphinx_doc/Inputs.rst (section "Dust Model").
# This deck documents the keys; it is not a tuned case. With every key at its
# default the dust model is off (erf.dust.enable = false), so running it is a
# plain neutral ABL. The prerequisites of the dust model are set below so the
# deck can be used as a starting point.
#
# Generated by the parameter list in the documentation; edit the list, not
# this file, when a key changes.
# ============================================================================

# ---------------------------------------------------------------------------
# Atmosphere set-up that the dust model requires
# ---------------------------------------------------------------------------
erf.prob_name = "ABL"
geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 3000.0 3000.0 1024.0
amr.n_cell           = 8 8 64
geometry.is_periodic = 1 1 0
amr.max_level        = 0
amr.max_grid_size    = 32
amr.max_grid_size_z  = 64            # no decomposition in z (dust prerequisite)
amrex.fpe_trap_invalid = 0

zlo.type = "surface_layer"           # a surface layer is required (u*, T_sfc, PBLH)
zhi.type = "SlipWall"
erf.most.z0   = 0.1
erf.most.zref = 24.0                 # match erf.dust.zref

erf.pbl_type   = "MRF"               # provides the PBL height and the scalar diffusivity
erf.transport_scalar = true          # the dust rides in a passive scalar
erf.molec_diff_type = "None"
erf.les_type        = "None"
erf.use_gravity = true
erf.init_type   = "input_sounding"
erf.input_sounding_file = "sounding_neutral_abl"
erf.theta_ref   = 300.0
erf.abl_geo_wind = 15.0 0.0 0.0

erf.fixed_dt = 0.5
max_step     = 20
erf.check_int = -1
erf.plot_int_1  = 10
erf.plot_file_1 = plt
erf.plot_vars_1 = density x_velocity y_velocity z_velocity theta rhoadv_0
erf.v = 0
amr.v = 0
""")
for title, intro, params in G:
    deck.append("# " + "-" * 75)
    deck.append("# " + title)
    deck.append("# " + "-" * 75)
    for key, d, a, dflt, deckv, comment in params:
        val = deckv if deckv is not None else dflt
        if val == "none":
            line = "# " + full(key) + " ="
        elif val.startswith("0 (Pr_t)"):
            line = full(key) + " = 0.0"
        else:
            line = full(key) + " = " + val
        desc = textwrap.wrap(d + (" [" + a + "]" if a else ""), 74)
        for s in desc:
            deck.append("# " + s)
        deck.append(line)
        deck.append("")
open(DECK, "w").write("\n".join(deck) + "\n")
n = sum(len(x[2]) for x in G)
print("dust inputs:", n, "keys ->", INPUTS_RST, "and", DECK)
