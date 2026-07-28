# ERF-SLUCM Future Features — Phase 7+ Candidate Roadmap

## Purpose

This document catalogs candidate features for ERF-SLUCM Phase 7 and beyond, targeting the post-v1.0 release window. All features listed here meet three criteria:

1. **Peer-reviewed literature basis** — physics is published and validated somewhere.
2. **Absent or awkward in WRF-UCM / BEP / BEM** — this is where ERF-SLUCM can differentiate.
3. **Fits within the SLUCM architectural pattern** — per-cell CSV inputs, 2D slab, N-facet Newton SEB, radiosity infrastructure by v1.0.

Features are grouped by implementation cost tier. Each entry describes the physics, the citation, why it fits ERF-SLUCM specifically, and the WRF gap.

Companion document: `UCM_DEVELOPMENT.md` (main roadmap, Phases 1.1 through 7.1b).

---

## Roadmap Positioning

Current Phase 7 in `UCM_DEVELOPMENT.md`:

- **7.1a** — worry-list audit + regression consolidation
- **7.1b** — documentation + release CI + v1.0 tag

Proposed extension for post-v1.0:

- **7.2** — Novel-feature planning gate: audit user feedback from v1.0, prioritize items from this document
- **v1.1** — Top 3–5 novel features (differentiation release)
- **v1.2+** — Remaining features via research collaborations, external contributions

This lets v1.0 remain a validated baseline; v1.1 becomes the "here's what ERF-SLUCM can do that BEP+BEM cannot" release, with a much stronger marketing narrative than jamming novel physics into v1.0 before validation.

---

## Tier 1 — Very Cheap (CSV-only or single-scalar edits, days of work each)

### 7.2 — Anisotropic pavement albedo (retroreflective / directional cool pavements)

**Physics.** Angular-dependent road albedo: `α_road = α_0 + Δα · f(zenith)`. Retroreflective cool pavements reflect downward (into sky) under high sun instead of horizontally (into pedestrians), reducing pedestrian UHI while maintaining rooftop cooling.

**Literature.** Sen & Roesler 2016; Rosado et al. 2017.

**Fit.** Single new column in `materials.csv` (`albedo_slope_per_rad`); one line in SEB SW input path multiplies existing per-cell albedo by the zenith-dependent factor.

**WRF gap.** No support in WRF-UCM/BEP/BEM.

---

### 7.3 — Time-varying anthropogenic heat with weekday / weekend / holiday distinction

**Physics.** Extends the existing Phase 2.3 diurnal AH profile from a single hourly curve to a `profile_id ∈ {weekday, weekend, holiday}` selector with a calendar CSV. Weekday peaks are morning + evening; weekends are single-peak + shifted; holidays are flat-suppressed.

**Literature.** Sailor 2011; Iamarino et al. 2012 (LUCY dataset).

**Fit.** Extend `ah_profile_id` iMultiFab from 1 index to 3; add day-of-week logic in `compute_anthropogenic_heat`. Calendar CSV maps `julian_day → profile_id`.

**WRF gap.** WRF-UCM uses a single AH profile per land-use category.

---

### 7.4 — Snow cover on roof / road

**Physics.** Time-varying `albedo`, `emissivity`, and (indirectly) `slab_L_effective` for roof and road facets driven by a simple mass-balance: snowfall input − melt from Newton residual (positive heat flux → melt) − sublimation. Snow-covered surfaces have α ≈ 0.7, ε ≈ 0.95, and act as a thermal insulator on top of the slab.

**Literature.** Järvi et al. 2014 (SUEWS snow module); Wang et al. 2013.

**Fit.** Adds one state variable (`snow_depth`) per facet; modifies existing albedo/emissivity/conduction fields in-place before SEB call. Precipitation phase (rain vs snow) from `T_air` threshold.

**WRF gap.** WRF has Noah-MP snow but does not couple it into the UCM path.

---

### 7.5 — Seasonal / heat-wave adaptive AC operation

**Physics.** Simple threshold model on top of Phase 5.2 COP-based rejection: HVAC waste heat is zero when `T_air < T_setpoint - 2 K`, nominal (Phase 5.2 baseline) when `T_setpoint - 2 K ≤ T_air ≤ T_setpoint + 2 K`, and doubles when `T_air > T_setpoint + 5 K` (during heatwaves). Adds hysteresis to prevent chatter.

**Literature.** Salamanca et al. 2014; de Munck et al. 2013.

**Fit.** Three lines in the AH computation once Phase 5.2 is in. Setpoint from `materials.csv` or a new `building_thermal.csv`.

**WRF gap.** WRF-BEM has a single setpoint, no hysteresis, no heatwave amplification.

---

### 7.6 — Wind-direction-dependent wall drag (windward vs leeward Cd)

**Physics.** Real building walls have asymmetric drag: windward face `Cd ≈ 0.8`, leeward face `Cd ≈ 0.2`. The Phase 2.8 kernel currently uses symmetric `Cd_wall`. Split into two Cd values, decompose ATM velocity onto wall-normal vectors (from `building_layout.csv` orientation), apply per-face.

**Literature.** Coceal & Belcher 2004; Santiago et al. 2013.

**Fit.** Existing scalar `Cd_wall` becomes two scalars (`Cd_wall_windward`, `Cd_wall_leeward`); one dot-product against wall-normal in the drag kernel selects which one applies to which momentum component.

**WRF gap.** WRF-BEP uses symmetric Cd.

---

## Tier 2 — Modest infrastructure (one new state variable or field, 1–2 weeks each)

### 7.7 — Diffuse SW component & anisotropic sky (direct/diffuse split)

**Physics.** Split `SW_down` into `SW_direct + SW_diffuse` with `f_direct = f(zenith, cf)` from Reindl 1990 or Erbs 1982. Direct term uses SVF weighting (line-of-sight from sun); diffuse term uses isotropic radiosity (reuses the Phase 5.1b 2×2 matrix). Both feed the SEB SW input path independently, then are summed.

**Literature.** Reindl et al. 1990; Perez et al. 1990; Erbs et al. 1982.

**Fit.** Two SW streams into the existing radiosity solver — the 2×2 matrix stays the same, just two independent RHS solves. Cost: minor.

**WRF gap.** WRF-BEP typically treats SW as lumped.

---

### 7.8 — Surface moisture on non-vegetated surfaces (wet pavement, dew)

**Physics.** Simple bucket model per facet: puddle depth updated by precipitation input − evaporation from the Newton LE branch (which is currently placeholder=0). Adds LE to the SEB energy balance. Wet pavement has ~30% higher effective emissivity and lower daytime T_skin peak.

**Literature.** Grimmond & Oke 1991; Ramamurthy & Bou-Zeid 2014.

**Fit.** Reuses the green-roof soil bucket infrastructure from Phase 5.3; applies to road/roof facets. Adds one state MF per facet (`water_depth_{road,roof,wall}`).

**WRF gap.** WRF-UCM has no surface moisture; WRF-BEP treats it partially.

---

### 7.9 — Cool pavement thermal fatigue (aging albedo)

**Physics.** High-albedo cool pavements soil over time: `α_road(t) = α_new · exp(-t/τ) + α_aged · (1 − exp(-t/τ))` with `τ ≈ 2 years`. Realistic modeling of urban planning interventions — a "new cool pavement" scenario cannot honestly claim the initial albedo for the whole simulation.

**Literature.** Levinson & Akbari 2002.

**Fit.** One line in the albedo lookup, driven by a simulation-start date (already present in Phase 3.5b as `julian_day`); τ from `materials.csv`.

**WRF gap.** No aging model anywhere in WRF-UCM/BEP.

---

### 7.10 — Storage-heat-flux memory (ΔQ_S via Objective Hysteresis Model)

**Physics.** The SEB currently closes instantaneously via Newton. Real urban surfaces show hysteresis between radiation input and storage — peak storage lags peak net radiation by 1–2 hours. Grimmond, Cleugh, Oke 1991 (OHM):

```
ΔQ_S = a1·Q* + a2·dQ*/dt + a3
```

with `(a1, a2, a3)` per-facet coefficients from calibrated flux-tower campaigns.

**Literature.** Grimmond et al. 1991; Meyn & Oke 2009.

**Fit.** Post-SEB diagnostic first (validates coefficients); then wired into the storage term for a physically more realistic closure. Needs one extra state variable per facet (previous-timestep `Q*`), 3 CSV coefficients per material.

**WRF gap.** WRF-SLUCM has no OHM; SUEWS-family standalone models do.

---

### 7.11 — Vertical variation of Cd within canyon (BEP-Tree P-parameter)

**Physics.** Current Cd_wall is uniform vertical. Real canyons show top-heavy drag from roof-level shear (P-parameter formulation). Replace scalar `Cd_wall` with a vertical profile `Cd_wall(z/H_bldg)` from a small polynomial or lookup.

**Literature.** Krayenhoff et al. 2015 (BEP-Tree); Simón-Moral et al. 2017.

**Fit.** Modest — the drag kernel already iterates over vertical layers. Replace one scalar lookup with an interpolated profile.

**WRF gap.** WRF-BEP uses uniform Cd; only the (limited) BEP-Tree extension has vertical variation.

---

### 7.12 — Roughness sublayer displacement correction

**Physics.** MOST assumes log-law above `d + z0`. The roughness sublayer extends to `~2·H_bldg` where log-law breaks down. Kanda 2006 correction shifts the wind profile inside the RSL:

```
u(z) = (u*/κ) · [ln((z-d)/z0) + Ψ_RSL(z/H_bldg)]
```

**Literature.** Harman & Finnigan 2007; Kanda 2006.

**Fit.** One extra term in the wind-extraction routine `interpolate_wind_to_zref`. RSL correction function tabulated or analytic.

**WRF gap.** WRF-BEP has no RSL correction.

---

## Tier 3 — Meaningful physics addition (new module, 2–4 weeks each)

### 7.13 — Urban precipitation interception & rooftop runoff

**Physics.** Full urban water balance: rainfall input → intercepted by roof/wall/road buckets → excess drains as runoff → surface moisture feeds LE. Runoff coefficient per material (concrete: 0.9; grass: 0.3; green roof: 0.1). Enables realistic diurnal Bowen ratio and Blue-Green infrastructure evaluation.

**Literature.** Wang et al. 2013 (UWRF); Järvi et al. 2011 (SUEWS).

**Fit.** Extends Feature 7.8 (surface moisture) to a full water balance. Requires precipitation as new ATM→UCM data stream (Phase 3.2 established the plumbing; add one field).

**WRF gap.** WRF-UCM/BEP: no water balance; SUEWS has full version but is standalone.

---

### 7.14 — Vertical air temperature profile within canyon (multi-layer canyon-air)

**Physics.** Currently one scalar `T_canyon_air`. Split into `T_canyon_air(k)` for `k = 1 … N_canyon_levels` (typically 3–5). Improves crown/tree coupling (Phase 6.2b), better indoor-outdoor thermal exchange, and captures inversion or unstable stratification within deep canyons. The existing first-order relaxation becomes a small tridiagonal solve.

**Literature.** Martilli 2002 (BEP); Krayenhoff et al. 2015.

**Fit.** `T_canyon_air` becomes a MultiFab with `ncomp = N`; existing scalar relaxation becomes a tridiagonal per column (reuse TDMA from slab conduction).

**WRF gap.** WRF-BEP has multi-layer canyon; WRF-SLUCM does not. **This is arguably the single biggest physics upgrade — it's the actual "M" (multi-layer) of BEP but implemented inside the SLUCM framework, so users don't have to switch models.**

---

### 7.15 — Local advection between adjacent UCM cells

**Physics.** Explicit horizontal advection of `T_canyon_air` between neighboring UCM cells driven by ATM wind direction and magnitude. Captures edge effects: a park cell downwind of dense urban is warmer than an equivalent park upwind; oasis effect at urban-rural boundaries.

**Literature.** Nazarian & Kleissl 2016; Theeuwes et al. 2013.

**Fit.** One advection term added to the canyon-air prognostic on the UCM grid; reuses ATM wind fields already aggregated to UCM. Upwind or centered difference on the 2D UCM slab.

**WRF gap.** WRF does not do horizontal advection at UCM level — only at the ATM level, which loses sub-grid heterogeneity.

---

### 7.16 — Kanda et al. 2013 z0/d formulation using building-height variance

**Physics.** Kusaka's simplified `z0 = 0.1·H`, `d = 0.7·H` shortcut ignores subgrid heterogeneity. Kanda et al. 2013 use `σ_H / H_mean` (variance-weighted formulation) for more accurate effective roughness in heterogeneous urban landscapes.

**Literature.** Kanda et al. 2013; Millward-Hopkins et al. 2011.

**Fit.** One-line change in `fill_ucm_z0_and_disp` — Phase 2.5 already computes `H_bldg_std_atm`; just consume it in the Kanda formula instead of the constant fraction.

**WRF gap.** WRF-SLUCM uses the Kusaka shortcut.

---

### 7.17 — Anisotropic diffuse SW distribution (Perez model)

**Physics.** Diffuse SW under cloud is not isotropic — it exhibits circumsolar brightening, horizon brightening, and zenith darkening depending on cloud type and sun position. Perez 1990 model captures these with three coefficients.

**Literature.** Perez et al. 1990.

**Fit.** One weighting function on the SW radiosity RHS (does not change the matrix). Requires Feature 7.7 (direct/diffuse split) as prerequisite.

**WRF gap.** All WRF radiation transfers assume isotropic diffuse.

---

## Tier 4 — Substantial features (4–8 weeks, post-v1.0 candidates)

### 7.18 — Ground heat flux to deeper subsurface (dynamic T_deep)

**Physics.** Currently road slab bottom boundary is `T_deep = const`. Add a physically motivated seasonal `T_deep(t)` from a simple 1D heat equation into the deeper soil column (below the road slab). Prevents summer heat storage from cumulative drift.

**Literature.** Best & Grimmond 2014 (JULES-Urban); Sun & Grimmond 2019.

**Fit.** One 1D column solver run monthly (or every N steps) to update `slab_T_deep_road` from surface flux history and a deep boundary at 5 m or 10 m depth.

**WRF gap.** WRF-UCM uses constant `T_deep`.

---

### 7.19 — Building thermal envelope with wall insulation layers (BEM-lite)

**Physics.** Currently walls are homogeneous N-layer slabs of one material. Real walls: outer render + brick + air gap + insulation + interior plaster + interior air. Multi-material slab with 5–7 layers, each with different `k`, `ρc_p`, thickness.

**Literature.** Salamanca et al. 2010 (BEM).

**Fit.** Extends the N-layer slab from single-material to a per-layer `material_id` iMultiFab; reuses the existing TDMA solver. **This is the "E" in BEM — worth adding since you already have the slab infrastructure and the material CSV catalog. This turns "SLUCM" into "SLUCM + BEM-lite" and gives you AC waste heat that's genuinely physical (indoor T + envelope resistance), not just a COP scaling.**

**WRF gap.** WRF-BEM has the full version but requires switching from SLUCM to BEP+BEM (a big infrastructure change for the user).

---

### 7.20 — Radiative cooling paints / passive daytime radiative cooling (PDRC) surfaces

**Physics.** Emerging cool-surface technology: high LW emissivity in the atmospheric window (8–13 μm) combined with high SW reflectivity gives surfaces that cool *below* ambient during the day. Modeled via band splitting: `ε_LW_window` separated from `ε_LW_out_of_window`.

**Literature.** Raman et al. 2014; Mandal et al. 2018.

**Fit.** Extends the Phase 5.1c LW radiosity with band splitting; adds a per-material `epsilon_window` scalar in `materials.csv`.

**WRF gap.** No support anywhere in WRF (WRF cannot represent PDRC without a custom fork). **This would be a headline v1.1 feature — no other operational urban model supports it and it's an emerging urban intervention.**

---

### 7.21 — Arbitrary canyon orientation (not just E–W)

**Physics.** Current SVF and view-factor formulae assume the canyon runs east–west. Real canyons have arbitrary orientation. The angle between canyon axis and solar azimuth modulates wall shading and self-shadowing.

**Literature.** Masson 2000 (TEB); Mills 1997.

**Fit.** Adds one field `canyon_orientation_deg` to `building_layout.csv`; modifies SVF and radiosity view-factor formulae to include orientation. Multiple orientations per cell → weighted average.

**WRF gap.** WRF-SLUCM assumes E–W canyon; WRF-BEP handles multiple orientations only by averaging over four canonical directions.

---

### 7.22 — Human thermal comfort diagnostics (UTCI, PET, WBGT)

**Physics.** Post-SEB diagnostic (no feedback to physics) at pedestrian height (`z = 1.5 m`). Computes Universal Thermal Climate Index, Physiological Equivalent Temperature, and Wet Bulb Globe Temperature from canyon T, MRT (from radiosity — already computed by Phase 5.1c), wind, and humidity. Outputs to plotfile.

**Literature.** Bröde et al. 2012 (UTCI); Höppe 1999 (PET); WBGT ISO 7243.

**Fit.** Three new plotfile fields; reads existing state; no new state variables. **Cheap and extraordinarily high-impact for the urban-planning user community — one of the most-requested outputs for design-scenario studies. Instant differentiator vs. WRF.**

**WRF gap.** Available only in post-processing external tools; not integrated in-model.

---

### 7.23 — Turbulent kinetic energy source from urban shear

**Physics.** Adds explicit TKE source to the ATM from wall/roof drag: `S_TKE = Cd · u³ / L_scale`. Improves PBL prediction over cities in TKE-based schemes (MYNN, MYJ).

**Literature.** Santiago & Martilli 2010; Nazarian et al. 2018.

**Fit.** One line added to the Phase 2.8 drag kernel, injecting into the TKE MF when the PBL scheme supports it. Requires TKE-based PBL to be active (currently ERF-SLUCM validates on MRF, which is TKE-diagnostic — needs MYNN activation first).

**WRF gap.** WRF-BEP has this for MYNN only; not for MYJ or others.

---

## Tier 5 — Frontier / research-grade (publication material, v1.2+ or external collaborations)

### 7.24 — Machine-learning-based SEB residual correction

**Physics.** Train a small neural network on flux-tower residuals (Urban Flux Network data: Basel, Marseille, Tokyo, Melbourne, etc.). Apply as a per-facet correction to the Newton SEB output. Captures unresolved physics (subgrid mixing, local advection details, occupancy specifics) that no explicit model can economically represent.

**Literature.** Meyer et al. 2022; Wouters et al. 2019.

**Fit.** Only if you want to explore ML-in-loop. Extends the Newton output through a torch/ONNX runtime call. Requires a training pipeline and a curated flux-tower dataset.

**WRF gap.** No ML-in-loop urban physics anywhere in operational WRF.

**Value.** Research collaboration opportunity, likely 2+ papers, high visibility.

---

### 7.25 — Green wall / vertical vegetation on facets

**Physics.** Not just green roof — vegetation as a partial cover on wall facets. Modifies wall albedo (~0.15 → 0.25 with vegetation), LW emissivity (higher), adds an LE branch on the wall via transpiration. Substantially different heat exchange than bare wall.

**Literature.** Pérez et al. 2011; Djedjig et al. 2013.

**Fit.** Extends Phase 5.3 (green roof) and Phase 6.3 (tree transpiration) to the wall facet. New columns in `materials.csv` (`green_wall_fraction`, `LAD_wall`); wall facet's SEB gets a modified LE branch.

**WRF gap.** No support anywhere in WRF.

---

### 7.26 — Dynamic albedo from wall/road wetness (Fresnel-like reflection)

**Physics.** When puddles form (from Feature 7.8), road/wall albedo becomes view-angle dependent via Fresnel-like reflection — mirror-like at grazing angles, low-albedo at normal incidence. Explains why wet roads glare at sunset.

**Literature.** Kondo et al. 2001.

**Fit.** Composites with 7.8 (surface moisture) — same water-depth trigger drives an albedo modification that varies with zenith angle. One extra term in the SW radiosity RHS.

**WRF gap.** No support anywhere in WRF.

---

### 7.27 — Sub-diurnal traffic-based AH from calibrated vehicle counts

**Physics.** Extends 7.3 further: not just weekday/weekend distinction but calibrated hour-by-hour traffic-count CSV per cell (from municipal traffic-monitoring data or OSM road-class + typical traffic curves). Traffic AH dominates rush-hour peaks in dense urban cores.

**Literature.** Sailor & Lu 2004; Nie et al. 2014.

**Fit.** Another AH profile channel with a per-cell scale factor from `building_layout.csv`. Traffic AH is 30–100 W/m² in rush hours in dense urban.

**WRF gap.** WRF-UCM uses a single lumped diurnal AH per city; no per-cell traffic variation.

---

## Recommended Top 5 for v1.1

If curating a headline "v1.1 novel-features release" — weighted by (novelty vs WRF) × (ease given existing infrastructure) × (user community impact):

| Rank | Feature | Why it lands |
|------|---------|--------------|
| 1 | **7.14 — Multi-layer canyon-air** | Closes the SLUCM-vs-BEP gap for canyon thermal stratification; makes Phase 6.2b (tree crown) genuinely physical; reuses TDMA. This alone justifies a release. |
| 2 | **7.22 — Human thermal comfort (UTCI/PET/WBGT)** | Post-processing diagnostic, extraordinarily cheap, huge user demand for urban-planning studies. Instant differentiator. |
| 3 | **7.19 — Building envelope multi-material walls (BEM-lite)** | Turns SLUCM into SLUCM+BEM-lite; makes Phase 5.2 AC waste heat genuinely physical (indoor T + envelope resistance, not just COP scaling). |
| 4 | **7.16 — Kanda z0/d from building-height variance** | Two-line consumption of already-computed variance from Phase 2.5; makes ERF-SLUCM demonstrably more accurate than WRF-SLUCM on heterogeneous grids. Cheapest possible novelty win. |
| 5 | **7.13 — Urban water balance / precipitation interception** | Completes the moisture story; delivers realistic diurnal Bowen ratios that WRF-SLUCM cannot produce; pairs with Phase 5.3 to match SUEWS moisture path. |

## Honorable mentions for v1.2 or research collaboration

- **7.20 (PDRC / radiative cooling paints)** — highest-novelty single feature; targets emerging urban intervention no operational model supports. Headline v1.2 feature.
- **7.11 (vertical Cd variation)** — lightweight lead-in to full BEP-Tree; pairs well with 7.14.
- **7.7 (direct/diffuse SW split)** — natural refinement once Phase 5.1c radiosity is stable.
- **7.24 (ML-based SEB residual correction)** — 2+ papers; requires flux-tower dataset curation.

## Deferred / de-prioritized items

Not on this list, but discussed and consciously excluded:

- **Full multi-band SW spectrum** — infrastructure cost too high for marginal benefit; the analytic + cloud path is adequate for Phase 4.x.
- **Coupled indoor air quality (CO2, PM2.5)** — outside SLUCM scope; belongs in a chemistry-transport model coupling.
- **Explicit building-shadow ray-tracing** — infrastructure cost too high; the analytic SVF + view-factor approach captures 90% of the effect at <1% of the cost.
- **Urban precipitation feedback (heat island → convective triggering)** — a coupling-side concern, not UCM-side.

---

## Implementation Notes

**For all Tier 1–2 features:** design new fields as per-cell CSV columns in existing readers (`materials.csv`, `building_layout.csv`) whenever possible. Preserve the "per-cell heterogeneity as first-class" design philosophy that distinguishes ERF-SLUCM from WRF-UCM.

**For all Tier 3–4 features:** add an explicit ParmParse selector `erf.ucm.<feature>_mode = {off, simple, full}` with `off` bit-identical to prior behavior. This is the pattern established by Contract #14 (cloud source), Contract #18 (radiosity mode), etc.

**For all Tier 5 features:** these are research collaborations by design. Treat as external contribution paths — provide clean plugin points in the SEB architecture (e.g., a `Vector<PostSEBHook>` that ML corrections, PDRC band-splitting, etc. can register into) rather than hard-coding.

**Documentation contract:** every landed feature must add to this document with the primary references, the resolved parameter defaults, and any new design contracts (numbered continuing from the master contracts index).

---

## References (consolidated)

- Best, M. J., and C. S. B. Grimmond (2014), Importance of initial state and atmospheric conditions for urban land surface models' performance, *Urban Climate*, 10, 387–406.
- Bröde, P., et al. (2012), Deriving the operational procedure for the Universal Thermal Climate Index (UTCI), *Int. J. Biometeorol.*, 56, 481–494.
- Coceal, O., and S. E. Belcher (2004), A canopy model of mean winds through urban areas, *Q. J. R. Meteorol. Soc.*, 130, 1349–1372.
- de Munck, C., et al. (2013), How much can air conditioning increase air temperatures for a city like Paris?, *Int. J. Climatol.*, 33, 210–227.
- Djedjig, R., et al. (2013), Development and validation of a coupled heat and mass transfer model for green roofs, *Int. Commun. Heat Mass Transf.*, 47, 65–72.
- Erbs, D. G., et al. (1982), Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation, *Sol. Energy*, 28, 293–302.
- Grimmond, C. S. B., H. A. Cleugh, and T. R. Oke (1991), An objective urban heat storage model and its comparison with other schemes, *Atmos. Environ. B*, 25, 311–326.
- Grimmond, C. S. B., and T. R. Oke (1991), An evapotranspiration-interception model for urban areas, *Water Resour. Res.*, 27, 1739–1755.
- Harman, I. N., and J. J. Finnigan (2007), A simple unified theory for flow in the canopy and roughness sublayer, *Bound.-Layer Meteorol.*, 123, 339–363.
- Höppe, P. (1999), The physiological equivalent temperature — a universal index for the biometeorological assessment of the thermal environment, *Int. J. Biometeorol.*, 43, 71–75.
- Iamarino, M., et al. (2012), A high-resolution anthropogenic heat emissions inventory for the London region, *Int. J. Climatol.*, 32, 1754–1767.
- Järvi, L., et al. (2011), The Surface Urban Energy and Water Balance Scheme (SUEWS), *J. Hydrol.*, 411, 219–237.
- Järvi, L., et al. (2014), Development of the Surface Urban Energy and Water balance Scheme (SUEWS) for cold climate cities, *Geosci. Model Dev.*, 7, 1691–1711.
- Kanda, M. (2006), Progress in the scale modeling of urban climate, *Theor. Appl. Climatol.*, 84, 23–33.
- Kanda, M., et al. (2013), A new aerodynamic parametrization for real urban surfaces, *Bound.-Layer Meteorol.*, 148, 357–377.
- Kondo, J., et al. (2001), A numerical experiment of the "extended mixed layer" over a heterogeneous surface, *J. Meteorol. Soc. Japan*, 79, 587–597.
- Krayenhoff, E. S., et al. (2015), A multi-layer radiation model for urban neighbourhoods with trees, *Bound.-Layer Meteorol.*, 151, 139–178.
- Levinson, R., and H. Akbari (2002), Effects of composition and exposure on the solar reflectance of Portland cement concrete, *Cement Concrete Res.*, 32, 1679–1698.
- Mandal, J., et al. (2018), Hierarchically porous polymer coatings for highly efficient passive daytime radiative cooling, *Science*, 362, 315–319.
- Martilli, A., et al. (2002), An urban surface exchange parameterisation for mesoscale models, *Bound.-Layer Meteorol.*, 104, 261–304.
- Masson, V. (2000), A physically-based scheme for the urban energy budget in atmospheric models, *Bound.-Layer Meteorol.*, 94, 357–397.
- Meyer, D., et al. (2022), Machine learning applications for urban modelling, *Environ. Model. Softw.*, 149, 105269.
- Meyn, S. K., and T. R. Oke (2009), Heat fluxes through roofs and their relevance to estimates of urban heat storage, *Energy Build.*, 41, 745–752.
- Mills, G. (1997), An urban canopy-layer climate model, *Theor. Appl. Climatol.*, 57, 229–244.
- Millward-Hopkins, J. T., et al. (2011), Estimating aerodynamic parameters of urban-like surfaces with heterogeneous building heights, *Bound.-Layer Meteorol.*, 141, 443–465.
- Nazarian, N., and J. Kleissl (2016), Realistic solar heating in urban areas: air exchange and street-canyon ventilation, *Build. Environ.*, 95, 75–93.
- Nazarian, N., et al. (2018), Vertical resolution of a large-eddy simulation of a heated urban canopy, *Bound.-Layer Meteorol.*, 166, 47–67.
- Nie, W., et al. (2014), Assessment of urban fluxes of anthropogenic heat in Chinese megacities, *Sci. China Earth Sci.*, 57, 2860–2870.
- Perez, R., et al. (1990), Modeling daylight availability and irradiance components from direct and global irradiance, *Sol. Energy*, 44, 271–289.
- Pérez, G., et al. (2011), Green vertical systems for buildings as passive systems for energy savings, *Appl. Energy*, 88, 4854–4859.
- Raman, A., et al. (2014), Passive radiative cooling below ambient air temperature under direct sunlight, *Nature*, 515, 540–544.
- Ramamurthy, P., and E. Bou-Zeid (2014), Contribution of impervious surfaces to urban evaporation, *Water Resour. Res.*, 50, 2889–2902.
- Reindl, D. T., et al. (1990), Diffuse fraction correlations, *Sol. Energy*, 45, 1–7.
- Rosado, P. J., et al. (2017), Influence of street trees on residential energy use in California, *Urban Forestry Urban Greening*, 21, 111–121.
- Sailor, D. J. (2011), A review of methods for estimating anthropogenic heat and moisture emissions in the urban environment, *Int. J. Climatol.*, 31, 189–199.
- Sailor, D. J., and L. Lu (2004), A top-down methodology for developing diurnal and seasonal anthropogenic heating profiles for urban areas, *Atmos. Environ.*, 38, 2737–2748.
- Salamanca, F., et al. (2010), A new building energy model coupled with an urban canopy parameterization for urban climate simulations, *Theor. Appl. Climatol.*, 99, 331–344.
- Salamanca, F., et al. (2014), Anthropogenic heating of the urban environment due to air conditioning, *J. Geophys. Res. Atmos.*, 119, 5949–5965.
- Santiago, J. L., and A. Martilli (2010), A dynamic urban canopy parameterization for mesoscale models based on computational fluid dynamics Reynolds-averaged Navier–Stokes microscale simulations, *Bound.-Layer Meteorol.*, 137, 417–439.
- Santiago, J. L., et al. (2013), CFD simulation of flow and dispersion around individual buildings with different orientation angles, *Environ. Fluid Mech.*, 13, 631–648.
- Sen, S., and J. Roesler (2016), Aging albedo model for asphalt pavement surfaces, *J. Cleaner Prod.*, 117, 169–175.
- Simón-Moral, A., et al. (2017), Sensitivity of radiation and turbulent fluxes to the vertical distribution of urban roughness elements, *Q. J. R. Meteorol. Soc.*, 143, 2653–2664.
- Sun, T., and C. S. B. Grimmond (2019), A Python-enhanced urban land surface model SuPy (SUEWS in Python), *Geosci. Model Dev.*, 12, 2781–2795.
- Theeuwes, N. E., et al. (2013), Modeling the influence of open water surfaces on the summertime temperature and thermal comfort in the city, *J. Geophys. Res. Atmos.*, 118, 8881–8896.
- Wang, Z. H., et al. (2013), A coupled energy transport and hydrological model for urban canopies evaluated using a wireless sensor network, *Q. J. R. Meteorol. Soc.*, 139, 1643–1657.
- Wouters, H., et al. (2019), Analytical support for the derivation of the urban canopy parameters used in the Community Land Model, *Geosci. Model Dev.*, 12, 4519–4544.

---

*Document status: living catalog. Update alongside any Phase 7+ implementation. Cross-reference with `UCM_DEVELOPMENT.md` for the master roadmap.*