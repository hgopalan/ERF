# UCM Post-v1.0 Roadmap

**Scope:** Complete roadmap for post-v1.0 development. Phase 7 is a dedicated data-center research track; Phases 8+ are thematic groupings of remaining features.

**Prerequisites:** v1.0 = Phase 6.4b complete (per-cell LAI + light-response stomata + Contract #38 SLUCM invariant).

**Status:** Roadmap only. No commitment to build until explicit go-ahead per phase.

---

## Phase 7 — Data centers (headline research track)

Every data-center-related feature, consolidated. This is a coherent research thread that could produce a dedicated paper series.

### Phase 7.1 — Release infrastructure

- **7.1a** — Worry-list audit + regression consolidation
- **7.1b** — Documentation + release CI + v1.0 tag
- **7.1c** — Zenodo DOI + code archival + citation file
- **7.1d** — GMD paper submission prep (demo canonicals frozen, plotting utilities)

### Phase 7.2 — Data-center foundation

- **7.2a** — Data-center CSV schema (`data_center.csv`, `dc_profile_id` in building_layout)
- **7.2b** — PUE model (Power Usage Effectiveness): IT_power → total_power via PUE
- **7.2c** — Sensible/latent split of rejected heat (dry-cooled vs evaporative cooling)
- **7.2d** — Static rejection-height injection (heat enters ATM at configurable z_reject)

### Phase 7.3 — Data-center feedbacks

- **7.3a** — T_air-dependent PUE (cooling less efficient in hot ambient)
- **7.3b** — Chiller COP degradation vs T_wetbulb (requires wet-bulb estimator)
- **7.3c** — Cooling-tower plume contribution to local LW downwelling (elevates sky emissivity near DCs)
- **7.3d** — Data-center IT-power → indoor T coupling (requires indoor T prognostic from 9.1c; scaffold now, wire when 9.1c lands)

### Phase 7.4 — Data-center plume dynamics

- **7.4a** — Analytic Briggs-style plume rise (buoyant plume from cooling towers reaches variable height)
- **7.4b** — Wind-tilted plume trajectory (advection of plume by ATM wind)
- **7.4c** — Multi-plume interaction (adjacent DCs' plumes merge)
- **7.4d** — Plume dispersion into ATM (Gaussian spread from plume centerline)

### Phase 7.5 — Data-center system-level features

- **7.5a** — Waste-heat recovery / district heating export (Q_reject scaled by recovery_frac)
- **7.5b** — DC-cluster growth scenario CSV (per-year IT-power multiplier for future scenarios)
- **7.5c** — Redundancy / N+1 cooling capacity modeling (peak-load safety margin)
- **7.5d** — Failure-mode scenarios (chiller outage → emergency air-side economizer → sensible flux spike)

### Phase 7.6 — Data-center validation & applications

- **7.6a** — Comparison against operational DC data (if available via industry partners)
- **7.6b** — Singapore DC cluster scenario runs (real-world case study)
- **7.6c** — Climate-adaptation scenarios (2050 warming + DC growth → UHI impact)
- **7.6d** — Policy diagnostics (per-DC emissions, water usage, heat rejection budget)

**v1.1 release target:** Phases 7.1–7.3 (DC foundation + feedbacks). Sufficient for first DC paper.
**v1.2 release target:** Phases 7.4–7.6 (plume dynamics + validation). Full DC research capability.

---

## Phase 8 — Anthropogenic heat + traffic + emissions

Non-DC anthropogenic sources, consolidated.

### Phase 8.1 — Temporal AH profiles

- **8.1a** — Weekday / weekend / holiday AH profiles _(was 7.3 in old roadmap)_
- **8.1b** — Sub-diurnal traffic-based AH from vehicle counts _(was 7.27)_
- **8.1c** — Seasonal / heat-wave adaptive AC operation _(was 7.5)_
- **8.1d** — Occupancy-driven office/residential AH split

### Phase 8.2 — Emissions accounting

- **8.2a** — Emissions from AH + HVAC + IT power (CO₂, requires grid-mix CSV) _(was 7.11d)_
- **8.2b** — Passive CO₂ tracer transport from AH cells _(was 7.14a; consolidates deferred Phase 6.5A)_
- **8.2c** — Building emissions inventory hookup (per-cell CO₂ from HVAC + IT) _(was 7.14b; consolidates deferred Phase 6.5B)_
- **8.2d** — Passive scalar advection in canyon air (canyon CO₂ budget)

### Phase 8.3 — Air quality

- **8.3a** — PM2.5 from traffic-density CSV _(was 7.14c)_
- **8.3b** — NOx tracer from combustion AH
- **8.3c** — Ozone precursor tracking (VOC + NOx co-emission)
- **8.3d** — Chemistry-transport model coupling stub (plugin interface only) _(was 7.14d)_

---

## Phase 9 — Building envelope + indoor coupling (BEM-lite)

### Phase 9.1 — Envelope prognostics

- **9.1a** — Multi-material wall slabs (BEM-lite outer/insulation/inner) _(was 7.19)_
- **9.1b** — Ground heat flux to deeper subsurface / dynamic T_deep _(was 7.18)_
- **9.1c** — Indoor air temperature prognostic (single-zone; couples Phase 5.2 HVAC to 9.1a envelope)
- **9.1d** — Multi-zone indoor T (per-floor split for tall buildings)

### Phase 9.2 — Envelope refinements

- **9.2a** — Window / fenestration SW/LW transmission
- **9.2b** — Thermal bridging at wall-floor junctions
- **9.2c** — Radiant heating/cooling systems (indoor surface T ≠ air T)
- **9.2d** — Occupant thermal load (metabolic heat + latent from breathing)

---

## Phase 10 — Canyon air structure (biggest physics upgrade)

⚠ **Contract #38 review required before implementation.** Multi-layer canyon violates the current SLUCM invariant. Either retire Contract #38 or spin off `Source/UrbanCanopyMultiLayer/` as a separate model track.

### Phase 10.1 — Vertical canyon structure

- **10.1a** — Multi-layer canyon air `T_canyon_air(k)` _(was 7.14)_
- **10.1b** — Canyon humidity `q_canyon_air(k)` companion to 10.1a
- **10.1c** — Vertical variation of Cd within canyon / BEP-Tree P-parameter _(was 7.11)_
- **10.1d** — Per-height wall SEB (T_wall(z) — full BEP-style)

### Phase 10.2 — Horizontal canyon coupling

- **10.2a** — Local advection between adjacent UCM cells _(was 7.15)_
- **10.2b** — Arbitrary canyon orientation _(was 7.21)_
- **10.2c** — Canyon corner effects (2×2 intersection physics)
- **10.2d** — Two-way coupling to building-resolving CFD at envelope

### Phase 10.3 — Alternative: diagnostic profile (Contract #38-preserving)

If Contract #38 is retained instead of retired, this sub-phase delivers vertical profiles as diagnostics only:

- **10.3a** — Log-linear diagnostic T_canyon(z) profile _(was Phase 6.6 in earlier roadmap)_
- **10.3b** — Krayenhoff-Voogt empirical profile (H/W-aware)
- **10.3c** — BEP-calibrated fit for T_canyon(z) shape (if BEP reference runs available)
- **10.3d** — Pedestrian-height (2 m) diagnostic output for downstream biometeorology coupling

---

## Phase 11 — Vegetation extensions (assumes Part 6 tree canopy landed)

### Phase 11.1 — Additional vegetation types

- **11.1a** — Green wall / vertical vegetation on facets _(was 7.25)_
- **11.1b** — Green roof refinements (already in Phase 5.3; polish + extended validation)
- **11.1c** — Extensive vs intensive green roof differentiation
- **11.1d** — Permeable pavement vegetation (grass pavers)

### Phase 11.2 — Species and phenology

- **11.2a** — Tree species → LAI/stomatal library (bulk lookup by species_id)
- **11.2b** — Deciduous seasonal LAI cycle
- **11.2c** — Species-specific Vcmax, Jmax parameter database (uses Phase 11.4 if built)
- **11.2d** — Tropical vs temperate species differentiation

### Phase 11.3 — Water balance

- **11.3a** — Soil moisture bucket for trees _(consolidates deferred Phase 6.5E-2)_
- **11.3b** — Irrigation stream for urban vegetation (soil moisture forcing)
- **11.3c** — Root-zone moisture profile (below bucket)
- **11.3d** — Runoff to storm-drain / groundwater interface

### Phase 11.4 — Advanced biophysics

- **11.4a** — Sunlit/shaded canopy partitioning _(consolidates deferred Phase 6.5E-1)_
- **11.4b** — Bulk-canopy Farquhar photosynthesis (diagnostic) _(consolidates deferred Phase 6.5C)_
- **11.4c** — Ball-Berry stomatal conductance (replaces Jarvis when enabled) _(consolidates deferred Phase 6.5D)_
- **11.4d** — Coupled biosphere: photosynthesis + stomata + CO2 uptake (requires Phase 8.2)

**Note:** Phase 11.4 is the "EnviMET-parity" biosphere work originally scoped in Phase 6.5C/D/E. Deferred here because no LANDSAT-observable target discriminates these models. Build when eddy-covariance or MODIS GPP validation becomes available.

---

## Phase 12 — Snow and wet surfaces

### Phase 12.1 — Snow

- **12.1a** — Snow cover on roof / road _(was 7.4)_
- **12.1b** — Snow albedo aging (fresh snow vs old snow)
- **12.1c** — Snow melt + refreeze cycle
- **12.1d** — Snowpack thermal insulation of underlying surface

### Phase 12.2 — Wet surfaces

- **12.2a** — Surface moisture on non-vegetated surfaces (wet pavement, dew) _(was 7.8)_
- **12.2b** — Dynamic albedo from wall/road wetness (Fresnel-like) _(was 7.26)_
- **12.2c** — Urban precipitation interception + rooftop runoff _(was 7.13)_
- **12.2d** — Storm-drain / urban hydrology coupling

---

## Phase 13 — Radiation refinements

### Phase 13.1 — SW improvements

- **13.1a** — Diffuse SW component & direct/diffuse split _(was 7.7)_
- **13.1b** — Anisotropic diffuse SW (Perez model) _(was 7.17)_
- **13.1c** — Anisotropic pavement albedo (retroreflective) _(was 7.2)_
- **13.1d** — Cool pavement thermal fatigue / aging albedo _(was 7.9)_

### Phase 13.2 — LW improvements

- **13.2a** — Spectral LW band splitting (8–13 μm atmospheric window)
- **13.2b** — Storage-heat-flux memory / OHM hysteresis _(was 7.10)_
- **13.2c** — Sky-view-factor refinements (obstacle-aware LW exchange)
- **13.2d** — 3D LW radiosity between building faces

### Phase 13.3 — Advanced radiative surfaces

- **13.3a** — Passive daytime radiative cooling (PDRC) surfaces _(was 7.20; requires 13.2a)_
- **13.3b** — Photovoltaic roof surfaces (electricity export + reduced sensible flux)
- **13.3c** — Photovoltaic array shading on ground (parking-lot PV canopy affects road SEB)
- **13.3d** — Solar water heater roof coupling

---

## Phase 14 — Aerodynamics + roughness

### Phase 14.1 — Roughness parameterization

- **14.1a** — Kanda z0/d from building-height variance _(was 7.16)_
- **14.1b** — Roughness sublayer displacement correction _(was 7.12)_
- **14.1c** — Wind-direction-dependent wall drag (windward vs leeward Cd) _(was 7.6)_
- **14.1d** — Frontal area index computation from building_layout

### Phase 14.2 — Turbulence

- **14.2a** — TKE source from urban shear _(was 7.23)_
- **14.2b** — Canopy-generated TKE from tree wakes
- **14.2c** — Building-wake turbulence parameterization
- **14.2d** — Turbulent transport of scalars in canyon (K-theory closure)

---

## Phase 15 — Human & policy diagnostics

### Phase 15.1 — Thermal comfort

- **15.1a** — UTCI / PET / WBGT thermal comfort _(was 7.22)_
- **15.1b** — Heat-stress hours / degree-hours diagnostic (integrated over sim time)
- **15.1c** — Cold-stress diagnostics (winter applications)
- **15.1d** — Vulnerability-weighted exposure (elderly, low-income neighborhoods)

### Phase 15.2 — Energy diagnostics

- **15.2a** — Energy-use intensity (EUI) diagnostic per building (integrated HVAC + IT power)
- **15.2b** — Peak-load timing per building type
- **15.2c** — Grid-mix carbon accounting (per-hour grid CO₂ intensity)
- **15.2d** — Renewable-energy offset accounting (PV + wind local generation)

### Phase 15.3 — Water diagnostics

- **15.3a** — Water use for cooling (DC + AC evaporative)
- **15.3b** — Stormwater management (runoff volumes, flooding risk)
- **15.3c** — Irrigation water demand (Phase 11.3b integration)
- **15.3d** — Total urban water balance closure

---

## Phase 16 — Machine learning & data assimilation

### Phase 16.1 — ML augmentation

- **16.1a** — ML-based SEB residual correction _(was 7.24)_
- **16.1b** — Neural surrogate for slow processes (soil heat, envelope conduction)
- **16.1c** — Bayesian parameter estimation from observations
- **16.1d** — Automated tuning of species/material parameters

### Phase 16.2 — Data assimilation

- **16.2a** — Flux-tower dataset curation (Basel, Marseille, Tokyo, Melbourne)
- **16.2b** — Observation nudging of T_canyon_air to ground-station data
- **16.2c** — Satellite LST assimilation (LANDSAT, MODIS, Sentinel)
- **16.2d** — Ensemble Kalman filter for state correction

### Phase 16.3 — Uncertainty quantification

- **16.3a** — Perturbed-parameter ensemble wrapper
- **16.3b** — Structural uncertainty (multi-model ensembles)
- **16.3c** — Sensitivity analysis (Sobol indices for key parameters)
- **16.3d** — Confidence bounds on diagnostics

---

## Phase 17 — Multi-scale / nesting (research)

### Phase 17.1 — Downscaling

- **17.1a** — LES-scale UCM (dx_ucm ≤ 50 m) — verify canyon-average physics still hold
- **17.1b** — UCM inside AMR-refined ATM patches (dynamic refinement over UHI hotspots)
- **17.1c** — Two-way coupling to a building-resolving model (CFD handoff at building envelope)
- **17.1d** — Nested urban-rural transition physics

### Phase 17.2 — Upscaling & regional

- **17.2a** — Regional climate coupling (10s of cities in one domain)
- **17.2b** — Coastal/marine boundary interaction (sea breeze into urban)
- **17.2c** — Mountain-urban interaction (topographic slope flows)
- **17.2d** — Long-term climate projection wrapper (multi-year continuous integration)

### Phase 17.3 — Ensemble scenarios

- **17.3a** — Ensemble stochastic UCM parameters for spread quantification
- **17.3b** — Multi-city scenario framework (Singapore + Jakarta + Bangkok comparison)
- **17.3c** — Climate-adaptation scenario matrix (albedo × vegetation × DC growth)
- **17.3d** — Policy intervention what-if runs

---

## Release trajectory

**v1.0** — Phase 6.4b complete. Core SLUCM+crown production-ready. First public release.

**v1.1** — Phase 7.1–7.3 (DC foundation + feedbacks). Headline: "First urban canopy model with integrated data center heat rejection physics."

**v1.2** — Phase 7.4–7.6 (DC plume dynamics + validation). Headline: "Data-center-aware urban climate for planning."

**v1.3** — Phase 8 (AH + emissions) + Phase 15 (diagnostics). Headline: "Policy-ready urban climate model."

**v2.0** — Phase 9 (BEM-lite) + Phase 12 (snow/wet) + Phase 13 (radiation). Headline: "Comprehensive urban surface physics."

**v2.1** — Phase 11 (vegetation extensions including deferred 6.5 biophysics). Headline: "EnviMET-parity urban ecology."

**v3.0** — Phase 10 (canyon vertical structure). Requires Contract #38 decision. Headline: "Multi-layer canyon capability."

**v3.1+** — Phases 14, 16, 17 (aerodynamics refinements, ML/DA, multi-scale). Ongoing research directions.

---

## Cross-references to `UCM_DEVELOPMENT.md`

**Deferred Phase 6 backlog items now consolidated into Phase 8/11:**
- Phase 6.5A/B (CO2 passive scalar) → **Phase 8.2b–8.2d**
- Phase 6.5C (Farquhar photosynthesis) → **Phase 11.4b**
- Phase 6.5D (Ball-Berry) → **Phase 11.4c**
- Phase 6.5E-1 (sunlit/shaded) → **Phase 11.4a**
- Phase 6.5E-2 (soil moisture bucket) → **Phase 11.3a**
- Phase 6.6 (diagnostic canyon profile) → **Phase 10.3** (as Contract #38-preserving alternative)

**Contracts that may need revision:**
- **Contract #38** (SLUCM architectural invariant) — MUST be reviewed before Phase 10.1–10.2. Either retire or spin off `Source/UrbanCanopyMultiLayer/`.
- **Contract #29** (3-var byte-identity) — applies to all future features. Any new gate must default to preserving prior behavior.
- **Contract #37** (opt-in preservation) — applies to all future features.

---

## Coverage check — original 27 features + new additions

| Original | New location | Original | New location |
|---|---|---|---|
| 7.2 anisotropic pavement | **13.1c** | 7.16 Kanda z0/d | **14.1a** |
| 7.3 weekday/weekend AH | **8.1a** | 7.17 Perez diffuse | **13.1b** |
| 7.4 snow | **12.1a** | 7.18 T_deep | **9.1b** |
| 7.5 adaptive AC | **8.1c** | 7.19 BEM-lite walls | **9.1a** |
| 7.6 windward/leeward Cd | **14.1c** | 7.20 PDRC | **13.3a** |
| 7.7 direct/diffuse SW | **13.1a** | 7.21 canyon orientation | **10.2b** |
| 7.8 surface moisture | **12.2a** | 7.22 UTCI/PET/WBGT | **15.1a** |
| 7.9 albedo aging | **13.1d** | 7.23 TKE source | **14.2a** |
| 7.10 OHM storage | **13.2b** | 7.24 ML residual | **16.1a** |
| 7.11 vertical Cd | **10.1c** | 7.25 green wall | **11.1a** |
| 7.12 RSL correction | **14.1b** | 7.26 wet-surface Fresnel | **12.2b** |
| 7.13 precip/runoff | **12.2c** | 7.27 traffic AH | **8.1b** |
| 7.14 multi-layer canyon | **10.1a** | data centers | **Phase 7 entire** |
| 7.15 UCM advection | **10.2a** | Phase 6.5A-E biophysics | **Phase 8.2, 11.3, 11.4** |

All 27 original features + all deferred Phase 6.5 backlog + all data-center features covered.