/**
 * @file ERF_DustLayer.cpp
 * @brief Implementation of the DustLayer container class.
 */

#include <ERF_DustLayer.H>
#include <fstream>
#include <cstdio>
#include <ERF_DustPrerequisites.H>
#include <ERF_DustGrid.H>
#include <ERF_DustSurfaceReader.H>
#include <ERF_PhreeqcReader.H>
#include <ERF_DustThreshold.H>
#include <ERF_DustTerrainSlope.H>
#include <ERF_DustEmission.H>
#include <ERF_DustSuppression.H>
#include <ERF_DustWindExtract.H>
#include <ERF_DustAtmCoupling.H>
#include <ERF_DustFireLofting.H>
#include <ERF_DustMSHA.H>
#include <ERF_DustMSHAOutput.H>
#include <ERF_FireWindExtract.H>
#include <ERF.H>
#include <ERF_SurfaceLayer.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFabUtil.H>
#include <cmath>

void
DustLayer::initialize(
  const ERF& erf,
  const SurfaceLayer* surface_layer,
  const amrex::MultiFab& z_phys_nd_atm,
  const DustParams& dust_params)
{
  verify_dust_prerequisites(erf, surface_layer, dust_params);
  m_geom_atm = erf.Geom(0);   // the coarsening to the atmosphere columns needs it before any advance (restart)

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 1: Verified all prerequisites\n";
  }

  m_dg = create_dust_grid(
    erf.boxArray(0), erf.DistributionMap(0), erf.Geom(0),
    dust_params.grid_ratio);

  if (dust_params.dust_debug) {
    amrex::Box dust_domain = m_dg.ba.minimalBox();
    int dust_nx = dust_domain.length(0);
    int dust_ny = dust_domain.length(1);
    amrex::Print() << "[DUST DEBUG] Created dust grid: " << dust_nx << "x"
                   << dust_ny << "x1 cells, "
                   << "grid_ratio=" << dust_params.grid_ratio << "\n";
    const amrex::Box& dom = m_dg.geom.Domain();
    amrex::Print() << "[DUST DEBUG] Phase 2: grid extent [" << dom.smallEnd()
                   << "," << dom.bigEnd()
                   << "] grid_ratio=" << dust_params.grid_ratio
                   << " boxes=" << m_dg.ba.size() << "\n";
  }

  m_params = dust_params;

  amrex::IntVect ng(1, 1, 0);
  dust_slopes        = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 2, ng);
  dust_curvature     = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, 0);
  dust_slopes->setVal(0.0);
  dust_curvature->setVal(0.0);
  compute_dust_terrain_slopes(
    *dust_slopes, z_phys_nd_atm, erf.Geom(0), m_dg, dust_params.terrain_file);
  dust_fill_boundary(*dust_slopes, m_dg.geom);
  compute_terrain_curvature(*dust_curvature, *dust_slopes, m_dg.geom);

  dust_ustar_t       = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_soil_type     = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_silt_fraction = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_crust_index   = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_moisture_flag = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_suppression   = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_emission_flux = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, dust_params.n_size_bins, ng);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Allocated MultiFabs: "
                   << "dust_ustar_t (1 comp), " << "dust_soil_type (1 comp), "
                   << "dust_silt_fraction (1 comp), "
                   << "dust_crust_index (1 comp), "
                   << "dust_moisture_flag (1 comp), "
                   << "dust_suppression (1 comp), " << "dust_emission_flux ("
                   << dust_params.n_size_bins << " comp)\n";
  }

  // The threshold belongs to the SALTATING grains (erf.dust.saltation_diameter,
  // 75 um by default), not to the emitted dust bins: Bagnold's formula at the
  // 7 um bin-0 diameter gave 0.0385 m/s, 13x below Shao-Lu at that size, and
  // every case ran saturated. The deck's air density is the one the emission
  // flux uses; erf.dust.ustar_t_base >= 0 replaces the formula.
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!dust_params.bin_diameters.empty(),
      "[DUST] erf.dust.bin_diameters must list at least one bin diameter [m]");
  const amrex::Real d_salt = dust_params.saltation_diameter;
  amrex::Real ustar_t = (dust_params.threshold_model == "bagnold")
      ? compute_ustar_t_bagnold(dust_params.threshold_A_coeff, dust_params.particle_density,
                                d_salt, dust_params.rho_air)
      : compute_ustar_t_shao_lu(dust_params.shao_lu_A_N, dust_params.shao_lu_gamma,
                                dust_params.particle_density, d_salt, dust_params.rho_air);
  if (dust_params.ustar_t_base >= 0.0) ustar_t = dust_params.ustar_t_base;
  dust_ustar_t->setVal(ustar_t);

  amrex::Print() << "[DUST] Base threshold u*_t = " << ustar_t << " m/s ("
                 << ((dust_params.ustar_t_base >= 0.0) ? std::string("erf.dust.ustar_t_base")
                                                        : dust_params.threshold_model)
                 << ", saltation diameter " << d_salt * 1.0e6 << " um, rho_p="
                 << dust_params.particle_density << " kg/m^3, rho_a=" << dust_params.rho_air << ")\n";

  dust_soil_type->setVal(0.0);
  dust_silt_fraction->setVal(dust_params.silt_fraction);
  dust_crust_index->setVal(dust_params.crust_index);
  dust_moisture_flag->setVal(0.0);
  dust_suppression->setVal(0.0);
  dust_emission_flux->setVal(0.0);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Set initial values: "
                   << "dust_soil_type=0.0, "
                   << "dust_silt_fraction=" << dust_params.silt_fraction << ", "
                   << "dust_crust_index=" << dust_params.crust_index << ", "
                   << "dust_moisture_flag=0.0, " << "dust_suppression=0.0, "
                   << "dust_emission_flux=0.0\n";
  }

  dust_efflor = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_ustar_base = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_efflor->setVal(0.0);
  amrex::MultiFab::Copy(
    *dust_ustar_base, *dust_ustar_t, 0, 0, 1, amrex::IntVect(1, 1, 0));

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Allocated internal MultiFabs: "
                   << "dust_efflor (1 comp), " << "dust_ustar_base (1 comp)\n";
  }

  dust_surf_moist = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_surf_moist->setVal(0.0);
  dust_surf_qflux = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_surf_qflux->setVal(0.0);
  dust_ustar_fire = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, ng);
  dust_ustar_fire->setVal(0.0);

  dust_ustar_in = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_ustar_in->setVal(dust_params.test_ustar);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 6: dust_ustar_in (placeholder) = "
                   << dust_params.test_ustar << " m/s\n";
  }

  dust_retreat_flag = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_retreat_flag->setVal(0.0);
  dust_treated_mask = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_treated_mask->setVal(0.0);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 8: dust_retreat_flag allocated, "
                   << "supp_tau_base_s=" << dust_params.supp_tau_base_s
                   << " s, test_surf_temp_K=" << dust_params.test_surf_temp_K
                   << " K, test_wind_speed=" << dust_params.test_wind_speed
                   << " m/s\n";
  }

  dust_wind_ref = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 2, amrex::IntVect(1, 1, 0));
  dust_pblh = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_tsfc = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1, 1, 0));
  dust_wind_ref->setVal(0.0);
  dust_pblh->setVal(0.0);
  dust_tsfc->setVal(dust_params.test_surf_temp_K);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 9: dust_wind_ref(2), dust_pblh, "
                      "dust_tsfc allocated\n";
    amrex::Print() << "[DUST DEBUG] Phase 9: zref=" << dust_params.zref
                   << " m for wind extraction\n";
  }

  {
    amrex::BoxArray ba_atm = erf.boxArray(0);
    amrex::Vector<amrex::Box> bl;
    for (int b = 0; b < ba_atm.size(); ++b) {
      amrex::Box bx = ba_atm[b];
      bx.setSmall(2, 0);
      bx.setBig(2, 0);
      bl.push_back(bx);
    }
    amrex::BoxArray ba2d(amrex::BoxList(std::move(bl)));
    dust_flux_atm = std::make_unique<amrex::MultiFab>(
      ba2d, erf.DistributionMap(0), 1, amrex::IntVect(1, 1, 0));
    dust_flux_atm->setVal(0.0);
  }

  m_dust_scalar_comp = RhoScalar_comp + 1;

  if (dust_params.dust_debug) {
    amrex::Print()
      << "[DUST DEBUG] Phase 10: dust_flux_atm allocated on atm grid."
      << " dust_scalar_comp=" << m_dust_scalar_comp << "\n";
  }

  dust_deposition_rate = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_deposition_rate->setVal(0.0);
  ustar_atm = std::make_unique<amrex::MultiFab>(
    dust_flux_atm->boxArray(), dust_flux_atm->DistributionMap(), 1, amrex::IntVect(0));
  ustar_atm->setVal(0.0);

  dep_flux_atm = std::make_unique<amrex::MultiFab>(
    dust_flux_atm->boxArray(),
    dust_flux_atm->DistributionMap(),
    1, amrex::IntVect(1,1,0));
  dep_flux_atm->setVal(0.0);
  dep_flux_step = std::make_unique<amrex::MultiFab>(
    dust_flux_atm->boxArray(),
    dust_flux_atm->DistributionMap(),
    1, amrex::IntVect(0));
  dep_flux_step->setVal(0.0);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 12: dust_deposition_rate"
                   << " and dep_flux_atm allocated\n";
  }

  dust_conc_sfc = std::make_unique<amrex::MultiFab>(
    m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_conc_sfc->setVal(0.0);
  dust_surf_moist->setVal(0.0);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 13: dust_conc_sfc and"
                   << " dust_surf_moist allocated on dust grid\n"
                   << "[DUST DEBUG] Phase 13: loading_feedback_coeff="
                   << dust_params.loading_feedback_coeff
                   << "\n";
  }

  dust_pm25      = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_pm10      = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_pm25_24h  = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_pm10_24h  = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_pm25_exceed = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_pm10_exceed = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  for (auto* mf : {dust_pm25.get(), dust_pm10.get(),
                   dust_pm25_24h.get(), dust_pm10_24h.get(),
                   dust_pm25_exceed.get(), dust_pm10_exceed.get()})
    mf->setVal(0.0);
  m_emitted_per_bin.assign(dust_params.n_size_bins, 0.0);   // whatever the averaging (it sat inside the window branch once)
  if (dust_params.averaging == "window") {
    m_pm25_window.define(m_dg.ba, m_dg.dm, 24, 86400.0);   // 24 hourly means
    m_pm10_window.define(m_dg.ba, m_dg.dm, 24, 86400.0);
    if (dust_params.stel_enable)
      m_stel_window.define(m_dg.ba, m_dg.dm, 15, dust_params.stel_averaging_s);   // 15 slots of the STEL window
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 17: PM2.5/PM10 MultiFabs allocated\n"
                   << "[DUST DEBUG] Phase 17: naaqs_file="
                   << dust_params.dust_naaqs_file << "\n";
  }

  dust_msha_dose      = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_msha_twa       = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_msha_exceed    = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_msha_shift_twa = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  for (auto* mf : {dust_msha_dose.get(), dust_msha_twa.get(),
                   dust_msha_exceed.get(), dust_msha_shift_twa.get()})
    mf->setVal(0.0);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 18: MSHA exposure MultiFabs allocated\n"
                   << "[DUST DEBUG] Phase 18: PEL=" << dust_params.msha_pel_mg_m3
                   << " mg/m3 (30 CFR 56.5001)\n"
                   << "[DUST DEBUG] Phase 18: shift_duration="
                   << dust_params.msha_shift_duration_s << " s\n"
                   << "[DUST DEBUG] Phase 18: n_receptors="
                   << dust_params.msha_receptor_names.size() << "\n";
    for (int r = 0; r < (int)dust_params.msha_receptor_names.size(); ++r) {
      amrex::Print() << "[DUST DEBUG] Phase 18: receptor " << r << ": "
                     << dust_params.msha_receptor_names[r] << " ("
                     << dust_params.msha_receptor_x[r] << ", "
                     << dust_params.msha_receptor_y[r] << ")\n";
    }
  }

#if defined(ERF_USE_PARTICLES)
  if (dust_params.enable_particles) {
    m_dust_pc = std::make_unique<ERFDustPC>(
        m_geom_atm, erf.DistributionMap(0), erf.boxArray(0));
    dust_source_map = std::make_unique<amrex::MultiFab>(
        m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
    dust_source_map->setVal(0.0);

    if (dust_params.dust_debug) {
      amrex::Print() << "[DUST DEBUG] Phase 19: ERFDustPC initialized"
                     << " on dust grid " << m_dg.ba.size() << " boxes\n";
    }
  }
#endif

  populate_dust_surface_maps(
    *dust_soil_type, *dust_silt_fraction, *dust_crust_index,
    *dust_moisture_flag, *dust_suppression, m_dg, dust_params);

  // Ensure ghost cells are synchronized after reading surface maps
  dust_crust_index->FillBoundary(m_dg.geom.periodicity());
  dust_silt_fraction->FillBoundary(m_dg.geom.periodicity());
  // cells that were treated at all: the re-treatment flag is for them (a cell
  // whose coverage decayed to zero used to lose the flag, and could not be
  // told from one never treated)
  for (amrex::MFIter mfi(*dust_treated_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.tilebox();
    auto tm = dust_treated_mask->array(mfi);
    auto sp = dust_suppression->const_array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
      tm(i,j,k) = (sp(i,j,k) > amrex::Real(0.0)) ? amrex::Real(1.0) : amrex::Real(0.0);
    });
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Surface maps populated: "
                   << "soil_type_file=\"" << dust_params.soil_type_file
                   << "\", " << "silt_fraction_file=\""
                   << dust_params.silt_fraction_file << "\", "
                   << "crust_index_file=\"" << dust_params.crust_index_file
                   << "\", " << "moisture_flag_file=\""
                   << dust_params.moisture_flag_file << "\", "
                   << "suppression_file=\"" << dust_params.suppression_file
                   << "\"\n";
    amrex::Print() << "[DUST DEBUG] Phase 3: silt_frac min="
                   << dust_silt_fraction->min(0)
                   << " max=" << dust_silt_fraction->max(0)
                   << " soil_type_max=" << dust_soil_type->max(0) << "\n";
  }

#ifdef ERF_USE_DUST
  if (!dust_params.blast_schedule_file.empty()) {
    load_blast_schedule(dust_params.blast_schedule_file, m_blast_schedule);
    m_has_blast_schedule = !m_blast_schedule.empty();
    if (m_has_blast_schedule) {
      amrex::Print() << "[DUST] Blast schedule loaded from: "
                     << dust_params.blast_schedule_file << "\n";
    }
    if (dust_params.dust_debug && m_has_blast_schedule) {
      amrex::Print() << "[DUST DEBUG] Phase 7: blast schedule loaded, "
                     << m_blast_schedule.events.size() << " events\n";
    }
  }

  // Phase 22: load haul road vehicle schedule, and count the cells each road
  // covers (the AP-42 mass rate is spread over them).
  load_road_schedule(dust_params.road_schedule_file, m_road_schedule);
  count_road_cells(m_road_schedule, m_dg.ba, m_dg.geom);
  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 22: road_schedule_file="
                   << dust_params.road_schedule_file
                   << " n_roads=" << m_road_schedule.roads.size() << "\n";
    for (int r = 0; r < (int)m_road_schedule.roads.size(); ++r) {
        const auto& ev = m_road_schedule.roads[r];
        amrex::Print() << "  road " << r << " name=" << ev.name
                       << " bbox=[" << ev.x_lo << "," << ev.y_lo << ","
                       << ev.x_hi << "," << ev.y_hi << "]"
                       << " W=" << ev.vehicle_weight_t << " t"
                       << " silt=" << ev.silt_pct << "%"
                       << " vkt=" << ev.vkt_per_h << "/h"
                       << " t=[" << ev.start_time_s << ","
                       << ev.end_time_s << "] s\n";
    }
  }
#endif

  recompute_dust_ustar_t(
    *dust_ustar_t, *dust_ustar_base, *dust_crust_index, *dust_efflor,
    *dust_surf_moist, *dust_suppression, dust_params.alpha_crust,
    dust_params.alpha_efflor, dust_slopes.get());

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 5: u*_t after init modifiers: "
                   << "min=" << dust_ustar_t->min(0) << " m/s, "
                   << "max=" << dust_ustar_t->max(0) << " m/s "
                   << "(USTAR_T_MIN=" << DustThresholdConst::USTAR_T_MIN
                   << " m/s)\n";
    amrex::Print() << "[DUST DEBUG] Phase 8: suppression_max="
                   << dust_suppression->max(0)
                   << " tau_base=" << dust_params.supp_tau_base_s << " s\n";
  }

  amrex::Box dust_domain2 = m_dg.ba.minimalBox();
  int dust_nx2 = dust_domain2.length(0);
  int dust_ny2 = dust_domain2.length(1);
  amrex::Print() << "[DUST] DustLayer initialized: grid_ratio="
                 << m_dg.grid_ratio << ", dust cells=" << dust_nx2 << "x"
                 << dust_ny2 << "x1, "
                 << "n_size_bins=" << dust_params.n_size_bins << ", "
                 << "z0_dust=" << dust_params.z0_dust << " m\n";

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] PHREEQC configuration: "
                   << "phreeqc_output_file=\""
                   << dust_params.phreeqc_output_file << "\", "
                   << "phreeqc_update_interval_s="
                   << dust_params.phreeqc_update_interval_s << " s\n";
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 11: bin_diameters [um]:";
    for (auto d : dust_params.bin_diameters)
      amrex::Print() << " " << d * 1e6;
    amrex::Print() << "\n"
                   << "[DUST DEBUG] Phase 11: transport_bins_separately="
                   << dust_params.transport_bins_separately << "\n";
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 17: bin PM classification:\n";
    for (int b = 0; b < (int)dust_params.bin_diameters.size(); ++b) {
      amrex::Real d = dust_params.bin_diameters[b];
      amrex::Print() << "  bin " << b << " d=" << d*1e6 << " um"
                     << " PM2.5=" << (is_pm25(d)?"yes":"no")
                     << " PM10=" << (is_pm10(d)?"yes":"no") << "\n";
    }
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] initialize complete:"
                   << " n_size_bins=" << dust_params.n_size_bins
                   << " grid_ratio=" << m_dg.grid_ratio
                   << " dust_scalar_comp=" << m_dust_scalar_comp
                   << " loading_feedback_coeff="
                   << dust_params.loading_feedback_coeff << "\n";
  }

  // The crust the burned-area reduction starts from each step. Refreshed after
  // a PHREEQC update and checkpointed (DustCrustBaseline): the copy made here
  // from the inputs and rasters is the fresh-run value, and the restart read
  // that follows initialize() replaces it with the checkpointed one.
  dust_crust_baseline = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, dust_crust_index->nGrowVect());
  amrex::MultiFab::Copy(*dust_crust_baseline, *dust_crust_index, 0, 0, 1, dust_crust_index->nGrowVect());

  write_dust_stats_header(m_params.dust_diag_file);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 16: dust_plot_int="
                   << dust_params.dust_plot_int
                   << " prefix=" << dust_params.dust_plot_prefix
                   << " diag_file=" << dust_params.dust_diag_file << "\n";
  }

  // Phase 20: allocate dust_site_id and populate from bounding boxes.
  dust_site_id = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_site_id->setVal(0.0);

  int n_sites = (int)dust_params.site_names.size();
  if (n_sites > 0) {
    populate_dust_site_id(*dust_site_id, m_dg.geom,
                          dust_params.site_x_lo, dust_params.site_y_lo,
                          dust_params.site_x_hi, dust_params.site_y_hi);

    auto counts = count_site_cells(*dust_site_id, n_sites);

    if (dust_params.dust_debug) {
      amrex::Print() << "[DUST DEBUG] Phase 20: " << n_sites
                     << " mine sites registered\n";
      amrex::Print() << "[DUST DEBUG] Phase 20: unassigned cells="
                     << counts[0] << "\n";
      for (int s = 0; s < n_sites; ++s) {
        amrex::Print() << "[DUST DEBUG] Phase 20: site " << (s+1)
                       << " name=" << dust_params.site_names[s]
                       << " file=" << (s < (int)dust_params.site_phreeqc_files.size()
                                       ? dust_params.site_phreeqc_files[s] : std::string("(none)"))
                       << " bbox=["
                       << dust_params.site_x_lo[s] << ","
                       << dust_params.site_y_lo[s] << ","
                       << dust_params.site_x_hi[s] << ","
                       << dust_params.site_y_hi[s] << "]"
                       << " cells=" << counts[s+1] << "\n";
      }
    }

    // (site_phreeqc_files may be absent, DustParams allows it; indexing it
    // for every site read past the Vector and segfaulted at start-up with
    // site_names alone -- the parity CTest's configuration)
  } else {
    if (dust_params.dust_debug)
      amrex::Print() << "[DUST DEBUG] Phase 20: no site bounding boxes"
                     << " defined; single-site mode (Phase 4 table)\n";
  }

  // Phase 21: PHREEQC deposition feedback file writer
  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 21: phreeqc_feedback_interval_s="
                   << dust_params.phreeqc_feedback_interval_s
                   << " feedback_file=" << dust_params.phreeqc_feedback_file
                   << " site_summary_file="
                   << dust_params.phreeqc_site_summary_file << "\n";
  }

  // Phase 23: critical material flux MultiFab.
  dust_cm_flux = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(1,1,0));
  dust_cm_flux->setVal(0.0);

  // Phase 7: visibility and health exposure diagnostics MultiFabs.
  m_visibility = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0,0,0));
  m_vis_closure = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0,0,0));
  m_vis_warning = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0,0,0));
  m_rcs_conc = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0,0,0));
  m_stel_avg = std::make_unique<amrex::MultiFab>(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0,0,0));

  // Initialize all Phase 7 diagnostics to zero
  for (auto* mf : {m_visibility.get(), m_vis_closure.get(), m_vis_warning.get(),
                   m_rcs_conc.get(), m_stel_avg.get()}) {
    if (mf) mf->setVal(0.0);
  }

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 7: visibility_enable=" << dust_params.visibility_enable
                   << " silica_enable=" << dust_params.silica_enable
                   << " stel_enable=" << dust_params.stel_enable << "\n";
    amrex::Print() << "[DUST DEBUG] Phase 23: cm_fractions=";
    for (auto f : dust_params.cm_fractions)
      amrex::Print() << f << " ";
    amrex::Print() << "\n"
                   << "[DUST DEBUG] Phase 23: cm_budget_file="
                   << dust_params.cm_budget_file << "\n";
    if (dust_params.cm_fractions.empty())
      amrex::Print() << "[DUST DEBUG] Phase 23: cm_fractions empty,"
                     << " CM tracking disabled\n";
  }
} // end initialize()

void
DustLayer::advance(
  amrex::Real dt,
  const DustParams& dust_params,
  SurfaceLayer* surface_layer,
  const amrex::MultiFab* xvel_mf,
  const amrex::MultiFab* yvel_mf,
  const amrex::MultiFab* zvel_mf,
  const amrex::MultiFab* z_phys_cc_mf,
  const amrex::Geometry* geom_atm,
  int nz,
  const amrex::MultiFab* cons_mf)
{
  check_settling_stability(dt);   // the step dt, every step (an adaptive dt can grow)
  // Used only in the ERF_USE_PARTICLES block near the end of this function, so
  // they are genuinely unused when particles are off. Naming them here keeps the
  // signature intact for both configurations.
  amrex::ignore_unused(zvel_mf);

  ++m_step;
  m_time += dt;

  // Deposited mass for the step just completed: the last RK stage's flux (summed
  // over bins in apply_deposition_bc) times the step. It used to be added inside
  // apply_deposition_bc at every stage with that stage's dt, 1.83 dt per step.
  if (dep_flux_step && dust_deposition_rate && dt > 0.0) {
    apply_deposition_to_dust_grid(*dust_deposition_rate, *dep_flux_step,
                                  m_dg.grid_ratio, dt);
    dep_flux_step->setVal(0.0);
  }

  if (m_params.dust_debug) {
    amrex::Real cs  = dust_conc_sfc  ? dust_conc_sfc->max(0) * 1e9 : 0.0;
    amrex::Real p10 = dust_pm10      ? dust_pm10->max(0) : 0.0;
    amrex::Real twa = dust_msha_twa  ? dust_msha_twa->max(0) : 0.0;
    amrex::Real site_max = dust_site_id ? dust_site_id->max(0) : 0.0;
    amrex::Real ef_max = dust_emission_flux ? dust_emission_flux->max(0) : 0.0;
    int  n_roads_active = 0;
    if (m_road_schedule.loaded) {
        for (const auto& ev : m_road_schedule.roads) {
            bool active = (m_time >= ev.start_time_s) &&
                          (ev.end_time_s < 0.0 || m_time <= ev.end_time_s);
            if (active) ++n_roads_active;
        }
    }
    amrex::Print() << "[DUST DEBUG] advance: step=" << m_step
                   << " emission_flux_max=" << ef_max << " kg/m^2/s"
                   << " active_roads=" << n_roads_active
                   << " conc_sfc=" << cs << " ug/m3"
                   << " PM10=" << p10 << " ug/m3"
                   << " MSHA_TWA=" << twa << " mg/m3"
                   << " site_id_max=" << (int)site_max
                   << " n_sites=" << (int)m_params.site_names.size() << "\n";
  }

  bool have_atm =
    (surface_layer && xvel_mf && yvel_mf && z_phys_cc_mf && nz > 0);

  if (have_atm) {
    if (surface_layer->get_u_star(0))
      fill_dust_ustar_from_surface_layer(
        *dust_ustar_in, *surface_layer->get_u_star(0), m_dg);
    fill_dust_wind_from_interpolation(
      *dust_wind_ref, *xvel_mf, *yvel_mf, *z_phys_cc_mf, m_dg, m_params.zref, nz);
    if (m_params.use_terrain_wind && dust_slopes && dust_curvature) {
      // The FARSITE factors change the wind at zref; u* follows by the same
      // factor (terrain_ustar = scale, u* linear in U in a neutral log law),
      // so a flat cell keeps the surface layer's value. The former log law on
      // z0_dust (terrain_ustar = loglaw) replaced it by a neutral one on
      // another roughness, 0.70x on flat ground.
      amrex::MultiFab wind_raw(dust_wind_ref->boxArray(), dust_wind_ref->DistributionMap(), 2, 0);
      amrex::MultiFab::Copy(wind_raw, *dust_wind_ref, 0, 0, 2, 0);
      apply_farsite_terrain_wind(
        *dust_wind_ref, *dust_slopes, *dust_curvature, m_params.k_ridge,
        m_params.k_shelter, m_params.k_valley, m_params.k_deflect);
      if (m_params.terrain_ustar == "loglaw") {
        compute_dust_ustar_from_wind(
          *dust_ustar_in, *dust_wind_ref, m_params.zref, m_params.z0_dust);
      } else {
        scale_dust_ustar_by_wind_ratio(*dust_ustar_in, *dust_wind_ref, wind_raw);
      }
    }
    // Fire-dust coupling: the fire-grid wind raises u* where it is stronger. This
    // has to come after the fills above, which overwrite dust_ustar_in.
    if (dust_ustar_fire) {
      for (amrex::MFIter mfi(*dust_ustar_in, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto ust = dust_ustar_in->array(mfi);
        auto usf = dust_ustar_fire->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
          ust(i,j,k) = amrex::max(ust(i,j,k), usf(i,j,k));
        });
      }
      dust_ustar_in->FillBoundary(m_dg.geom.periodicity());
      dust_ustar_fire->setVal(0.0);   // never reuse a stale fire wind
    }
    if (surface_layer->get_t_surf(0)) {
      // The surface layer's t_surf is a potential temperature; the suppression
      // decay wants a temperature (exp(0.05 (T_s - 293.15))), and at a 1500 m
      // mine theta_s - T_s = 14 K, a factor 2 on the decay rate from the site
      // elevation alone. Convert with the Exner function of the lowest cell's
      // pressure (half a cell above the surface, 0.05 % of it) when the state is
      // at hand; without it the value is taken as a temperature.
      if (cons_mf && geom_atm) {
        amrex::MultiFab T_atm(surface_layer->get_t_surf(0)->boxArray(),
                              surface_layer->get_t_surf(0)->DistributionMap(), 1, 0);
        const int klo = geom_atm->Domain().smallEnd(2);
        constexpr amrex::Real rdOcp = R_d / Cp_d;
        for (amrex::MFIter mfi(T_atm, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          // the surface layer's flux BoxArray is a k = 0 slab, except on EB
          // terrain where it is the full 3-D array: read the slab only
          amrex::Box bx = mfi.tilebox();
          if (!bx.contains(amrex::IntVect(bx.smallEnd(0), bx.smallEnd(1), 0))) continue;
          bx.setRange(2, 0, 1);
          auto T   = T_atm.array(mfi);
          auto th  = surface_layer->get_t_surf(0)->const_array(mfi);
          auto S   = cons_mf->const_array(mfi);
          amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real qv = amrex::Real(0.0);
            if (RhoQ1_comp < S.nComp()) qv = S(i, j, klo, RhoQ1_comp) / S(i, j, klo, Rho_comp);
            const amrex::Real p     = getPgivenRTh(S(i, j, klo, RhoTheta_comp), qv);
            const amrex::Real exner = getExnergivenP(p, rdOcp);
            T(i, j, k) = th(i, j, 0) * exner;
          });
        }
        fill_dust_scalar_from_atm(*dust_tsfc, T_atm, m_dg);
      } else {
        fill_dust_scalar_from_atm(*dust_tsfc, *surface_layer->get_t_surf(0), m_dg);
      }
    }
    if (surface_layer->get_pblh(0))
      fill_dust_scalar_from_atm(*dust_pblh, *surface_layer->get_pblh(0), m_dg);

    if (m_params.dust_debug)
      amrex::Print() << "[DUST DEBUG] Phase 9: step=" << m_step
                     << " u*_max=" << dust_ustar_in->max(0)
                     << " u_10m_max=" << dust_wind_ref->max(0)
                     << " PBLH_max=" << dust_pblh->max(0) << "\n";
  } else {
    dust_ustar_in->setVal(m_params.test_ustar);
    dust_tsfc->setVal(m_params.test_surf_temp_K);
    dust_wind_ref->setVal(m_params.test_wind_speed);
    if (m_params.dust_debug)
      amrex::Print() << "[DUST DEBUG] Phase 9: placeholder path"
                     << " test_ustar=" << m_params.test_ustar << "\n";
  }

  if (m_params.dust_debug && have_atm) {
    amrex::Print() << "[DUST DEBUG] Phase 9: T_sfc_max=" << dust_tsfc->max(0)
                   << " K  PBLH_max=" << dust_pblh->max(0) << " m\n";
  }

  amrex::Real T_sfc = have_atm ? dust_tsfc->max(0) : m_params.test_surf_temp_K;
  // the domain-maximum wind SPEED at zref: max(0) of the (u, v) field is the
  // maximum u component, which is <= 0 for a wind along -x or along y and
  // switched the wind enhancement of the suppression decay off
  amrex::Real u_10m = m_params.test_wind_speed;
  if (have_atm) {
    amrex::MultiFab spd(dust_wind_ref->boxArray(), dust_wind_ref->DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(spd); mfi.isValid(); ++mfi) {
      auto const& w = dust_wind_ref->const_array(mfi);
      auto const& s = spd.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        s(i, j, k) = std::sqrt(w(i, j, k, 0) * w(i, j, k, 0) + w(i, j, k, 1) * w(i, j, k, 1));
      });
    }
    u_10m = spd.max(0);
  }

  bool do_phreeqc =
    (m_last_phreeqc_update < 0.0) ||
    (m_time - m_last_phreeqc_update >= dust_params.phreeqc_update_interval_s);

  if (dust_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 4: PHREEQC reader check at step="
                   << m_step << ", time=" << m_time << " s, "
                   << "do_phreeqc=" << do_phreeqc << "\n";
  }

  if (do_phreeqc && !dust_params.phreeqc_output_file.empty()) {
    if (dust_params.dust_debug)
      amrex::Print() << "[DUST DEBUG] PHREEQC update triggered at step="
                     << m_step << ", time=" << m_time << " s\n";
    // Start from the baseline crust: with the fire coupling on, dust_crust_index
    // still carries the previous step's burned-area reduction here, and copying
    // it back into the baseline below compounded the reduction once per PHREEQC
    // interval whenever the table had no crust column.
    if (dust_crust_baseline)
      amrex::MultiFab::Copy(*dust_crust_index, *dust_crust_baseline, 0, 0, 1, dust_crust_index->nGrowVect());
    update_dust_from_phreeqc(
      *dust_crust_index, *dust_silt_fraction, *dust_efflor, *dust_suppression,
      dust_site_id.get(), m_dg, dust_params);
    dust_crust_index->FillBoundary(m_dg.geom.periodicity());
    dust_silt_fraction->FillBoundary(m_dg.geom.periodicity());
    // the table may have treated cells the raster did not
    for (amrex::MFIter mfi(*dust_treated_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto tm = dust_treated_mask->array(mfi);
      auto sp = dust_suppression->const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        if (sp(i,j,k) > amrex::Real(0.0)) tm(i,j,k) = amrex::Real(1.0);
      });
    }
    m_last_phreeqc_update = m_time;
    if (dust_crust_baseline)
      amrex::MultiFab::Copy(*dust_crust_baseline, *dust_crust_index, 0, 0, 1, dust_crust_index->nGrowVect());
    if (dust_params.dust_debug)
      amrex::Print() << "[DUST DEBUG] PHREEQC update completed\n";
  }

  amrex::MultiFab::Copy(
    *dust_surf_moist, *dust_moisture_flag, 0, 0, 1, amrex::IntVect(1, 1, 0));

  if (dt > 0.0) {
    advance_dust_suppression(
      *dust_suppression, *dust_retreat_flag, *dust_treated_mask, T_sfc, u_10m, dt,
      m_params.supp_tau_base_s);
  }

  /*if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 8: suppression coverage max="
                   << dust_suppression->max(0)
                   << ", retreat_flag sum=" << dust_retreat_flag->sum(0)
                   << " at step=" << m_step << "\n";
  }*/

    if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 8: suppression coverage max="
                   << dust_suppression->max(0)
                   << ", retreat_flag sum=" << dust_retreat_flag->sum(0)
                   << " at step=" << m_step << "\n";
  }

  // -----------------------------------------------------------------------
  // Fire-dust coupling: reset crust to baseline each step, then re-apply
  // burned-area reduction so u*_t is recomputed from the current crust.
  // Must run BEFORE recompute_dust_ustar_t so the updated crust is used.
  // -----------------------------------------------------------------------
#if defined(ERF_ENABLE_FIRE) && defined(ERF_USE_DUST)
  if (m_fire_dust_coupling && m_fire_dust_coupling->enabled
      && dust_crust_index)
  {
      // Reset the crust to its baseline (inputs, rasters, last PHREEQC update)
      // each step so the reduction is applied once, not compounded. The reset
      // used to be setVal(crust_index), which wiped a crust raster.
      amrex::MultiFab::Copy(*dust_crust_index, *dust_crust_baseline, 0, 0, 1, dust_crust_index->nGrowVect());
      // Re-apply burned-area reduction using current fire phi field.
      m_fire_dust_coupling->apply_burned_area_to_crust(*dust_crust_index);
      dust_crust_index->FillBoundary(m_dg.geom.periodicity());
  }
#endif

  if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 7: crust_index before u*_t computation at step=" << m_step
                   << " min=" << dust_crust_index->min(0)
                   << " max=" << dust_crust_index->max(0) << "\n";
  }


  recompute_dust_ustar_t(
    *dust_ustar_t, *dust_ustar_base, *dust_crust_index, *dust_efflor,
    *dust_surf_moist, *dust_suppression, m_params.alpha_crust,
    m_params.alpha_efflor, dust_slopes.get(), dust_wind_ref.get());

  if (m_params.dust_debug) {
    /*amrex::Print() << "[DUST DEBUG] Phase 7: crust_index before u*_t computation at step=" << m_step
                   << " min=" << dust_crust_index->min(0)
                   << " max=" << dust_crust_index->max(0) << "\n";*/
    amrex::Print() << "[DUST DEBUG] Phase 5: u*_t after crust modulation at step=" << m_step
                   << " u*_t_min=" << dust_ustar_t->min(0)
                   << " u*_t_max=" << dust_ustar_t->max(0) << " [m/s]\n";
  }

  if (m_params.loading_feedback_coeff > 0.0 && dust_conc_sfc) {
    for (amrex::MFIter mfi(*dust_ustar_t, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto ust  = dust_ustar_t->array(mfi);
      auto conc = dust_conc_sfc->const_array(mfi);
      const amrex::Real alpha = m_params.loading_feedback_coeff;
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        ust(i,j,k) *= (1.0 + alpha * amrex::max(static_cast<amrex::Real>(conc(i,j,k)), static_cast<amrex::Real>(0.0)));
      });
    }
    if (m_params.dust_debug) {
      amrex::Real ust_max = dust_ustar_t->max(0);
      amrex::Print() << "[DUST DEBUG] Phase 13: loading feedback applied"
                     << " ustar_t_max=" << ust_max << " m/s\n";
    }
  }


  if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 6: Before emission computation: u*_t_min="
                   << dust_ustar_t->min(0) << " u*_t_max=" << dust_ustar_t->max(0)
                   << " [m/s], u*_in_min=" << dust_ustar_in->min(0)
                   << " u*_in_max=" << dust_ustar_in->max(0) << " [m/s]\n";
  }

  compute_dust_emission_flux(
    *dust_emission_flux, *dust_ustar_t, *dust_ustar_in, *dust_silt_fraction,
    m_params.n_size_bins, m_params.rho_air);

  // Soil code 0 of a soil-type raster means "undefined: no emission"
  // (dust_sources.rst), which nothing implemented; without a raster the field
  // is 0 everywhere and means nothing.
  if (!m_params.soil_type_file.empty()) {
    for (amrex::MFIter mfi(*dust_emission_flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto fl = dust_emission_flux->array(mfi);
      auto st = dust_soil_type->const_array(mfi);
      const int nb = m_params.n_size_bins;
      amrex::ParallelFor(bx, nb, [=] AMREX_GPU_DEVICE (int i, int j, int k, int b) noexcept {
        if (st(i,j,k) < amrex::Real(0.5)) fl(i,j,k,b) = amrex::Real(0.0);
      });
    }
  }

  if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 6: emission_flux bin0 at step="
                   << m_step << " max=" << dust_emission_flux->max(0)
                   << " sum=" << dust_emission_flux->sum(0) << " [kg/m^2/s]\n";
  }

#ifdef ERF_USE_DUST
  if (m_has_blast_schedule && dt > 0.0) {
    apply_blast_schedule(
      *dust_emission_flux, m_dg, m_blast_schedule, m_time, m_time - dt, dt,
      m_params.n_size_bins, m_params.blast_reactivity);
    if (m_params.dust_debug)
      amrex::Print()
        << "[DUST DEBUG] Phase 7: emission_flux bin0 after blast step="
        << m_step << " max=" << dust_emission_flux->max(0) << " [kg/m^2/s]\n";
  }

  // Phase 22: apply haul road vehicle resuspension (additive to emission flux).
  apply_road_schedule(*dust_emission_flux, m_dg.geom, m_road_schedule,
                      m_time, dt, m_params.dust_debug,
                      m_params.road_diag_file, m_step);

  // Fire lofting of this step's flux, before the budget, the particles and the
  // diagnostics read it (it used to be applied by ERF after advance() returned).
  if (m_loft_heat) {
    apply_fire_lofting_to_emission_flux(*dust_emission_flux, *m_loft_heat,
                                        m_params.n_size_bins, m_loft_k,
                                        m_loft_Q_threshold, m_loft_Q_ref,
                                        m_params.dust_debug, m_step);
    m_loft_heat = nullptr;
  }

  // Phase 23: compute critical material flux and write budget.
  if (!m_params.cm_fractions.empty() && dust_emission_flux && dust_cm_flux) {
    // the emission flux carries every bin whichever way they are transported,
    // and the atmosphere receives their sum: the budget sums them all too
    // (bin 0 alone used to be counted when the bins travel together)
    compute_cm_flux(*dust_cm_flux, *dust_emission_flux,
                    m_params.cm_fractions, m_params.n_size_bins);
    if (m_step % m_params.cm_budget_int == 0)
      append_cm_budget(m_params.cm_budget_file,
                       *dust_cm_flux,
                       dust_site_id.get(),
                       m_dg.geom,
                       m_time, m_step,
                       m_params.site_names);
    if (m_params.dust_debug) {
      amrex::Real cm_max = dust_cm_flux->max(0);
      amrex::Real cm_sum = dust_cm_flux->sum(0);
      amrex::Print() << "[DUST DEBUG] Phase 23: step=" << m_step
                     << " cm_flux_max=" << cm_max << " kg_CM/m^2/s"
                     << " cm_flux_sum=" << cm_sum << " kg_CM/m^2/s\n";
    }
  }

  // end-of-step time, the same stamp as dust_diag.dat (the start-of-step time
  // was written next to an end-of-step concentration until October 2026, and
  // the MSHA shift boundary was found one step late)
  compute_naaqs_diagnostics(dt, m_time, m_step);
  compute_msha_exposure(dt, m_time, m_step);

  // The mass each bin emitted this step (every source stamps the flux array
  // before this point): the bins' shares feed the PM classes and the mean
  // settling of the lumped scalar.
  if (dust_emission_flux) {
    const amrex::Real cell_area = m_dg.geom.CellSize(0) * m_dg.geom.CellSize(1);
    for (int bb = 0; bb < m_params.n_size_bins && bb < (int)m_emitted_per_bin.size(); ++bb)
      m_emitted_per_bin[bb] += dust_emission_flux->sum(bb) * cell_area * dt;
  }

  // Coarsen this step's emission flux and friction velocity to the atmosphere
  // columns once; the RK stages used to redo both average_downs (with their
  // allocations and, on a mismatched DM, a ParallelCopy) at every stage.
  coarsen_to_atm_columns();
#endif

#if defined(ERF_USE_PARTICLES)
  // Phase 19 diagnostic: check why block may not execute
  if (m_params.dust_debug) {
    amrex::Print() << "[DUST DEBUG] Phase 19 check: step=" << m_step
                   << " enable_particles=" << m_params.enable_particles
                   << " m_dust_pc=" << (m_dust_pc ? "valid" : "null")
                   << " xvel_mf=" << (xvel_mf ? "valid" : "null")
                   << " yvel_mf=" << (yvel_mf ? "valid" : "null")
                   << " zvel_mf=" << (zvel_mf ? "valid" : "null")
                   << " geom_atm=" << (geom_atm ? "valid" : "null")
                   << " interval_check=" << (m_step % m_params.particle_release_interval)
                   << " emission_max=" << dust_emission_flux->max(0) << "\n";
  }
  if (m_params.enable_particles && m_dust_pc && xvel_mf && yvel_mf && zvel_mf
      && geom_atm && (m_step % m_params.particle_release_interval == 0)) {
    amrex::Real d_m   = m_params.bin_diameters.empty() ? 7.0e-6 : m_params.bin_diameters[0];
    amrex::Real rho_p = m_params.particle_density;
    // a particle released every N steps carries N steps of emission
    m_dust_pc->ReleaseParticles(*dust_emission_flux, *geom_atm, m_dg.geom,
                                 dt * m_params.particle_release_interval, d_m, rho_p);
    if (dust_source_map) {
      m_dust_pc->AdvanceParticles(*xvel_mf, *yvel_mf, *zvel_mf,
                                   *dust_source_map, *geom_atm, m_dg.geom, dt);
    }
    if (m_params.dust_debug) {
      amrex::Long np = m_dust_pc->TotalNumberOfParticles();
      amrex::Real sm = dust_source_map ? dust_source_map->sum(0) : 0.0;
      amrex::Print() << "[DUST DEBUG] Phase 19: step=" << m_step
                     << " n_particles=" << np
                     << " source_map_sum=" << sm << " kg/m^2\n";
    }
  }
#endif

} // end advance()

#ifdef ERF_USE_DUST

void
DustLayer::apply_to_cc_source(
  amrex::MultiFab& cc_source,
  const amrex::MultiFab& detJ_cc,
  const amrex::Geometry& geom_atm)
{
  if (!dust_flux_atm) return;
  if (m_params.atm_feedback <= 0.0) return;

  // dust_flux_atm is the flux coarsened once per step (coarsen_to_atm_columns).
  // The state carries one dust scalar: separately transported bins are refused
  // at start-up above one bin (DustParams), so one coarsened flux is the whole
  // story here. A per-bin branch that overwrote dust_flux_atm with the last
  // bin's flux (and fed it to scalar 0 at the next stage) was removed in
  // October 2026; one coarsened MultiFab per bin is the way to lift the guard.
  int n_active = m_params.transport_bins_separately ? m_params.n_size_bins : 1;
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_active == 1,
      "[DUST] apply_to_cc_source: one coarsened flux serves one transported scalar");
  for (int b = 0; b < n_active; ++b) {
    apply_dust_tendency_to_cc_source(
      cc_source, *dust_flux_atm, detJ_cc, geom_atm,
      m_dust_scalar_comp + b, m_params.atm_feedback, m_params.dust_debug);
  }

  if (m_params.dust_debug) {
    amrex::Real F_max = dust_flux_atm->max(0);
    amrex::Print() << "[DUST DEBUG] Phase 10: step=" << m_step
                   << " F_dust_atm_max=" << F_max
                   << " kg/m^2/s  n_active_bins=" << n_active << "\n";
  }
}

void
DustLayer::coarsen_to_atm_columns ()
{
  if (!(dust_flux_atm && ustar_atm && !m_geom_atm.Domain().isEmpty() && m_params.atm_feedback > 0.0)) return;
  if (!m_dust_bin_tmp.ok()) m_dust_bin_tmp.define(m_dg.ba, m_dg.dm, 1, amrex::IntVect(0));
  m_dust_bin_tmp.setVal(0.0);
  if (m_params.transport_bins_separately) {
    amrex::MultiFab::Copy(m_dust_bin_tmp, *dust_emission_flux, 0, 0, 1, amrex::IntVect(0));
  } else {
    for (int bb = 0; bb < m_params.n_size_bins; ++bb)
      amrex::MultiFab::Add(m_dust_bin_tmp, *dust_emission_flux, bb, 0, 1, amrex::IntVect(0));
  }
  amrex::Box atm_domain_2d = m_geom_atm.Domain();
  atm_domain_2d.setSmall(2, 0);
  atm_domain_2d.setBig(2, 0);
  amrex::RealBox prob_2d = m_geom_atm.ProbDomain();
  prob_2d.setHi(2, prob_2d.lo(2) + 1.0);
  amrex::Geometry geom_atm_2d(atm_domain_2d, prob_2d, amrex::CoordSys::cartesian, {false, false, false});
  dust_flux_atm->setVal(0.0);
  amrex::average_down(m_dust_bin_tmp, *dust_flux_atm, m_dg.geom, geom_atm_2d, 0, 1,
                      amrex::IntVect(m_dg.grid_ratio, m_dg.grid_ratio, 1));
  ustar_atm->setVal(0.0);
  amrex::average_down(*dust_ustar_in, *ustar_atm, m_dg.geom, geom_atm_2d, 0, 1,
                      amrex::IntVect(m_dg.grid_ratio, m_dg.grid_ratio, 1));
}

void
DustLayer::check_settling_stability (amrex::Real dt) const
{
    // Explicit first-order settling is stable for v_s dt / h_0 <= 1; a 100 um
    // bin (0.8 m/s) with dt = 10 s on 5 m cells is not, and nothing refused
    // it. Checked every step with that step's dt (an adaptive dt can grow).
    if (!(m_h0_min > 0.0) || !(dt > 0.0)) return;
    amrex::Real vs_max = 0.0;
    for (auto d : m_params.bin_diameters)
        vs_max = amrex::max(vs_max, compute_stokes_settling(d, m_params.particle_density, amrex::Real(1.225),
                                                            DustSettlingConst::MU_AIR_STD));
    const amrex::Real cfl = vs_max * dt / m_h0_min;
    if (cfl > 1.0) {
        amrex::Abort("[DUST] explicit settling is unstable: max v_s * dt / h_0 = "
                     + std::to_string(cfl) + " (v_s = " + std::to_string(vs_max)
                     + " m/s, dt = " + std::to_string(dt) + " s, thinnest cell "
                     + std::to_string(m_h0_min) + " m); shorten erf.fixed_dt or drop the coarse bin");
    }
}

std::vector<amrex::Real>
DustLayer::emitted_shares () const
{
    const int nd = (int)m_params.bin_diameters.size();
    std::vector<amrex::Real> s(nd, 1.0 / amrex::Real(amrex::max(nd, 1)));
    amrex::Real tot = 0.0;
    for (int b = 0; b < nd && b < (int)m_emitted_per_bin.size(); ++b) tot += m_emitted_per_bin[b];
    if (tot > 0.0)
        for (int b = 0; b < nd && b < (int)m_emitted_per_bin.size(); ++b) s[b] = m_emitted_per_bin[b] / tot;
    return s;
}

DustBinDiameters
DustLayer::scalar_bin_diameters (int b, int& nb, DustBinWeights& w) const
{
    DustBinDiameters d{};
    for (int i = 0; i < DustSettlingConst::MAX_BINS; ++i) { d[i] = 0.0; w[i] = 0.0; }
    w[0] = 1.0;
    const int nd = (int)m_params.bin_diameters.size();
    if (m_params.transport_bins_separately) {
        // scalar b carries bin b alone
        const int idx = amrex::min(b, nd - 1);
        d[0] = m_params.bin_diameters[idx];
        nb = 1;
    } else if (m_params.lumped_settling == "bin0") {
        d[0] = m_params.bin_diameters[0];
        nb = 1;
    } else {
        // the lumped scalar carries every bin with its share of the emitted mass
        nb = amrex::min(nd, DustSettlingConst::MAX_BINS);
        const std::vector<amrex::Real> s = emitted_shares();
        for (int i = 0; i < nb; ++i) { d[i] = m_params.bin_diameters[i]; w[i] = s[i]; }
    }
    return d;
}

void
DustLayer::apply_settling_to_cc_source(
    amrex::MultiFab& cc_source,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& detJ_cc,
    const amrex::Geometry& geom_atm,
    amrex::Real dt)
{
    if (m_params.bin_diameters.empty()) return;

    int n_active = m_params.transport_bins_separately ? m_params.n_size_bins : 1;
    const amrex::Real rhop = m_params.particle_density;

    // The thinnest first cell for the stability check in advance() (the step
    // dt lives there; this hook sees the stage dt, dt/3 at the first stage,
    // which under-read the condition 3x until October 2026)
    if (m_h0_min < 0.0) {
        // the thinnest cell anywhere (a mid-level cell can be thinner than the
        // surface cell on a fitted mesh): the smallest detJ above the kernels'
        // floor, so that covered EB cells (detJ 0, treated as full cells by the
        // kernels) neither abort the run nor hide a thin cut cell
        amrex::Real dj = amrex::ReduceMin(detJ_cc, 0,
            [=] AMREX_GPU_HOST_DEVICE (amrex::Box const& bx, amrex::Array4<amrex::Real const> const& a) -> amrex::Real {
                amrex::Real m = 1.0e30;
                amrex::Loop(bx, [&] (int i, int j, int k) {
                    const amrex::Real v = a(i,j,k);
                    if (v > 1.0e-10 && v < m) m = v;
                });
                return m;
            });
        amrex::ParallelAllReduce::Min(dj, amrex::ParallelContext::CommunicatorSub());
        m_h0_min = geom_atm.CellSize(2) * ((dj < 1.0e29) ? dj : amrex::Real(1.0));
    }
    amrex::ignore_unused(dt);

    for (int b = 0; b < n_active; ++b) {
        int nb = 1;
        DustBinWeights w{};
        const DustBinDiameters bins = scalar_bin_diameters(b, nb, w);
        int comp = m_dust_scalar_comp + b;
        AMREX_ASSERT(comp < cc_source.nComp());

        apply_dust_settling_to_cc_source(cc_source, S_old, detJ_cc,
                                         geom_atm, bins, w, nb, rhop, comp,
                                         m_params.dust_debug);
    }

    if (m_params.dust_debug) {
        amrex::Print() << "[DUST DEBUG] Phase 11: step=" << m_step
                       << " settling applied n_active=" << n_active
                       << " (erf.transport_scalar=true required)\n";
    }
}

void
DustLayer::apply_deposition_bc(
   amrex::MultiFab& cc_source, const amrex::MultiFab& S_old,
   const amrex::MultiFab& detJ_cc, const amrex::Geometry& geom_atm,
   amrex::Real /*dt*/)
{
   if (!dep_flux_atm || !dust_ustar_in || !ustar_atm) return;
   if (m_params.atm_feedback <= 0.0) return;

   // The deposition kernel walks the atmosphere's cells, so it needs u* on the
   // atmosphere grid: the mean over the C x C dust cells of each atmosphere
   // cell, coarsened once per step in advance() (ustar_atm). It used to be
   // handed dust_ustar_in itself and read it with atmosphere indices, which
   // with grid_ratio > 1 is the wrong dust cell, and then re-coarsened at
   // every RK stage.

   int n_active = m_params.transport_bins_separately ? m_params.n_size_bins : 1;

   if (dep_flux_step) dep_flux_step->setVal(0.0);
   for (int b = 0; b < n_active; ++b) {
       int nb = 1;
       DustBinWeights w{};
       const DustBinDiameters bins = scalar_bin_diameters(b, nb, w);
       amrex::Real rhop = m_params.particle_density;
       amrex::Real E_0  = m_params.deposition_E0;
       int comp = m_dust_scalar_comp + b;

       apply_dust_deposition_bc(cc_source, *dep_flux_atm,
                                 S_old, *ustar_atm,
                                 detJ_cc, geom_atm,
                                 bins, w, nb, rhop, E_0, comp,
                                 m_params.dust_debug);

       if (dep_flux_step)
         amrex::MultiFab::Add(*dep_flux_step, *dep_flux_atm, 0, 0, 1, 0);
   }

   if (m_params.dust_debug) {
       amrex::Real dep_sum = dust_deposition_rate->sum(0);
       amrex::Print() << "[DUST DEBUG] Phase 12: step=" << m_step
                      << " deposition_rate_sum=" << dep_sum
                      << " kg/m^2 (accumulated total)\n";
   }
}

void
DustLayer::extract_atm_return_fields(
    const amrex::MultiFab& S_new_cons,
    const amrex::MultiFab* Q1fx3,
    const amrex::Geometry& geom_atm)
{
    fill_dust_conc_from_atm(*dust_conc_sfc, S_new_cons,
                             m_dust_scalar_comp, geom_atm, m_dg.grid_ratio);

    const amrex::MultiFab* q1fx3_ptr = Q1fx3;   // null without a moisture scheme; the flux is output only
    fill_dust_moist_from_atm(*dust_surf_qflux, q1fx3_ptr,
                              geom_atm, m_dg.grid_ratio);

    if (m_params.dust_debug) {
        amrex::Real conc_max  = dust_conc_sfc->max(0);
        amrex::Real conc_sum  = dust_conc_sfc->sum(0);
        amrex::Real moist_max = dust_surf_qflux->max(0);
        amrex::Real dep_total = dust_deposition_rate ? dust_deposition_rate->sum(0) : 0.0;
        bool q1fx3_active = (Q1fx3 != nullptr);
        amrex::Print() << "[DUST DEBUG] Phase 13: step=" << m_step
                       << " conc_sfc_max=" << conc_max
                       << " kg/m^3  conc_sfc_sum=" << conc_sum
                       << "\n[DUST DEBUG] Phase 13:"
                       << " moist_flux_max=" << moist_max
                       << " kg/m^2/s  moisture_path_active=" << q1fx3_active
                       << "\n[DUST DEBUG] Phase 13:"
                       << " dep_total=" << dep_total
                       << " kg/m^2 (Phase 12 accumulator)\n";
        amrex::Print() << "[DUST DEBUG] Phase 14: MRF diffusion active"
                       << " (erf.transport_scalar=true, EddyDiff::Scalar_v"
                       << " set by ComputeDiffusivityMRF)\n"
                       << "[DUST DEBUG] Phase 14: gamma_dust=0"
                       << " (no countergradient term for dust scalar)\n";
    }
}

void
DustLayer::compute_naaqs_diagnostics(amrex::Real dt, amrex::Real cur_time, int nstep)
{
    if (!dust_conc_sfc) return;

    int n_active = m_params.transport_bins_separately ? m_params.n_size_bins : 1;

    compute_pm_concentrations(*dust_pm25, *dust_pm10,
                               *dust_conc_sfc,
                               m_params.bin_diameters, n_active,
                               /*lumped=*/!m_params.transport_bins_separately,
                               emitted_shares());

    if (m_params.averaging == "window") {
        m_pm25_window.update(*dust_pm25, dt, *dust_pm25_24h);
        m_pm10_window.update(*dust_pm10, dt, *dust_pm10_24h);
    } else {
        update_running_average(*dust_pm25_24h, *dust_pm25, dt, 86400.0);
        update_running_average(*dust_pm10_24h, *dust_pm10, dt, 86400.0);
    }

    compute_exceedance_flag(*dust_pm25_exceed, *dust_pm25_24h,
                             DustPMConst::PM25_24H_NAAQS);
    compute_exceedance_flag(*dust_pm10_exceed, *dust_pm10_24h,
                             DustPMConst::PM10_24H_NAAQS);
    // A 24-hour standard is compared once 24 hours are covered: the block mean
    // of a partial window is the mean over the time covered, so after one step
    // it is the instantaneous value and a one-step spike would flag the day.
    if (m_params.averaging == "window" && m_pm25_window.n_filled < m_pm25_window.n_slots) {
        dust_pm25_exceed->setVal(0.0);
        dust_pm10_exceed->setVal(0.0);
    }

    append_naaqs_stats(nstep, cur_time, m_params.dust_naaqs_file,
                       *dust_pm25, *dust_pm25_24h,
                       *dust_pm10, *dust_pm10_24h,
                       *dust_pm25_exceed, *dust_pm10_exceed);

    if (m_params.dust_debug) {
        amrex::Real pm25_max    = dust_pm25->max(0);
        amrex::Real pm25_24h_mx = dust_pm25_24h->max(0);
        amrex::Real pm10_max    = dust_pm10->max(0);
        amrex::Real n_ex25      = dust_pm25_exceed->sum(0);
        amrex::Real n_ex10      = dust_pm10_exceed->sum(0);
        amrex::Print() << "[DUST DEBUG] Phase 17: step=" << nstep
                       << " PM25_max=" << pm25_max << " ug/m^3"
                       << " PM25_24h_max=" << pm25_24h_mx << " ug/m^3"
                       << " PM10_max=" << pm10_max << " ug/m^3"
                       << " PM25_exceed_cells=" << (long)n_ex25
                       << " PM10_exceed_cells=" << (long)n_ex10 << "\n";
    }
}

void
DustLayer::compute_msha_exposure(amrex::Real dt, amrex::Real cur_time, int nstep)
{
    if (!dust_pm10) return;

    using namespace amrex;

    update_msha_dose(*dust_msha_dose, *dust_msha_twa, *dust_pm10, dt);
    compute_msha_exceed(*dust_msha_exceed, *dust_msha_twa, m_params.msha_pel_mg_m3);

    // (sd > 0 is enforced in DustParams; the floor keeps the speculated x/0
    // of the unselected && arm trap-free)
    const Real sd  = m_params.msha_shift_duration_s;
    const Real sdf = amrex::max(sd, Real(1.0e-30));
    if (sd > 0.0 && cur_time > dt &&
        std::floor(cur_time / sdf) > std::floor((cur_time - dt) / sdf)) {
        MultiFab::Copy(*dust_msha_shift_twa, *dust_msha_twa, 0, 0, 1, 0);
        write_msha_shift_summary(++m_msha_shift_count, cur_time,
                                 m_params.msha_shift_file,
                                 *dust_msha_twa, *dust_msha_exceed);
        dust_msha_dose->setVal(0.0);
        if (m_params.dust_debug) {
            amrex::Print() << "[DUST DEBUG] Phase 18: shift " << m_msha_shift_count
                           << " ended t=" << cur_time << " s  dose reset\n";
        }
    }

    append_msha_stats(nstep, cur_time, m_params.msha_exposure_file,
                      *dust_msha_twa, *dust_msha_exceed, *dust_msha_dose);

    for (int r = 0; r < (int)m_params.msha_receptor_names.size(); ++r) {
        append_receptor_sample(nstep, cur_time,
            "msha_receptor_" + m_params.msha_receptor_names[r] + ".csv",
            m_params.msha_receptor_names[r],
            m_params.msha_receptor_x[r], m_params.msha_receptor_y[r],
            *dust_pm10, m_dg.geom);
    }

    if (m_params.dust_debug) {
        Real tmax = dust_msha_twa->max(0);
        Real nex  = dust_msha_exceed->sum(0);
        amrex::Print() << "[DUST DEBUG] Phase 18: step=" << nstep
                       << " TWA_max=" << tmax << " mg/m3  exceed=" << (long)nex
                       << " shift=" << m_msha_shift_count << "\n";
    }

    // Phase 7: Visibility diagnostics
    if (m_params.visibility_enable && m_visibility && m_vis_closure && m_vis_warning && dust_pm10) {
        compute_visibility(*m_visibility, *dust_pm10, m_params.visibility_k_ext, 1.0e5);
        compute_visibility_flags(*m_vis_closure, *m_vis_warning, *m_visibility,
                                 m_params.visibility_road_closure_m,
                                 m_params.visibility_warning_m);
        write_visibility_stats(nstep, cur_time, *m_visibility, *m_vis_closure,
                               *m_vis_warning, m_params.visibility_diag_file);
    }

    // Phase 7: Silica RCS diagnostics
    if (m_params.silica_enable && m_rcs_conc && dust_pm10) {
        compute_silica_concentration(*m_rcs_conc, *dust_pm10, m_params.silica_fraction_rcs);
        write_silica_stats(nstep, cur_time, *m_rcs_conc,
                           m_params.silica_osha_pel_mg_m3, m_params.silica_diag_file);
    }

    // Phase 7: STEL diagnostics
    if (m_params.stel_enable && m_stel_avg && dust_pm10) {
        update_stel_average(*m_stel_avg, *dust_pm10, dt, m_params.stel_averaging_s,
                            (m_params.averaging == "window") ? &m_stel_window : nullptr);
        write_stel_stats(nstep, cur_time, *m_stel_avg,
                         m_params.stel_threshold_mg_m3, m_params.stel_diag_file);
    }
}

void
DustLayer::write_output(int nstep, amrex::Real cur_time, bool is_final)
{
    // A step reaches here twice when the run ends on it (the time loop, then
    // WriteAtFinalTime) and when a restart starts on it (the original run,
    // then InitData). The second call appends no second row and rewrites no
    // plotfile the interval already wrote. (One case is left: a checkpoint the
    // time loop writes on the run's last step is saved before WriteAtFinalTime
    // adds the final plotfile, so a restart from it at that same step writes
    // that plotfile again.)
    const bool step_written = (nstep == m_last_output_step);
    if (!step_written) {
        append_dust_stats(nstep, cur_time,
                          m_params.dust_diag_file,
                          get_emission_flux(),
                          get_deposition_rate(),
                          get_ustar_in(),
                          get_conc_sfc(),
                          m_dg.geom.CellSize(0) * m_dg.geom.CellSize(1));
    }
    m_last_output_step = nstep;

    bool write_plt = false;
    if (m_params.dust_plot_int > 0)
        write_plt = (nstep % m_params.dust_plot_int == 0) && (nstep != m_last_dust_plot_step);
    if (is_final && nstep > m_last_dust_plot_step)
        write_plt = true;

    if (write_plt) {
        WriteDustPlotfile(m_params.dust_plot_prefix, *this, cur_time, nstep);
        m_last_dust_plot_step = nstep;

        if (m_params.dust_debug) {
            amrex::Print() << "[DUST DEBUG] Phase 16: plotfile written"
                           << " step=" << nstep
                           << " is_final=" << is_final << "\n";
        }
    }

    // Phase 21: PHREEQC deposition feedback file writer
    write_phreeqc_feedback(nstep, cur_time, is_final);

    if (m_params.dust_debug) {
        amrex::Real em = dust_emission_flux   ? dust_emission_flux->sum(0)   : 0.0;
        amrex::Real dp = dust_deposition_rate ? dust_deposition_rate->sum(0) : 0.0;
        amrex::Print() << "[DUST DEBUG] Phase 16: step=" << nstep
                       << " emission_flux_sum(bin 0)=" << em << " kg/m^2/s"
                       << " dep_sum=" << dp << " kg/m^2\n";
    }
}

void
DustLayer::write_phreeqc_feedback(int nstep, amrex::Real cur_time,
                                   bool is_final)
{
    if (!dust_deposition_rate) return;
    const amrex::Real interval = m_params.phreeqc_feedback_interval_s;
    if (interval <= 0.0 && !is_final) return;

    if (nstep == m_last_phreeqc_write_step) return;

    bool do_write = false;
    if (interval > 0.0 &&
        (cur_time - m_last_phreeqc_write_time) >= interval - 0.5*interval*1e-6)
        do_write = true;
    if (is_final && nstep > m_last_phreeqc_write_step)
        do_write = true;

    if (!do_write) return;

    m_last_phreeqc_write_step = nstep;
    m_last_phreeqc_write_time = cur_time;

    write_phreeqc_deposition_file(
        m_params.phreeqc_feedback_file,
        *dust_deposition_rate,
        m_dg.geom,
        cur_time, nstep);

    append_phreeqc_site_summary(
        m_params.phreeqc_site_summary_file,
        *dust_deposition_rate,
        dust_site_id.get(),
        m_dg.geom,
        cur_time,
        m_params.site_names);

    if (m_params.dust_debug) {
        amrex::Real dep_sum = dust_deposition_rate->sum(0);
        amrex::Print() << "[DUST DEBUG] Phase 21: PHREEQC feedback written"
                       << " step=" << nstep
                       << " time=" << cur_time
                       << " dep_sum=" << dep_sum << " kg/m^2"
                       << " is_final=" << is_final << "\n";
    }
}


#endif // ERF_USE_DUST
// ---------------------------------------------------------------------------
// Checkpoint and restart
// ---------------------------------------------------------------------------

amrex::Vector<std::pair<std::string, amrex::MultiFab*>>
DustLayer::checkpoint_fields ()
{
    amrex::Vector<std::pair<std::string, amrex::MultiFab*>> fields;
    auto add = [&] (const char* name, amrex::MultiFab* mf) {
        if (mf) { fields.push_back({std::string(name), mf}); }
    };
    // Surface state evolved by PHREEQC, the suppression decay and the fire coupling.
    add("DustUstarT",         dust_ustar_t.get());
    add("DustUstarBase",      dust_ustar_base.get());
    add("DustCrustIndex",     dust_crust_index.get());
    // The baseline the fire coupling resets the crust to every step: the raster
    // or input value until the first PHREEQC update, that table's crust after
    // it. initialize() rebuilds it from the inputs before the restart read, so
    // without this field a restart between two PHREEQC updates threw the table's
    // crust away (the per-step reset overwrote the restored DustCrustIndex).
    add("DustCrustBaseline",  dust_crust_baseline.get());
    add("DustSiltFraction",   dust_silt_fraction.get());
    add("DustEfflor",         dust_efflor.get());
    add("DustSuppression",    dust_suppression.get());
    add("DustRetreatFlag",    dust_retreat_flag.get());
    add("DustTreatedMask",    dust_treated_mask.get());
    // Accumulators, running averages and the flags derived from them.
    add("DustDepositionRate", dust_deposition_rate.get());
    add("DustPM25",           dust_pm25.get());
    add("DustPM10",           dust_pm10.get());
    add("DustPM25_24h",       dust_pm25_24h.get());
    add("DustPM10_24h",       dust_pm10_24h.get());
    add("DustPM25Exceed",     dust_pm25_exceed.get());
    add("DustPM10Exceed",     dust_pm10_exceed.get());
    add("DustMSHADose",       dust_msha_dose.get());
    add("DustMSHATWA",        dust_msha_twa.get());
    add("DustMSHAExceed",     dust_msha_exceed.get());
    add("DustMSHAShiftTWA",   dust_msha_shift_twa.get());
    add("DustSTELAvg",        m_stel_avg.get());
    // the rings and open slots behind the window means (window averaging only)
    if (m_pm25_window.ring.ok()) {
        add("DustPM25Ring",   &m_pm25_window.ring);  add("DustPM25Accum", &m_pm25_window.accum);
        add("DustPM10Ring",   &m_pm10_window.ring);  add("DustPM10Accum", &m_pm10_window.accum);
    }
    if (m_stel_window.ring.ok()) {
        add("DustSTELRing",   &m_stel_window.ring);  add("DustSTELAccum", &m_stel_window.accum);
    }
    // The dust step runs after the dycore of the same step, so the first dycore
    // after a restart injects the emission flux and deposits with the friction
    // velocity that the last step before the checkpoint computed, and reads the
    // surface concentration returned after its slow right-hand side. Without
    // these three the first restarted step injected nothing, deposited nothing
    // (u* was still zero) and the deposition accumulator carried that offset
    // for the rest of the run.
    add("DustEmissionFlux",   dust_emission_flux.get());
    add("DustUstarIn",        dust_ustar_in.get());
    add("DustConcSfc",        dust_conc_sfc.get());
#if defined(ERF_USE_PARTICLES)
    add("DustSourceMap",      dust_source_map.get());
#endif
    return fields;
}

void
DustLayer::write_window_state (std::ostream& f, const char* name, const DustWindowMean& w) const
{
    // self-describing: the slot count and length first, so a restart whose
    // deck defines the window differently aborts naming it instead of
    // reading the next key as a number
    f << name << "_layout " << w.n_slots << " " << w.slot_dur << "\n"
      << name << "_slot_elapsed " << w.slot_elapsed << "\n"
      << name << "_slot_index "   << w.slot_index   << "\n"
      << name << "_n_filled "     << w.n_filled     << "\n"
      << name << "_slot_len " << w.n_slots;
    for (int s = 0; s < w.n_slots; ++s) f << " " << w.slot_len[s];
    f << "\n";
}

void
DustLayer::read_window_state (const std::string& key, std::istream& f, DustWindowMean& w)
{
    if (key.size() > 7 && key.compare(key.size() - 7, 7, "_layout") == 0) {
        int n; amrex::Real sd; f >> n >> sd;
        if (n != w.n_slots || std::abs(sd - w.slot_dur) > 1.0e-6 * amrex::max(sd, w.slot_dur))
            amrex::Abort("[DUST] the checkpoint's " + key + " has " + std::to_string(n) + " slots of "
                         + std::to_string(sd) + " s, the deck defines " + std::to_string(w.n_slots)
                         + " slots of " + std::to_string(w.slot_dur)
                         + " s; erf.dust.averaging, erf.dust.stel_enable and erf.dust.stel_averaging_s must"
                         " match the checkpoint");
    }
    else if (key.size() > 13 && key.compare(key.size() - 13, 13, "_slot_elapsed") == 0) { f >> w.slot_elapsed; }
    else if (key.size() > 11 && key.compare(key.size() - 11, 11, "_slot_index") == 0)   { f >> w.slot_index; }
    else if (key.size() >  9 && key.compare(key.size() -  9,  9, "_n_filled") == 0)     { f >> w.n_filled; }
    else if (key.size() >  9 && key.compare(key.size() -  9,  9, "_slot_len") == 0) {
        int n; f >> n;
        for (int s = 0; s < n; ++s) { amrex::Real v; f >> v; if (s < w.n_slots) w.slot_len[s] = v; }
    }
    else { std::string skip; std::getline(f, skip); }   // a future key: consume its line, keep the parser aligned
}

void
DustLayer::write_checkpoint_state (const std::string& checkpointname) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream f(checkpointname + "/DustState");
    f.precision(17);
    // the layout the fields were written with: a restart with another bin count
    // or grid ratio used to die inside VisMF::Read with no mention of dust
    f << "n_size_bins " << m_params.n_size_bins << "\n"
      << "grid_ratio " << m_dg.grid_ratio << "\n"
      << "step " << m_step << "\n"
      << "time " << m_time << "\n"
      << "last_phreeqc_update " << m_last_phreeqc_update << "\n"
      << "msha_shift_count " << m_msha_shift_count << "\n"
      << "last_phreeqc_write_time " << m_last_phreeqc_write_time << "\n"
      << "last_phreeqc_write_step " << m_last_phreeqc_write_step << "\n"
      << "last_dust_plot_step " << m_last_dust_plot_step << "\n"
      << "last_output_step " << m_last_output_step << "\n"
      << "averaging " << m_params.averaging << "\n"
      << "emitted_per_bin " << m_emitted_per_bin.size();
    for (auto m : m_emitted_per_bin) f << " " << m;
    f << "\n";
    write_window_state(f, "pm25_window", m_pm25_window);
    write_window_state(f, "pm10_window", m_pm10_window);
    write_window_state(f, "stel_window", m_stel_window);
}

void
DustLayer::remove_outputs_for_fresh_start () const
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::vector<std::string> files = {
            m_params.dust_diag_file, m_params.dust_naaqs_file, m_params.msha_exposure_file,
            m_params.msha_shift_file, m_params.stel_diag_file, m_params.silica_diag_file,
            m_params.visibility_diag_file, m_params.cm_budget_file, m_params.road_diag_file,
            m_params.phreeqc_feedback_file, m_params.phreeqc_site_summary_file };
        for (const auto& name : m_params.msha_receptor_names) files.push_back("msha_receptor_" + name + ".csv");
        for (const auto& f : files) {
            if (f.empty()) continue;
            if (std::remove(f.c_str()) == 0)
                amrex::Print() << "[DUST] fresh start: removed the earlier " << f << "\n";
        }
    }
    amrex::ParallelDescriptor::Barrier();
}

void
DustLayer::trim_outputs_after_restart (int step) const
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::vector<std::string> files = {
            m_params.dust_diag_file, m_params.dust_naaqs_file, m_params.msha_exposure_file,
            m_params.stel_diag_file, m_params.silica_diag_file, m_params.visibility_diag_file,
            m_params.cm_budget_file, m_params.road_diag_file };
        for (const auto& name : m_params.msha_receptor_names) files.push_back("msha_receptor_" + name + ".csv");
        for (const auto& fname : files) {
            if (fname.empty()) continue;
            std::ifstream in(fname);
            if (!in) continue;
            std::vector<std::string> keep;
            std::string line;
            bool keyed_by_step = false, seen_header = false;
            int dropped = 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') { keep.push_back(line); continue; }
                if (!seen_header) {
                    seen_header = true;
                    keyed_by_step = (line.rfind("step", 0) == 0);
                    keep.push_back(line);
                    continue;
                }
                if (!keyed_by_step) { keep.push_back(line); continue; }
                int s = 0;
                try { s = std::stoi(line.substr(0, line.find(','))); } catch (...) { keep.push_back(line); continue; }
                if (s <= step) keep.push_back(line); else ++dropped;
            }
            in.close();
            if (!keyed_by_step || dropped == 0) continue;
            std::ofstream out(fname, std::ios::out | std::ios::trunc);
            for (const auto& l : keep) out << l << "\n";
            amrex::Print() << "[DUST] restart: dropped " << dropped << " rows past step " << step
                           << " from " << fname << "\n";
        }
    }
    amrex::ParallelDescriptor::Barrier();
}

void
DustLayer::check_checkpoint_layout (const std::string& restart_chkfile) const
{
    std::ifstream f(restart_chkfile + "/DustState");
    if (!f) return;
    std::string key;
    while (f >> key) {
        if (key == "n_size_bins") {
            int n; f >> n;
            if (n != m_params.n_size_bins)
                amrex::Abort("[DUST] the checkpoint was written with erf.dust.n_size_bins = " + std::to_string(n)
                             + ", the deck has " + std::to_string(m_params.n_size_bins)
                             + "; the emission flux has one component per bin and cannot be reread");
        } else if (key == "grid_ratio") {
            int g; f >> g;
            if (g != m_dg.grid_ratio)
                amrex::Abort("[DUST] the checkpoint was written with erf.dust.grid_ratio = " + std::to_string(g)
                             + ", the deck has " + std::to_string(m_dg.grid_ratio)
                             + "; the dust fields live on that grid and cannot be reread");
        } else if (key == "averaging") {
            std::string a; f >> a;
            if (a != m_params.averaging)
                amrex::Abort("[DUST] the checkpoint was written with erf.dust.averaging = " + a
                             + ", the deck has " + m_params.averaging + "; the 24-hour and STEL means"
                             " cannot be carried across the change");
        } else { std::string skip; std::getline(f, skip); }
    }
}

void
DustLayer::read_checkpoint_state (const std::string& restart_chkfile,
                                  int step, amrex::Real time)
{
    // The dust layer counts its own steps and accumulates its own time from
    // zero, so without this a restarted run would see PHREEQC intervals, MSHA
    // shifts and output intervals measured from the restart instead of from
    // the start of the run.
    m_step = step;
    m_time = time;
    std::ifstream f(restart_chkfile + "/DustState");
    if (f) {
        std::string key;
        while (f >> key) {
            if      (key == "n_size_bins") {
                int n; f >> n;
                if (n != m_params.n_size_bins)
                    amrex::Abort("[DUST] the checkpoint was written with erf.dust.n_size_bins = " + std::to_string(n)
                                 + ", the deck has " + std::to_string(m_params.n_size_bins)
                                 + "; the emission flux has one component per bin and cannot be reread");
            }
            else if (key == "grid_ratio") {
                int g; f >> g;
                if (g != m_dg.grid_ratio)
                    amrex::Abort("[DUST] the checkpoint was written with erf.dust.grid_ratio = " + std::to_string(g)
                                 + ", the deck has " + std::to_string(m_dg.grid_ratio)
                                 + "; the dust fields live on that grid and cannot be reread");
            }
            else if (key == "averaging") {
                std::string a; f >> a;
                if (a != m_params.averaging)
                    amrex::Abort("[DUST] the checkpoint was written with erf.dust.averaging = " + a
                                 + ", the deck has " + m_params.averaging + "; the 24-hour and STEL means"
                                 " cannot be carried across the change");
            }
            else if (key == "emitted_per_bin") {
                int n; f >> n;
                for (int b = 0; b < n; ++b) { amrex::Real v; f >> v; if (b < (int)m_emitted_per_bin.size()) m_emitted_per_bin[b] = v; }
            }
            else if (key.rfind("pm25_window", 0) == 0) { read_window_state(key, f, m_pm25_window); }
            else if (key.rfind("pm10_window", 0) == 0) { read_window_state(key, f, m_pm10_window); }
            else if (key.rfind("stel_window", 0) == 0) { read_window_state(key, f, m_stel_window); }
            else if (key == "step")                    { f >> m_step; }
            else if (key == "time")                    { f >> m_time; }
            else if (key == "last_phreeqc_update")     { f >> m_last_phreeqc_update; }
            else if (key == "msha_shift_count")        { f >> m_msha_shift_count; }
            else if (key == "last_phreeqc_write_time") { f >> m_last_phreeqc_write_time; }
            else if (key == "last_phreeqc_write_step") { f >> m_last_phreeqc_write_step; }
            else if (key == "last_dust_plot_step")     { f >> m_last_dust_plot_step; }
            else if (key == "last_output_step")        { f >> m_last_output_step; }
            else { std::string skip; std::getline(f, skip); }   // a future key: consume its line
        }
    } else {
        amrex::Print() << "[DUST] Checkpoint has no DustState; taking step and time"
                       << " from the atmosphere.\n";
    }
    // Events dated before the restart were applied by the run that wrote the
    // checkpoint; the schedule only fires events inside the current step's
    // interval, so this keeps the debug output and the fired flags consistent.
    for (auto& ev : m_blast_schedule.events) {
        ev.fired = (ev.time_s <= m_time);
    }
    // the first restarted dycore injects and deposits with the restored fields
    coarsen_to_atm_columns();
}

#if defined(ERF_USE_PARTICLES)
void
DustLayer::checkpoint_particles (const std::string& checkpointname) const
{
    if (!m_dust_pc) { return; }
    m_dust_pc->Checkpoint(checkpointname, "DustParticles", true, m_dust_pc->varNames());
}

void
DustLayer::restart_particles (const std::string& restart_chkfile)
{
    if (!m_dust_pc) { return; }
    if (!amrex::FileExists(restart_chkfile + "/DustParticles/Header")) {
        amrex::Print() << "[DUST] Checkpoint has no DustParticles; starting with none.\n";
        return;
    }
    m_dust_pc->Restart(restart_chkfile, "DustParticles");
}
#endif
