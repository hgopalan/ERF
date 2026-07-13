#ifdef ERF_USE_DUST

#include <ERF_DustLayer.H>
#include <ERF_PhreeqcReader.H>
#include <AMReX_ParallelFor.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>
#include <cmath>

using namespace amrex;

DustLayer::DustLayer(const BoxArray&       atm_ba,
                     const DistributionMapping& atm_dm,
                     const DustParams&    dust_params)
{
    // Create dust grid based on grid ratio
    int grid_ratio = dust_params.grid_ratio;
    
    // Get atmospheric domain box
    Box atm_domain = atm_ba.minimalEnclosingBox();
    
    // Create dust domain box (coarsened by grid_ratio in x,y)
    IntVect lo = atm_domain.smallEnd();
    IntVect hi = atm_domain.bigEnd();
    
    // Coarsen in x and y only (z unchanged for dust layer)
    lo[0] /= grid_ratio;
    lo[1] /= grid_ratio;
    hi[0] = (hi[0] + 1) / grid_ratio - 1;
    hi[1] = (hi[1] + 1) / grid_ratio - 1;
    
    Box dust_domain(lo, hi);
    
    // Create dust BoxArray (single box for simplicity)
    BoxArray dust_ba(dust_domain);
    DistributionMapping dust_dm(dust_ba);
    
    m_dg.ba = dust_ba;
    m_dg.dm = dust_dm;
}

void DustLayer::initialize(const DustParams& dust_params)
{
    // Allocate MultiFabs on dust grid
    // 1 component, no ghost cells in horizontal (IntVect(1,1,0) is ghost specification)
    
    int n_bins = dust_params.n_size_bins;
    
    dust_ustar_t = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    dust_crust_index = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    dust_silt_fraction = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    dust_suppression = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    dust_emission_flux = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, n_bins, IntVect(1,1,0));
    
    // Initialize with default values
    dust_ustar_t->setVal(0.0);
    dust_crust_index->setVal(dust_params.crust_index);
    dust_silt_fraction->setVal(dust_params.silt_fraction);
    dust_suppression->setVal(0.0);
    dust_emission_flux->setVal(0.0);
    
    // Compute u*_t from Bagnold formula for bin 0
    // u*_t = A * sqrt(rho_p * g * d / rho_a)
    // For simplicity, use a single representative diameter
    const Real g = 9.81;
    const Real rho_a = 1.225;  // Air density at sea level [kg/m^3]
    const Real d = 7.0e-6;     // Representative diameter [m] (7 microns)
    const Real rho_p = dust_params.particle_density;
    const Real A = dust_params.threshold_A_coeff;
    
    Real ustar_t_computed = A * std::sqrt(rho_p * g * d / rho_a);
    
    // Set u*_t based on ustar_t_base parameter
    Real ustar_t_value = ustar_t_computed;
    if (dust_params.ustar_t_base > 0.0) {
        ustar_t_value = dust_params.ustar_t_base;
    }
    
    // Clamp to minimum
    ustar_t_value = std::max(ustar_t_value, PhreeqcDustConst::USTAR_T_MIN);
    
    dust_ustar_t->setVal(ustar_t_value);
    
    // Allocate efflorescence and base u*_t MultiFabs (Phase 4)
    dust_efflor = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    dust_ustar_base = std::make_unique<MultiFab>(m_dg.ba, m_dg.dm, 1, IntVect(1,1,0));
    
    dust_efflor->setVal(0.0);
    // Store the Bagnold u*_t computed from DustParams as the base value.
    // update_ustar_t_from_chemistry modifies dust_ustar_t; dust_ustar_base is read-only.
    dust_ustar_base->Copy(*dust_ustar_t, 0, 0, 1, IntVect(1,1,0));
}

void DustLayer::advance(Real dt, const DustParams& dust_params)
{
    // Physics inserted in Phases 5-13. PHREEQC reader called here (Phase 4).
    m_time += dt;
    m_step++;

    // Call PHREEQC reader if update interval has elapsed.
    // The interval is set by dust_params.phreeqc_update_interval_s.
    // File-based coupling is appropriate because geochemical processes
    // evolve on timescales of days to weeks, much longer than the
    // atmospheric timestep.
    bool do_phreeqc = (m_last_phreeqc_update < 0.0) ||
                      (m_time - m_last_phreeqc_update >=
                       dust_params.phreeqc_update_interval_s);

    if (do_phreeqc && !dust_params.phreeqc_output_file.empty()) {
        update_dust_from_phreeqc(*dust_ustar_t,
                                 *dust_ustar_base,
                                 *dust_crust_index,
                                 *dust_silt_fraction,
                                 *dust_efflor,
                                 *dust_suppression,
                                 *dust_emission_flux,
                                 m_dg,
                                 dust_params);
        m_last_phreeqc_update = m_time;
    }
}

#endif  // ERF_USE_DUST
