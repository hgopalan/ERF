/*
 * RTE-RRTMGP radiation model interface to ERF
 * The original code is developed by RobertPincus, and the code is open source available at:
 *                        https://github.com/earth-system-radiation/rte-rrtmgp
 * Please reference to the following paper,
 *                        https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001621
 * NOTE: we use the C++ version of RTE-RRTMGP, which is reimplemented the original Fortran
 * code using C++ YAKL for CUDA, HiP and SYCL application by E3SM ECP team, the C++ version
 * of the rte-rrtmgp code is located at:
 *                       https://github.com/E3SM-Project/rte-rrtmgp
 * The RTE-RRTMGP uses BSD-3-Clause Open Source License, if you want to make changes,
 * and modifications to the code, please refer to BSD-3-Clause Open Source License.
 */
#include <string>
#include <vector>
#include <memory>

#include "ERF_Radiation.H"
#include "ERF_m2005_effradius.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_TableData.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include "ERF_Constants.H"
#include "ERF_IndexDefines.H"
#include "ERF_DataStruct.H"
#include "ERF_EOS.H"
#include "ERF_TileNoZ.H"
#include "ERF_Orbit.H"

using namespace amrex;
using yakl::intrinsics::size;
using yakl::fortran::parallel_for;
using yakl::fortran::SimpleBounds;

namespace internal {
    void initial_fluxes (int nz, int nlay, int nbands, FluxesByband& fluxes)
    {
        fluxes.flux_up     = real2d("flux_up"    , nz, nlay+1);
        fluxes.flux_dn     = real2d("flux_dn"    , nz, nlay+1);
        fluxes.flux_net    = real2d("flux_net"   , nz, nlay+1);
        fluxes.flux_dn_dir = real2d("flux_dn_dir", nz, nlay+1);

        fluxes.bnd_flux_up     = real3d("flux_up"    , nz, nlay+1, nbands);
        fluxes.bnd_flux_dn     = real3d("flux_dn"    , nz, nlay+1, nbands);
        fluxes.bnd_flux_net    = real3d("flux_net"   , nz, nlay+1, nbands);
        fluxes.bnd_flux_dn_dir = real3d("flux_dn_dir", nz, nlay+1, nbands);
    }

    void expand_day_fluxes (const FluxesByband& daytime_fluxes,
                            FluxesByband& expanded_fluxes,
                            const int1d& day_indices)
    {
        auto ncol  = size(daytime_fluxes.bnd_flux_up, 1);
        auto nlev  = size(daytime_fluxes.bnd_flux_up, 2);
        auto nbnds = size(daytime_fluxes.bnd_flux_up, 3);

        int1d nday_1d("nday_1d", 1),nday_host("nday_host",1);
        yakl::memset(nday_1d, 0);
        parallel_for(SimpleBounds<1>(ncol), YAKL_LAMBDA (int icol)
        {
            if (day_indices(icol) > 0) nday_1d(1)++;
            //printf("daynight indices(check): %d, %d, %d\n",icol,day_indices(icol),nday_1d(1));
        });

        nday_1d.deep_copy_to(nday_host);
        auto nday = nday_host(1);
        AMREX_ASSERT_WITH_MESSAGE((nday>0) && (nday<=ncol), "RADIATION: Invalid number of days!");
        parallel_for(SimpleBounds<3>(nday, nlev, nbnds), YAKL_LAMBDA (int iday, int ilev, int ibnd)
        {
            // Map daytime index to proper column index
            auto icol = day_indices(iday);
            //auto icol = iday;
            // Expand broadband fluxes
            expanded_fluxes.flux_up(icol,ilev)     = daytime_fluxes.flux_up(iday,ilev);
            expanded_fluxes.flux_dn(icol,ilev)     = daytime_fluxes.flux_dn(iday,ilev);
            expanded_fluxes.flux_net(icol,ilev)    = daytime_fluxes.flux_net(iday,ilev);
            expanded_fluxes.flux_dn_dir(icol,ilev) = daytime_fluxes.flux_dn_dir(iday,ilev);

            // Expand band-by-band fluxes
            expanded_fluxes.bnd_flux_up(icol,ilev,ibnd)     = daytime_fluxes.bnd_flux_up(iday,ilev,ibnd);
            expanded_fluxes.bnd_flux_dn(icol,ilev,ibnd)     = daytime_fluxes.bnd_flux_dn(iday,ilev,ibnd);
            expanded_fluxes.bnd_flux_net(icol,ilev,ibnd)    = daytime_fluxes.bnd_flux_net(iday,ilev,ibnd);
            expanded_fluxes.bnd_flux_dn_dir(icol,ilev,ibnd) = daytime_fluxes.bnd_flux_dn_dir(iday,ilev,ibnd);
        });
    }

    // Utility function to reorder an array given a new indexing
    void reordered (const real1d& array_in, const int1d& new_indexing, const real1d& array_out)
    {
        // Reorder array based on input index mapping, which maps old indices to new
        parallel_for(SimpleBounds<1>(size(array_in, 1)), YAKL_LAMBDA (int i)
        {
            array_out(i) = array_in(new_indexing(i));
        });
    }
}

// init
void Radiation::initialize (const MultiFab& cons_in,
                            MultiFab* lsm_fluxes,
                            MultiFab* lsm_zenith,
                            MultiFab* qheating_rates,
                            MultiFab* lat,
                            MultiFab* lon,
                            Vector<MultiFab*> qmoist,
                            const BoxArray& grids,
                            const Geometry& geom,
                            const Real& dt_advance,
                            const bool& do_sw_rad,
                            const bool& do_lw_rad,
                            const bool& do_aero_rad,
                            const bool& do_snow_opt,
                            const bool& is_cmip6_volcano)
{
    m_geom = geom;
    m_box = grids;

    qrad_src = qheating_rates;

    auto dz   = m_geom.CellSize(2);
    auto lowz = m_geom.ProbLo(2);

    dt = dt_advance;

    do_short_wave_rad = do_sw_rad;
    do_long_wave_rad  = do_lw_rad;
    do_aerosol_rad    = do_aero_rad;
    do_snow_optics    = do_snow_opt;
    is_cmip6_volc     = is_cmip6_volcano;

    m_lat = lat;
    m_lon = lon;

    m_lsm_fluxes = lsm_fluxes;
    m_lsm_zenith = lsm_zenith;

    rrtmgp_data_path = getRadiationDataDir() + "/";
    rrtmgp_coefficients_file_sw = rrtmgp_data_path + rrtmgp_coefficients_file_name_sw;
    rrtmgp_coefficients_file_lw = rrtmgp_data_path + rrtmgp_coefficients_file_name_lw;

    ParmParse pp("erf");
    pp.query("fixed_total_solar_irradiance", fixed_total_solar_irradiance);
    pp.query("radiation_uniform_angle"     , uniform_angle);
    pp.query("moisture_model", moisture_type); // TODO: get from SolverChoice?
    has_qmoist = (moisture_type != "None");

    nlev = geom.Domain().length(2);
    ncol = 0;
    rank_offsets.resize(cons_in.local_size());
    for (MFIter mfi(cons_in, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();
        int nx = box3d.length(0);
        int ny = box3d.length(1);
        rank_offsets[mfi.LocalIndex()] = ncol;
        ncol += nx * ny;
    }

    ngas = active_gases.size();

    // initialize cloud, aerosol, and radiation
    radiation.initialize(ngas, active_gases,
                         rrtmgp_coefficients_file_sw.c_str(),
                         rrtmgp_coefficients_file_lw.c_str());

    // initialize the radiation data
    nswbands = radiation.get_nband_sw();
    nswgpts  = radiation.get_ngpt_sw();
    nlwbands = radiation.get_nband_lw();
    nlwgpts  = radiation.get_ngpt_lw();

    rrtmg_to_rrtmgp = int1d("rrtmg_to_rrtmgp",14);
    parallel_for(14, YAKL_LAMBDA (int i)
    {
        if (i == 1) {
            rrtmg_to_rrtmgp(i) = 13;
        } else {
            rrtmg_to_rrtmgp(i) = i - 1;
        }
    });

    tmid = real2d("tmid", ncol, nlev);
    pmid = real2d("pmid", ncol, nlev);
    pdel = real2d("pdel", ncol, nlev);

    pint = real2d("pint", ncol, nlev+1);
    tint = real2d("tint", ncol, nlev+1);

    qt   = real2d("qt", ncol, nlev);
    qc   = real2d("qc", ncol, nlev);
    qi   = real2d("qi", ncol, nlev);
    qn   = real2d("qn", ncol, nlev);
    zi   = real2d("zi", ncol, nlev);

    // Get the temperature, density, theta, qt and qp from input
    for (MFIter mfi(cons_in, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();
        auto nx = box3d.length(0);

        auto states_array = cons_in.array(mfi);
        auto qt_array = (has_qmoist) ? qmoist[0]->array(mfi) : Array4<Real> {};
        auto qv_array = (has_qmoist) ? qmoist[1]->array(mfi) : Array4<Real> {};
        auto qc_array = (has_qmoist) ? qmoist[2]->array(mfi) : Array4<Real> {};
        auto qi_array = (has_qmoist && qmoist.size()>=8) ? qmoist[3]->array(mfi) : Array4<Real> {};
        const int offset = rank_offsets[mfi.LocalIndex()];

        // Get pressure, theta, temperature, density, and qt, qp
        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            auto icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
            auto ilev = k+1;
            Real qv         = (qv_array) ? qv_array(i,j,k): 0.0;
            qt(icol,ilev)   = (qt_array) ? qt_array(i,j,k): 0.0;
            qc(icol,ilev)   = (qc_array) ? qc_array(i,j,k): 0.0;
            qi(icol,ilev)   = (qi_array) ? qi_array(i,j,k): 0.0;
            qn(icol,ilev)   = qc(icol,ilev) + qi(icol,ilev);
            tmid(icol,ilev) = getTgivenRandRTh(states_array(i,j,k,Rho_comp),states_array(i,j,k,RhoTheta_comp),qv);
            // NOTE: RRTMGP code expects pressure in pa
            pmid(icol,ilev) = getPgivenRTh(states_array(i,j,k,RhoTheta_comp),qv);
        });
    }

    parallel_for(SimpleBounds<2>(ncol, nlev+1), YAKL_LAMBDA (int icol, int ilev)
    {
        if (ilev == 1) {
            pint(icol, 1) = -0.5*pmid(icol, 2) + 1.5*pmid(icol, 1);
            tint(icol, 1) = -0.5*tmid(icol, 2) + 1.5*tmid(icol, 1);
        } else if (ilev <= nlev) {
            pint(icol, ilev) = 0.5*(pmid(icol, ilev-1) + pmid(icol, ilev));
            tint(icol, ilev) = 0.5*(tmid(icol, ilev-1) + tmid(icol, ilev));
        } else {
            pint(icol, nlev+1) = -0.5*pmid(icol, nlev-1) + 1.5*pmid(icol, nlev);
            tint(icol, nlev+1) = -0.5*tmid(icol, nlev-1) + 1.5*tmid(icol, nlev);
        }
    });

    parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
    {
        zi(icol, ilev)  = lowz + (ilev+0.5)*dz;
        pdel(icol,ilev) = pint(icol,ilev+1) - pint(icol,ilev);
    });

    albedo_dir = real2d("albedo_dir", nswbands, ncol);
    albedo_dif = real2d("albedo_dif", nswbands, ncol);

    qrs = real2d("qrs", ncol, nlev);   // shortwave radiative heating rate
    qrl = real2d("qrl", ncol, nlev);   // longwave  radiative heating rate

    // Clear-sky heating rates are not on the physics buffer, and we have no
    // reason to put them there, so declare these are regular arrays here
    qrsc = real2d("qrsc", ncol, nlev);
    qrlc = real2d("qrlc", ncol, nlev);

    int nmodes = 3;
    int nrh = 1;
    int top_lev = 1;
    naer = 4;
    std::vector<std::string> aero_names {"H2O", "N2", "O2", "O3"};
    auto geom_radius = real2d("geom_radius", ncol, nlev);
    yakl::memset(geom_radius, 0.1);

    optics.initialize(ngas, nmodes, naer, nswbands, nlwbands,
                      ncol, nlev, nrh, top_lev, aero_names, zi,
                      pmid, pdel, tmid, qt, geom_radius);

    amrex::Print() << "LW coefficients file: " << rrtmgp_coefficients_file_lw
                   << "\nSW coefficients file: " << rrtmgp_coefficients_file_sw
                   << "\nFrequency (timesteps) of Shortwave Radiation calc: " << dt
                   << "\nFrequency (timesteps) of Longwave Radiation calc:  " << dt
                   << "\nDo aerosol radiative calculations: " << do_aerosol_rad << std::endl;

}


// run radiation model
void Radiation::run ()
{
    // Cosine solar zenith angle for all columns in chunk
    coszrs = real1d("coszrs", ncol);

    // Pointers to fields on the physics buffer
    cld = real2d("cld", ncol, nlev);
    cldfsnow = real2d("cldfsnow", ncol, nlev);
    iclwp = real2d("iclwp", ncol, nlev);
    iciwp = real2d("iciwp", ncol, nlev);
    icswp = real2d("icswp", ncol, nlev);
    dei = real2d("dei", ncol, nlev);
    des = real2d("des", ncol, nlev);
    lambdac = real2d("lambdac", ncol, nlev);
    mu = real2d("mu", ncol, nlev);
    rei = real2d("rei", ncol, nlev);
    rel = real2d("rel", ncol, nlev);

    // Cloud, snow, and aerosol optical properties
    cld_tau_gpt_sw = real3d("cld_tau_gpt_sw", ncol, nlev, nswgpts);
    cld_ssa_gpt_sw = real3d("cld_ssa_gpt_sw", ncol, nlev, nswgpts);
    cld_asm_gpt_sw = real3d("cld_asm_gpt_sw", ncol, nlev, nswgpts);

    cld_tau_bnd_sw = real3d("cld_tau_bnd_sw", ncol, nlev, nswbands);
    cld_ssa_bnd_sw = real3d("cld_ssa_bnd_sw", ncol, nlev, nswbands);
    cld_asm_bnd_sw = real3d("cld_asm_bnd_sw", ncol, nlev, nswbands);

    aer_tau_bnd_sw = real3d("aer_tau_bnd_sw", ncol, nlev, nswbands);
    aer_ssa_bnd_sw = real3d("aer_ssa_bnd_sw", ncol, nlev, nswbands);
    aer_asm_bnd_sw = real3d("aer_asm_bnd_sw", ncol, nlev, nswbands);

    cld_tau_bnd_lw = real3d("cld_tau_bnd_lw", ncol, nlev, nlwbands);
    aer_tau_bnd_lw = real3d("aer_tau_bnd_lw", ncol, nlev, nlwbands);

    cld_tau_gpt_lw = real3d("cld_tau_gpt_lw", ncol, nlev, nlwgpts);

    // NOTE: these are diagnostic only
    liq_tau_bnd_sw = real3d("liq_tau_bnd_sw", ncol, nlev, nswbands);
    ice_tau_bnd_sw = real3d("ice_tau_bnd_sw", ncol, nlev, nswbands);
    snw_tau_bnd_sw = real3d("snw_tau_bnd_sw", ncol, nlev, nswbands);
    liq_tau_bnd_lw = real3d("liq_tau_bnd_lw", ncol, nlev, nlwbands);
    ice_tau_bnd_lw = real3d("ice_tau_bnd_lw", ncol, nlev, nlwbands);
    snw_tau_bnd_lw = real3d("snw_tau_bnd_lw", ncol, nlev, nlwbands);

    // Gas volume mixing ratios
    gas_vmr = real3d("gas_vmr", ngas, ncol, nlev);

    // Needed for shortwave aerosol;
    //int nday, nnight;     // Number of daylight columns
    int1d day_indices("day_indices", ncol), night_indices("night_indices", ncol);   // Indices of daylight coumns

    // Flag to carry (QRS,QRL)*dp across time steps.
    // TODO: what does this mean?
    bool conserve_energy = true;

    // For loops over diagnostic calls
    //bool active_calls(0:N_DIAG)

    // Zero-array for cloud properties if not diagnosed by microphysics
    real2d zeros("zeros", ncol, nlev);

    gpoint_bands_sw = int1d("gpoint_bands_sw", nswgpts);
    gpoint_bands_lw = int1d("gpoint_bands_lw", nlwgpts);

    // Do shortwave stuff...
    if (do_short_wave_rad) {
        // Radiative fluxes
        internal::initial_fluxes(ncol, nlev+1, nswbands, sw_fluxes_allsky);
        internal::initial_fluxes(ncol, nlev+1, nswbands, sw_fluxes_clrsky);

        // TODO: Integrate calendar day computation
        int calday = 1;
        // Get cosine solar zenith angle for current time step.
        if (m_lat) {
            zenith(calday, m_lat, m_lon, rank_offsets, coszrs, ncol,
                   eccen,  mvelpp, lambm0, obliqr);
        } else {
            zenith(calday, m_lat, m_lon, rank_offsets, coszrs, ncol,
                   eccen,  mvelpp, lambm0, obliqr, uniform_angle);
        }

        // Get albedo. This uses CAM routines internally and just provides a
        // wrapper to improve readability of the code here.
        set_albedo(coszrs, albedo_dir, albedo_dif);

        // Do shortwave cloud optics calculations
        yakl::memset(cld_tau_gpt_sw, 0.);
        yakl::memset(cld_ssa_gpt_sw, 0.);
        yakl::memset(cld_asm_gpt_sw, 0.);

        // set cloud fraction to be 1, and snow fraction 0
        yakl::memset(cldfsnow, 0.0);
        yakl::memset(cld, 1.0);

        parallel_for (SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int i, int k)
        {
            iciwp(i,k) = std::min(qi(i,k)/std::max(1.0e-4,cld(i,k)),0.005)*pmid(i,k)/CONST_GRAV;
            iclwp(i,k) = std::min(qt(i,k)/std::max(1.0e-4,cld(i,k)),0.005)*pmid(i,k)/CONST_GRAV;
            icswp(i,k) = qn(i,k)/std::max(1.0e-4,cldfsnow(i,k))*pmid(i,k)/CONST_GRAV;
        });

        m2005_effradius(qc, qc, qi, qi, qt, qt, cld, pmid, tmid,
                        rel, rei, dei, lambdac, mu, des);

        // calculate the cloud radiation
        optics.get_cloud_optics_sw(ncol, nlev, nswbands, do_snow_optics, cld,
                                   cldfsnow, iclwp, iciwp, icswp,
                                   lambdac, mu, dei, des, rel, rei,
                                   cld_tau_bnd_sw, cld_ssa_bnd_sw, cld_asm_bnd_sw,
                                   liq_tau_bnd_sw, ice_tau_bnd_sw, snw_tau_bnd_sw);

        // Now reorder bands to be consistent with RRTMGP
        // We need to fix band ordering because the old input files assume RRTMG
        // band ordering, but this has changed in RRTMGP.
        // TODO: fix the input files themselves!
        cld_tau_bnd_sw_1d = real1d("cld_tau_bnd_sw_1d", nswbands);
        cld_ssa_bnd_sw_1d = real1d("cld_ssa_bnd_sw_1d", nswbands);
        cld_asm_bnd_sw_1d = real1d("cld_asm_bnd_sw_1d", nswbands);
        cld_tau_bnd_sw_o_1d = real1d("cld_tau_bnd_sw_1d", nswbands);
        cld_ssa_bnd_sw_o_1d = real1d("cld_ssa_bnd_sw_1d", nswbands);
        cld_asm_bnd_sw_o_1d = real1d("cld_asm_bnd_sw_1d", nswbands);

        parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilay)
        {
            for (auto ibnd = 1; ibnd <= nswbands; ++ibnd) {
                cld_tau_bnd_sw_1d(ibnd) = cld_tau_bnd_sw(icol,ilay,ibnd);
                cld_ssa_bnd_sw_1d(ibnd) = cld_ssa_bnd_sw(icol,ilay,ibnd);
                cld_asm_bnd_sw_1d(ibnd) = cld_asm_bnd_sw(icol,ilay,ibnd);
            }
            internal::reordered(cld_tau_bnd_sw_1d, rrtmg_to_rrtmgp, cld_tau_bnd_sw_o_1d);
            internal::reordered(cld_ssa_bnd_sw_1d, rrtmg_to_rrtmgp, cld_ssa_bnd_sw_o_1d);
            internal::reordered(cld_asm_bnd_sw_1d, rrtmg_to_rrtmgp, cld_asm_bnd_sw_o_1d);
            for (auto ibnd = 1; ibnd <= nswbands; ++ibnd) {
                cld_tau_bnd_sw(icol,ilay,ibnd) = cld_tau_bnd_sw_o_1d(ibnd);
                cld_ssa_bnd_sw(icol,ilay,ibnd) = cld_ssa_bnd_sw_o_1d(ibnd);
                cld_asm_bnd_sw(icol,ilay,ibnd) = cld_asm_bnd_sw_o_1d(ibnd);
            }
        });

        // And now do the MCICA sampling to get cloud optical properties by
        // gpoint/cloud state
        radiation.get_gpoint_bands_sw(gpoint_bands_sw);

        optics.sample_cloud_optics_sw(ncol, nlev, nswgpts, gpoint_bands_sw,
                                      pmid, cld, cldfsnow,
                                      cld_tau_bnd_sw, cld_ssa_bnd_sw, cld_asm_bnd_sw,
                                      cld_tau_gpt_sw, cld_ssa_gpt_sw, cld_asm_gpt_sw);

        // Aerosol needs night indices
        // TODO: remove this dependency, it's just used to mask aerosol outputs
        set_daynight_indices(coszrs, day_indices, night_indices);
        int1d nday("nday",1);
        int1d nnight("nnight",1);
        yakl::memset(nday, 0);
        yakl::memset(nnight, 0);
        for (auto icol=1; icol<=ncol; ++icol) {
            if (day_indices(icol) > 0) nday(1)++;
            if (night_indices(icol) > 0) nnight(1)++;
        }

        AMREX_ALWAYS_ASSERT(nday(1) + nnight(1) == ncol);

        // get aerosol optics
        do_aerosol_rad = false; // TODO: this causes issues if enabled
        {
            // Get gas concentrations
            get_gas_vmr(active_gases, gas_vmr);

            // Get aerosol optics
            if (do_aerosol_rad) {
                yakl::memset(aer_tau_bnd_sw, 0.);
                yakl::memset(aer_ssa_bnd_sw, 0.);
                yakl::memset(aer_asm_bnd_sw, 0.);

                clear_rh = real2d("clear_rh",ncol, nswbands);
                yakl::memset(clear_rh, 0.01);

                optics.set_aerosol_optics_sw(0, ncol, nlev, nswbands, dt, night_indices,
                                             is_cmip6_volc, aer_tau_bnd_sw, aer_ssa_bnd_sw, aer_asm_bnd_sw, clear_rh);

                // Now reorder bands to be consistent with RRTMGP
                // TODO: fix the input files themselves!
                aer_tau_bnd_sw_1d = real1d("cld_tau_bnd_sw_1d", nswbands);
                aer_ssa_bnd_sw_1d = real1d("cld_ssa_bnd_sw_1d", nswbands);
                aer_asm_bnd_sw_1d = real1d("cld_asm_bnd_sw_1d", nswbands);
                aer_tau_bnd_sw_o_1d = real1d("cld_tau_bnd_sw_1d", nswbands);
                aer_ssa_bnd_sw_o_1d = real1d("cld_ssa_bnd_sw_1d", nswbands);
                aer_asm_bnd_sw_o_1d = real1d("cld_asm_bnd_sw_1d", nswbands);

                parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilay)
                {
                    for (auto ibnd = 1; ibnd <= nswbands; ++ibnd) {
                        aer_tau_bnd_sw_1d(ibnd) = aer_tau_bnd_sw(icol,ilay,ibnd);
                        aer_ssa_bnd_sw_1d(ibnd) = aer_ssa_bnd_sw(icol,ilay,ibnd);
                        aer_asm_bnd_sw_1d(ibnd) = aer_asm_bnd_sw(icol,ilay,ibnd);
                    }
                    internal::reordered(aer_tau_bnd_sw_1d, rrtmg_to_rrtmgp, aer_tau_bnd_sw_o_1d);
                    internal::reordered(aer_ssa_bnd_sw_1d, rrtmg_to_rrtmgp, aer_ssa_bnd_sw_o_1d);
                    internal::reordered(aer_asm_bnd_sw_1d, rrtmg_to_rrtmgp, aer_asm_bnd_sw_o_1d);
                    for (auto ibnd = 1; ibnd <= nswbands; ++ibnd) {
                        aer_tau_bnd_sw(icol,ilay,ibnd) = aer_tau_bnd_sw_o_1d(ibnd);
                        aer_ssa_bnd_sw(icol,ilay,ibnd) = aer_ssa_bnd_sw_o_1d(ibnd);
                        aer_asm_bnd_sw(icol,ilay,ibnd) = aer_asm_bnd_sw_o_1d(ibnd);
                    }
                });
            } else {
                yakl::memset(aer_tau_bnd_sw, 0.);
                yakl::memset(aer_ssa_bnd_sw, 0.);
                yakl::memset(aer_asm_bnd_sw, 0.);
            }

         yakl::memset(cld_tau_gpt_sw, 0.);
         yakl::memset(cld_ssa_gpt_sw, 0.);
         yakl::memset(cld_asm_gpt_sw, 0.);

         // Call the shortwave radiation driver
         radiation_driver_sw(ncol, gas_vmr,
                             pmid, pint, tmid, albedo_dir, albedo_dif, coszrs,
                             cld_tau_gpt_sw, cld_ssa_gpt_sw, cld_asm_gpt_sw,
                             aer_tau_bnd_sw, aer_ssa_bnd_sw, aer_asm_bnd_sw,
                             sw_fluxes_allsky, sw_fluxes_clrsky, qrs, qrsc);
        }

        // Set surface fluxes that are used by the land model
        export_surface_fluxes(sw_fluxes_allsky, "shortwave");

    } else {
        // Conserve energy
        if (conserve_energy) {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                qrs(icol,ilev) = qrs(icol,ilev)/pdel(icol,ilev);
            });
        }
    }  // dosw

    // Do longwave stuff...
    if (do_long_wave_rad) {
        // Allocate longwave outputs; why is this not part of the fluxes_t object?
        internal::initial_fluxes(ncol, nlev, nlwbands, lw_fluxes_allsky);
        internal::initial_fluxes(ncol, nlev, nlwbands, lw_fluxes_clrsky);

        // NOTE: fluxes defined at interfaces, so initialize to have vertical dimension nlev_rad+1
        yakl::memset(cld_tau_gpt_lw, 0.);

        optics.get_cloud_optics_lw(ncol, nlev, nlwbands, do_snow_optics, cld, cldfsnow, iclwp, iciwp, icswp,
                                   lambdac, mu, dei, des, rei,
                                   cld_tau_bnd_lw, liq_tau_bnd_lw, ice_tau_bnd_lw, snw_tau_bnd_lw);

        radiation.get_gpoint_bands_lw(gpoint_bands_lw);

        optics.sample_cloud_optics_lw(ncol, nlev, nlwgpts, gpoint_bands_lw,
                                      pmid, cld, cldfsnow,
                                      cld_tau_bnd_lw, cld_tau_gpt_lw);

        // Get gas concentrations
        get_gas_vmr(active_gases, gas_vmr);

        // Get aerosol optics
        yakl::memset(aer_tau_bnd_lw, 0.);
        if (do_aerosol_rad) {
            aer_rad.aer_rad_props_lw(is_cmip6_volc, 0, dt, zi, aer_tau_bnd_lw, clear_rh);
        }

        // Call the longwave radiation driver to calculate fluxes and heating rates
        radiation_driver_lw(ncol, nlev, gas_vmr, pmid, pint, tmid, tint, cld_tau_gpt_lw, aer_tau_bnd_lw,
                            lw_fluxes_allsky, lw_fluxes_clrsky, qrl, qrlc);

        // Set surface fluxes that are used by the land model
        export_surface_fluxes(lw_fluxes_allsky, "longwave");

    }
    else {
        // Conserve energy (what does this mean exactly?)
        if (conserve_energy) {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                qrl(icol,ilev) = qrl(icol,ilev)/pdel(icol,ilev);
            });
        }
    } // dolw

    // Populate source term for theta dycore variable
    for (MFIter mfi(*(qrad_src)); mfi.isValid(); ++mfi) {
        auto qrad_src_array = qrad_src->array(mfi);
        const auto& box3d = mfi.tilebox();
        auto nx = box3d.length(0);
        int const offset = rank_offsets[mfi.LocalIndex()];
        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Map (col,lev) to (i,j,k)
            auto icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
            auto ilev = k+1;

            // TODO: We do not include the cloud source term qrsc/qrlc.
            //       Do these simply sum for a net source or do we pick one?

            // SW and LW sources
            qrad_src_array(i,j,k,0) = qrs(icol,ilev);
            qrad_src_array(i,j,k,1) = qrl(icol,ilev);
        });
    }
}

void Radiation::radiation_driver_sw (int ncol, const real3d& gas_vmr,
                                     const real2d& pmid, const real2d& pint, const real2d& tmid,
                                     const real2d& albedo_dir, const real2d& albedo_dif, const real1d& coszrs,
                                     const real3d& cld_tau_gpt, const real3d& cld_ssa_gpt, const real3d& cld_asm_gpt,
                                     const real3d& aer_tau_bnd, const real3d& aer_ssa_bnd, const real3d& aer_asm_bnd,
                                     FluxesByband& fluxes_clrsky, FluxesByband& fluxes_allsky, const real2d& qrs,
                                     const real2d& qrsc)
{
    // Incoming solar radiation, scaled for solar zenith angle
    // and earth-sun distance
    real2d solar_irradiance_by_gpt("solar_irradiance_by_gpt",ncol,nswgpts);

    // Gathered indices of day and night columns
    // chunk_column_index = day_indices(daylight_column_index)
    int1d day_indices("day_indices",ncol), night_indices("night_indices", ncol);   // Indices of daylight coumns

    real1d coszrs_day("coszrs_day", ncol);
    real2d albedo_dir_day("albedo_dir_day", nswbands, ncol), albedo_dif_day("albedo_dif_day", nswbands, ncol);
    real2d pmid_day("pmid_day", ncol, nlev);
    real2d tmid_day("tmid_day", ncol, nlev);
    real2d pint_day("pint_day", ncol, nlev+1);

    real3d gas_vmr_day("gas_vmr_day", ngas, ncol, nlev);

    real3d cld_tau_gpt_day("cld_tau_gpt_day", ncol, nlev, nswgpts);
    real3d cld_ssa_gpt_day("cld_ssa_gpt_day", ncol, nlev, nswgpts);
    real3d cld_asm_gpt_day("cld_asm_gpt_day", ncol, nlev, nswgpts);
    real3d aer_tau_bnd_day("aer_tau_bnd_day", ncol, nlev, nswbands);
    real3d aer_ssa_bnd_day("aer_ssa_bnd_day", ncol, nlev, nswbands);
    real3d aer_asm_bnd_day("aer_asm_bnd_day", ncol, nlev, nswbands);

    real3d cld_tau_gpt_rad("cld_tau_gpt_rad", ncol, nlev+1, nswgpts);
    real3d cld_ssa_gpt_rad("cld_ssa_gpt_rad", ncol, nlev+1, nswgpts);
    real3d cld_asm_gpt_rad("cld_asm_gpt_rad", ncol, nlev+1, nswgpts);
    real3d aer_tau_bnd_rad("aer_tau_bnd_rad", ncol, nlev+1, nswgpts);
    real3d aer_ssa_bnd_rad("aer_ssa_bnd_rad", ncol, nlev+1, nswgpts);
    real3d aer_asm_bnd_rad("aer_asm_bnd_rad", ncol, nlev+1, nswgpts);

    // Scaling factor for total sky irradiance; used to account for orbital
    // eccentricity, and could be used to scale total sky irradiance for different
    // climates as well (i.e., paleoclimate simulations)
    real tsi_scaling;
    real solar_declination;

    if (fixed_total_solar_irradiance<0) {
        // TODO: Integrate calendar day computation
        int calday = 1;
        // Get orbital eccentricity factor to scale total sky irradiance
        shr_orb_decl(calday, eccen, mvelpp, lambm0, obliqr, solar_declination, tsi_scaling);
    } else {
        // For fixed TSI we divide by the default solar constant of 1360.9
        // At some point we will want to replace this with a method that
        // retrieves the solar constant
        tsi_scaling = fixed_total_solar_irradiance / 1360.9;
    }

    // Gather night/day column indices for subsetting SW inputs; we only want to
    // do the shortwave radiative transfer during the daytime to save
    // computational cost (and because RRTMGP will fail for cosine solar zenith
    // angles less than or equal to zero)
    set_daynight_indices(coszrs, day_indices, night_indices);
    int1d nday("nday",1);
    int1d nnight("nnight", 1);
    yakl::memset(nday, 0);
    yakl::memset(nnight, 0);
    parallel_for(SimpleBounds<1>(ncol), YAKL_LAMBDA (int icol)
    {
        if (day_indices(icol) > 0) nday(1)++;
        if (night_indices(icol) > 0) nnight(1)++;
    });

    AMREX_ASSERT(nday(1) + nnight(1) == ncol);

    intHost1d num_day("num_day",1);
    intHost1d num_night("num_night",1);
    nday.deep_copy_to(num_day);
    nnight.deep_copy_to(num_night);

    // If no daytime columns in this chunk, then we return zeros
    if (num_day(1) == 0) {
        //    reset_fluxes(fluxes_allsky)
        //    reset_fluxes(fluxes_clrsky)
        yakl::memset(qrs, 0.);
        yakl::memset(qrsc, 0.);
        return;
    }

    // Compress to daytime-only arrays
    parallel_for(SimpleBounds<2>(num_day(1), nlev), YAKL_LAMBDA (int iday, int ilev)
    {
        // 2D arrays
        auto icol = day_indices(iday);
        tmid_day(iday,ilev) = tmid(icol,ilev);
        pmid_day(iday,ilev) = pmid(icol,ilev);
        pint_day(iday,ilev) = pint(icol,ilev);
    });
    parallel_for(SimpleBounds<1>(num_day(1)), YAKL_LAMBDA (int iday)
    {
        // copy extra level for pmid
        auto icol = day_indices(iday);
        pint_day(iday,nlev+1) = pint(icol,nlev+1);

        coszrs_day(iday) = coszrs(icol);
        AMREX_ASSERT(coszrs_day(iday) > 0.0);
    });
    parallel_for(SimpleBounds<3>(num_day(1), nlev, nswgpts), YAKL_LAMBDA (int iday, int ilev, int igpt)
    {
        auto icol = day_indices(iday);
        cld_tau_gpt_day(iday,ilev,igpt) = cld_tau_gpt(icol,ilev,igpt);
        cld_ssa_gpt_day(iday,ilev,igpt) = cld_ssa_gpt(icol,ilev,igpt);
        cld_asm_gpt_day(iday,ilev,igpt) = cld_asm_gpt(icol,ilev,igpt);
    });
    parallel_for(SimpleBounds<2>(num_day(1), nswbands), YAKL_LAMBDA (int iday, int ibnd)
    {
        // albedo dims: [nswbands, ncol]
        auto icol = day_indices(iday);
        albedo_dir_day(ibnd,iday) = albedo_dir(ibnd,icol);
        albedo_dif_day(ibnd,iday) = albedo_dif(ibnd,icol);
    });

    parallel_for(SimpleBounds<3>(num_day(1), nlev, nswbands), YAKL_LAMBDA (int iday, int ilev, int ibnd)
    {
        auto icol = day_indices(iday);
        aer_tau_bnd_day(iday,ilev,ibnd) = aer_tau_bnd(icol,ilev,ibnd);
        aer_ssa_bnd_day(iday,ilev,ibnd) = aer_ssa_bnd(icol,ilev,ibnd);
        aer_asm_bnd_day(iday,ilev,ibnd) = aer_asm_bnd(icol,ilev,ibnd);
    });

    // Allocate shortwave fluxes (allsky and clearsky)
    // NOTE: fluxes defined at interfaces, so initialize to have vertical
    // dimension nlev_rad+1, while we initialized the RRTMGP input variables to
    // have vertical dimension nlev_rad (defined at midpoints).
    FluxesByband fluxes_clrsky_day, fluxes_allsky_day;
    internal::initial_fluxes(num_day(1), nlev+1, nswbands, fluxes_allsky_day);
    internal::initial_fluxes(num_day(1), nlev+1, nswbands, fluxes_clrsky_day);

    // Add an empty level above model top
    // TODO: combine with day compression above
    yakl::memset(cld_tau_gpt_rad, 0.);
    yakl::memset(cld_ssa_gpt_rad, 0.);
    yakl::memset(cld_asm_gpt_rad, 0.);

    yakl::memset(aer_tau_bnd_rad, 0.);
    yakl::memset(aer_ssa_bnd_rad, 0.);
    yakl::memset(aer_asm_bnd_rad, 0.);

    parallel_for(SimpleBounds<3>(num_day(1), nlev, nswgpts), YAKL_LAMBDA (int iday, int ilev, int igpt)
    {
        cld_tau_gpt_rad(iday,ilev,igpt) = cld_tau_gpt_day(iday,ilev,igpt);
        cld_ssa_gpt_rad(iday,ilev,igpt) = cld_ssa_gpt_day(iday,ilev,igpt);
        cld_asm_gpt_rad(iday,ilev,igpt) = cld_asm_gpt_day(iday,ilev,igpt);
    });

    parallel_for(SimpleBounds<3>(num_day(1), nlev, ngas), YAKL_LAMBDA (int iday, int ilev, int igas)
    {
        auto icol = day_indices(iday);
        gas_vmr_day(igas,iday,ilev) = gas_vmr(igas,icol,ilev);
    });

    parallel_for(SimpleBounds<3>(num_day(1), nlev, nswbands), YAKL_LAMBDA (int iday, int ilev, int ibnd)
    {
        aer_tau_bnd_rad(iday,ilev,ibnd) = aer_tau_bnd_day(iday,ilev,ibnd);
        aer_ssa_bnd_rad(iday,ilev,ibnd) = aer_ssa_bnd_day(iday,ilev,ibnd);
        aer_asm_bnd_rad(iday,ilev,ibnd) = aer_asm_bnd_day(iday,ilev,ibnd);
    });

    // Do shortwave radiative transfer calculations
    radiation.run_shortwave_rrtmgp(ngas, num_day(1), nlev, gas_vmr_day, pmid_day,
                                   tmid_day, pint_day, coszrs_day, albedo_dir_day, albedo_dif_day,
                                   cld_tau_gpt_rad, cld_ssa_gpt_rad, cld_asm_gpt_rad, aer_tau_bnd_rad, aer_ssa_bnd_rad, aer_asm_bnd_rad,
                                   fluxes_allsky_day.flux_up    , fluxes_allsky_day.flux_dn    , fluxes_allsky_day.flux_net    , fluxes_allsky_day.flux_dn_dir    ,
                                   fluxes_allsky_day.bnd_flux_up, fluxes_allsky_day.bnd_flux_dn, fluxes_allsky_day.bnd_flux_net, fluxes_allsky_day.bnd_flux_dn_dir,
                                   fluxes_clrsky_day.flux_up    , fluxes_clrsky_day.flux_dn    , fluxes_clrsky_day.flux_net    , fluxes_clrsky_day.flux_dn_dir    ,
                                   fluxes_clrsky_day.bnd_flux_up, fluxes_clrsky_day.bnd_flux_dn, fluxes_clrsky_day.bnd_flux_net, fluxes_clrsky_day.bnd_flux_dn_dir,
                                   tsi_scaling);

    // Expand fluxes from daytime-only arrays to full chunk arrays
    internal::expand_day_fluxes(fluxes_allsky_day, fluxes_allsky, day_indices);
    internal::expand_day_fluxes(fluxes_clrsky_day, fluxes_clrsky, day_indices);

    // Calculate heating rates
    calculate_heating_rate(fluxes_allsky.flux_up,
                           fluxes_allsky.flux_dn,
                           pint, qrs);

    calculate_heating_rate(fluxes_clrsky.flux_up,
                           fluxes_allsky.flux_dn,
                           pint, qrsc);
}

void Radiation::radiation_driver_lw (int ncol, int nlev,
                                     const real3d& gas_vmr,
                                     const real2d& pmid, const real2d& pint, const real2d& tmid, const real2d& tint,
                                     const real3d& cld_tau_gpt, const real3d& aer_tau_bnd, FluxesByband& fluxes_clrsky,
                                     FluxesByband& fluxes_allsky, const real2d& qrl, const real2d& qrlc)
{
    real3d cld_tau_gpt_rad("cld_tau_gpt_rad", ncol, nlev+1, nlwgpts);
    real3d aer_tau_bnd_rad("aer_tau_bnd_rad", ncol, nlev+1, nlwgpts);

    // Surface emissivity needed for longwave
    real2d surface_emissivity("surface_emissivity", nlwbands, ncol);

    // Temporary heating rates on radiation vertical grid
    real3d gas_vmr_rad("gas_vmr_rad", ngas, ncol, nlev);

    // Set surface emissivity to 1 here. There is a note in the RRTMG
    // implementation that this is treated in the land model, but the old
    // RRTMG implementation also sets this to 1. This probably does not make
    // a lot of difference either way, but if a more intelligent value
    // exists or is assumed in the model we should use it here as well.
    // TODO: set this more intelligently?
    yakl::memset(surface_emissivity, 1.0);

    // Add an empty level above model top
    yakl::memset(cld_tau_gpt_rad, 0.);
    yakl::memset(aer_tau_bnd_rad, 0.);

    parallel_for(SimpleBounds<3>(ncol, nlev, nlwgpts), YAKL_LAMBDA (int icol, int ilev, int igpt)
    {
        cld_tau_gpt_rad(icol,ilev,igpt) = cld_tau_gpt(icol,ilev,igpt);
        aer_tau_bnd_rad(icol,ilev,igpt) = aer_tau_bnd(icol,ilev,igpt);
    });

    parallel_for(SimpleBounds<3>(ncol, nlev, ngas), YAKL_LAMBDA (int icol, int ilev, int igas)
    {
        gas_vmr_rad(igas,icol,ilev) = gas_vmr(igas,icol,ilev);
    });

    // Do longwave radiative transfer calculations
    radiation.run_longwave_rrtmgp(ngas, ncol, nlev,
                                  gas_vmr_rad, pmid, tmid, pint, tint,
                                  surface_emissivity, cld_tau_gpt_rad, aer_tau_bnd_rad,
                                  fluxes_allsky.flux_up    , fluxes_allsky.flux_dn    , fluxes_allsky.flux_net    ,
                                  fluxes_allsky.bnd_flux_up, fluxes_allsky.bnd_flux_dn, fluxes_allsky.bnd_flux_net,
                                  fluxes_clrsky.flux_up    , fluxes_clrsky.flux_dn    , fluxes_clrsky.flux_net    ,
                                  fluxes_clrsky.bnd_flux_up, fluxes_clrsky.bnd_flux_dn, fluxes_clrsky.bnd_flux_net);

    // Calculate heating rates
    calculate_heating_rate(fluxes_allsky.flux_up,
                           fluxes_allsky.flux_dn,
                           pint, qrl);

    calculate_heating_rate(fluxes_allsky.flux_up,
                           fluxes_allsky.flux_dn,
                           pint, qrlc);
}

// Initialize array of daytime indices to be all zero. If any zeros exist when
// we are done, something went wrong.
void Radiation::set_daynight_indices (const real1d& coszrs, const int1d& day_indices, const int1d& night_indices)
{
    // Loop over columns and identify daytime columns as those where the cosine
    // solar zenith angle exceeds zero. Note that we wrap the setting of
    // day_indices in an if-then to make sure we are not accessing day_indices out
    // of bounds, and stopping with an informative error message if we do for some reason.
    int1d iday("iday", 1);
    int1d inight("inight",1);
    yakl::memset(iday, 0);
    yakl::memset(inight, 0);
    yakl::memset(day_indices, 0);
    yakl::memset(night_indices, 0);
    parallel_for(SimpleBounds<1>(ncol), YAKL_LAMBDA (int icol)
    {
        if (coszrs(icol) > 0.) {
            iday(1) += 1;
            day_indices(iday(1)) = icol;
        } else {
            inight(1) += 1;
            night_indices(inight(1)) = icol;
        }
    });
}

void Radiation::get_gas_vmr (const std::vector<std::string>& gas_names, const real3d& gas_vmr)
{
    // Mass mixing ratio
    real2d mmr("mmr", ncol, nlev);

    // Gases and molecular weights. Note that we do NOT have CFCs yet (I think
    // this is coming soon in RRTMGP). RRTMGP also allows for absorption due to
    // CO and N2, which RRTMG did not have.
    const std::vector<std::string> gas_species = {"H2O", "CO2", "O3", "N2O",
                                                  "CO" , "CH4", "O2", "N2"};
    const std::vector<real> mol_weight_gas = {18.01528, 44.00950, 47.9982, 44.0128,
                                              28.01010, 16.04246, 31.9980, 28.0134}; // g/mol
    // Molar weight of air
    //const real mol_weight_air = 28.97; // g/mol
    // Defaults for gases that are not available (TODO: is this still accurate?)
    const real co_vol_mix_ratio = 1.0e-7;
    const real n2_vol_mix_ratio = 0.7906;

    // initialize
    yakl::memset(gas_vmr, 0.);

    // For each gas species needed for RRTMGP, read the mass mixing ratio from the
    // CAM rad_constituents interface, convert to volume mixing ratios, and
    // subset for daytime-only indices if needed.
    for (auto igas = 0; igas < gas_names.size(); ++igas) {

        std::cout << "gas_name: " << gas_names[igas] << "; igas: " << igas << std::endl;

        if (gas_names[igas] == "CO"){
            // CO not available, use default
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = co_vol_mix_ratio;
            });
        } else if (gas_names[igas] == "N2") {
            // N2 not available, use default
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = n2_vol_mix_ratio;
            });
        } else if (gas_names[igas] == "H2O") {
            // Water vapor is represented as specific humidity in CAM, so we
            // need to handle water a little differently
            //        rad_cnst_get_gas(icall, gas_species[igas], mmr);

            // Convert to volume mixing ratio by multiplying by the ratio of
            // molecular weight of dry air to molecular weight of gas. Note that
            // first specific humidity (held in the mass_mix_ratio array read
            // from rad_constituents) is converted to an actual mass mixing ratio.
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = qt(icol,ilev); //mmr(icol,ilev) / (
                //                  1. - mmr(icol,ilev))*mol_weight_air / mol_weight_gas[igas];
            });
        } else if (gas_names[igas] == "CO2") {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 3.8868676125307193E-4;
            });
        } else if (gas_names[igas] == "O3") {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 1.8868676125307193E-7;
            });
        } else if (gas_names[igas] == "N2O") {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 3.8868676125307193E-7;
            });
        } else if (gas_names[igas] == "CH4") {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 1.8868676125307193E-6;
            });
        } else if (gas_names[igas] == "O2") {
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 0.2095;
            });
        } else {
            // Get mass mixing ratio from the rad_constituents interface
            //        rad_cnst_get_gas(icall, gas_species[igas], mmr);

            // Convert to volume mixing ratio by multiplying by the ratio of
            // molecular weight of dry air to molecular weight of gas
            parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
            {
                gas_vmr(igas+1,icol,ilev) = 1.0e-6; //mmr(icol,ilev)
                //                                     * mol_weight_air / mol_weight_gas[igas];
            });
        }
    } // igas
}

// Loop over levels and calculate heating rates; note that the fluxes *should*
// be defined at interfaces, so the loop ktop,kbot and grabbing the current
// and next value of k should be safe. ktop should be the top interface, and
// kbot + 1 should be the bottom interface.
// NOTE: to get heating rate in K/day, normally we would use:
//     H = dF / dp * g * (sec/day) * (1e-5) / (cpair)
// Here we just use
//     H = dF / dp * g
// Why? Something to do with convenience with applying the fluxes to the
// heating tendency?
void Radiation::calculate_heating_rate (const real2d& flux_up,
                                        const real2d& flux_dn,
                                        const real2d& pint,
                                        const real2d& heating_rate)
{
    // NOTE: The pressure is in [pa] for RRTMGP to use.
    //       The fluxes are in [W/m^2] and gravity is [m/s^2].
    //       The heating rate is {dF/dP * g / Cp} with units [K/s]
    real1d heatfac("heatfac",1);
    yakl::memset(heatfac, 1.0/Cp_d);
    parallel_for(SimpleBounds<2>(ncol, nlev), YAKL_LAMBDA (int icol, int ilev)
    {
        heating_rate(icol,ilev) = heatfac(1) * ( (flux_up(icol,ilev+1) - flux_dn(icol,ilev+1))
                                               - (flux_up(icol,ilev  ) - flux_dn(icol,ilev  )) )
                                               *  CONST_GRAV/(pint(icol,ilev+1)-pint(icol,ilev));
        /*
        if (icol==1) {
            amrex::Print() << "HR: " << ilev << ' '
                           << heating_rate(icol,ilev) << ' '
                           << (flux_up(icol,ilev+1) - flux_dn(icol,ilev+1)) << ' '
                           << (flux_up(icol,ilev  ) - flux_dn(icol,ilev  )) << ' '
                           << flux_up(icol,ilev+1) << ' '
                           << flux_dn(icol,ilev+1) << ' '
                           << (flux_up(icol,ilev+1) - flux_dn(icol,ilev+1))
                            - (flux_up(icol,ilev  ) - flux_dn(icol,ilev  )) << ' '
                           << (pint(icol,ilev+1)-pint(icol,ilev)) << ' '
                           << CONST_GRAV << "\n";
        }
        */
    });
}

void
Radiation::export_surface_fluxes(FluxesByband& fluxes,
                                 std::string band)
{
    // No work to be done if we don't have valid pointers
    if (!m_lsm_fluxes) return;

    if (band == "shortwave") {
        real3d flux_dn_diffuse("flux_dn_diffuse", ncol, nlev+1, nswbands);

        // Calculate diffuse flux from total and direct
        // This only occurs at the bottom level (k index)
        parallel_for (SimpleBounds<3>(ncol, 1, nswbands), YAKL_LAMBDA (int icol, int ilev, int ibnd)
        {
            flux_dn_diffuse(icol,ilev,ibnd) = fluxes.bnd_flux_dn(icol,ilev,ibnd)
                                            - fluxes.bnd_flux_dn_dir(icol,ilev,ibnd);
        });

        // Populate the LSM data structure (this is a 2D MF)
        for (MFIter mfi(*(m_lsm_fluxes)); mfi.isValid(); ++mfi) {
            auto lsm_array = m_lsm_fluxes->array(mfi);
            const auto& box3d = mfi.tilebox();
            auto nx = box3d.length(0);
            const int offset = rank_offsets[mfi.LocalIndex()];
            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Map (col,lev) to (i,j,k)
                auto icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
                auto ilev = k+1;

                // Direct fluxes
                Real sum1(0.0), sum2(0.0);
                for (int ibnd(1); ibnd<=9; ++ibnd) {
                    sum1 += fluxes.bnd_flux_dn_dir(icol,ilev,ibnd);
                }
                for (int ibnd(11); ibnd<=14; ++ibnd) {
                    sum2 += fluxes.bnd_flux_dn_dir(icol,ilev,ibnd);
                }
                sum1 += 0.5 * fluxes.bnd_flux_dn_dir(icol,ilev,10);
                sum2 += 0.5 * fluxes.bnd_flux_dn_dir(icol,ilev,10);
                lsm_array(i,j,k,0) = sum1;
                lsm_array(i,j,k,1) = sum2;

                // Diffuse fluxes
                sum1=0.0; sum2=0.0;
                for (int ibnd(1); ibnd<=9; ++ibnd) {
                    sum1 += flux_dn_diffuse(icol,ilev,ibnd);
                }
                for (int ibnd(11); ibnd<=14; ++ibnd) {
                    sum2 += flux_dn_diffuse(icol,ilev,ibnd);
                }
                sum1 += 0.5 * flux_dn_diffuse(icol,ilev,10);
                sum2 += 0.5 * flux_dn_diffuse(icol,ilev,10);
                lsm_array(i,j,k,2) = sum1;
                lsm_array(i,j,k,3) = sum2;

                // Net fluxes
                lsm_array(i,j,k,4) = fluxes.flux_net(icol,ilev);
            });
        }
    } else if (band == "longwave") {
        // Populate the LSM data structure (this is a 2D MF)
        for (MFIter mfi(*(m_lsm_fluxes)); mfi.isValid(); ++mfi) {
            auto lsm_array = m_lsm_fluxes->array(mfi);
            const auto& box3d = mfi.tilebox();
            auto nx = box3d.length(0);
            const int offset = rank_offsets[mfi.LocalIndex()];
            amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Map (col,lev) to (i,j,k)
                auto icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
                auto ilev = k+1;

                // Net fluxes
                lsm_array(i,j,k,5) = fluxes.flux_dn(icol,ilev);
            });
        }
    } else {
         amrex::Abort("Unknown radiation band type!");
    }
}

// call back
void Radiation::on_complete () { }

void Radiation::yakl_to_mf(const real2d &data, amrex::MultiFab &mf)
{
    // creates a MF from a YAKL real2d mapping from [col, lev] to [x,y,z]
    // by reshaping the yakl array to the geometry of the output qsrc multifab
    mf = amrex::MultiFab(m_box, qrad_src->DistributionMap(), 1, 0);
    if (!data.initialized())
    {
        // yakl array hasn't been created yet, so create an empty MF
        mf.setVal(0.0);
        return;
    }

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto mf_arr = mf.array(mfi);
        const auto& box3d = mfi.tilebox();
        const int nx = box3d.length(0);
        const int offset = rank_offsets[mfi.LocalIndex()];
        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // map [i,j,k] 0-based to [icol, ilev] 1-based
            const int icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
            const int ilev = k+1;
            AMREX_ASSERT(icol <= static_cast<int>(data.get_dimensions()(1)));
            AMREX_ASSERT(ilev <= static_cast<int>(data.get_dimensions()(2)));
            mf_arr(i, j, k) = data(icol, ilev);
        });
    }
}

void Radiation::expand_yakl1d_to_mf(const real1d &data, amrex::MultiFab &mf)
{
    // copies the 1D yakl data to a 3D MF
    AMREX_ASSERT(data.get_dimensions()(1) == ncol);
    mf = amrex::MultiFab(m_box, qrad_src->DistributionMap(), 1, 0);
    if (!data.initialized())
    {
        // yakl array hasn't been created yet, so create an empty MF
        mf.setVal(0.0);
        return;
    }

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto mf_arr = mf.array(mfi);
        const auto& box3d = mfi.tilebox();
        const int nx = box3d.length(0);
        const int offset = rank_offsets[mfi.LocalIndex()];
        amrex::ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // map [i,j,k] 0-based to [icol, ilev] 1-based
            const int icol = (j-box3d.smallEnd(1))*nx + (i-box3d.smallEnd(0)) + 1 + offset;
            AMREX_ASSERT(icol <= static_cast<int>(data.get_dimensions()(1)));
            mf_arr(i, j, k) = data(icol);
        });
    }
}

void Radiation::writePlotfile(const std::string& plot_prefix, const amrex::Real time, const int level_step)
{
    // Note: Radiation::initialize() is not called until the first timestep, so skip over the initial file write at t=0
    if (!qrad_src)
    {
        return;
    }

    std::string plotfilename = amrex::Concatenate(plot_prefix + "_rad", level_step, 5);

    // list of real2d (3D) variables to plot
    amrex::Vector<real2d*> plotvars_2d;
    plotvars_2d.push_back(&cld);
    plotvars_2d.push_back(&cldfsnow);
    plotvars_2d.push_back(&iclwp);
    plotvars_2d.push_back(&iciwp);
    plotvars_2d.push_back(&icswp);
    plotvars_2d.push_back(&dei);
    plotvars_2d.push_back(&des);
    plotvars_2d.push_back(&lambdac);
    plotvars_2d.push_back(&mu);
    plotvars_2d.push_back(&rei);
    plotvars_2d.push_back(&rel);

    //   SW allsky
    plotvars_2d.push_back(&sw_fluxes_allsky.flux_up);
    plotvars_2d.push_back(&sw_fluxes_allsky.flux_dn);
    plotvars_2d.push_back(&sw_fluxes_allsky.flux_net);
    plotvars_2d.push_back(&sw_fluxes_allsky.flux_dn_dir);
    //   SW clearsky
    plotvars_2d.push_back(&sw_fluxes_clrsky.flux_up);
    plotvars_2d.push_back(&sw_fluxes_clrsky.flux_dn);
    plotvars_2d.push_back(&sw_fluxes_clrsky.flux_net);
    plotvars_2d.push_back(&sw_fluxes_clrsky.flux_dn_dir);

    //   LW allsky
    plotvars_2d.push_back(&lw_fluxes_allsky.flux_up);
    plotvars_2d.push_back(&lw_fluxes_allsky.flux_dn);
    plotvars_2d.push_back(&lw_fluxes_allsky.flux_net);
    //   LW clearsky
    plotvars_2d.push_back(&lw_fluxes_clrsky.flux_up);
    plotvars_2d.push_back(&lw_fluxes_clrsky.flux_dn);
    plotvars_2d.push_back(&lw_fluxes_clrsky.flux_net);

    plotvars_2d.push_back(&qrs);
    plotvars_2d.push_back(&qrl);
    plotvars_2d.push_back(&zi);
    plotvars_2d.push_back(&clear_rh);
    plotvars_2d.push_back(&qrsc);
    plotvars_2d.push_back(&qrlc);
    plotvars_2d.push_back(&qt);
    plotvars_2d.push_back(&qi);
    plotvars_2d.push_back(&qc);
    plotvars_2d.push_back(&qn);
    plotvars_2d.push_back(&tmid);
    plotvars_2d.push_back(&pmid);
    plotvars_2d.push_back(&pdel);

    //plotvars_2d.push_back(&pint); // NOTE these have nlev + 1
    //plotvars_2d.push_back(&tint);

    //plotvars_2d.push_back(&albedo_dir); // [nswbands, ncol]
    //plotvars_2d.push_back(&albedo_dif);

    // names of plotted variables
    amrex::Vector<std::string> varnames_2d;
    varnames_2d.push_back("cld");
    varnames_2d.push_back("cldfsnow");
    varnames_2d.push_back("iclwp");
    varnames_2d.push_back("iciwp");
    varnames_2d.push_back("icswp");
    varnames_2d.push_back("dei");
    varnames_2d.push_back("des");
    varnames_2d.push_back("lambdac");
    varnames_2d.push_back("mu");
    varnames_2d.push_back("rei");
    varnames_2d.push_back("rel");

    //   SW allsky
    varnames_2d.push_back("sw_fluxes_allsky.flux_up");
    varnames_2d.push_back("sw_fluxes_allsky.flux_dn");
    varnames_2d.push_back("sw_fluxes_allsky.flux_net");
    varnames_2d.push_back("sw_fluxes_allsky.flux_dn_dir");
    //   SW clearsky
    varnames_2d.push_back("sw_fluxes_clrsky.flux_up");
    varnames_2d.push_back("sw_fluxes_clrsky.flux_dn");
    varnames_2d.push_back("sw_fluxes_clrsky.flux_net");
    varnames_2d.push_back("sw_fluxes_clrsky.flux_dn_dir");

    //   LW allsky
    varnames_2d.push_back("lw_fluxes_allsky.flux_up");
    varnames_2d.push_back("lw_fluxes_allsky.flux_dn");
    varnames_2d.push_back("lw_fluxes_allsky.flux_net");
    //   LW clearsky
    varnames_2d.push_back("lw_fluxes_clrsky.flux_up");
    varnames_2d.push_back("lw_fluxes_clrsky.flux_dn");
    varnames_2d.push_back("lw_fluxes_clrsky.flux_net");

    varnames_2d.push_back("qrs");
    varnames_2d.push_back("qrl");
    varnames_2d.push_back("zi");
    varnames_2d.push_back("clear_rh");
    varnames_2d.push_back("qrsc");
    varnames_2d.push_back("qrlc");
    varnames_2d.push_back("qt");
    varnames_2d.push_back("qi");
    varnames_2d.push_back("qc");
    varnames_2d.push_back("qn");
    varnames_2d.push_back("tmid");
    varnames_2d.push_back("pmid");
    varnames_2d.push_back("pdel");

    //varnames_2d.push_back("pint"); // NOTE these have nlev + 1
    //varnames_2d.push_back("tint");

    //varnames_2d.push_back("albedo_dir");
    //varnames_2d.push_back("albedo_dif");

    // list 3D variables defined over bands (SW and LW)
    //   these are 4D fields split into 3D so they can use the same AMReX plotfile
    amrex::Vector<real3d*> plotvars_3d;
    amrex::Vector<std::string> varnames_3d;
    plotvars_3d.push_back(&cld_tau_bnd_sw);
    plotvars_3d.push_back(&cld_ssa_bnd_sw);
    plotvars_3d.push_back(&cld_asm_bnd_sw);
    plotvars_3d.push_back(&aer_tau_bnd_sw);
    plotvars_3d.push_back(&aer_ssa_bnd_sw);
    plotvars_3d.push_back(&aer_asm_bnd_sw);
    plotvars_3d.push_back(&cld_tau_bnd_lw);
    plotvars_3d.push_back(&aer_tau_bnd_lw);
    //   diagnostic variables:
    plotvars_3d.push_back(&liq_tau_bnd_sw);
    plotvars_3d.push_back(&ice_tau_bnd_sw);
    plotvars_3d.push_back(&snw_tau_bnd_sw);
    plotvars_3d.push_back(&liq_tau_bnd_lw);
    plotvars_3d.push_back(&ice_tau_bnd_lw);
    plotvars_3d.push_back(&snw_tau_bnd_lw);

    varnames_3d.push_back("cld_tau_bnd_sw");
    varnames_3d.push_back("cld_ssa_bnd_sw");
    varnames_3d.push_back("cld_asm_bnd_sw");
    varnames_3d.push_back("aer_tau_bnd_sw");
    varnames_3d.push_back("aer_ssa_bnd_sw");
    varnames_3d.push_back("aer_asm_bnd_sw");
    varnames_3d.push_back("cld_tau_bnd_lw");
    varnames_3d.push_back("aer_tau_bnd_lw");
    //   diagnostic variables:
    varnames_3d.push_back("liq_tau_bnd_sw");
    varnames_3d.push_back("ice_tau_bnd_sw");
    varnames_3d.push_back("snw_tau_bnd_sw");
    varnames_3d.push_back("liq_tau_bnd_lw");
    varnames_3d.push_back("ice_tau_bnd_lw");
    varnames_3d.push_back("snw_tau_bnd_lw");

    AMREX_ASSERT(varnames_2d.size() == plotvars_2d.size());
    AMREX_ASSERT(varnames_3d.size() == plotvars_3d.size());

    int output_size = plotvars_2d.size() + 1; // 2D vars + coszrs
    //  add in total output size of all 3D vars
    for (int i = 0; i < plotvars_3d.size(); i++)
    {
        output_size += plotvars_3d[i]->get_dimensions()(3);
    }

    // convert each YAKL array to a MF and combine into a single MF for plotting
    MultiFab fab(m_box, qrad_src->DistributionMap(), output_size, 0);
    // copy 2D vars first
    for (int i = 0; i < plotvars_2d.size(); i++)
    {
        amrex::MultiFab mf;
        yakl_to_mf(*plotvars_2d[i], mf);
        MultiFab::Copy(fab, mf, 0, i, 1, 0);
    }

    int dst = plotvars_2d.size(); // output component index

    // expand coszrs from 2D to 3D so it can be saved with the same file
    varnames_2d.push_back("coszrs");
    amrex::MultiFab coszrs_mf;
    expand_yakl1d_to_mf(coszrs, coszrs_mf);
    MultiFab::Copy(fab, coszrs_mf, 0, dst, 1, 0);
    dst++;

    // copy 3D vars
    for (int i = 0; i < plotvars_3d.size(); i++)
    {
        // split real3ds into real2d for each band for output
        // TODO: find a better way to do this
        const int var_nbands = plotvars_3d[i]->get_dimensions()(3); // either nswbands or nlwbands
        for (int bnd = 1; bnd <= var_nbands; bnd++)
        {
            // add variable name to 2d list
            varnames_2d.push_back(varnames_3d[i] + "_" + std::to_string(bnd));

            amrex::MultiFab mf;
            real2d band_var = plotvars_3d[i]->slice<2>(yakl::COLON, yakl::COLON, bnd);
            yakl_to_mf(band_var, mf);
            MultiFab::Copy(fab, mf, 0, dst, 1, 0);

            dst++;
        }
    }
    AMREX_ASSERT(dst == fab.nComp());

    // this should now match full output size with all 3D variables expanded
    AMREX_ASSERT(varnames_2d.size() == output_size);


    amrex::WriteSingleLevelPlotfile(plotfilename, fab, varnames_2d, m_geom, time, level_step);
}

