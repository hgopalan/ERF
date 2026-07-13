#ifndef ERF_DUST_PARAMS_H
#define ERF_DUST_PARAMS_H

#ifdef ERF_USE_DUST

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <string>
#include <vector>

/**
 * @brief Parameters for ERF-Dust module.
 *
 * Phase 3 parameters (surface properties):
 *   - Grid ratio, particle properties, roughness, thresholds
 *
 * Phase 4 parameters (PHREEQC geochemical coupling):
 *   - File paths, variable names, reduction coefficients
 */
struct DustParams {
    // Phase 1-3: Core dust parameters
    int n_size_bins = 3;                  ///< Number of dust size bins.
    int grid_ratio = 1;                   ///< Dust grid / atmospheric grid ratio.
    amrex::Real particle_density = 2650.0; ///< Particle density [kg/m^3].
    amrex::Real z0_dust = 0.01;           ///< Dust roughness length [m].
    amrex::Real silt_fraction = 0.10;     ///< Silt mass fraction [-].
    amrex::Real threshold_A_coeff = 0.0123; ///< Bagnold u*_t coefficient [-].
    amrex::Real crust_index = 0.0;        ///< Initial crust strength index [0,1].
    std::string surface_map_file = "";    ///< Path to surface property map.
    std::string soil_type_file = "";      ///< Path to soil type map.
    std::string silt_fraction_file = "";  ///< Path to silt fraction map.
    std::string crust_index_file = "";    ///< Path to crust index map.
    std::string moisture_flag_file = "";  ///< Path to moisture flag map.
    std::string suppression_file = "";    ///< Path to suppression coverage map.

    // Phase 4: PHREEQC coupling parameters
    amrex::Real phreeqc_update_interval_s = 86400.0; ///< PHREEQC update interval [s]. Default = 1 day.
    std::string phreeqc_output_file = "";            ///< Path to PHREEQC output file (.csv or .nc).
    std::string phreeqc_crust_var = "crust_index";   ///< Variable/column name for crust index.
    std::string phreeqc_silt_var = "silt_fraction";  ///< Variable/column name for silt fraction.
    std::string phreeqc_efflor_var = "efflorescence"; ///< Variable/column name for efflorescence.
    std::string phreeqc_supp_var = "suppression_mod"; ///< Variable/column name for suppression modifier.
    std::string phreeqc_metal_var = "metal_as_bin0"; ///< Variable/column name for As mass fraction bin 0.
    amrex::Real alpha_crust = 0.5;       ///< Crust strength reduction coefficient for u*_t [-].
                                          ///<   u*_t_new = u*_t_base * (1 - alpha_crust * crust_index).
                                          ///<   Reference: Marticorena & Bergametti (1995).
    amrex::Real alpha_efflor = 0.3;      ///< Efflorescence reduction coefficient for u*_t [-].
    amrex::Real ustar_t_base = -1.0;     ///< Base u*_t before chemistry [m/s].
                                          ///<   Negative means compute from Bagnold at init.

    /**
     * @brief Constructor: read parameters from ParmParse.
     */
    DustParams()
    {
        amrex::ParmParse pp("erf.dust");

        pp.query("n_size_bins", n_size_bins);
        pp.query("grid_ratio", grid_ratio);
        pp.query("particle_density", particle_density);
        pp.query("z0_dust", z0_dust);
        pp.query("silt_fraction", silt_fraction);
        pp.query("threshold_A_coeff", threshold_A_coeff);
        pp.query("crust_index", crust_index);
        pp.query("surface_map_file", surface_map_file);
        pp.query("soil_type_file", soil_type_file);
        pp.query("silt_fraction_file", silt_fraction_file);
        pp.query("crust_index_file", crust_index_file);
        pp.query("moisture_flag_file", moisture_flag_file);
        pp.query("suppression_file", suppression_file);

        // Phase 4 parameters
        pp.query("phreeqc_update_interval_s", phreeqc_update_interval_s);
        pp.query("phreeqc_output_file", phreeqc_output_file);
        pp.query("phreeqc_crust_var", phreeqc_crust_var);
        pp.query("phreeqc_silt_var", phreeqc_silt_var);
        pp.query("phreeqc_efflor_var", phreeqc_efflor_var);
        pp.query("phreeqc_supp_var", phreeqc_supp_var);
        pp.query("phreeqc_metal_var", phreeqc_metal_var);
        pp.query("alpha_crust", alpha_crust);
        pp.query("alpha_efflor", alpha_efflor);
        pp.query("ustar_t_base", ustar_t_base);
    }
};

#endif  // ERF_USE_DUST
#endif  // ERF_DUST_PARAMS_H
