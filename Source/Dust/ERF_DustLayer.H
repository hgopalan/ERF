#ifndef ERF_DUST_LAYER_H
#define ERF_DUST_LAYER_H

#ifdef ERF_USE_DUST

#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <memory>
#include <ERF_DustParams.H>

/**
 * @brief Dust layer for atmospheric transport and emission of dust particles.
 *
 * Phase 3 manages dust grid, surface properties, and emission computations.
 * Phase 4 adds PHREEQC geochemical coupling for mineral crust and efflorescence.
 *
 * **MultiFab layout (dust grid, 1 component each, no ghost cells in horizontal):**
 *
 * | Field | Symbol | Units | Source | Updated by |
 * |-------|--------|-------|--------|------------|
 * | dust_ustar_t | u*_t | m/s | Phase 3 (Bagnold) | Phase 4 (chemistry) |
 * | dust_ustar_base | u*_t,base | m/s | Phase 3 (Bagnold) | (read-only) |
 * | dust_crust_index | C_I | [0,1] | Phase 3 init | Phase 4 (PHREEQC) |
 * | dust_silt_fraction | f_silt | [0,1] | Phase 3 init | Phase 4 (PHREEQC) |
 * | dust_efflor | E_f | [0,1] | Phase 4 init | Phase 4 (PHREEQC) |
 * | dust_suppression | S | [-] | Phase 3 init | Phase 4 (PHREEQC) |
 * | dust_emission_flux | Q | [kg/m^2/s] | Phase 3 | (per size bin) |
 *
 */
struct DustGrid {
    /// Cell-centered box array on dust grid.
    amrex::BoxArray ba;
    
    /// Distribution mapping.
    amrex::DistributionMapping dm;
};

class DustLayer {
public:
    /**
     * @brief Constructor: allocate dust grid and MultiFabs.
     *
     * @param[in] atm_ba         Atmospheric grid BoxArray.
     * @param[in] atm_dm         Atmospheric DistributionMapping.
     * @param[in] dust_params    DustParams struct.
     */
    DustLayer(const amrex::BoxArray&       atm_ba,
              const amrex::DistributionMapping& atm_dm,
              const DustParams&            dust_params);

    /**
     * @brief Destructor.
     */
    ~DustLayer() = default;

    /**
     * @brief Initialize dust layer at simulation start.
     *
     * - Compute u*_t from Bagnold formula for each size bin.
     * - Set crust_index, silt_fraction from parameters or files.
     * - Initialize efflorescence and suppression to zero.
     * - Store Bagnold u*_t as base value (dust_ustar_base).
     *
     * Phase 4: Also allocate dust_efflor and dust_ustar_base MultiFabs.
     *
     * @param[in] dust_params DustParams.
     */
    void initialize(const DustParams& dust_params);

    /**
     * @brief Advance dust layer by one timestep.
     *
     * Phase 3: Physics placeholder (stub).
     * Phase 4: Check PHREEQC update interval and call reader if elapsed.
     *
     * @param[in] dt           Timestep size [s].
     * @param[in] dust_params  DustParams.
     */
    void advance(amrex::Real dt, const DustParams& dust_params);

    /**
     * @brief Get pointer to dust_ustar_t MultiFab (const).
     */
    const amrex::MultiFab* get_ustar_t() const { return dust_ustar_t.get(); }

    /**
     * @brief Get pointer to dust_crust_index MultiFab (const).
     */
    const amrex::MultiFab* get_crust_index() const { return dust_crust_index.get(); }

    /**
     * @brief Get pointer to dust_silt_fraction MultiFab (const).
     */
    const amrex::MultiFab* get_silt_fraction() const { return dust_silt_fraction.get(); }

    /**
     * @brief Get pointer to dust_efflor MultiFab (const).
     * Phase 4.
     */
    const amrex::MultiFab* get_efflor() const { return dust_efflor.get(); }

    /**
     * @brief Get pointer to dust_suppression MultiFab (const).
     */
    const amrex::MultiFab* get_suppression() const { return dust_suppression.get(); }

    /**
     * @brief Get pointer to dust_emission_flux MultiFab (const).
     */
    const amrex::MultiFab* get_emission_flux() const { return dust_emission_flux.get(); }

    /**
     * @brief Get DustGrid (const).
     */
    const DustGrid& get_dust_grid() const { return m_dg; }

    /**
     * @brief Get dust_ustar_t MultiFab (non-const).
     */
    amrex::MultiFab* get_ustar_t_mutable() { return dust_ustar_t.get(); }

private:
    DustGrid m_dg;  ///< Dust grid layout.
    int m_step = 0; ///< Timestep counter.

    // Phase 3: Core dust MultiFabs
    std::unique_ptr<amrex::MultiFab> dust_ustar_t;        ///< Threshold friction velocity [m/s]. Updated by PHREEQC.
    std::unique_ptr<amrex::MultiFab> dust_crust_index;    ///< Crust strength index [0,1].
    std::unique_ptr<amrex::MultiFab> dust_silt_fraction;  ///< Silt mass fraction [-].
    std::unique_ptr<amrex::MultiFab> dust_suppression;    ///< Suppression coverage [-].
    std::unique_ptr<amrex::MultiFab> dust_emission_flux;  ///< Emission flux [kg/m^2/s], one component per size bin.

    // Phase 4: Efflorescence and base ustar
    std::unique_ptr<amrex::MultiFab> dust_efflor;         ///< Efflorescence fraction [0,1]. Updated by PHREEQC reader.
    std::unique_ptr<amrex::MultiFab> dust_ustar_base;     ///< Base threshold friction velocity before chemistry [m/s]. Set at init, not modified.

    // Phase 4: Time tracking for PHREEQC updates
    amrex::Real m_time = 0.0;                             ///< Simulation time [s].
    amrex::Real m_last_phreeqc_update = -1.0e30;          ///< Time of last PHREEQC update [s].
};

#endif  // ERF_USE_DUST
#endif  // ERF_DUST_LAYER_H
