/**
 * @file ERF_UCMPlotfile.H
 * @brief Plotfile writer for UCM diagnostics
 *
 * Writes UCM state variables to AMReX plotfile format (plt_ucm_NNNNN) on the native
 * UCM 2D slab grid. Called at specified intervals during integration.
 *
 * References:
 *  - Source/Dust/ERF_DustPlotfile.H
 *  - Source/LNG/LNG*.H (diagnostic output pattern)
 *  - AMReX VisMF documentation
 */

#ifndef ERF_UCM_PLOTFILE_H
#define ERF_UCM_PLOTFILE_H

#include <UrbanCanopy/ERF_UCMParams.H>
#include <UrbanCanopy/ERF_UCMFields.H>
#include <UrbanCanopy/ERF_UCMGrid.H>
#include <string>
#include <memory>

/**
 * @class UCMPlotfile
 * @brief Writes UCM 2D field diagnostics to AMReX plotfile format
 *
 * Aggregates all UCM MultiFabs (morphology, radiative properties, temperatures,
 * fluxes, is_urban mask) into a single plt_ucm_NNNNN directory containing
 * binary (or ASCII) data readable by standard AMReX tools (AMRVis, yt, etc).
 *
 * **File naming:** `plt_ucm_NNNNNN` where NNNNNN = zero-padded step number.
 * **Location:** Same output directory as main ATM plotfiles (erf.plot_file_base).
 *
 * **Duplicate-write guard:** Tracks last_write_step to prevent re-writing
 * if write() is called twice for the same nstep (e.g., intermediate + final).
 */
class UCMPlotfile
{
public:
    /**
     * @brief Construct UCMPlotfile writer
     *
     * @param[in] params  UCMParams configuration (stored by reference)
     * @param[in] lev     AMR level (default 0)
     */
    UCMPlotfile(const UCMParams& params, int lev = 0);

    /**
     * @brief Destructor
     */
    ~UCMPlotfile();

    /**
     * @brief Write plotfile at current step
     *
     * Gathers all UCM fields, packs into a temporary MultiFab with UCMPlot_ncomp
     * components, and writes to plt_ucm_NNNNN using WriteSingleLevelPlotfile.
     *
     * **Debug trace format (IO rank only):**
     * ```
     * [UCM][1.4][UCMPlotfile::write]
     *   step={nstep} time={time} s
     *   filename: {full_path}/plt_ucm_NNNNNN
     *   ncomp={ncomp} (H_bldg, W_road, W_roof, albedos, emissivities, T_skin_*, T_canyon, H_sensible, LE_latent, is_urban)
     *   grid: {nx_ucm} x {ny_ucm} x 1, {nx_atm} x {ny_atm} (ATM for reference)
     * ```
     *
     * Duplicate-write check: if nstep == m_last_write_step, function returns
     * early without writing (or optionally logs a warning).
     *
     * @param[in] fields           UCMFields container (all MultiFabs on UCM grid)
     * @param[in] grid             UCMGrid container (geometry and BoxArray)
     * @param[in] nstep            Current timestep number
     * @param[in] time             Simulation time [s]
     * @param[in] is_final         If true, writes regardless of ucm_plot_int
     * @param[in] lev              AMR level (default 0)
     */
    void write(const UCMFields& fields, const UCMGrid& grid,
               int nstep, amrex::Real time, bool is_final = false,
               int lev = 0);

private:
    const UCMParams& m_params;          ///< Reference to parameters
    int              m_lev;             ///< AMR level
    int              m_last_write_step{-1};  ///< Track last written step (duplicate guard)

    /**
     * @brief Build full plotfile pathname
     *
     * Combines erf.plot_file_base (or current directory if empty) with
     * "plt_ucm_NNNNNN" where NNNNNN is zero-padded nstep.
     *
     * @param[in] nstep  Step number
     * @return           Full path string
     */
    std::string get_plotfile_name(int nstep) const;
};

#endif  // ERF_UCM_PLOTFILE_H
