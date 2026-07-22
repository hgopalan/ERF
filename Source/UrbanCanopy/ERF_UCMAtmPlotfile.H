/**
 * @file ERF_UCMAtmPlotfile.H
 * @brief Plotfile writer for UCM ATM-grid aggregates
 *
 * Writes 5 aggregated UCM scalar fields to AMReX plotfile format (plt_ucm_atm_NNNNN) on the native
 * ATM 2D slab grid. Called at specified intervals during integration.
 *
 * References:
 *  - Source/Fire/ERF_FirePlotfile.H/.cpp on ERF-Hazard (the reference for ATM-grid native-grid writing)
 *  - Source/UrbanCanopy/ERF_UCMPlotfile.H (the UCM-grid version)
 *  - AMReX VisMF documentation
 */

#ifndef ERF_UCM_ATM_PLOTFILE_H
#define ERF_UCM_ATM_PLOTFILE_H

#include <AMReX_MultiFab.h>
#include <AMReX_Geometry.H>
#include <string>
#include <memory>

/**
 * @class UCMAtmPlotfile
 * @brief Writes ATM 2D aggregated UCM field diagnostics to AMReX plotfile format
 *
 * Takes 5 pre-computed aggregates on the ATM 2D slab grid (f_urb, H_bldg_mean,
 * H_bldg_std, lambda_p, lambda_f) and writes them to a plt_ucm_atm_NNNNN directory
 * containing binary (or ASCII) data readable by standard AMReX tools (AMRVis, yt, etc).
 *
 * **File naming:** `plt_ucm_atm_NNNNNN` where NNNNNN = zero-padded step number.
 * **Location:** Same output directory as main ATM plotfiles (erf.plot_file_base).
 *
 * **Duplicate-write guard:** Tracks last_write_step to prevent re-writing
 * if write() is called twice for the same step (e.g., intermediate + final).
 */
class UCMAtmPlotfile
{
public:
    /**
     * @brief Construct UCMAtmPlotfile writer
     */
    UCMAtmPlotfile() = default;

    /**
     * @brief Destructor
     */
    ~UCMAtmPlotfile() = default;

    /**
     * @brief Write plotfile at current step
     *
     * Gathers 5 aggregated UCM scalar fields, packs into a temporary MultiFab with 5
     * components, and writes to plt_ucm_atm_NNNNN using WriteSingleLevelPlotfile.
     *
     * **Debug trace format (IO rank only):**
     * ```
     * [UCM][2.5-followup][UCMAtmPlotfile::write]
     *   step={step} time={time} s
     *   filename: {full_path}/plt_ucm_atm_NNNNNN
     *   ncomp=5 (f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f)
     *   grid: {nx_atm} x {ny_atm} x 1 (ATM grid)
     * ```
     *
     * Duplicate-write check: if step == m_last_write_step, function returns
     * early without writing (or optionally logs a warning).
     *
     * @param[in] step                    Current timestep number
     * @param[in] time                    Simulation time [s]
     * @param[in] f_urb_atm               Urban fraction on ATM grid [-]
     * @param[in] H_bldg_mean_atm        Mean building height on ATM grid [m]
     * @param[in] H_bldg_std_atm         Std dev building height on ATM grid [m]
     * @param[in] lambda_p_atm           Mean plan-area density on ATM grid [-]
     * @param[in] lambda_f_atm           Mean frontal-area density on ATM grid [-]
     * @param[in] geom                    ATM geometry
     * @param[in] ucm_debug              Print debug info (default false)
     * @param[in] lev                    AMR level (default 0)
     */
    void write(int                       step,
               amrex::Real               time,
               const amrex::MultiFab&    f_urb_atm,
               const amrex::MultiFab&    H_bldg_mean_atm,
               const amrex::MultiFab&    H_bldg_std_atm,
               const amrex::MultiFab&    lambda_p_atm,
               const amrex::MultiFab&    lambda_f_atm,
               const amrex::Geometry&    geom,
               bool                      ucm_debug = false,
               int                       lev = 0);

private:
    int m_last_write_step{-1};  ///< Track last written step (duplicate guard)

    /**
     * @brief Build full plotfile pathname
     *
     * Combines erf.plot_file_base (or current directory if empty) with
     * "plt_ucm_atm_NNNNNN" where NNNNNN is zero-padded step.
     *
     * @param[in] step   Step number
     * @return           Full path string
     */
    std::string get_plotfile_name(int step) const;
};

#endif  // ERF_UCM_ATM_PLOTFILE_H
