/**
 * @file ERF_UCMDiagnostics.H
 * @brief Diagnostics CSV writer for UCM statistics
 *
 * Writes per-timestep statistics to a CSV file (default: ucm_diag.dat)
 * containing key scalar quantities that characterize the UCM state:
 * max/min skin temperatures, max canyon temperature, fluxes, etc.
 *
 * **CSV Format:**
 * ```
 * step, time_s, T_skin_roof_max, T_skin_wall_max, T_skin_road_max,
 * T_canyon_max, H_sensible_max, H_sensible_sum, LE_latent_max
 * ```
 *
 * One row per call, on IO rank only.
 *
 * References:
 *  - Source/Dust/ERF_DustDiagnostics.H
 *  - Source/LNG/LNG*.H (diagnostic output pattern)
 */

#ifndef ERF_UCM_DIAGNOSTICS_H
#define ERF_UCM_DIAGNOSTICS_H

#include <UrbanCanopy/ERF_UCMParams.H>
#include <UrbanCanopy/ERF_UCMFields.H>
#include <string>
#include <memory>

/**
 * @class UCMDiagnostics
 * @brief Collects and writes UCM statistics to CSV file
 *
 * Instantiated once per AMR level. Each call to append() computes
 * min/max/sum statistics from UCM MultiFabs and appends a CSV row
 * to the output file.
 *
 * **File naming:** From params.ucm_diag_file (default "ucm_diag.dat").
 * **Location:** Same output directory as main plotfiles.
 * **IO rank only:** All writes occur on rank 0 only; MPI-safe.
 *
 * **Duplicate-write guard:** Tracks last_write_step to prevent duplicate
 * rows if append() is called multiple times for the same step.
 */
class UCMDiagnostics
{
public:
    /**
     * @brief Construct UCMDiagnostics writer
     *
     * Opens (or creates if doesn't exist) the diagnostics CSV file.
     * Writes header row if file is empty.
     *
     * @param[in] params  UCMParams configuration (stored by reference)
     * @param[in] lev     AMR level (default 0)
     */
    UCMDiagnostics(const UCMParams& params, int lev = 0);

    /**
     * @brief Destructor
     *
     * Closes file handle.
     */
    ~UCMDiagnostics();

    /**
     * @brief Compute and write one CSV row
     *
     * Iterates over UCM MultiFabs to compute:
     * - T_skin_roof_max: maximum roof skin temperature [K]
     * - T_skin_wall_max: maximum wall skin temperature [K]
     * - T_skin_road_max: maximum road skin temperature [K]
     * - T_canyon_max:    maximum canyon-air temperature [K]
     * - H_sensible_max:  maximum sensible heat flux [W/m²]
     * - H_sensible_sum:  domain-sum sensible heat flux [W]
     * - LE_latent_max:   maximum latent heat flux [W/m²]
     *
     * Only processes cells where is_urban(i,j,0) == 1.
     *
     * **Debug trace format (IO rank only):**
     * ```
     * [UCM][1.4][UCMDiagnostics::append] step={nstep} time={time}
     *   T_skin_roof_max={T_roof_max} T_canyon_max={T_canyon_max}
     *   H_sensible_max={H_max} H_sensible_sum={H_sum}
     * ```
     *
     * Duplicate-write check: if nstep == m_last_write_step, function
     * returns early without appending.
     *
     * @param[in] fields   UCMFields container (all MultiFabs on UCM grid)
     * @param[in] nstep    Current timestep number
     * @param[in] time     Simulation time [s]
     * @param[in] lev      AMR level (default 0)
     */
    void append(const UCMFields& fields, int nstep, amrex::Real time, int lev = 0);

private:
    const UCMParams& m_params;          ///< Reference to parameters
    int              m_lev;             ///< AMR level
    int              m_last_write_step{-1};  ///< Track last written step (duplicate guard)
    std::string      m_diag_file;       ///< Full path to CSV file

    /**
     * @brief Initialize diagnostics file and write header if needed
     *
     * Called from constructor. If file doesn't exist, creates and writes
     * CSV header row. If file exists, appends to it.
     */
    void initialize_file();

    /**
     * @brief Append one CSV row to file
     *
     * Called from append() after statistics are computed.
     *
     * @param[in] nstep         Step number
     * @param[in] time          Simulation time [s]
     * @param[in] T_roof_max    Max roof skin temp [K]
     * @param[in] T_wall_max    Max wall skin temp [K]
     * @param[in] T_road_max    Max road skin temp [K]
     * @param[in] T_canyon_max  Max canyon-air temp [K]
     * @param[in] H_max         Max sensible flux [W/m²]
     * @param[in] H_sum         Sum sensible flux [W]
     * @param[in] LE_max        Max latent flux [W/m²]
     */
    void write_row(int nstep, amrex::Real time,
                  amrex::Real T_roof_max, amrex::Real T_wall_max, amrex::Real T_road_max,
                  amrex::Real T_canyon_max, amrex::Real H_max, amrex::Real H_sum,
                  amrex::Real LE_max);
};

#endif  // ERF_UCM_DIAGNOSTICS_H
