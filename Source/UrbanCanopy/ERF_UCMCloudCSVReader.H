/**
 * @file ERF_UCMCloudCSVReader.H
 * @brief CSV reader for hourly cloud fraction time series (Phase 4.2)
 *
 * Reads a two-column CSV file with time [seconds] and cloud fraction [0–1].
 * Provides linear interpolation between hourly samples and 24-hour periodic wrap.
 *
 * CSV format:
 *   # Optional comment lines starting with #
 *   time_s, cloud_fraction
 *   0,      0.50
 *   3600,   0.45
 *   ...
 *   82800,  0.60
 *
 * The CSV is loaded once at simulation start. Query via get_cloud_fraction_at(sim_time_s)
 * with time wrapping modulo 86400 seconds (24 hours).
 *
 * References:
 *   - Phase 4.2 problem statement (cloud-aware analytical radiation)
 *   - Boston diurnal cloud profile (coastal New England climatology)
 */

#ifndef ERF_UCM_CLOUD_CSV_READER_H
#define ERF_UCM_CLOUD_CSV_READER_H

#include <AMReX_REAL.H>
#include <string>
#include <vector>

/**
 * @class UCMCloudCSVReader
 * @brief Load and interpolate cloud fraction from CSV time series
 *
 * Single-use loader: construct with CSV path, then query cloud fraction.
 * Loads the entire file into memory at construction. Typically instantiated
 * once per UCM init at the simulation start.
 */
class UCMCloudCSVReader
{
private:
    std::vector<amrex::Real> time_s_;         ///< Time samples [seconds from 00:00]
    std::vector<amrex::Real> cloud_fraction_; ///< Cloud fraction [0–1] at each time sample

public:
    /**
     * @brief Construct and load CSV file
     *
     * Reads all rows from the CSV, skipping comment lines starting with '#'.
     * Populates time_s_ and cloud_fraction_ vectors.
     *
     * @param csv_path  Path to CSV file (relative or absolute)
     */
    UCMCloudCSVReader(const std::string& csv_path);

    /**
     * @brief Get cloud fraction at a given simulation time (with interpolation)
     *
     * Wraps sim_time_s modulo 86400 (24 hours), then linearly interpolates
     * between the two bracketing CSV rows. If sim_time_s falls before the first
     * row, wraps to the end of the 24-hour cycle.
     *
     * When ucm_debug=1, prints bracketing row indices for tracing.
     *
     * @param sim_time_s      Simulation time [seconds from t=0]
     * @param ucm_debug       Debug flag; if 1, prints row bracket info
     *
     * @return Cloud fraction [0–1] at the effective time (after wrap and interpolation)
     */
    amrex::Real get_cloud_fraction_at(amrex::Real sim_time_s, int ucm_debug = 0) const;
};

#endif  // ERF_UCM_CLOUD_CSV_READER_H
