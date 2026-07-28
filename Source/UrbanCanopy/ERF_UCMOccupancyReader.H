/**
 * @file ERF_UCMOccupancyReader.H
 * @brief Occupancy profile CSV reader for Phase 5.2
 *
 * Reads a CSV file with per-hour occupancy fractions (0–1) for each profile.
 * Used to gate HVAC operation based on occupancy time-of-day.
 *
 * CSV format (header row required):
 *   occupancy_profile_id,hour_of_day,occupancy_fraction
 *   0,0,0.05
 *   0,1,0.05
 *   ...
 *   0,23,0.10
 *   1,0,0.02
 *   ...
 *
 * Each profile must have exactly 24 entries (one per hour of the day 0–23).
 * Occupancy fractions must be in [0, 1].
 *
 * References:
 *   - Phase 5.2 problem statement (HVAC waste heat, Contract #21)
 *   - ERF_UCMHVACReader.H (sibling CSV reader)
 */

#ifndef ERF_UCM_OCCUPANCY_READER_H
#define ERF_UCM_OCCUPANCY_READER_H

#include <AMReX_REAL.H>
#include <array>
#include <string>
#include <vector>

/**
 * @struct OccupancyProfile
 * @brief POD struct for a single occupancy profile (24-hour diurnal)
 *
 * Safe for storage in std::vector and device capture.
 */
struct OccupancyProfile
{
    int id;                                    ///< occupancy_profile_id (primary key)
    std::array<amrex::Real, 24> hourly_frac;  ///< Occupancy fraction per hour [0–1]
};

/**
 * @class UCMOccupancyReader
 * @brief Load occupancy profiles from CSV
 *
 * Single-use loader: construct with CSV path, then query profiles by ID and hour.
 * Loads the entire file into memory at construction.
 */
class UCMOccupancyReader
{
private:
    std::vector<OccupancyProfile> profiles_;  ///< All loaded occupancy profiles

public:
    /**
     * @brief Construct and load CSV file
     *
     * Reads all rows from the CSV (skipping header and comments).
     * Validates that each profile has 24 entries (hours 0–23) with
     * occupancy fractions in [0, 1].
     *
     * @param csv_path  Path to occupancy.csv file
     */
    UCMOccupancyReader(const std::string& csv_path);

    /**
     * @brief Get occupancy fraction for a profile at a given hour
     *
     * @param profile_id     The occupancy_profile_id to look up
     * @param hour_of_day    Hour of day (0–23)
     * @return Occupancy fraction [0–1], or 1.0 if profile not found
     */
    amrex::Real get_occupancy(int profile_id, int hour_of_day) const;

    /**
     * @brief Get occupancy profile by ID
     *
     * @param profile_id  The occupancy_profile_id to look up
     * @return Pointer to OccupancyProfile if found, nullptr otherwise
     */
    const OccupancyProfile* get_profile(int profile_id) const;

    /**
     * @brief Get all profiles (for device capture prep)
     *
     * @return Vector of all loaded profiles
     */
    const std::vector<OccupancyProfile>& get_all_profiles() const { return profiles_; }

    /**
     * @brief Get number of profiles
     *
     * @return Number of loaded profiles
     */
    int size() const { return static_cast<int>(profiles_.size()); }
};

#endif  // ERF_UCM_OCCUPANCY_READER_H
