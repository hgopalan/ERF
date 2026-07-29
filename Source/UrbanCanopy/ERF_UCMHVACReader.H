/**
 * @file ERF_UCMHVACReader.H
 * @brief HVAC profile CSV reader for Phase 5.2 (extended Phase 5.5)
 *
 * Reads a CSV file with HVAC system properties (COP, setpoint, occupancy_profile_id,
 * sensible_fraction, rejection_facet) keyed by hvac_profile_id. Used with building_layout.csv
 * to associate each cell with its HVAC profile.
 *
 * CSV format (header row required):
 *   hvac_profile_id,cop,t_setpoint_K,occupancy_profile_id,description,sensible_fraction,rejection_facet
 *   0,3.0,297.15,0,office_baseline,1.0,roof
 *   1,4.0,297.15,1,office_efficient,0.8,distributed
 *   2,2.5,299.15,2,residential_older_stock,0.7,road
 *
 * Phase 5.5 additions:
 *   - sensible_fraction [0,1]: fraction of Q_HVAC that is sensible; default 1.0 (backward-compat)
 *   - rejection_facet: roof|road|distributed; default roof (backward-compat)
 *
 * References:
 *   - Phase 5.2 problem statement (HVAC waste heat, Contract #21)
 *   - Phase 5.5 problem statement (extended physics, sensible/latent split)
 *   - ERF_UCMOccupancyReader.H (sibling CSV reader)
 */

#ifndef ERF_UCM_HVAC_READER_H
#define ERF_UCM_HVAC_READER_H

#include <AMReX_REAL.H>
#include <string>
#include <vector>

/**
 * @struct HVACProfile
 * @brief POD struct for a single HVAC profile row
 *
 * Safe for storage in std::vector and device capture.
 */
struct HVACProfile
{
    int id;                          ///< hvac_profile_id (primary key)
    amrex::Real cop;                 ///< Coefficient of performance [-]
    amrex::Real t_setpoint_K;        ///< Setpoint temperature [K]
    int occupancy_profile_id;        ///< Link to occupancy.csv profile
    std::string description;         ///< Human-readable description
    amrex::Real sensible_fraction;   ///< Phase 5.5: fraction of Q_HVAC that is sensible (0-1)
    int rejection_facet;             ///< Phase 5.5: 0=roof, 1=road, 2=distributed
};

/**
 * @class UCMHVACReader
 * @brief Load HVAC profiles from CSV
 *
 * Single-use loader: construct with CSV path, then query profiles by ID.
 * Loads the entire file into memory at construction.
 */
class UCMHVACReader
{
private:
    std::vector<HVACProfile> profiles_;  ///< All loaded HVAC profiles

public:
    /**
     * @brief Construct and load CSV file
     *
     * Reads all rows from the CSV (skipping header and comments).
     * Populates profiles_ vector.
     *
     * @param csv_path  Path to hvac.csv file
     */
    UCMHVACReader(const std::string& csv_path);

    /**
     * @brief Get HVAC profile by ID
     *
     * @param profile_id  The hvac_profile_id to look up
     * @return Pointer to HVACProfile if found, nullptr otherwise
     */
    const HVACProfile* get_profile(int profile_id) const;

    /**
     * @brief Get all profiles (for device capture prep)
     *
     * @return Vector of all loaded profiles
     */
    const std::vector<HVACProfile>& get_all_profiles() const { return profiles_; }

    /**
     * @brief Get number of profiles
     *
     * @return Number of loaded profiles
     */
    int size() const { return static_cast<int>(profiles_.size()); }
};

#endif  // ERF_UCM_HVAC_READER_H
