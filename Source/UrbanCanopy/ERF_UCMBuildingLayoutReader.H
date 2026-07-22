/**
 * @file ERF_UCMBuildingLayoutReader.H
 * @brief Building-layout CSV reader for the ERF-SLUCM module (Phase 2.1)
 *
 * Provides a POD struct for building rows and a rank-0-read + MPI_Bcast CSV
 * reader following the pattern from `Source/LNG/ERF_LNGSpillSchedule.H/.cpp`.
 *
 * ## CSV Schema
 * Expected header (exact string, whitespace tolerated):
 * ```
 * i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,is_urban
 * ```
 *
 * Each data row contains:
 * - i, j: grid indices (integers)
 * - bldg_id: building identifier (integer)
 * - height_m: building height in meters (float)
 * - plan_area_frac: plan area fraction (float, used for heterogeneous morphology Phase 2.2+)
 * - W_road_m: road width in meters (float)
 * - W_roof_m: roof width in meters (float)
 * - roof_mat_id, wall_mat_id, road_mat_id: indices into UCMMaterialRegistry (integers)
 * - orientation_deg: canyon orientation in degrees (float, Phase 2.2+)
 * - ah_profile_id: anthropogenic heat profile id (integer, Phase 2.3+)
 * - is_urban: urban/non-urban flag (0 or 1); controls LSM/MOST bypass (Phase 4.1)
 *
 * ## References
 *  - Source/LNG/ERF_LNGSpillSchedule.H (rank-0-read + MPI_Bcast pattern)
 *  - Source/LNG/LNG_MPI_SKILLS.md (MPI safety guidelines)
 */

#ifndef ERF_UCM_BUILDING_LAYOUT_READER_H
#define ERF_UCM_BUILDING_LAYOUT_READER_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>

/**
 * @struct UCMBuildingRow
 * @brief POD struct for a single building-layout CSV row
 *
 * Safe for MPI_Bcast as raw bytes (std::is_trivially_copyable_v<UCMBuildingRow> == true).
 * All fields are POD types (int, amrex::Real) with no pointers or virtual functions.
 *
 * Phase 2.1: used for basic morphology (height, road/roof widths, is_urban mask).
 * Phase 2.2+: orientation_deg, plan_area_frac used for heterogeneous canopy wind.
 * Phase 2.3+: ah_profile_id used for heterogeneous anthropogenic heat.
 * Phase 4.1: is_urban used to bypass LSM/MOST.
 */
struct UCMBuildingRow
{
    int    i;                  ///< Grid index (x direction)
    int    j;                  ///< Grid index (y direction)
    int    bldg_id;            ///< Building identifier (informational)
    amrex::Real height_m;      ///< Building height [m]
    amrex::Real plan_area_frac;///< Plan area fraction [-], future heterogeneous morphology
    amrex::Real W_road_m;      ///< Road width (canyon street width) [m]
    amrex::Real W_roof_m;      ///< Roof width (building footprint) [m]
    int    roof_mat_id;        ///< Index into UCMMaterialRegistry for roof
    int    wall_mat_id;        ///< Index into UCMMaterialRegistry for walls
    int    road_mat_id;        ///< Index into UCMMaterialRegistry for road
    amrex::Real orientation_deg;///< Canyon orientation [degrees], Phase 2.2+
    int    ah_profile_id;      ///< Anthropogenic heat profile ID, Phase 2.3+
    int    is_urban;           ///< Urban mask [0 or 1]; 0 = LSM/MOST, 1 = UCM
};

/**
 * @class UCMBuildingLayoutReader
 * @brief CSV reader for heterogeneous building-layout data
 *
 * Implements rank-0-read + MPI_Bcast pattern for building-layout CSV.
 * On rank 0, parses CSV and stores rows in m_rows vector. Then broadcasts
 * row count and data to all ranks via MPI_Bcast.
 *
 * Enforces:
 *  - No duplicate (i, j) pairs (aborts with error if found)
 *  - All mat_ids >= 0 (basic sanity; material registry lookup happens later)
 *  - is_urban in {0, 1} (enforced at read time)
 *
 * Example usage:
 * ```cpp
 * UCMBuildingLayoutReader reader;
 * reader.read_and_broadcast("building_layout.csv");
 * for (const auto& row : reader.rows()) {
 *     // Process row: fill MultiFabs based on (i,j) match
 * }
 * ```
 */
class UCMBuildingLayoutReader
{
public:
    /**
     * @brief Rank-0 read CSV, then MPI_Bcast POD struct array to all ranks
     *
     * Execution:
     *  - Rank 0: open file, parse header (must match schema), parse data rows
     *  - Rank 0: validate rows (no duplicate (i,j), check mat_ids >= 0, is_urban in {0,1})
     *  - Rank 0: abort if any validation fails
     *  - Rank 0: emit debug trace (path, row count, min/max ranges)
     *  - All ranks: MPI_Bcast row count, then row data as bytes
     *  - All ranks: call rows() to access the broadcast data
     *
     * Debug output (IO-rank only, if ucm_debug enabled):
     *  ```
     *  [UCM][2.1][UCMBuildingLayoutReader::read_and_broadcast]
     *    path = /path/to/building_layout.csv
     *    rows_parsed = 64
     *    bldg_id: min=1, max=8
     *    height_m: min=5.0, max=15.0
     *    plan_area_frac: min=0.4, max=0.6
     *    is_urban=0 count = 2
     *    duplicate (i,j) detected: (5, 3) appears 2 times → abort
     *  ```
     *
     * @param[in] path    Path to building-layout CSV file
     * @param[in] nx_ucm  Expected UCM grid size in x direction (for row-count validation)
     * @param[in] ny_ucm  Expected UCM grid size in y direction (for row-count validation)
     * @param[in] lev     AMR level (default 0; kept for generality with other readers)
     *
     * @throws Aborts via amrex::Abort() if:
     *  - File cannot be opened (file not found or permission denied)
     *  - Header line does not match expected schema
     *  - Parsing error (non-numeric field, malformed CSV)
     *  - Duplicate (i, j) pair found
     *  - is_urban not in {0, 1}
     *  - Any mat_id < 0
     *  - Row count != nx_ucm * ny_ucm (Phase 2.3 enforcement: CSV is on UCM grid)
     *  - Any (i,j) outside [0, nx_ucm) × [0, ny_ucm) range
     */
    void read_and_broadcast(const std::string& path,
                            int nx_ucm, int ny_ucm,
                            int lev = 0, bool ucm_debug = false);

    /**
     * @brief Direct const access to broadcast row table
     *
     * Valid on all ranks after read_and_broadcast() completes.
     *
     * @return Reference to const vector of UCMBuildingRow structs
     */
    const amrex::Vector<UCMBuildingRow>& rows() const { return m_rows; }

    /**
     * @brief Number of rows in the table
     *
     * Valid on all ranks after read_and_broadcast() completes.
     *
     * @return Number of rows (same on all ranks after broadcast)
     */
    int size() const { return static_cast<int>(m_rows.size()); }

private:
    /// Broadcast row table (same on all ranks after read_and_broadcast)
    amrex::Vector<UCMBuildingRow> m_rows;
};

#endif // ERF_UCM_BUILDING_LAYOUT_READER_H
