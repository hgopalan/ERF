/** @file ERF_UCMBuildingLayoutReader.H
 * @brief Building-layout CSV reader for the ERF-SLUCM module (Phase 2.1 → Phase 3.7)
 *
 * Provides a POD struct for building rows and a rank-0-read + MPI_Bcast CSV
 * reader following the pattern from `Source/LNG/ERF_LNGSpillSchedule.H/.cpp`.
 *
 * ## CSV Schema (Phase 3.7: support both legacy index and new physical modes)
 *
 * **Legacy mode (index-based, Phase 2.1–3.6):**
 * ```
 * i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban
 * 0,0,1,15.0,0.5,10.0,6.0,2,2,2,0.0,0,30.0,1
 * ```
 *
 * **Physical mode (new, Phase 3.7+, default going forward):**
 * ```
 * x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban
 * 125.0,125.0,1,15.0,0.5,10.0,6.0,2,2,2,0.0,0,30.0,1
 * ```
 * where x_m, y_m are physical cell-center coordinates in meters.
 *
 * ### Physical mode behavior (Phase 3.7):
 * - Load all rows with min/max bounding box (x_min, x_max, y_min, y_max)
 * - For each UCM cell (i,j), compute cell center: x_c = prob_lo[0] + (i+0.5)*dx_ucm, etc.
 * - Find CSV row with minimum Euclidean distance (linear scan)
 * - If cell center outside bounding box, fall back to uniform defaults with ONE warning per rank
 * - Row-count assertion is REMOVED (sparse CSV samples OK)
 *
 * ### Legacy mode behavior (Phase 2.1–3.6, preserved):
 * - CSV must have exactly nx_ucm * ny_ucm rows
 * - Row-count assertion ENFORCED
 * - (i,j) indices directly map to UCM grid
 *
 * Each data row contains:
 * - i or x_m (mode-dependent): grid index or physical x coordinate
 * - j or y_m (mode-dependent): grid index or physical y coordinate
 * - bldg_id: building identifier (integer)
 * - height_m: building height in meters (float)
 * - plan_area_frac: plan area fraction (float, used for heterogeneous morphology Phase 2.2+)
 * - W_road_m: road width in meters (float)
 * - W_roof_m: roof width in meters (float)
 * - roof_mat_id, wall_mat_id, road_mat_id: indices into UCMMaterialRegistry (integers)
 * - orientation_deg: canyon orientation in degrees (float, Phase 2.2+)
 * - ah_profile_id: anthropogenic heat profile id (integer, Phase 2.3+)
 * - AH_Wm2: per-cell anthropogenic heat override in W/m^2 (float, Phase 2.9+); 0 → use ParmParse fallback
 * - is_urban: urban/non-urban flag (0 or 1); controls LSM/MOST bypass (Phase 4.1)
 *
 * ## References
 *  - Source/LNG/ERF_LNGSpillSchedule.H (rank-0-read + MPI_Bcast pattern)
 *  - Source/LNG/LNG_MPI_SKILLS.md (MPI safety guidelines)
 *  - Source/UrbanCanopy/UCM_DEVELOPMENT.md (Phase 3.7 requirements)
 */

#ifndef ERF_UCM_BUILDING_LAYOUT_READER_H
#define ERF_UCM_BUILDING_LAYOUT_READER_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>

/** @struct UCMBuildingRow
 * @brief POD struct for a single building-layout CSV row (both legacy and physical modes)
 *
 * Safe for MPI_Bcast as raw bytes (std::is_trivially_copyable_v<UCMBuildingRow> == true).
 * All fields are POD types (int, amrex::Real) with no pointers or virtual functions.
 *
 * Phase 2.1: used for basic morphology (height, road/roof widths, is_urban mask).
 * Phase 2.2+: orientation_deg, plan_area_frac used for heterogeneous canopy wind.
 * Phase 2.3+: ah_profile_id used for heterogeneous anthropogenic heat.
 * Phase 2.9+: AH_Wm2 used for per-cell anthropogenic heat override.
 * Phase 3.7+: x, y store physical coordinates (Phase 3.7 physical mode) or grid indices (legacy mode)
 * Phase 4.1: is_urban used to bypass LSM/MOST.
 *
 * ### Phase 3.7 extension:
 * In physical mode, fields x and y hold physical coordinates [m] instead of grid indices.
 * The loader detects mode from CSV header (x_m, y_m vs i, j).
 */
struct UCMBuildingRow
{
    amrex::Real x;             ///< Grid index i OR physical x coordinate [m], depending on CSV mode (Phase 3.7)
    amrex::Real y;             ///< Grid index j OR physical y coordinate [m], depending on CSV mode (Phase 3.7)
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
    amrex::Real AH_Wm2;        ///< Per-cell AH override [W/m^2], Phase 2.9+; 0 → use ParmParse fallback
    int    is_urban;           ///< Urban mask [0 or 1]; 0 = LSM/MOST, 1 = UCM
    int    hvac_profile_id;    ///< HVAC profile ID (Phase 5.2+), defaults to 0 if column missing
};

/**
 * @class UCMBuildingLayoutReader
 * @brief CSV reader for heterogeneous building-layout data (Phase 2.1 → 3.7)
 *
 * Implements rank-0-read + MPI_Bcast pattern for building-layout CSV.
 * On rank 0, parses CSV and stores rows in m_rows vector. Then broadcasts
 * row count and data to all ranks via MPI_Bcast.
 *
 * ### Phase 3.7: Support both legacy and physical-coordinate modes
 * - **Legacy mode** (i,j columns): Enforces row count = nx_ucm * ny_ucm, direct index mapping
 * - **Physical mode** (x_m,y_m columns): Sparse CSV OK, nearest-neighbor lookup by Euclidean distance
 *
 * Enforces:
 *  - No duplicate (i, j) pairs in legacy mode (aborts with error if found)
 *  - All mat_ids >= 0 (basic sanity; material registry lookup happens later)
 *  - is_urban in {0, 1} (enforced at read time)
 *  - In physical mode: cells outside CSV bounding box fall back to uniform defaults
 *
 * Example usage:
 * ```cpp
 * UCMBuildingLayoutReader reader;
 * reader.read_and_broadcast("building_layout.csv", nx_ucm, ny_ucm, lev, ucm_debug);
 * for (const auto& row : reader.rows()) {
 *     // Process row: in physical mode, row.x, row.y are coordinates;
 *     //              in legacy mode, row.x, row.y are i,j indices
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
     * ### Phase 3.7 behavior:
     *  - Detects CSV mode from header: x_m,y_m (physical) or i,j (legacy)
     *  - **Physical mode**: Row count may be < nx_ucm * ny_ucm; sparse CSV allowed
     *  - **Legacy mode**: Row count must equal nx_ucm * ny_ucm; duplicates forbidden
     *
     * @throws Aborts via amrex::Abort() if:
     *  - File cannot be opened (file not found or permission denied)
     *  - Header line does not match expected schema (for either mode)
     *  - Parsing error (non-numeric field, malformed CSV)
     *  - Duplicate (i, j) pair found (legacy mode only)
     *  - is_urban not in {0, 1}
     *  - Any mat_id < 0
     *  - Row count != nx_ucm * ny_ucm (legacy mode only, Phase 3.7 removes for physical)
     *  - Any (i,j) outside [0, nx_ucm) × [0, ny_ucm) range (legacy mode only)
     */
    void read_and_broadcast(const std::string& path,
                            int nx_ucm, int ny_ucm,
                            int lev, bool ucm_debug = false);

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
