/** @file ERF_UCMTreeLayoutReader.H
 * @brief Tree-layout CSV reader for the ERF-SLUCM module (Phase 6.1)
 *
 * Provides a POD struct for tree rows and a rank-0-read + MPI_Bcast CSV
 * reader following the pattern from `ERF_UCMBuildingLayoutReader.H` (Phase 2.1 → 3.7).
 *
 * ## CSV Schema (Physical mode ONLY, Phase 6.1)
 *
 * ```
 * x_m,y_m,tree_id,H_tree_m,H_crown_base_m,LAD_bulk,crown_area_frac,Cd_leaf,is_tree
 * 125.0,125.0,1,15.0,5.0,2.0,0.3,0.2,1
 * ```
 * where x_m, y_m are physical cell-center coordinates in meters.
 *
 * Each data row contains:
 * - x_m: physical x coordinate [m]
 * - y_m: physical y coordinate [m]
 * - tree_id: tree identifier (integer, informational)
 * - H_tree_m: crown top height above ground [m]
 * - H_crown_base_m: crown base height above ground [m]
 * - LAD_bulk: leaf area density [m²/m³]; uniform profile inside [H_crown_base_m, H_tree_m]
 * - crown_area_frac: crown footprint / cell area [-], typically ∈ [0,1] (Phase 6.1: supports > 1 for multiple trees per cell)
 * - Cd_leaf: leaf drag coefficient [-], typical ∈ [0.1, 0.3]; overrides Cd_leaf_default if > 0
 * - is_tree: tree mask {0, 1}; 0 = no tree, 1 = tree present
 *
 * ## References
 *  - Source/UrbanCanopy/ERF_UCMBuildingLayoutReader.H (rank-0-read + MPI_Bcast pattern)
 *  - Source/LNG/ERF_LNGSpillSchedule.H (MPI broadcast pattern)
 */

#ifndef ERF_UCM_TREE_LAYOUT_READER_H
#define ERF_UCM_TREE_LAYOUT_READER_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>

/** @struct UCMTreeRow
 * @brief POD struct for a single tree-layout CSV row (Phase 6.1)
 *
 * Safe for MPI_Bcast as raw bytes (std::is_trivially_copyable_v<UCMTreeRow> == true).
 * All fields are POD types (int, amrex::Real) with no pointers or virtual functions.
 *
 * Stores tree morphology data: position (x_m, y_m), height envelope 
 * (H_tree_m, H_crown_base_m), leaf area density (LAD_bulk), crown footprint 
 * (crown_area_frac), and drag coefficient (Cd_leaf).
 */
struct UCMTreeRow
{
    amrex::Real x_m;             ///< Physical x coordinate [m]
    amrex::Real y_m;             ///< Physical y coordinate [m]
    int         tree_id;         ///< Informational tree identifier
    amrex::Real H_tree_m;        ///< Crown top height above ground [m]
    amrex::Real H_crown_base_m;  ///< Crown base height above ground [m]
    amrex::Real LAD_bulk;        ///< Leaf area density [m^2/m^3]
    amrex::Real crown_area_frac; ///< Crown footprint / cell area [-]
    amrex::Real Cd_leaf;         ///< Leaf drag coefficient [-]
    int         is_tree;         ///< Tree mask {0, 1}
};

/**
 * @class UCMTreeLayoutReader
 * @brief CSV reader for tree-layout data (Phase 6.1)
 *
 * Implements rank-0-read + MPI_Bcast pattern for tree-layout CSV (physical mode only).
 * On rank 0, parses CSV and stores rows in m_rows vector. Then broadcasts
 * row count and data to all ranks via MPI_Bcast.
 *
 * Enforces:
 *  - All heights ∈ [0, H_tree_max_bound]
 *  - H_crown_base_m ≤ H_tree_m
 *  - LAD_bulk ∈ [0, LAD_max_bound]
 *  - crown_area_frac ∈ [0, ∞) (allows multiple trees per cell)
 *  - Cd_leaf ∈ [0, 1.0]
 *  - is_tree ∈ {0, 1}
 *
 * Example usage:
 * ```cpp
 * UCMTreeLayoutReader reader;
 * reader.read_and_broadcast("tree_layout.csv", nx_ucm, ny_ucm, 
 *                           H_tree_max, LAD_max, lev, ucm_debug);
 * for (const auto& row : reader.rows()) {
 *     // Process row: x_m, y_m are cell-center coordinates
 * }
 * ```
 */
class UCMTreeLayoutReader
{
public:
    /**
     * @brief Rank-0 read CSV, then MPI_Bcast POD struct array to all ranks
     *
     * Execution:
     *  - Rank 0: open file, parse header (must match physical mode schema), parse data rows
     *  - Rank 0: validate rows (sanity bounds on all fields)
     *  - Rank 0: abort if any validation fails
     *  - Rank 0: emit debug trace (path, row count, min/max ranges)
     *  - All ranks: MPI_Bcast row count, then row data as bytes
     *  - All ranks: call rows() to access the broadcast data
     *
     * Debug output (IO-rank only, if ucm_debug enabled):
     *  ```
     *  [UCM][6.1][UCMTreeLayoutReader::read_and_broadcast]
     *    path = /path/to/tree_layout.csv
     *    rows_parsed = 64
     *    tree_id: min=1, max=100
     *    H_tree_m: min=5.0, max=30.0
     *    LAD_bulk: min=0.5, max=3.0
     *    is_tree=1 count = 60
     *  ```
     *
     * @param[in] path    Path to tree-layout CSV file
     * @param[in] nx_ucm  Expected UCM grid size in x direction (for reference; not enforced)
     * @param[in] ny_ucm  Expected UCM grid size in y direction (for reference; not enforced)
     * @param[in] H_tree_max  Maximum allowed tree height [m]
     * @param[in] LAD_max     Maximum allowed LAD [m^2/m^3]
     * @param[in] lev     AMR level (default 0; kept for generality with other readers)
     * @param[in] ucm_debug   Enable debug output
     *
     * @throws Aborts via amrex::Abort() if:
     *  - File cannot be opened (file not found or permission denied)
     *  - Header line does not match expected physical-mode schema
     *  - Parsing error (non-numeric field, malformed CSV)
     *  - Any field outside sanity bounds
     */
    void read_and_broadcast(const std::string& path,
                            int nx_ucm, int ny_ucm,
                            amrex::Real H_tree_max,
                            amrex::Real LAD_max,
                            int lev, bool ucm_debug = false);

    /**
     * @brief Direct const access to broadcast row table
     *
     * Valid on all ranks after read_and_broadcast() completes.
     *
     * @return Reference to const vector of UCMTreeRow structs
     */
    const amrex::Vector<UCMTreeRow>& rows() const { return m_rows; }

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
    amrex::Vector<UCMTreeRow> m_rows;
};

#endif // ERF_UCM_TREE_LAYOUT_READER_H
