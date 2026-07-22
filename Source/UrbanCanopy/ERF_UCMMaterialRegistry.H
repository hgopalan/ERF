/**
 * @file ERF_UCMMaterialRegistry.H
 * @brief Material-library CSV reader and registry for the ERF-SLUCM module (Phase 2.1)
 *
 * Provides a POD struct for material properties and a rank-0-read + MPI_Bcast registry
 * following the pattern from `Source/LNG/ERF_LNGRegulatory.H/.cpp` for lookup tables.
 *
 * ## CSV Schema
 * Expected header (exact string, whitespace tolerated):
 * ```
 * mat_id,name,albedo,emissivity,k_therm_W_per_mK,rho_cp_J_per_m3K,thickness_m,description
 * ```
 *
 * Each data row contains:
 * - mat_id: unique material identifier (integer, typically 1, 2, 3, ...)
 * - name: material name (string, fixed 64 chars for POD safety)
 * - albedo: shortwave reflectance [-] (float, typically 0.05–0.95)
 * - emissivity: longwave emissivity [-] (float, typically 0.8–0.95)
 * - k_therm_W_per_mK: thermal conductivity [W/(m·K)] (float)
 * - rho_cp_J_per_m3K: volumetric heat capacity [J/(m³·K)] (float)
 * - thickness_m: typical slab thickness [m] (float, may inform slab_L per material Phase 2.2+)
 * - description: optional description (string, fixed 128 chars for POD safety)
 *
 * ## References
 *  - Source/LNG/ERF_LNGRegulatory.H (lookup-table registry pattern)
 *  - Source/Dust/ERF_DustSurfaceReader.H (factory/registry pattern)
 *  - Source/LNG/LNG_MPI_SKILLS.md (MPI safety guidelines)
 */

#ifndef ERF_UCM_MATERIAL_REGISTRY_H
#define ERF_UCM_MATERIAL_REGISTRY_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>
#include <unordered_map>
#include <unordered_set>

/**
 * @struct UCMMaterial
 * @brief POD struct for a single material library entry
 *
 * Safe for MPI_Bcast as raw bytes (std::is_trivially_copyable_v<UCMMaterial> == true).
 * All fields are POD types (int, amrex::Real, char arrays) with no pointers or virtuals.
 *
 * Fixed-size char arrays (name[64], description[128]) ensure deterministic size and
 * alignment for MPI_Bcast. Strings are null-terminated but may be truncated if input
 * exceeds fixed size.
 *
 * Phase 2.1: used to store per-material optical + thermal properties.
 * Phase 2.2+: thickness_m may inform per-material slab depths.
 */
struct UCMMaterial
{
    int    mat_id;                     ///< Unique material ID (typically >= 1)
    char   name[64];                   ///< Material name (fixed for POD safety)
    amrex::Real albedo;                ///< Shortwave albedo [-]
    amrex::Real emissivity;            ///< Longwave emissivity [-]
    amrex::Real k_therm_W_per_mK;      ///< Thermal conductivity [W/(m·K)]
    amrex::Real rho_cp_J_per_m3K;      ///< Volumetric heat capacity [J/(m³·K)]
    amrex::Real thickness_m;           ///< Slab thickness [m] (informational)
    char   description[128];           ///< Material description (fixed for POD safety)
};

/**
 * @class UCMMaterialRegistry
 * @brief In-memory lookup table for material properties
 *
 * Implements rank-0-read + MPI_Bcast pattern for material-library CSV.
 * On rank 0, parses CSV and stores materials in m_table vector, then builds
 * m_id_to_idx map for O(1) lookup. Broadcasts data to all ranks via MPI_Bcast.
 *
 * Enforces:
 *  - No duplicate mat_id (aborts with error if found)
 *  - All values in physically plausible ranges (albedo, emissivity ∈ [0, 1];
 *    k_therm, rho_cp, thickness > 0)
 *  - lookup(mat_id) returns UCMMaterial& or aborts if mat_id not in registry
 *
 * Example usage:
 * ```cpp
 * UCMMaterialRegistry registry;
 * registry.load_and_broadcast("materials.csv");
 * const UCMMaterial& mat = registry.lookup(1);  // Look up mat_id=1
 * double albedo = mat.albedo;
 * ```
 */
class UCMMaterialRegistry
{
public:
    /**
     * @brief Rank-0 read CSV, then MPI_Bcast POD struct array to all ranks
     *
     * Execution:
     *  - Rank 0: open file, parse header (must match schema), parse data rows
     *  - Rank 0: validate rows (no duplicate mat_id, check value ranges)
     *  - Rank 0: build m_id_to_idx map for fast lookup
     *  - Rank 0: abort if any validation fails
     *  - Rank 0: emit debug trace (path, material count, per-entry details)
     *  - All ranks: MPI_Bcast material count, then material data as bytes
     *  - All ranks: reconstruct m_id_to_idx map
     *  - All ranks: call lookup(mat_id) to retrieve materials
     *
     * Debug output (IO-rank only, if ucm_debug enabled):
     *  ```
     *  [UCM][2.1][UCMMaterialRegistry::load_and_broadcast]
     *    path = /path/to/materials.csv
     *    n_materials = 3
     *    mat_id=1: name="Asphalt", albedo=0.15, emissivity=0.90, k=1.2, rho_cp=1.5e6, thickness=0.2
     *    mat_id=2: name="Concrete", albedo=0.35, emissivity=0.93, k=1.4, rho_cp=1.9e6, thickness=0.3
     *    mat_id=3: name="Brick", albedo=0.40, emissivity=0.90, k=0.9, rho_cp=1.6e6, thickness=0.15
     *  ```
     *
     * @param[in] path Path to material-library CSV file
     * @param[in] lev  AMR level (default 0; kept for generality with other readers)
     * @param[in] ucm_debug If true, emit debug output on IO rank
     *
     * @throws Aborts via amrex::Abort() if:
     *  - File cannot be opened (file not found or permission denied)
     *  - Header line does not match expected schema
     *  - Parsing error (non-numeric field, malformed CSV)
     *  - Duplicate mat_id found
     *  - Albedo or emissivity not in [0, 1]
     *  - k_therm, rho_cp, or thickness not > 0
     */
    void load_and_broadcast(const std::string& path, int lev = 0, bool ucm_debug = false);

    /**
     * @brief Look up material by ID
     *
     * Returns a const reference to the UCMMaterial struct for the given mat_id.
     * Aborts with a clear message if mat_id is not in the registry.
     *
     * On first call for a given mat_id (per rank), emits a debug trace if
     * ucm_debug is enabled (one-time per unique mat_id across the application).
     *
     * Debug output format:
     *  ```
     *  [UCM][2.1][UCMMaterialRegistry::lookup] mat_id=5 first call → name="Custom", albedo=0.30
     *  ```
     *
     * @param[in] mat_id Material ID to look up
     * @return Const reference to UCMMaterial; aborts if not found
     * @throws Aborts if mat_id not in registry
     */
    const UCMMaterial& lookup(int mat_id) const;

    /**
     * @brief Number of materials in the registry
     *
     * Valid on all ranks after load_and_broadcast() completes.
     *
     * @return Number of materials (same on all ranks after broadcast)
     */
    int size() const { return static_cast<int>(m_table.size()); }

    /**
     * @brief Const access to the full material table
     *
     * Valid on all ranks after load_and_broadcast() completes.
     *
     * @return Reference to const vector of UCMMaterial structs
     */
    const amrex::Vector<UCMMaterial>& table() const { return m_table; }

private:
    /// Material table (indexed by position; use m_id_to_idx for mat_id → index lookup)
    amrex::Vector<UCMMaterial> m_table;

    /// Lookup map: mat_id → index in m_table for O(1) retrieval
    std::unordered_map<int, int> m_id_to_idx;

    /// Debug flag (set via load_and_broadcast; persistent for lookup() calls)
    bool m_ucm_debug = false;

    /// Set of material IDs for which we've already printed a lookup trace
    /// (used to ensure one-time-per-unique-mat_id debug output)
    mutable std::unordered_set<int> m_lookup_trace_emitted;
};

#endif // ERF_UCM_MATERIAL_REGISTRY_H
