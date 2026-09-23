#ifndef ERF_CUSTOM_FUEL_H
#define ERF_CUSTOM_FUEL_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <ERF_FuelModels.H>

/**
 * @file ERF_CustomFuel.H
 * @brief User-defined fuel models: fuel codes 1000-1015 described in the deck
 *
 * Every rate-of-spread model in ERF-Fire reads its fuel through
 * FuelModelParams, and the two published sets that fill it — the Anderson 13
 * and the Scott and Burgan 40 — are compiled-in tables. A fuel complex that is
 * in neither set therefore had no way in. This class is that way in: codes
 * CUSTOM_FUEL_CODE_BASE to CUSTOM_FUEL_CODE_BASE + CUSTOM_FUEL_MAX - 1 carry
 * their properties from the input deck and take the fuel table slots above the
 * published sets.
 *
 * How far a deck-defined code reaches depends on how it is used, and the two
 * cases are not the same:
 *
 *  - As the uniform model, erf.fire.fuel_model_id, it is a published model's
 *    equal everywhere: every rate-of-spread model (Rothermel, BEHAVE, Balbi,
 *    MacArthur, Cheney-Gould, FBP) builds its uniform state from the same
 *    FuelModelParams, so all six respond to it.
 *  - From a spatial fuel map, per cell, only the models that have a per-fuel
 *    coefficient table evaluate it: Rothermel (with
 *    erf.fire.rothermel_per_fuel, on the isotropic and the level-set paths) and
 *    Balbi. BEHAVE, MacArthur, Cheney-Gould and FBP keep one uniform state for
 *    the whole grid, whatever the map says, so a deck-defined code in a raster
 *    does not change their rate. That is a property of those models' plumbing,
 *    not of this class.
 *
 * The heat flux, the per-cell fuel load, the flame residence time, the burnout
 * time, the wind adjustment factor (from the cell's bed depth) and the Byram
 * intensity and flame length (from the cell's heat content, and its initial
 * load under erf.fire.fuel_map.load_from_map) read the slot tables per cell in
 * every case, whatever rate-of-spread model is in use.
 *
 * erf.fire.fuel_model_id then names only the fuel a deck without a map burns
 * and, with erf.fire.fuel_map.load_from_map off, the load every cell starts at.
 *
 * Properties are given in SI, the units a fuel measurement is reported in, and
 * converted once here to the US units FuelModelParams carries internally.
 *
 *   erf.fire.custom_fuel.codes = 1000 1001
 *   erf.fire.custom_fuel.1000.name                  = coastal_scrub
 *   erf.fire.custom_fuel.1000.w_1h_kg_m2            = 0.60
 *   erf.fire.custom_fuel.1000.sav_1h_1_m            = 5000.0
 *   erf.fire.custom_fuel.1000.depth_m               = 0.60
 *   erf.fire.custom_fuel.1000.moisture_ext          = 0.25
 *   erf.fire.custom_fuel.1000.heat_content_J_kg     = 18600000.0
 *   erf.fire.custom_fuel.1000.density_kg_m3 = 512.0
 *
 * A custom code is recognised under either fuel set, so a deck may mix custom
 * codes with the Anderson 13 or with the Scott-Burgan 40 in one fuel raster.
 *
 * The table is built from the inputs and holds no evolving state, so it is
 * rebuilt identically on restart and is deliberately not checkpointed.
 */

/**
 * @struct CustomFuelEntry
 * @brief One deck-defined fuel model
 *
 * fp is in the US units of FuelModelParams, converted from the deck's SI
 * values when the entry is read, so the entry drops into any table slot the
 * published sets fill.
 */
struct CustomFuelEntry
{
    FuelModelParams fp{};                ///< Properties in FuelModelParams units
    amrex::Real     burnout_time_s = 0;  ///< Flaming burn time [s] for erf.fire.burnout_model = sfire
    std::string     name;                ///< Deck label, for the start-up summary
};

/**
 * @class CustomFuelTable
 * @brief The deck-defined fuel models, their validation and their lookups
 *
 * Owns nothing but the parsed entries: every consumer reaches them either
 * through params()/burnout_time_s() on the host, or through the slot-indexed
 * FuelModelParams table that build_fuel_params_slot_table() fills and the fire
 * layer copies to the device.
 */
class CustomFuelTable
{
public:
    /// Read erf.fire.custom_fuel.* and validate every property, aborting on a bad value.
    void init_from_inputs ();

    /// Whether the deck defines any custom fuel model.
    [[nodiscard]] bool active () const noexcept { return m_count > 0; }

    /// How many custom fuel models the deck defines.
    [[nodiscard]] int count () const noexcept { return m_count; }

    /// Whether this code is a custom code the deck defines.
    [[nodiscard]] bool has_code (int code) const noexcept
    {
        return custom_fuel_code(code) && m_defined[code - CUSTOM_FUEL_CODE_BASE];
    }

    /// Properties of a defined custom code; aborts when the code is not defined.
    [[nodiscard]] const FuelModelParams& params (int code) const;

    /// Flaming burn time [s] of a defined custom code; aborts when the code is not defined.
    [[nodiscard]] amrex::Real burnout_time_s (int code) const;

    /// One line per defined model, at start-up under erf.fire.fire_debug.
    void print_summary () const;

    /**
     * @brief Abort when a fuel raster holds a custom code the deck does not define.
     *
     * Without this a raster cell carrying, say, 1004 with no
     * erf.fire.custom_fuel.1004 block would fall through to the fuel set's
     * unknown-code default (Anderson model 1 or GR1) and burn as grass.
     *
     * A code the deck lists in erf.fire.fuel_map.nonburnable_codes is exempt:
     * it never spreads, so it needs no properties, and a raster that carries a
     * code the run is meant to ignore does not have to be edited.
     *
     * @param[in] fuel_model      Per-cell fuel code field
     * @param[in] nonburnable_codes erf.fire.fuel_map.nonburnable_codes
     */
    void validate_map_codes (const amrex::MultiFab& fuel_model,
                             const std::vector<int>& nonburnable_codes) const;

private:
    std::array<CustomFuelEntry, CUSTOM_FUEL_MAX> m_entry{};
    std::array<bool, CUSTOM_FUEL_MAX>            m_defined{};
    int                                          m_count = 0;
};

/**
 * @brief Fuel properties of a code, custom codes included.
 *
 * The host-side counterpart of get_fuel_params(): identical for every
 * published code, and the deck's entry for a custom code.
 *
 * A code in the custom range that the deck never defines aborts rather than
 * falling through to get_fuel_params(), which would hand back the fuel set's
 * unknown-code default (Anderson model 1 or GR1) and burn a structural fuel as
 * grass. The fuel raster has the same guard in validate_map_codes(); this one
 * covers erf.fire.fuel_model_id, which the raster check never sees.
 *
 * @param[in] where Which input named this code, for the abort message
 */
inline FuelModelParams fuel_params_with_custom (int code, int fuel_set, amrex::Real M_live,
                                                const CustomFuelTable& cft,
                                                const char* where = "erf.fire.fuel_model_id")
{
    if (cft.has_code(code)) { return cft.params(code); }
    if (custom_fuel_code(code)) {
        amrex::Abort("ERF-Fire custom fuel: " + std::string(where) + " = " + std::to_string(code)
                     + " is in the custom range " + std::to_string(CUSTOM_FUEL_CODE_BASE) + "-"
                     + std::to_string(CUSTOM_FUEL_CODE_BASE + CUSTOM_FUEL_MAX - 1)
                     + " but no erf.fire.custom_fuel." + std::to_string(code)
                     + ".* block defines it; add one or use a published fuel code");
    }
    return get_fuel_params(code, fuel_set, M_live);
}

/**
 * @brief Fuel properties of every table slot, for the device lookup.
 *
 * Slot s holds the properties of fuel_code_from_slot(s) under the set that slot
 * belongs to — the same expression the Rothermel and Balbi coefficient tables
 * are built with — so a kernel indexing this table by fuel_table_index() gets
 * exactly what a kernel calling get_fuel_params() got before the table existed.
 *
 * @param[in] fuel_set Fuel model set id of the deck's own codes
 * @param[in] M_live   Live herbaceous moisture for the dynamic models' curing transfer
 * @param[in] cft      The deck-defined models, for the slots above the published sets
 * @return host-side vector of size FUEL_SLOT_COUNT
 */
inline std::vector<FuelModelParams> build_fuel_params_slot_table (int fuel_set, amrex::Real M_live,
                                                                  const CustomFuelTable& cft)
{
    std::vector<FuelModelParams> table(FUEL_SLOT_COUNT);

    // A custom slot the deck does not define is non-burnable rather than the
    // fuel set's unknown-code default, so a code that slipped past
    // validate_map_codes cannot burn as grass. get_fuel_params(0) under the
    // Scott-Burgan set is that entry: zero loads on GR1's ratios, so nothing
    // downstream divides by zero.
    const FuelModelParams fp_none = get_fuel_params(0, FUEL_SET_SCOTT_BURGAN40, M_live);

    for (int slot = 0; slot < FUEL_SLOT_COUNT; ++slot) {
        if (slot >= FUEL_SLOT_CUSTOM_BASE) {
            const int code = CUSTOM_FUEL_CODE_BASE + (slot - FUEL_SLOT_CUSTOM_BASE);
            table[slot] = cft.has_code(code) ? cft.params(code) : fp_none;
            continue;
        }
        const int set = (slot >= FUEL_SLOT_SB40_BASE) ? FUEL_SET_SCOTT_BURGAN40 : fuel_set;
        table[slot] = get_fuel_params(fuel_code_from_slot(slot), set, M_live);
    }
    return table;
}

#endif  // ERF_CUSTOM_FUEL_H
