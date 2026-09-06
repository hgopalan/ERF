#ifndef ERF_FUEL_MODELS_H
#define ERF_FUEL_MODELS_H

#include <ERF_Rothermel.H>

/**
 * @file ERF_FuelModels.H
 * @brief Fuel model sets: the Anderson 13 and the Scott and Burgan 40
 *
 * The Anderson (1982) 13 are the historical set (codes 1-13, anything else
 * falling to model 1). The Scott and Burgan (2005) 40 are the set LANDFIRE's
 * FBFM40 rasters use: 101-109 GR grass, 121-124 GS grass-shrub, 141-149 SH
 * shrub, 161-165 TU timber-understory, 181-189 TL timber litter, 201-204 SB
 * slash-blowdown, and 91-99 non-burnable (91 urban, 92 snow/ice, 93
 * agriculture, 98 water, 99 barren). Loads are in the table's tons per acre
 * converted to lb/ft2, surface-area-to-volume ratios in 1/ft, depth in ft,
 * moisture of extinction as a fraction and heat content in BTU/lb, the units
 * of FuelModelParams. Every model uses the standard particle density of
 * 32 lb/ft3.
 *
 * The dynamic models (those with a live herbaceous load: GR, GS, SH1, SH9,
 * TU1, TU3) move their cured herbaceous load to the 1-h dead class, fully
 * cured at 30% live moisture and not at all at 120% (the BEHAVE/FARSITE
 * transfer); the transfer is applied when the parameters are built, at the
 * deck's erf.fire.moisture_live. The 1-h surface-area-to-volume ratio is
 * kept for the transferred load, since the single-class Rothermel path
 * here carries one dead ratio.
 *
 * Tables indexed by fuel are sized FUEL_SLOT_COUNT and addressed by
 * fuel_slot(code): 0 non-burnable, 1-13 the Anderson models at their own
 * codes (so the historical lookups are unchanged), 14-53 the Scott-Burgan
 * models in code order. fuel_table_index() gives a table index with the
 * set's semantics: with the Anderson set a code outside 0-13 is outside the
 * table, as before; with the Scott-Burgan set the non-burnable codes land on
 * slot 0 and unknown codes are outside the table.
 *
 * erf.fire.fuel_map.sb40_crosswalk translates the 40 to the 13 at load
 * instead, the way the Community Fire Behavior Model runs LANDFIRE data,
 * with the commonly used crosswalk in sb40_to_anderson().
 */

inline constexpr int FUEL_SET_ANDERSON13     = 0;
inline constexpr int FUEL_SET_SCOTT_BURGAN40 = 1;
inline constexpr int FUEL_SLOT_COUNT         = 54;   ///< 0 non-burnable, 1-13 Anderson, 14-53 Scott-Burgan

/// Non-burnable code of the Scott-Burgan set (91-99), or the map's 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool sb40_nonburnable(int code) noexcept { return code == 0 || (code >= 91 && code <= 99); }

/// Table slot of a fuel code: 1-13 Anderson, 14-53 Scott-Burgan, 0 otherwise.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int fuel_slot(int code) noexcept
{
    if (code >= 1 && code <= 13) return code;
    switch (code) {
    case 101: return 14;
    case 102: return 15;
    case 103: return 16;
    case 104: return 17;
    case 105: return 18;
    case 106: return 19;
    case 107: return 20;
    case 108: return 21;
    case 109: return 22;
    case 121: return 23;
    case 122: return 24;
    case 123: return 25;
    case 124: return 26;
    case 141: return 27;
    case 142: return 28;
    case 143: return 29;
    case 144: return 30;
    case 145: return 31;
    case 146: return 32;
    case 147: return 33;
    case 148: return 34;
    case 149: return 35;
    case 161: return 36;
    case 162: return 37;
    case 163: return 38;
    case 164: return 39;
    case 165: return 40;
    case 181: return 41;
    case 182: return 42;
    case 183: return 43;
    case 184: return 44;
    case 185: return 45;
    case 186: return 46;
    case 187: return 47;
    case 188: return 48;
    case 189: return 49;
    case 201: return 50;
    case 202: return 51;
    case 203: return 52;
    case 204: return 53;
    default: return 0;
    }
}

/// Fuel code held by a table slot (0 for slot 0).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int fuel_code_from_slot(int slot) noexcept
{
    constexpr int sb[40] = {
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        121,
        122,
        123,
        124,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        161,
        162,
        163,
        164,
        165,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        201,
        202,
        203,
        204,
    };
    if (slot >= 1 && slot <= 13) return slot;
    if (slot >= 14 && slot < FUEL_SLOT_COUNT) return sb[slot - 14];
    return 0;
}

/**
 * @brief Index into a slot-sized table with the fuel set's semantics.
 * @return The slot, or -1 when the code is outside the table (caller falls back).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int fuel_table_index(int code, int fuel_set) noexcept
{
    if (fuel_set == FUEL_SET_ANDERSON13) return (code >= 0 && code <= 13) ? code : -1;   // as the 14-entry tables behaved
    if (sb40_nonburnable(code)) return 0;
    const int s = fuel_slot(code);
    return (s > 0) ? s : -1;
}

/// The commonly used crosswalk from the Scott-Burgan 40 to the Anderson 13; non-burnable to 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int sb40_to_anderson(int code) noexcept
{
    if (code >= 1 && code <= 13) return code;
    if (sb40_nonburnable(code)) return 0;
    switch (code) {
    case 101: return 1;
    case 102: return 1;
    case 103: return 3;
    case 104: return 3;
    case 105: return 3;
    case 106: return 3;
    case 107: return 3;
    case 108: return 3;
    case 109: return 3;
    case 121: return 2;
    case 122: return 2;
    case 123: return 2;
    case 124: return 4;
    case 141: return 5;
    case 142: return 6;
    case 143: return 6;
    case 144: return 6;
    case 145: return 4;
    case 146: return 5;
    case 147: return 4;
    case 148: return 6;
    case 149: return 4;
    case 161: return 8;
    case 162: return 10;
    case 163: return 10;
    case 164: return 10;
    case 165: return 10;
    case 181: return 8;
    case 182: return 8;
    case 183: return 8;
    case 184: return 8;
    case 185: return 9;
    case 186: return 9;
    case 187: return 8;
    case 188: return 9;
    case 189: return 9;
    case 201: return 11;
    case 202: return 11;
    case 203: return 12;
    case 204: return 13;
    default: return code;
    }
}

/**
 * @brief Scott and Burgan (2005) fuel model parameters.
 * @param code   101-204
 * @param M_live Live herbaceous moisture [fraction] for the curing transfer of the
 *               dynamic models; negative leaves the herbaceous load live
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FuelModelParams get_scott_burgan_fuel_params(int code, amrex::Real M_live = -1.0)
{
    constexpr amrex::Real TPA = 2000.0 / 43560.0;   // tons per acre to lb/ft2
    FuelModelParams fp;
    fp.rho_p = 32.0;
    switch (code) {
    case 101:  // GR1
        fp.w_d1 = 0.10 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.30 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2200.0; fp.sigma_lh = 2000.0; fp.sigma_lw = 0.0;
        fp.delta = 0.40; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 102:  // GR2
        fp.w_d1 = 0.10 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 1.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 0.0;
        fp.delta = 1.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 103:  // GR3
        fp.w_d1 = 0.10 * TPA; fp.w_d10 = 0.40 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 1.50 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1500.0; fp.sigma_lh = 1300.0; fp.sigma_lw = 0.0;
        fp.delta = 2.00; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 104:  // GR4
        fp.w_d1 = 0.25 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 1.90 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 0.0;
        fp.delta = 2.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 105:  // GR5
        fp.w_d1 = 0.40 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 2.50 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1600.0; fp.sigma_lw = 0.0;
        fp.delta = 1.50; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 106:  // GR6
        fp.w_d1 = 0.10 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 3.40 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2200.0; fp.sigma_lh = 2000.0; fp.sigma_lw = 0.0;
        fp.delta = 1.50; fp.Mx = 0.40; fp.heat_content = 9000.0;
        break;
    case 107:  // GR7
        fp.w_d1 = 1.00 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 5.40 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 0.0;
        fp.delta = 3.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 108:  // GR8
        fp.w_d1 = 0.50 * TPA; fp.w_d10 = 1.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 7.30 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1500.0; fp.sigma_lh = 1300.0; fp.sigma_lw = 0.0;
        fp.delta = 4.00; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 109:  // GR9
        fp.w_d1 = 1.00 * TPA; fp.w_d10 = 1.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 9.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1600.0; fp.sigma_lw = 0.0;
        fp.delta = 5.00; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 121:  // GS1
        fp.w_d1 = 0.20 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.50 * TPA; fp.w_lw = 0.65 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1800.0;
        fp.delta = 0.90; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 122:  // GS2
        fp.w_d1 = 0.50 * TPA; fp.w_d10 = 0.50 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.60 * TPA; fp.w_lw = 1.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1800.0;
        fp.delta = 1.50; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 123:  // GS3
        fp.w_d1 = 0.30 * TPA; fp.w_d10 = 0.25 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 1.45 * TPA; fp.w_lw = 1.25 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1600.0; fp.sigma_lw = 1600.0;
        fp.delta = 1.80; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 124:  // GS4
        fp.w_d1 = 1.90 * TPA; fp.w_d10 = 0.30 * TPA; fp.w_d100 = 0.10 * TPA;
        fp.w_lh = 3.40 * TPA; fp.w_lw = 7.10 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1600.0; fp.sigma_lw = 1600.0;
        fp.delta = 2.10; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 141:  // SH1
        fp.w_d1 = 0.25 * TPA; fp.w_d10 = 0.25 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.15 * TPA; fp.w_lw = 1.30 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1600.0;
        fp.delta = 1.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 142:  // SH2
        fp.w_d1 = 1.35 * TPA; fp.w_d10 = 2.40 * TPA; fp.w_d100 = 0.75 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 3.85 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 1.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 143:  // SH3
        fp.w_d1 = 0.45 * TPA; fp.w_d10 = 3.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 6.20 * TPA;
        fp.sigma_d1 = 1600.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1400.0;
        fp.delta = 2.40; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 144:  // SH4
        fp.w_d1 = 0.85 * TPA; fp.w_d10 = 1.15 * TPA; fp.w_d100 = 0.20 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 2.55 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1600.0;
        fp.delta = 3.00; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 145:  // SH5
        fp.w_d1 = 3.60 * TPA; fp.w_d10 = 2.10 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 2.90 * TPA;
        fp.sigma_d1 = 750.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 6.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 146:  // SH6
        fp.w_d1 = 2.90 * TPA; fp.w_d10 = 1.45 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 1.40 * TPA;
        fp.sigma_d1 = 750.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 2.00; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 147:  // SH7
        fp.w_d1 = 3.50 * TPA; fp.w_d10 = 5.30 * TPA; fp.w_d100 = 2.20 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 3.40 * TPA;
        fp.sigma_d1 = 750.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 6.00; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    case 148:  // SH8
        fp.w_d1 = 2.05 * TPA; fp.w_d10 = 3.40 * TPA; fp.w_d100 = 0.85 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 4.35 * TPA;
        fp.sigma_d1 = 750.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 3.00; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 149:  // SH9
        fp.w_d1 = 4.50 * TPA; fp.w_d10 = 2.45 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 1.55 * TPA; fp.w_lw = 7.00 * TPA;
        fp.sigma_d1 = 750.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1500.0;
        fp.delta = 4.40; fp.Mx = 0.40; fp.heat_content = 8000.0;
        break;
    case 161:  // TU1
        fp.w_d1 = 0.20 * TPA; fp.w_d10 = 0.90 * TPA; fp.w_d100 = 1.50 * TPA;
        fp.w_lh = 0.20 * TPA; fp.w_lw = 0.90 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1800.0; fp.sigma_lw = 1600.0;
        fp.delta = 0.60; fp.Mx = 0.20; fp.heat_content = 8000.0;
        break;
    case 162:  // TU2
        fp.w_d1 = 0.95 * TPA; fp.w_d10 = 1.80 * TPA; fp.w_d100 = 1.25 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.20 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1600.0;
        fp.delta = 1.00; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 163:  // TU3
        fp.w_d1 = 1.10 * TPA; fp.w_d10 = 0.15 * TPA; fp.w_d100 = 0.25 * TPA;
        fp.w_lh = 0.65 * TPA; fp.w_lw = 1.10 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1600.0; fp.sigma_lw = 1400.0;
        fp.delta = 1.30; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 164:  // TU4
        fp.w_d1 = 4.50 * TPA; fp.w_d10 = 0.00 * TPA; fp.w_d100 = 0.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 2.00 * TPA;
        fp.sigma_d1 = 2300.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 2000.0;
        fp.delta = 0.50; fp.Mx = 0.12; fp.heat_content = 8000.0;
        break;
    case 165:  // TU5
        fp.w_d1 = 4.00 * TPA; fp.w_d10 = 4.00 * TPA; fp.w_d100 = 3.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 3.00 * TPA;
        fp.sigma_d1 = 1500.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 750.0;
        fp.delta = 1.00; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 181:  // TL1
        fp.w_d1 = 1.00 * TPA; fp.w_d10 = 2.20 * TPA; fp.w_d100 = 3.60 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.20; fp.Mx = 0.30; fp.heat_content = 8000.0;
        break;
    case 182:  // TL2
        fp.w_d1 = 1.40 * TPA; fp.w_d10 = 2.30 * TPA; fp.w_d100 = 2.20 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.20; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 183:  // TL3
        fp.w_d1 = 0.50 * TPA; fp.w_d10 = 2.20 * TPA; fp.w_d100 = 2.80 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.30; fp.Mx = 0.20; fp.heat_content = 8000.0;
        break;
    case 184:  // TL4
        fp.w_d1 = 0.50 * TPA; fp.w_d10 = 1.50 * TPA; fp.w_d100 = 4.20 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.40; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 185:  // TL5
        fp.w_d1 = 1.15 * TPA; fp.w_d10 = 2.50 * TPA; fp.w_d100 = 4.40 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.60; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 186:  // TL6
        fp.w_d1 = 2.40 * TPA; fp.w_d10 = 1.20 * TPA; fp.w_d100 = 1.20 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.30; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 187:  // TL7
        fp.w_d1 = 0.30 * TPA; fp.w_d10 = 1.40 * TPA; fp.w_d100 = 8.10 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.40; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 188:  // TL8
        fp.w_d1 = 5.80 * TPA; fp.w_d10 = 1.40 * TPA; fp.w_d100 = 1.10 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.30; fp.Mx = 0.35; fp.heat_content = 8000.0;
        break;
    case 189:  // TL9
        fp.w_d1 = 6.65 * TPA; fp.w_d10 = 3.30 * TPA; fp.w_d100 = 4.15 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 1800.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 0.60; fp.Mx = 0.35; fp.heat_content = 8000.0;
        break;
    case 201:  // SB1
        fp.w_d1 = 1.50 * TPA; fp.w_d10 = 3.00 * TPA; fp.w_d100 = 11.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 1.00; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 202:  // SB2
        fp.w_d1 = 4.50 * TPA; fp.w_d10 = 4.25 * TPA; fp.w_d100 = 4.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 1.00; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 203:  // SB3
        fp.w_d1 = 5.50 * TPA; fp.w_d10 = 2.75 * TPA; fp.w_d100 = 3.00 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 1.20; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    case 204:  // SB4
        fp.w_d1 = 5.25 * TPA; fp.w_d10 = 3.50 * TPA; fp.w_d100 = 5.25 * TPA;
        fp.w_lh = 0.00 * TPA; fp.w_lw = 0.00 * TPA;
        fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 0.0;
        fp.delta = 2.70; fp.Mx = 0.25; fp.heat_content = 8000.0;
        break;
    default:   // unknown: GR1, the lightest model, rather than nothing
        fp.w_d1 = 0.10 * TPA; fp.w_d10 = 0.0; fp.w_d100 = 0.0; fp.w_lh = 0.30 * TPA; fp.w_lw = 0.0;
        fp.sigma_d1 = 2200.0; fp.sigma_lh = 2000.0; fp.sigma_lw = 0.0;
        fp.delta = 0.40; fp.Mx = 0.15; fp.heat_content = 8000.0;
        break;
    }
    if (fp.w_lh > 0.0 && M_live >= 0.0) {
        // Dynamic model: the cured fraction of the herbaceous load joins the 1-h dead class.
        const amrex::Real T = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), (1.2 - M_live) / 0.9));
        fp.w_d1 += T * fp.w_lh;
        fp.w_lh *= (1.0 - T);
    }
    return fp;
}

/**
 * @brief Fuel parameters of a code under a fuel set.
 *
 * Anderson set: get_anderson_fuel_params(code), with its fall-through to
 * model 1 for unknown codes. Scott-Burgan set: 1-13 Anderson, 101-204
 * Scott-Burgan with the curing transfer at M_live, non-burnable codes a zero
 * load (GR1's ratios, so nothing divides by zero), unknown codes GR1.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FuelModelParams get_fuel_params(int code, int fuel_set, amrex::Real M_live = -1.0)
{
    if (fuel_set != FUEL_SET_SCOTT_BURGAN40) return get_anderson_fuel_params(code);
    if (code >= 1 && code <= 13) return get_anderson_fuel_params(code);
    if (sb40_nonburnable(code)) {
        FuelModelParams fp = get_scott_burgan_fuel_params(101);
        fp.w_d1 = fp.w_d10 = fp.w_d100 = fp.w_lh = fp.w_lw = 0.0;
        return fp;
    }
    return get_scott_burgan_fuel_params(code, M_live);
}

/// Total oven-dry load of a fuel model [kg/m2] from its lb/ft2 loads.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real fuel_total_load_kg_m2(const FuelModelParams& fp) noexcept
{
    return (fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw) * 4.88243;
}

#endif
