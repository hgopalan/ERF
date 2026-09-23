#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <AMReX_ParmParse.H>

#include <ERF_CustomFuel.H>
#include <ERF_FuelModels.H>
#include <ERF_Rothermel.H>
#include <ERF_BalbiModel.H>

using amrex::Real;
using amrex::ParmParse;

namespace {

/// Round-off only: the conversions are a divide here and a multiply back.
constexpr Real TOL = (sizeof(Real) == 8) ? Real(1.0e-10) : Real(1.0e-4);

constexpr Real M_DEAD = 0.08;
constexpr Real M_LIVE_OFF = -1.0;

/// SI values of Scott-Burgan GR2, so a deck entry can reproduce it exactly.
constexpr Real GR2_W_1H_KG_M2      = 0.10 * (2000.0 / 43560.0) * 4.88243;
constexpr Real GR2_W_LH_KG_M2      = 1.00 * (2000.0 / 43560.0) * 4.88243;
constexpr Real GR2_SAV_1H_INV_M    = 2000.0 * 3.28084;
constexpr Real GR2_SAV_LH_INV_M    = 1800.0 * 3.28084;
constexpr Real GR2_DEPTH_M         = 1.00 * 0.3048;
constexpr Real GR2_MX              = 0.15;
constexpr Real GR2_HEAT_J_KG       = 8000.0 * 2326.0;
constexpr Real GR2_RHO_P_KG_M3     = 32.0 * 16.0185;

/// Adds erf.fire.custom_fuel.* keys and removes them again, so one test's deck
/// cannot leak into the next through ParmParse's global table.
class ScopedCustomFuel
{
public:
    explicit ScopedCustomFuel (const std::vector<int>& codes)
    {
        m_pp.addarr("custom_fuel.codes", codes);
        m_names.emplace_back("custom_fuel.codes");
    }

    ~ScopedCustomFuel ()
    {
        for (const auto& n : m_names) { m_pp.remove(n.c_str()); }
    }

    ScopedCustomFuel (const ScopedCustomFuel&) = delete;
    ScopedCustomFuel& operator= (const ScopedCustomFuel&) = delete;

    void add (int code, const char* prop, Real value)
    {
        const std::string name = "custom_fuel." + std::to_string(code) + "." + prop;
        m_pp.add(name.c_str(), double(value));
        m_names.emplace_back(name);
    }

private:
    ParmParse                m_pp{"erf.fire"};
    std::vector<std::string> m_names;
};

/// The full GR2-equivalent block at one custom code.
void add_gr2_block (ScopedCustomFuel& deck, int code)
{
    deck.add(code, "w_1h_kg_m2",             GR2_W_1H_KG_M2);
    deck.add(code, "w_live_herb_kg_m2",      GR2_W_LH_KG_M2);
    deck.add(code, "sav_1h_1_m",             GR2_SAV_1H_INV_M);
    deck.add(code, "sav_live_herb_1_m",      GR2_SAV_LH_INV_M);
    deck.add(code, "depth_m",                GR2_DEPTH_M);
    deck.add(code, "moisture_ext",           GR2_MX);
    deck.add(code, "heat_content_J_kg",      GR2_HEAT_J_KG);
    deck.add(code, "density_kg_m3", GR2_RHO_P_KG_M3);
}

void expect_same_fuel (const FuelModelParams& a, const FuelModelParams& b, const char* what)
{
    auto rel = [&] (Real x, Real y, const char* field) {
        EXPECT_NEAR(x, y, TOL * std::max(Real(1.0), std::abs(y))) << what << " field " << field;
    };
    rel(a.w_d1,   b.w_d1,   "w_d1");
    rel(a.w_d10,  b.w_d10,  "w_d10");
    rel(a.w_d100, b.w_d100, "w_d100");
    rel(a.w_lh,   b.w_lh,   "w_lh");
    rel(a.w_lw,   b.w_lw,   "w_lw");
    rel(a.sigma_d1, b.sigma_d1, "sigma_d1");
    rel(a.sigma_lh, b.sigma_lh, "sigma_lh");
    rel(a.sigma_lw, b.sigma_lw, "sigma_lw");
    rel(a.delta,  b.delta,  "delta");
    rel(a.Mx,     b.Mx,     "Mx");
    rel(a.heat_content, b.heat_content, "heat_content");
    rel(a.rho_p,  b.rho_p,  "rho_p");
}

}  // namespace

// Motivation: the custom codes take the table slots above the published sets.
// A mapping that overlapped a published slot would silently replace a
// Scott-Burgan model with a deck-defined one.
TEST(CustomFuel, SlotMappingIsDisjointFromThePublishedSets)
{
    for (int i = 0; i < CUSTOM_FUEL_MAX; ++i) {
        const int code = CUSTOM_FUEL_CODE_BASE + i;
        EXPECT_TRUE(custom_fuel_code(code)) << "code " << code;
        EXPECT_EQ(fuel_slot(code), FUEL_SLOT_CUSTOM_BASE + i) << "code " << code;
        EXPECT_EQ(fuel_code_from_slot(FUEL_SLOT_CUSTOM_BASE + i), code);
        EXPECT_LT(fuel_slot(code), FUEL_SLOT_COUNT) << "code " << code;
    }

    EXPECT_FALSE(custom_fuel_code(CUSTOM_FUEL_CODE_BASE - 1));
    EXPECT_FALSE(custom_fuel_code(CUSTOM_FUEL_CODE_BASE + CUSTOM_FUEL_MAX));

    // No published code may land on a custom slot, and none is non-burnable.
    for (int code = 1; code <= 13; ++code) {
        EXPECT_LT(fuel_slot(code), FUEL_SLOT_CUSTOM_BASE) << "Anderson " << code;
        EXPECT_FALSE(custom_fuel_code(code));
    }
    for (int slot = FUEL_SLOT_SB40_BASE; slot < FUEL_SLOT_CUSTOM_BASE; ++slot) {
        const int code = fuel_code_from_slot(slot);
        EXPECT_EQ(fuel_slot(code), slot) << "Scott-Burgan " << code;
        EXPECT_FALSE(custom_fuel_code(code)) << "Scott-Burgan " << code;
        EXPECT_FALSE(sb40_nonburnable(code)) << "Scott-Burgan " << code;
    }
}

// Motivation: a custom code has to resolve under either fuel set, so a deck may
// mix it with the Anderson 13 as well as with the Scott-Burgan 40. The Anderson
// branch of fuel_table_index() returns -1 for everything above 13, which would
// have sent every custom code to the uniform fallback.
TEST(CustomFuel, TableIndexResolvesCustomCodesUnderEitherSet)
{
    for (int i = 0; i < CUSTOM_FUEL_MAX; ++i) {
        const int code = CUSTOM_FUEL_CODE_BASE + i;
        EXPECT_EQ(fuel_table_index(code, FUEL_SET_ANDERSON13),     FUEL_SLOT_CUSTOM_BASE + i);
        EXPECT_EQ(fuel_table_index(code, FUEL_SET_SCOTT_BURGAN40), FUEL_SLOT_CUSTOM_BASE + i);
    }
    // Codes outside both the published sets and the custom range stay outside.
    EXPECT_EQ(fuel_table_index(500, FUEL_SET_ANDERSON13),     -1);
    EXPECT_EQ(fuel_table_index(500, FUEL_SET_SCOTT_BURGAN40),  -1);
    EXPECT_EQ(fuel_table_index(CUSTOM_FUEL_CODE_BASE + CUSTOM_FUEL_MAX, FUEL_SET_SCOTT_BURGAN40), -1);
}

// Motivation: the slot table is what the device kernels read instead of the
// get_fuel_params() switch. Any disagreement would change every existing run.
TEST(CustomFuel, SlotTableReproducesGetFuelParamsForEveryPublishedCode)
{
    const CustomFuelTable none;   // no deck block

    for (int fuel_set : {FUEL_SET_ANDERSON13, FUEL_SET_SCOTT_BURGAN40}) {
        const auto tbl = build_fuel_params_slot_table(fuel_set, M_LIVE_OFF, none);
        ASSERT_EQ(static_cast<int>(tbl.size()), FUEL_SLOT_COUNT);

        for (int code = 0; code <= 13; ++code) {
            const int idx = fuel_table_index(code, fuel_set);
            ASSERT_GE(idx, 0) << "code " << code;
            expect_same_fuel(tbl[idx], get_fuel_params(code, fuel_set, M_LIVE_OFF), "Anderson");
        }
        for (int slot = FUEL_SLOT_SB40_BASE; slot < FUEL_SLOT_CUSTOM_BASE; ++slot) {
            const int code = fuel_code_from_slot(slot);
            expect_same_fuel(tbl[slot],
                             get_fuel_params(code, FUEL_SET_SCOTT_BURGAN40, M_LIVE_OFF),
                             "Scott-Burgan");
        }
    }
}

// Motivation: a custom slot with no deck block must be non-burnable, not the
// fuel set's unknown-code default, which is grass. A fuel raster that slipped a
// stray 1000 past validate_map_codes would otherwise spread at GR1's rate.
TEST(CustomFuel, UndefinedCustomSlotIsNonBurnable)
{
    const CustomFuelTable none;
    EXPECT_FALSE(none.active());
    EXPECT_EQ(none.count(), 0);
    EXPECT_FALSE(none.has_code(CUSTOM_FUEL_CODE_BASE));

    const auto tbl = build_fuel_params_slot_table(FUEL_SET_SCOTT_BURGAN40, M_LIVE_OFF, none);
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        EXPECT_EQ(fuel_total_load_kg_m2(tbl[slot]), Real(0.0)) << "slot " << slot;
    }

    // And with the slot table in hand, the coefficients give no spread and stay
    // finite: the coefficients of a zero load divide by the packing ratio.
    const auto rc = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, FUEL_SET_SCOTT_BURGAN40,
                                               M_LIVE_OFF, true, tbl.data());
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        EXPECT_EQ(rc[slot].R0, Real(0.0)) << "slot " << slot;
        EXPECT_TRUE(std::isfinite(rc[slot].beta_ratio_E)) << "slot " << slot;
        EXPECT_TRUE(std::isfinite(rc[slot].I_R))          << "slot " << slot;
    }

    FireParams::BalbiParams bp;
    const auto bc = build_fuel_balbi_table(bp, M_DEAD, Real(-1.0), FUEL_SET_SCOTT_BURGAN40,
                                           M_LIVE_OFF, tbl.data());
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        EXPECT_EQ(bc[slot].A_coeff, Real(0.0))       << "slot " << slot;
        EXPECT_TRUE(std::isfinite(bc[slot].Rb_coef)) << "slot " << slot;
        EXPECT_TRUE(std::isfinite(bc[slot].Rc_coef)) << "slot " << slot;
    }
}

// Motivation: the whole feature is the SI-to-FuelModelParams conversion. A deck
// entry whose SI numbers are GR2's must come back as GR2 in every field and
// produce GR2's Rothermel coefficients, which pins all five conversion factors
// and the field order at once.
TEST(CustomFuel, DeckEntryInSIReproducesScottBurganGR2)
{
    constexpr int CODE = CUSTOM_FUEL_CODE_BASE;

    ScopedCustomFuel deck({CODE});
    add_gr2_block(deck, CODE);
    deck.add(CODE, "burnout_time_s", Real(42.0));

    CustomFuelTable cft;
    cft.init_from_inputs();
    ASSERT_TRUE(cft.active());
    EXPECT_EQ(cft.count(), 1);
    ASSERT_TRUE(cft.has_code(CODE));
    EXPECT_FALSE(cft.has_code(CODE + 1));

    // GR2 untransferred: a deck-defined model has no curing transfer, so the
    // comparison is against the table entry, not get_fuel_params at an M_live.
    const FuelModelParams gr2 = get_scott_burgan_fuel_params(102);
    expect_same_fuel(cft.params(CODE), gr2, "GR2 round trip");

    // The deck's burn time is what the sfire burnout model gets, not grass's 7 s.
    EXPECT_EQ(cft.burnout_time_s(CODE), Real(42.0));

    // Same properties, so same coefficients: this is the path a run takes.
    const auto rc_custom = compute_rothermel_params(cft.params(CODE), M_DEAD, M_DEAD, M_DEAD, true);
    const auto rc_gr2    = compute_rothermel_params(gr2, M_DEAD, M_DEAD, M_DEAD, true);
    EXPECT_GT(rc_gr2.R0, Real(0.0));   // non-vacuous: GR2 does spread
    EXPECT_NEAR(rc_custom.R0, rc_gr2.R0, TOL * std::max(Real(1.0), rc_gr2.R0));
    EXPECT_NEAR(rc_custom.C,           rc_gr2.C,           TOL * std::max(Real(1.0), rc_gr2.C));
    EXPECT_NEAR(rc_custom.B,           rc_gr2.B,           TOL * std::max(Real(1.0), rc_gr2.B));
    EXPECT_NEAR(rc_custom.beta,        rc_gr2.beta,        TOL * std::max(Real(1.0), rc_gr2.beta));
    EXPECT_NEAR(rc_custom.phi_s_const, rc_gr2.phi_s_const, TOL * std::max(Real(1.0), rc_gr2.phi_s_const));
    EXPECT_NEAR(rc_custom.I_R,         rc_gr2.I_R,         TOL * std::max(Real(1.0), rc_gr2.I_R));

    // And it reaches the slot the kernels index.
    const auto tbl = build_fuel_params_slot_table(FUEL_SET_SCOTT_BURGAN40, M_LIVE_OFF, cft);
    expect_same_fuel(tbl[fuel_slot(CODE)], gr2, "GR2 at its slot");
}

// Motivation: a fuel model the published sets cannot express is the reason the
// feature exists. A heavy, coarse, deep bed must spread slower than grass; if
// the entry were quietly falling back to a published model this would not hold.
TEST(CustomFuel, CoarseHeavyDeckFuelSpreadsSlowerThanGrass)
{
    constexpr int CODE = CUSTOM_FUEL_CODE_BASE + 3;

    ScopedCustomFuel deck({CODE});
    // Dimensional lumber rather than fine fuel: a low surface-area-to-volume
    // ratio and a heavy load, which Rothermel's reaction term punishes.
    deck.add(CODE, "w_1h_kg_m2",             Real(2.0));
    deck.add(CODE, "w_10h_kg_m2",            Real(20.0));
    deck.add(CODE, "sav_1h_1_m",             Real(200.0));
    deck.add(CODE, "depth_m",                Real(2.0));
    deck.add(CODE, "moisture_ext",           Real(0.35));
    deck.add(CODE, "heat_content_J_kg",      Real(1.86e7));
    deck.add(CODE, "density_kg_m3", Real(512.0));

    CustomFuelTable cft;
    cft.init_from_inputs();
    ASSERT_TRUE(cft.has_code(CODE));

    const FuelModelParams heavy = cft.params(CODE);
    const FuelModelParams grass = get_fuel_params(1, FUEL_SET_ANDERSON13, M_LIVE_OFF);

    // The entry is genuinely its own fuel, not a published one in disguise.
    EXPECT_GT(fuel_total_load_kg_m2(heavy), fuel_total_load_kg_m2(grass) * Real(5.0));
    EXPECT_LT(heavy.sigma_d1, grass.sigma_d1 * Real(0.2));

    const auto rc_heavy = compute_rothermel_params(heavy, M_DEAD, M_DEAD, M_DEAD, true);
    const auto rc_grass = compute_rothermel_params(grass, M_DEAD, M_DEAD, M_DEAD, true);
    EXPECT_GT(rc_grass.R0, Real(0.0));
    EXPECT_GT(rc_heavy.R0, Real(0.0));
    EXPECT_LT(rc_heavy.R0, rc_grass.R0);
}

// Motivation: two codes in one deck must stay in their own slots. A shared
// buffer or an off-by-one in the slot arithmetic would give both the same fuel.
TEST(CustomFuel, TwoDeckEntriesKeepTheirOwnSlots)
{
    constexpr int CODE_A = CUSTOM_FUEL_CODE_BASE + 7;
    constexpr int CODE_B = CUSTOM_FUEL_CODE_BASE + 9;

    ScopedCustomFuel deck({CODE_A, CODE_B});
    add_gr2_block(deck, CODE_A);
    deck.add(CODE_B, "w_1h_kg_m2",             Real(5.0));
    deck.add(CODE_B, "sav_1h_1_m",             Real(400.0));
    deck.add(CODE_B, "depth_m",                Real(1.5));
    deck.add(CODE_B, "moisture_ext",           Real(0.30));
    deck.add(CODE_B, "heat_content_J_kg",      Real(2.0e7));
    deck.add(CODE_B, "density_kg_m3", Real(600.0));

    CustomFuelTable cft;
    cft.init_from_inputs();
    EXPECT_EQ(cft.count(), 2);
    ASSERT_TRUE(cft.has_code(CODE_A));
    ASSERT_TRUE(cft.has_code(CODE_B));
    EXPECT_FALSE(cft.has_code(CUSTOM_FUEL_CODE_BASE + 8));

    expect_same_fuel(cft.params(CODE_A), get_scott_burgan_fuel_params(102), "code A");
    EXPECT_GT(cft.params(CODE_B).w_d1, cft.params(CODE_A).w_d1);

    const auto tbl = build_fuel_params_slot_table(FUEL_SET_ANDERSON13, M_LIVE_OFF, cft);
    expect_same_fuel(tbl[fuel_slot(CODE_A)], cft.params(CODE_A), "slot A");
    expect_same_fuel(tbl[fuel_slot(CODE_B)], cft.params(CODE_B), "slot B");
    // The undefined slot between them is still non-burnable.
    EXPECT_EQ(fuel_total_load_kg_m2(tbl[fuel_slot(CUSTOM_FUEL_CODE_BASE + 8)]), Real(0.0));
}
