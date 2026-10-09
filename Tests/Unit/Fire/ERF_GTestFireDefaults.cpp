#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <string>
#include <vector>

#include "ERF_FireParams.H"
#include "ERF_Rothermel.H"
#include "ERF_BehaveModel.H"
#include "ERF_FarsiteEllipse.H"
#include "ERF_CheneyGouldModel.H"

/**
 * @file ERF_GTestFireDefaults.cpp
 * @brief The defaults the 2026-10 validation changed, and the keys that keep
 *        the earlier forms: an empty deck takes the reviewed physics, and each
 *        legacy value is read back as its selector.
 */

using namespace amrex;

namespace {

/// Keys added to erf.fire for one test and removed on exit
class ScopedKeys
{
public:
    ScopedKeys () = default;
    ~ScopedKeys () { for (const auto& n : m_names) { m_pp.remove(n.c_str()); } }
    ScopedKeys (const ScopedKeys&) = delete;
    ScopedKeys& operator= (const ScopedKeys&) = delete;

    void add (const char* key, const std::string& v) { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, double v)             { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, int v)                { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, bool v)               { m_pp.add(key, v); m_names.emplace_back(key); }

private:
    ParmParse                m_pp{"erf.fire"};
    std::vector<std::string> m_names;
};

} // namespace

TEST(FireDefaults, AnEmptyDeckTakesTheReviewedForms)
{
    FireParams fp;
    EXPECT_EQ(fp.behave.net_load_mode, behave_net_load::weighted) << "BEHAVE net load: surface-area weighted";
    EXPECT_EQ(fp.farsite_shape, farsite_shape::ellipse)            << "FARSITE shape: the Richards ellipse";
    EXPECT_TRUE(fp.firebreak_use_mask)                             << "firebreaks mask the fuel";
    EXPECT_FALSE(fp.use_terrain_wind)                              << "terrain wind factors off";
    EXPECT_TRUE(fp.use_wind_limit);
    EXPECT_EQ(fp.wind_limit_mode, fire_wind_limit::rothermel)      << "the wind limit is Rothermel's 0.9 I_R";
    EXPECT_FALSE(fp.wind_below_first_log)                          << "the centre's wind below the first cell centre (log is opt-in)";
    EXPECT_FALSE(fp.heat_tendency_exner)                           << "WRF-SFIRE's tendency without the Exner factor";
    EXPECT_NEAR(fp.macarthur_ros_max, 6.0, 1.0e-12)                << "WRF-Fire's 6 m/s cap";
    EXPECT_TRUE(fp.rothermel_cell_moisture);
    EXPECT_EQ(fp.ros_model, "rothermel");
    EXPECT_EQ(fp.cheney_gould.pasture, cheney_gould::pasture_natural);
    EXPECT_EQ(fp.cheney_gould.curing_curve, cheney_gould::curing_cruz2015);
    EXPECT_EQ(fp.cheney_gould.wind_source, 0) << "the 1998 model reads the 10 m wind";
}

TEST(FireDefaults, LegacyKeysSelectTheEarlierForms)
{
    ScopedKeys deck;
    deck.add("farsite.shape", std::string("rectangle"));
    deck.add("wind_limit", std::string("fuel_class"));
    deck.add("wind_below_first_cell", std::string("log"));
    deck.add("use_terrain_wind", true);
    deck.add("heat_tendency_exner", true);
    deck.add("rothermel_cell_moisture", false);
    deck.add("firebreak.use_mask", false);

    FireParams fp;
    EXPECT_EQ(fp.farsite_shape, farsite_shape::rectangle);
    EXPECT_EQ(fp.wind_limit_mode, fire_wind_limit::fuel_class);
    EXPECT_TRUE(fp.wind_below_first_log);
    EXPECT_TRUE(fp.use_terrain_wind);
    EXPECT_TRUE(fp.heat_tendency_exner);
    EXPECT_FALSE(fp.rothermel_cell_moisture);
    EXPECT_FALSE(fp.firebreak_use_mask);
}

TEST(FireDefaults, TheBehaveNetLoadIsReadWithItsModel)
{
    ScopedKeys deck;
    deck.add("ros_model", std::string("behave"));
    deck.add("behave.net_load", std::string("sum"));
    FireParams fp;
    EXPECT_EQ(fp.behave.net_load_mode, behave_net_load::sum);
}

TEST(FireDefaults, TheMcArthurCapIsReadWithItsModel)
{
    ScopedKeys deck;
    deck.add("ros_model", std::string("macarthur"));
    deck.add("macarthur.ros_max", 0.0);
    FireParams fp;
    EXPECT_NEAR(fp.macarthur_ros_max, 0.0, 1.0e-12) << "0 removes the cap";
}

TEST(FireDefaults, GrassModelsAndTheirKeys)
{
    {
        ScopedKeys deck;
        deck.add("ros_model", std::string("grass_simple"));
        FireParams fp;
        EXPECT_EQ(fp.ros_model, "grass_simple") << "the pre-2026-10 fit is kept under its own name";
    }
    {
        ScopedKeys deck;
        deck.add("ros_model", std::string("cheney_gould"));
        deck.add("cheney_gould.pasture", std::string("grazed"));
        deck.add("cheney_gould.curing_curve", std::string("cheney1998"));
        deck.add("cheney_gould.wind_source", std::string("midflame"));
        FireParams fp;
        EXPECT_EQ(fp.cheney_gould.pasture, cheney_gould::pasture_grazed);
        EXPECT_EQ(fp.cheney_gould.curing_curve, cheney_gould::curing_cheney1998);
        EXPECT_EQ(fp.cheney_gould.wind_source, 1);
    }
}
