#include <gtest/gtest.h>
#include <AMReX_ParmParse.H>
#include <map>
#include <string>
#include <vector>

#include "ERF_FireModelSummary.H"

/**
 * @file ERF_GTestFireModelSummary.cpp
 * @brief The start-up summary names the model or option of every piece of
 *        fire physics with the value FireParams read, and lists a model's
 *        own options only when the model is in use.
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
private:
    ParmParse                m_pp{"erf.fire"};
    std::vector<std::string> m_names;
};

std::map<std::string, std::string> by_key (const FireParams& fp)
{
    std::map<std::string, std::string> m;
    for (const auto& l : fire_model_summary(fp)) { m[l.key] = l.value; }
    return m;
}

} // namespace

// One line per piece of physics on an empty deck, each with the default the
// constructor sets (a key missing here is physics the log would not name)
TEST(FireModelSummary, AnEmptyDeckNamesEveryPhysics)
{
    FireParams fp;
    const auto m = by_key(fp);
    for (const char* k : {"ros_model", "reaction_velocity_formula", "wrf_bmst_compat", "rothermel_per_fuel",
                          "rothermel_cell_moisture", "use_wind_limit", "wind_limit", "propagation_method",
                          "farsite.shape", "farsite.front_update", "wind_interp", "wind_ref_ht", "wind_sample_ht",
                          "wind_below_first_cell", "use_waf", "waf_formula", "use_terrain_wind", "prescribed_wind",
                          "fuel_model_id", "fuel_map.fuel_set", "fuel_map.file", "burnout_model", "moisture_dynamic",
                          "moisture_model", "emc_model", "precip_source", "moisture_live_model", "coupling_type",
                          "fire_atm_feedback", "heat_flux_partition", "inject_latent", "heat_tendency_exner",
                          "heat_flux_alfg", "source_mode", "heat_tendency_density", "heat_open_fraction",
                          "ignition_r", "ignition.polygon_file", "ignition.schedule_file", "terrain_file_name",
                          "farsite.use_anderson_lw", "tau_residence_s",
                          "flame_temp_method", "accel.enable", "spotting.enable", "crown.enable",
                          "ignition.threshold_enable", "smoke_enable", "structures.enable", "suppression.enable"}) {
        EXPECT_EQ(m.count(k), 1u) << "no summary line for erf.fire." << k;
    }
    EXPECT_EQ(m.at("ros_model"), "rothermel");
    EXPECT_EQ(m.at("wind_limit"), "rothermel");
    EXPECT_EQ(m.at("propagation_method"), "farsite");
    EXPECT_EQ(m.at("farsite.shape"), "ellipse");
    EXPECT_EQ(m.at("coupling_type"), "lagged");
    EXPECT_EQ(m.at("ignition_r"), "20");
    EXPECT_EQ(m.count("moisture_live"), 0u) << "with dynamic moisture the live model is listed, not the value";
    EXPECT_EQ(m.count("cheney_gould.pasture"), 0u) << "Cheney-Gould's options only when it runs";
    EXPECT_EQ(m.count("levelset.gradient"), 0u) << "level-set options only on the level-set path";
}

// A model's own options follow the model, and an integer-coded selector is
// written with the word that selects it
TEST(FireModelSummary, OptionsFollowTheModelInUse)
{
    ScopedKeys deck;
    deck.add("ros_model", std::string("cheney_gould"));
    deck.add("propagation_method", std::string("levelset"));
    deck.add("cheney_gould.pasture", std::string("grazed"));
    deck.add("directional_shape", std::string("ellipse"));
    deck.add("wind_limit", std::string("fuel_class"));
    FireParams fp;
    const auto m = by_key(fp);
    EXPECT_EQ(m.at("ros_model"), "cheney_gould");
    EXPECT_EQ(m.at("cheney_gould.pasture"), "grazed");
    EXPECT_EQ(m.at("cheney_gould.curing_curve"), "cruz2015");
    EXPECT_EQ(m.at("levelset.gradient"), "weno5z_front");
    EXPECT_EQ(m.at("directional_shape"), "ellipse");
    EXPECT_EQ(m.count("reaction_velocity_formula"), 0u) << "Rothermel's options only when Rothermel runs";
    EXPECT_EQ(m.count("wind_limit"), 0u) << "the midflame wind limit is Rothermel's and BEHAVE's";
    EXPECT_EQ(m.count("farsite.shape"), 0u) << "FARSITE's options only on the FARSITE path";
}

// An option that does not act in the run is not listed, and a value the code
// forced says so
TEST(FireModelSummary, InactiveOptionsAreLeftOutAndForcedValuesMarked)
{
    {
        ScopedKeys deck;
        deck.add("coupling_type", std::string("passive"));
        deck.add("prescribed_wind", std::string("true"));
        deck.add("moisture_dynamic", std::string("false"));
        FireParams fp;
        const auto m = by_key(fp);
        EXPECT_EQ(m.count("heat_flux_partition"), 0u) << "passive coupling sends no heat";
        EXPECT_EQ(m.count("fire_atm_feedback"), 0u);
        EXPECT_EQ(m.count("wind_interp"), 0u) << "a prescribed wind is not interpolated";
        EXPECT_EQ(m.count("moisture_live_model"), 0u) << "the live model runs only with dynamic moisture";
        EXPECT_EQ(m.at("moisture_live"), "0.6");
        EXPECT_EQ(m.count("rothermel_cell_moisture"), 0u) << "per-cell moisture needs dynamic moisture";
    }
    {
        ScopedKeys deck;
        deck.add("propagation_method", std::string("levelset"));
        deck.add("levelset.ellipse", std::string("true"));
        FireParams fp;
        const auto m = by_key(fp);
        EXPECT_EQ(m.at("directional_ros"), "false (off: levelset.ellipse)");
        EXPECT_EQ(m.count("directional_shape"), 0u);
    }
}
