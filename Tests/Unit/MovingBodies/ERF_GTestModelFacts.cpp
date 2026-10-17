// What start-up takes from each turbine model (erf_openfast::gather_model_facts): each model is read on
// one rank, and every rank must end up with exactly what reading the model itself gives, the induction
// and tower-shadow texts and the rotor radius and time step, whichever rank read it.

#include <string>
#include <vector>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_OpenFASTAudit.H"
#include "ERF_OpenFASTDriver.H"

TEST(ModelFacts, EveryRankHoldsWhatReadingEachModelGives)
{
    const std::string cases = ERF_GTEST_MOVING_BODIES_CASES;
    // read on ranks 0, 1, 2, 0, ...: on two ranks rank 1 alone reads the BEM model, the one whose checks give texts
    const std::vector<std::string> paths{cases + "/OpenFAST_ALM_Tower/stub_turbine.fst", cases + "/IEA15_land_model/IEA15-land-bem.fst",
                                         cases + "/IEA15_land_model/IEA15-land.fst", cases + "/OpenFAST_ADM_TwoTurbines/stub_turbine_dt002.fst"};
    const std::vector<erf_openfast::ModelFacts> facts = erf_openfast::gather_model_facts(paths);
    ASSERT_EQ(facts.size(), paths.size());
    for (std::size_t p = 0; p < paths.size(); ++p) {
        EXPECT_EQ(facts[p].induction_off, erf_openfast::check_induction(paths[p], true)) << paths[p];
        EXPECT_EQ(facts[p].induction_on, erf_openfast::check_induction(paths[p], false)) << paths[p];
        EXPECT_EQ(facts[p].tower_shadow, erf_openfast::check_tower_shadow_off(paths[p])) << paths[p];
        EXPECT_EQ(facts[p].rotor_radius, erf_openfast::model_rotor_radius(paths[p])) << paths[p];
        EXPECT_EQ(facts[p].time_step, erf_openfast::model_time_step(paths[p])) << paths[p];
    }
    // the texts are there to be shared: the BEM model's checks give one each
    EXPECT_FALSE(facts[1].induction_off.empty());
    EXPECT_FALSE(facts[1].tower_shadow.empty());
    EXPECT_FALSE(facts[2].induction_on.empty());
    EXPECT_EQ(facts[3].time_step, amrex::Real(0.02));
    // no models: nothing to read, on every rank alike
    EXPECT_TRUE(erf_openfast::gather_model_facts({}).empty());
}
