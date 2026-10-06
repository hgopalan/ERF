// Every conductor test starts with no erf.conductors.* inputs.
//
// The tests set their inputs in AMReX's global ParmParse table, and a key one test sets stays for
// the tests after it: a test that reads a default (the air density, a line's output root) then
// sees another test's value. This listener removes every erf.conductors.* entry before each test,
// so the tests pass in any order and under --gtest_repeat (the erf_unit_tests_shuffle_repeat CTest).

#include <string>

#include <AMReX_ParmParse.H>

#include <gtest/gtest.h>

namespace {

class ClearConductorInputs : public ::testing::EmptyTestEventListener
{
    void OnTestStart (const ::testing::TestInfo& /*info*/) override
    {
        amrex::ParmParse pp;
        for (const std::string& key : amrex::ParmParse::getEntries("erf.conductors")) {
            pp.remove(key);
        }
    }
};

// gtest takes ownership of the listener
const bool registered = [] {
    ::testing::UnitTest::GetInstance()->listeners().Append(new ClearConductorInputs);
    return true;
}();

} // namespace
