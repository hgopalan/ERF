// Contract of CheckForDuplicateInputs: a deck may include several files with FILE = <name>, the
// AMReX ParmParse include, which is not a parameter and so never a duplicate. (A parameter given
// twice aborts, which a gtest cannot run.)

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <ERF_InputsName.H>

#include "../ERF_GTestTempDir.H"

TEST(InputsName, SeveralFileIncludesAreNotDuplicates)
{
    const auto dir = erf_gtest_temp_path("erf_gtest_inputs_name");
    std::filesystem::create_directories(dir);
    const std::string deck = (dir / "inputs").string();
    {
        std::ofstream out(deck);
        out << "# a deck of shared settings and a generated block\n"
            << "FILE = flow.inputs\n"
            << "max_step = 10\n"
            << "FILE = network.inputs   # the generated block\n"
            << "erf.v = 1\n";
    }
    EXPECT_NO_FATAL_FAILURE(CheckForDuplicateInputs(deck));
    std::filesystem::remove_all(dir);
}
