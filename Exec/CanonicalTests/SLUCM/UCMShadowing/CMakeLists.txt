cmake_minimum_required(VERSION 3.20)
project(UCMShadowing)

# All inputs and support files needed for the test
set(CANONICAL_TEST_INPUT_FILES
    inputs
    materials.csv
    building_layout.csv
    sounding_neutral_abl
)

# Copy all input files to the build directory
foreach(test_input_file ${CANONICAL_TEST_INPUT_FILES})
  configure_file(${test_input_file} ${test_input_file} COPYONLY)
endforeach()

# Provide information about this test
message(STATUS "Canonical test: UCMShadowing (Phase 2.4 canyon shadowing with SVF)")
