if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR
    "erf_ucm_unit_tests expects the vendored googletest target from "
    "Tests/Unit/CMakeLists.txt. Configure unit tests through the top-level "
    "ERF test build so Submodules/googletest is added first.")
endif()

if(NOT ERF_ENABLE_UCM)
  message(STATUS "Skipping UrbanCanopy unit tests (ERF_ENABLE_UCM is OFF)")
  return()
endif()

include(${PROJECT_SOURCE_DIR}/CMake/BuildERFExe.cmake)
include(${PROJECT_SOURCE_DIR}/CMake/SetERFCompileFlags.cmake)

add_executable(erf_ucm_unit_tests
  ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
  ${PROJECT_SOURCE_DIR}/Tests/Unit/ERF_GTestRuntimeStubs.cpp
  ERF_GTestUCMUnit.cpp
)

build_erf_exe(erf_ucm_unit_tests NO_ERF_MAIN)

target_include_directories(erf_ucm_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/Source
  ${CMAKE_SOURCE_DIR}/Source/DataStructs
  ${CMAKE_SOURCE_DIR}/Source/UrbanCanopy
  ${CMAKE_SOURCE_DIR}/Source/Utils
)

target_compile_definitions(erf_ucm_unit_tests PRIVATE
  ERF_HAS_UCM
)

target_link_libraries(erf_ucm_unit_tests PRIVATE GTest::gtest)

erf_register_gtest_suite(erf_ucm_unit_tests
  LABELS unit ucm
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DISCOVERY_TIMEOUT 60
  EXECUTION_TIMEOUT 900
)
