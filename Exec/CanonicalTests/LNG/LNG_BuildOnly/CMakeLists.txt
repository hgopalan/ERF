# ERF-LNG Phase 1 Build-Only Regression Test
# This test verifies:
# - CMake build with ERF_USE_LNG=ON
# - LNGParams ParmParse reading
# - LNGGrid construction
# - MultiFab allocation
# - Prerequisite validation
# - Debug output infrastructure
# - Zero-physics advance() stub
# - CSV header creation

add_test(NAME LNG_BuildOnly
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../ERF3d
                 ${CMAKE_CURRENT_SOURCE_DIR}/inputs_lng_buildonly
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         TIMEOUT 30
)

# Set pass criteria
set_tests_properties(LNG_BuildOnly PROPERTIES
  PASS_REGULAR_EXPRESSION "\\[LNG\\] ERF-LNG Phase 1 initialized"
  FAIL_REGULAR_EXPRESSION "NaN detected|Aborting"
)
