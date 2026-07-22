cmake_minimum_required(VERSION 3.14)
project(erf_ucm_facet_split)

set(EXE_NAME erf_ucm_facet_split)
set(erf_ucm_facet_split_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the executable
add_executable(${EXE_NAME})

# Link to ERF library
target_link_libraries(${EXE_NAME} PRIVATE erf)

# Copy input files to build directory
add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_facet_split_SOURCE_DIR}/inputs"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_facet_split_SOURCE_DIR}/sounding_neutral_abl"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_neutral_abl"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_facet_split_SOURCE_DIR}/building_hetero_mat.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/building_hetero_mat.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_facet_split_SOURCE_DIR}/materials.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/materials.csv"
    COMMENT "Copying UCMFacetSplit input files"
)

# Install target (optional, for testing infrastructure)
install(TARGETS ${EXE_NAME} DESTINATION bin)
