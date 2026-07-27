cmake_minimum_required(VERSION 3.14)
project(erf_ucm_scaffold)

set(EXE_NAME erf_ucm_scaffold)
set(erf_ucm_scaffold_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the executable
add_executable(${EXE_NAME})

# Link to ERF library
target_link_libraries(${EXE_NAME} PRIVATE erf)

# Copy inputs and sounding files to build directory
add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_scaffold_SOURCE_DIR}/inputs"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_scaffold_SOURCE_DIR}/sounding_neutral_abl"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_neutral_abl"
    COMMENT "Copying UCMScaffold input files"
)

# Install target (optional, for testing infrastructure)
install(TARGETS ${EXE_NAME} DESTINATION bin)
