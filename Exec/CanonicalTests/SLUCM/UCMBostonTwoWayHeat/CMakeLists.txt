cmake_minimum_required(VERSION 3.14)
project(erf_ucm_boston)

set(EXE_NAME erf_ucm_boston)
set(erf_ucm_boston_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE erf)

add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/inputs_singlelevel"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_singlelevel"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/inputs_multilevel_oneway"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_multilevel_oneway"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/inputs_multilevel_twoway"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_multilevel_twoway"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/sounding_boston"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_boston"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/inflow_boston.txt"
            "${CMAKE_CURRENT_BINARY_DIR}/inflow_boston.txt"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/gen_boston.py"
            "${CMAKE_CURRENT_BINARY_DIR}/gen_boston.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/check_boston_singlelevel.py"
            "${CMAKE_CURRENT_BINARY_DIR}/check_boston_singlelevel.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/building_layout.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/building_layout.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_boston_SOURCE_DIR}/materials.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/materials.csv"
    COMMENT "Copying UCMBoston input files"
)

install(TARGETS ${EXE_NAME} DESTINATION bin)
