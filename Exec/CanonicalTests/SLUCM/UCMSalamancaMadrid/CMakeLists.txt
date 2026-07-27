cmake_minimum_required(VERSION 3.14)
project(erf_ucm_salamanca_madrid)

set(EXE_NAME erf_ucm_salamanca_madrid)
set(erf_ucm_salamanca_madrid_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE erf)

add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/inputs"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/sounding_madrid"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_madrid"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/inflow_madrid.txt"
            "${CMAKE_CURRENT_BINARY_DIR}/inflow_madrid.txt"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/gen_csv.py"
            "${CMAKE_CURRENT_BINARY_DIR}/gen_csv.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/check_salamanca.py"
            "${CMAKE_CURRENT_BINARY_DIR}/check_salamanca.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/building_layout.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/building_layout.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_salamanca_madrid_SOURCE_DIR}/materials.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/materials.csv"
    COMMENT "Copying UCMSalamancaMadrid input files"
)

install(TARGETS ${EXE_NAME} DESTINATION bin)
