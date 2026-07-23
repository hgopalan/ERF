cmake_minimum_required(VERSION 3.14)
project(erf_ucm_kanda_wind_tunnel)

set(EXE_NAME erf_ucm_kanda_wind_tunnel)
set(erf_ucm_kanda_wind_tunnel_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE erf)

add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/inputs"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/sounding_kanda"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_kanda"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/inflow_kanda.txt"
            "${CMAKE_CURRENT_BINARY_DIR}/inflow_kanda.txt"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/gen_csv.py"
            "${CMAKE_CURRENT_BINARY_DIR}/gen_csv.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/check_kanda.py"
            "${CMAKE_CURRENT_BINARY_DIR}/check_kanda.py"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/building_layout.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/building_layout.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_kanda_wind_tunnel_SOURCE_DIR}/materials.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/materials.csv"
    COMMENT "Copying UCMKandaWindTunnel input files"
)

install(TARGETS ${EXE_NAME} DESTINATION bin)
