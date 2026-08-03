cmake_minimum_required(VERSION 3.14)
project(erf_ucm_tree_drag_unit)

set(EXE_NAME erf_ucm_tree_drag_unit)
set(erf_ucm_tree_drag_unit_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${EXE_NAME})
target_link_libraries(${EXE_NAME} PRIVATE erf)

add_custom_command(
    TARGET ${EXE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/inputs_off"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_off"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/inputs_on"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_on"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/inputs_on_dense"
            "${CMAKE_CURRENT_BINARY_DIR}/inputs_on_dense"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/sounding_neutral_abl"
            "${CMAKE_CURRENT_BINARY_DIR}/sounding_neutral_abl"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/building_layout.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/building_layout.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/materials.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/materials.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/tree_layout.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/tree_layout.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/tree_layout_dense.csv"
            "${CMAKE_CURRENT_BINARY_DIR}/tree_layout_dense.csv"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${erf_ucm_tree_drag_unit_SOURCE_DIR}/check_tree_drag.py"
            "${CMAKE_CURRENT_BINARY_DIR}/check_tree_drag.py"
    COMMENT "Copying UCMTreeDragUnit input files"
)

install(TARGETS ${EXE_NAME} DESTINATION bin)
