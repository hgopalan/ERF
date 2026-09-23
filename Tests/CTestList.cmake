# Have CMake discover the number of cores on the node
include(ProcessorCount)
ProcessorCount(PROCESSES)

#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================
function(resolve_test_exe TEST_DIR TEST_EXE OUT_VAR)
    if(WIN32)
        # Multi-config generators place binaries in a config subdir.
        set(${OUT_VAR} "${CMAKE_BINARY_DIR}/Exec/${TEST_DIR}/*/${TEST_EXE}.exe" PARENT_SCOPE)
    else()
        set(_exe_in_subdir "${CMAKE_BINARY_DIR}/Exec/${TEST_DIR}/${TEST_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
        set(_exe_in_root  "${CMAKE_BINARY_DIR}/Exec/${TEST_EXE}${CMAKE_EXECUTABLE_SUFFIX}")
        if(EXISTS "${_exe_in_subdir}")
            set(${OUT_VAR} "${_exe_in_subdir}" PARENT_SCOPE)
        elseif(EXISTS "${_exe_in_root}")
            set(${OUT_VAR} "${_exe_in_root}" PARENT_SCOPE)
        else()
            # Keep the historical path so the error message is still informative.
            set(${OUT_VAR} "${_exe_in_subdir}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

macro(setup_test)
    if(DEFINED TEST_FILES_DIR AND NOT "${TEST_FILES_DIR}" STREQUAL "")
        set(_test_source_dir_name "${TEST_FILES_DIR}")
    else()
        set(_test_source_dir_name "${TEST_NAME}")
    endif()
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${_test_source_dir_name})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    set(PLOT_GOLD ${ERF_TEST_GOLD_FILES_DIRECTORY}/${TEST_NAME})

    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        set(NP ${ERF_TEST_NRANKS})
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
        set(MPI_FCOMP_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
        unset(MPI_FCOMP_COMMANDS)
    endif()
endmacro(setup_test)

# Production contract test for native 3D plotfile names and unavailable-name warnings.
function(add_test_plotfile_header TEST_NAME TEST_DIR TEST_EXE PLTFILE)
    setup_test()

    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)
    set(header_checker "${PROJECT_SOURCE_DIR}/Tests/CheckPlotfileHeader.cmake")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(header_file "${CURRENT_TEST_BINARY_DIR}/${PLTFILE}/Header")
    # Regression motivation: this test is launched through `sh -c`, while on
    # Windows CMAKE_COMMAND normally resides below "C:/Program Files". Keep
    # checker executable and path-bearing arguments shell-quoted.
    set(check_command
        "\"${CMAKE_COMMAND}\""
        "\"-DHEADER=${header_file}\""
        "\"-DEXPECTED_NAMES_FILE=${CURRENT_TEST_BINARY_DIR}/expected_names.txt\""
        "\"-DLOG=${test_log}\""
        "\"-DEXPECTED_UNAVAILABLE_FILE=${CURRENT_TEST_BINARY_DIR}/expected_unavailable.txt\""
        "-P"
        "\"${header_checker}\"")
    list(JOIN check_command " " check_command_string)
    set(test_input "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    set(test_command sh -c
        "${MPI_COMMANDS} ${TEST_EXE} \"${test_input}\" > \"${test_log}\" 2>&1 && ${check_command_string}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 300
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;plotfile"
        ATTACHED_FILES_ON_FAIL "${test_log};${header_file}"
    )
endfunction(add_test_plotfile_header)

# Standard regression test
function(add_test_r TEST_NAME TEST_DIR TEST_EXE PLTFILE)
    set(options )
    set(oneValueArgs "INPUT_SOUNDING" "RUNTIME_OPTIONS" "FCOMPARE_RTOL" "FCOMPARE_ATOL")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_R "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    setup_test()

    set(RUNTIME_OPTIONS "${ADD_TEST_R_RUNTIME_OPTIONS}")
    if(NOT "${ADD_TEST_R_INPUT_SOUNDING}" STREQUAL "")
      string(APPEND RUNTIME_OPTIONS "erf.input_sounding_file=${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_R_INPUT_SOUNDING}")
    endif()

    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)

    set(_fcompare_rtol "${ERF_TEST_FCOMPARE_RTOL}")
    set(_fcompare_atol "${ERF_TEST_FCOMPARE_ATOL}")
    if(NOT "${ADD_TEST_R_FCOMPARE_RTOL}" STREQUAL "")
        set(_fcompare_rtol "${ADD_TEST_R_FCOMPARE_RTOL}")
    endif()
    if(NOT "${ADD_TEST_R_FCOMPARE_ATOL}" STREQUAL "")
        set(_fcompare_atol "${ADD_TEST_R_FCOMPARE_ATOL}")
    endif()

    set(FCOMPARE_TOLERANCE "-r ${_fcompare_rtol} --abs_tol ${_fcompare_atol}")
    set(FCOMPARE_FLAGS "--abort_if_not_all_found -a ${FCOMPARE_TOLERANCE}")
    set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i ${RUNTIME_OPTIONS} > ${TEST_NAME}.log && ${MPI_FCOMP_COMMANDS} ${FCOMPARE_EXE} ${FCOMPARE_FLAGS} ${PLOT_GOLD} ${CURRENT_TEST_BINARY_DIR}/${PLTFILE}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 5400
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log"
    )
endfunction(add_test_r)

# Rotated six-wall anelastic manufactured regression. Each case keeps a
# linear theta profile stationary and checks the full field inventory.
function(add_test_anelastic_wall_diffusion TEST_NAME TEST_AXIS)
    set(TEST_FILES_DIR "${TEST_NAME}")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(test_input "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    set(test_simulation_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.simulation.log")
    set(test_checker_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.checker.log")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DSIMULATION_LOG=${test_simulation_log}
        -DCHECKER_LOG=${test_checker_log}
        -DCHECKER=${ANELASTIC_WALL_DIFFUSION_CHECKER}
        -DPLOTFILE=${CURRENT_TEST_BINARY_DIR}/plt00002
        -DAXIS=${TEST_AXIS}
        -DTHETA_LO=300.0
        -DTHETA_HI=301.0
        -P ${PROJECT_SOURCE_DIR}/Tests/RunAnelasticWallDiffusion.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 900
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;anelastic;wall-diffusion"
        ATTACHED_FILES_ON_FAIL "${test_simulation_log};${test_checker_log}")
endfunction(add_test_anelastic_wall_diffusion)

# Checker-driven Cloud Chamber tests.  The short run checks the exact initial
# conserved-state correction and a bounded early buoyant response; it
# intentionally avoids a fragile turbulent gold file.
function(add_test_cloud_chamber TEST_NAME MODE)
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(test_input "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    set(test_simulation_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.simulation.log")
    set(test_checker_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.checker.log")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DSIMULATION_LOG=${test_simulation_log}
        -DCHECKER_LOG=${test_checker_log}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -DMODE=${MODE}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamber.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 900
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber"
        ATTACHED_FILES_ON_FAIL "${test_simulation_log};${test_checker_log}")
endfunction(add_test_cloud_chamber)

# Gold-free TwoStream radiation regression: run a short SW + LW column case
# and verify the vertical structure of qsrc_sw / qsrc_lw in the plotfile
# (surface at k = 0, cooling to space from the top layer).
function(add_test_two_stream_radiation TEST_NAME PLTFILE)
    set(oneValueArgs "RUNTIME_OPTIONS")
    cmake_parse_arguments(ADD_TEST_TSR "" "${oneValueArgs}" "" ${ARGN})
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(test_input "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    set(test_simulation_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.simulation.log")
    set(test_checker_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.checker.log")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DSIMULATION_LOG=${test_simulation_log}
        -DCHECKER_LOG=${test_checker_log}
        -DCHECKER=${TWO_STREAM_RADIATION_CHECKER}
        -DPLOTFILE=${CURRENT_TEST_BINARY_DIR}/${PLTFILE}
        "-DRUNTIME_OPTIONS=${ADD_TEST_TSR_RUNTIME_OPTIONS}"
        -P ${PROJECT_SOURCE_DIR}/Tests/RunTwoStreamRadiation.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 600
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;radiation"
        ATTACHED_FILES_ON_FAIL "${test_simulation_log};${test_checker_log}")
endfunction(add_test_two_stream_radiation)

function(add_test_cloud_chamber_parity TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_SatAdj")
    if (ARGC GREATER 1)
        set(TEST_FILES_DIR "${ARGV1}")
    endif()
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_FILES_DIR}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberParity.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1200
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/budget_off/simulation.log;${CURRENT_TEST_BINARY_DIR}/budget_on/simulation.log;${CURRENT_TEST_BINARY_DIR}/parity.log")
endfunction(add_test_cloud_chamber_parity)

# Run the deck in test_files/<TEST_FILES_DIR> on a single box (one rank) and on a split
# BoxArray (ERF_TEST_NRANKS ranks) and compare the two plotfiles PLTFILE with fcompare.
# COMMON_OPTIONS go to both runs, REFERENCE_OPTIONS must make the grid a single box and
# SPLIT_OPTIONS give the split (the deck's own grid when empty).
function(add_test_box_parity TEST_NAME TEST_FILES_DIR PLTFILE)
    set(oneValueArgs "COMMON_OPTIONS" "REFERENCE_OPTIONS" "SPLIT_OPTIONS" "FCOMPARE_RTOL" "FCOMPARE_ATOL")
    cmake_parse_arguments(ADD_TEST_BP "" "${oneValueArgs}" "" ${ARGN})
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(_fcompare_rtol "${ERF_TEST_FCOMPARE_RTOL}")
    set(_fcompare_atol "${ERF_TEST_FCOMPARE_ATOL}")
    if(NOT "${ADD_TEST_BP_FCOMPARE_RTOL}" STREQUAL "")
        set(_fcompare_rtol "${ADD_TEST_BP_FCOMPARE_RTOL}")
    endif()
    if(NOT "${ADD_TEST_BP_FCOMPARE_ATOL}" STREQUAL "")
        set(_fcompare_atol "${ADD_TEST_BP_FCOMPARE_ATOL}")
    endif()

    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_FILES_DIR}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DFCOMPARE=${FCOMPARE_EXE}
        -DPLTFILE=${PLTFILE}
        -DRTOL=${_fcompare_rtol}
        -DATOL=${_fcompare_atol}
        "-DCOMMON_OPTIONS=${ADD_TEST_BP_COMMON_OPTIONS}"
        "-DREFERENCE_OPTIONS=${ADD_TEST_BP_REFERENCE_OPTIONS}"
        "-DSPLIT_OPTIONS=${ADD_TEST_BP_SPLIT_OPTIONS}"
        -P ${PROJECT_SOURCE_DIR}/Tests/RunBoxParity.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1200
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;box-parity"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/one_box/simulation.log;${CURRENT_TEST_BINARY_DIR}/split/simulation.log;${CURRENT_TEST_BINARY_DIR}/parity.log")
endfunction(add_test_box_parity)

# Tiling parity: run one deck with MFIter tiling on and off and require identical
# 3D and 2D plotfiles (no gold file). Catches kernels that loop over the valid box
# while indexing per-tile work arrays. VARYING_3D / VARYING_2D list fields (space
# separated) that must take more than one value in the untiled run, so the
# agreement is not between two copies of a constant.
function(add_test_tiling_parity TEST_NAME TEST_FILES_DIR PLTFILE PLT2DFILE)
    set(options )
    set(oneValueArgs "RUNTIME_OPTIONS" "VARYING_3D" "VARYING_2D")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_TP "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_FILES_DIR}.i
        -DRUNTIME_OPTIONS=${ADD_TEST_TP_RUNTIME_OPTIONS}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DFCOMPARE=${FCOMPARE_EXE}
        -DFEXTREMA=${FEXTREMA_EXE}
        -DRTOL=${ERF_TEST_FCOMPARE_RTOL}
        -DATOL=${ERF_TEST_FCOMPARE_ATOL}
        -DPLTFILE=${PLTFILE}
        -DPLT2DFILE=${PLT2DFILE}
        -DVARYING_3D=${ADD_TEST_TP_VARYING_3D}
        -DVARYING_2D=${ADD_TEST_TP_VARYING_2D}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunTilingParity.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1200
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/tiled.log;${CURRENT_TEST_BINARY_DIR}/untiled.log;${CURRENT_TEST_BINARY_DIR}/fcompare_plt.log;${CURRENT_TEST_BINARY_DIR}/fcompare_plt2d.log;${CURRENT_TEST_BINARY_DIR}/fextrema_plt.log;${CURRENT_TEST_BINARY_DIR}/fextrema_plt2d.log")
endfunction(add_test_tiling_parity)

function(add_test_cloud_chamber_budget TEST_NAME MODE SOURCE_NAME)
    set(_cloud_chamber_input_name "${SOURCE_NAME}")
    set(TEST_FILES_DIR "${SOURCE_NAME}")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${_cloud_chamber_input_name}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -DMODE=${MODE}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberBudget.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 900
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/simulation.log;${CURRENT_TEST_BINARY_DIR}/checker.log;${CURRENT_TEST_BINARY_DIR}/cloud_chamber_budget.dat")
endfunction(add_test_cloud_chamber_budget)

# Parity test: the deck with whole-height fine grids and with the fine grids split in z
# must give identical plotfiles
function(add_test_terrain_zsplit_parity TEST_NAME PLTFILE)
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DFCOMPARE=${FCOMPARE_EXE}
        -DPLTFILE=${PLTFILE}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunTerrainZSplitParity.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 600
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/full_columns/simulation.log;${CURRENT_TEST_BINARY_DIR}/split_in_z/simulation.log;${CURRENT_TEST_BINARY_DIR}/parity.log")
endfunction(add_test_terrain_zsplit_parity)

# At-rest test: a hydrostatic atmosphere over terrain must stay at rest with lateral
# outflow boundaries, where the mesh is extrapolated past the domain and the base state in
# the ghost cells has to be built at the height the mesh puts them at rather than copied.
function(add_test_at_rest_terrain_outflow TEST_NAME PLTFILE TOLERANCE GRADP_TOLERANCE)
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DFEXTREMA=${FEXTREMA_EXE}
        -DPLTFILE=${PLTFILE}
        -DTOLERANCE=${TOLERANCE}
        -DGRADP_TOLERANCE=${GRADP_TOLERANCE}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunAtRestTerrainOutflow.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 600
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/symmetry/simulation.log;${CURRENT_TEST_BINARY_DIR}/outflow/simulation.log;${CURRENT_TEST_BINARY_DIR}/at_rest.log")
endfunction(add_test_at_rest_terrain_outflow)

# Positive startup regression for the retained legacy theta/qv parser path.
# This intentionally has no physical-temperature or physical-wall keys.
function(add_test_cloud_chamber_legacy_config TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_Legacy_Config")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(test_input "${CURRENT_TEST_BINARY_DIR}/CloudChamber_Legacy_Config.i")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(output_directory "${CURRENT_TEST_BINARY_DIR}/legacy_plt00000")
    set(output_artifact "${output_directory}/Header")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DLOG=${test_log}
        -DOUTPUT_DIRECTORY=${output_directory}
        -DOUTPUT_ARTIFACT=${output_artifact}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberConfigSuccess.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 180
        PROCESSORS 1
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;configuration"
        ATTACHED_FILES_ON_FAIL "${test_log};${output_artifact}")
endfunction(add_test_cloud_chamber_legacy_config)
# Negative startup tests for the Native SHOC transport modes removed from the
# production input contract.  The shared fixture supplies a complete Native
# SHOC run, while the runtime option exercises the real ParmParse reader path.
function(add_test_shoc_removed_transport TEST_NAME RUNTIME_OPTION EXPECTED_MESSAGE
        EXPECTED_GUIDANCE_1 EXPECTED_GUIDANCE_2)
    set(TEST_FILES_DIR "SHOC_Stable_Clear")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(test_input "${CURRENT_TEST_BINARY_DIR}/SHOC_Stable_Clear.i")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DRUNTIME_OPTIONS=${RUNTIME_OPTION}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DLOG=${test_log}
        -DEXPECTED_MESSAGE=${EXPECTED_MESSAGE}
        -DEXPECTED_GUIDANCE_1=${EXPECTED_GUIDANCE_1}
        -DEXPECTED_GUIDANCE_2=${EXPECTED_GUIDANCE_2}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunShocRemovedTransportConfig.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 180
        PROCESSORS 1
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;shoc;configuration"
        ATTACHED_FILES_ON_FAIL "${test_log}")
endfunction(add_test_shoc_removed_transport)

add_test_shoc_removed_transport(SHOC_Removed_Scalar_Host_Diffusion
    "erf.shoc.transport_mode=host_diffusion"
    "erf.shoc.transport_mode = host_diffusion has been removed for native SHOC"
    "Use erf.shoc.transport_mode = state_update"
    "")
add_test_shoc_removed_transport(SHOC_Removed_Momentum_Host_Diffusion
    "erf.shoc.momentum_transport=host_diffusion"
    "erf.shoc.momentum_transport = host_diffusion has been removed for native SHOC"
    "state_update"
    "none")

# Production wiring regression: two dry runs differ only in xlo roughness;
# the checker requires finite output and a resolvable z0_m response.

function(add_test_cloud_chamber_neutral_momentum TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_Dry_NeutralMomentum")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/CloudChamber_Dry_NeutralMomentum.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberNeutralMomentum.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;neutral-roughness"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/z0_baseline/simulation.log;${CURRENT_TEST_BINARY_DIR}/z0_changed/simulation.log;${CURRENT_TEST_BINARY_DIR}/neutral_momentum_checker.log")
endfunction(add_test_cloud_chamber_neutral_momentum)

# Production wiring regression for fixed bulk aerodynamic momentum.  The
# harness changes only C_D and requires a measurable velocity response.
function(add_test_cloud_chamber_fixed_momentum TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_Dry_FixedMomentum")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/CloudChamber_Dry_FixedMomentum.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberFixedMomentum.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;bulk-momentum"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/cd_baseline/simulation.log;${CURRENT_TEST_BINARY_DIR}/cd_changed/simulation.log;${CURRENT_TEST_BINARY_DIR}/fixed_momentum_checker.log")
endfunction(add_test_cloud_chamber_fixed_momentum)

# Production wiring regression for all-channel horizontal MOST.  The harness
# checks wet-wall budgets in both runs and changes only horizontal momentum
# transfer to require an observable production-path momentum response.
function(add_test_cloud_chamber_most TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_SatAdj_MOSTMixedWalls")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/CloudChamber_SatAdj_MOSTMixedWalls.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberMOST.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;most"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/most_baseline/simulation.log;${CURRENT_TEST_BINARY_DIR}/most_changed/simulation.log;${CURRENT_TEST_BINARY_DIR}/most_momentum_checker.log")
endfunction(add_test_cloud_chamber_most)

function(add_test_cloud_chamber_fixed_dt_guard TEST_NAME)
    set(test_log "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.log")
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        "-DTEST_EXE=$<TARGET_FILE:erf_cloud_chamber_wall_dt_guard_check>"
        -DLOG=${test_log}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberWallDtGuardFailure.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 120
        PROCESSORS 1
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;configuration"
        ATTACHED_FILES_ON_FAIL "${test_log}")
endfunction(add_test_cloud_chamber_fixed_dt_guard)

function(add_test_cloud_chamber_openmp TEST_NAME)
    set(TEST_FILES_DIR "CloudChamber_SatAdj")
    if (ARGC GREATER 1)
        set(TEST_FILES_DIR "${ARGV1}")
    endif()
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${TEST_FILES_DIR}.i
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DCHECKER=${CLOUD_CHAMBER_CHECKER}
        -DCMAKE_COMMAND=${CMAKE_COMMAND}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunCloudChamberOpenMP.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1200
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;cloud-chamber;openmp"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/omp_1_thread/simulation.log;${CURRENT_TEST_BINARY_DIR}/omp_2_threads/simulation.log;${CURRENT_TEST_BINARY_DIR}/openmp_parity.log")
endfunction(add_test_cloud_chamber_openmp)

# Native SHOC regression test.  This intentionally remains separate from
# add_test_r so existing registrations retain their exact command and
# fixture behaviour.  TEST_FILES_DIR and INPUT_FILE allow the small SHOC
# matrix to reuse a physical fixture while keeping unique binary and gold
# directories.  The checker runs before the selected gold comparison path.
function(add_test_shoc_r TEST_NAME TEST_DIR TEST_EXE PLTFILE)
    set(options SKIP_GOLD)
    set(oneValueArgs "TEST_FILES_DIR" "INPUT_FILE" "CHECK_MODE" "RUNTIME_OPTIONS" "TIMEOUT"
        "GOLD_COMPARISON" "GOLD_MODE")
    set(multiValueArgs "LABELS")
    cmake_parse_arguments(ADD_TEST_SHOC_R "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(TEST_FILES_DIR "${ADD_TEST_SHOC_R_TEST_FILES_DIR}")
    setup_test()

    if("${ADD_TEST_SHOC_R_INPUT_FILE}" STREQUAL "")
        set(test_input "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i")
    else()
        set(test_input "${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_SHOC_R_INPUT_FILE}")
    endif()

    set(RUNTIME_OPTIONS "${ADD_TEST_SHOC_R_RUNTIME_OPTIONS}")
    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)

    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    if(ADD_TEST_SHOC_R_GOLD_COMPARISON)
        set(_shoc_gold_comparison "${ADD_TEST_SHOC_R_GOLD_COMPARISON}")
    else()
        set(_shoc_gold_comparison "fcompare")
    endif()
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DINPUT=${test_input}
        -DRUNTIME_OPTIONS=${RUNTIME_OPTIONS}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DLOG=${test_log}
        -DCHECKER=${SHOC_PLOTFILE_CHECKER}
        -DCHECK_MODE=${ADD_TEST_SHOC_R_CHECK_MODE}
        -DINITIAL=${CURRENT_TEST_BINARY_DIR}/plt00000
        -DMIDPOINT=${CURRENT_TEST_BINARY_DIR}/plt00010
        -DFINAL=${CURRENT_TEST_BINARY_DIR}/${PLTFILE}
        -DFCOMPARE=${FCOMPARE_EXE}
        -DGOLD_DIFFERENTIAL=${SHOC_GOLD_DIFFERENTIAL}
        -DGOLD_COMPARISON=${_shoc_gold_comparison}
        -DGOLD_MODE=${ADD_TEST_SHOC_R_GOLD_MODE}
        -DRTOL=${ERF_TEST_FCOMPARE_RTOL}
        -DATOL=${ERF_TEST_FCOMPARE_ATOL}
        -DGOLD=${PLOT_GOLD}
        -DSKIP_GOLD=${ADD_TEST_SHOC_R_SKIP_GOLD}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunShocRegression.cmake)
    if(ADD_TEST_SHOC_R_TIMEOUT)
        set(_shoc_timeout "${ADD_TEST_SHOC_R_TIMEOUT}")
    else()
        set(_shoc_timeout 900)
    endif()
    if(ADD_TEST_SHOC_R_LABELS)
        set(_shoc_labels "${ADD_TEST_SHOC_R_LABELS}")
    else()
        set(_shoc_labels "regression;shoc")
    endif()
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT ${_shoc_timeout}
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "${_shoc_labels}"
        ATTACHED_FILES_ON_FAIL "${test_log}")
endfunction(add_test_shoc_r)

function(add_test_shoc_mutation TEST_NAME MUTATION_OPTION TARGET_FIELD
        MIN_FINAL_DIFFERENCE MIN_BASELINE_EVOLUTION MAX_MUTANT_TO_BASELINE_EVOLUTION_RATIO)
    set(TEST_FILES_DIR "SHOC_Stable_Clear")
    setup_test()
    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(_baseline_dir "${CURRENT_TEST_BINARY_DIR}/baseline")
    set(_mutant_dir "${CURRENT_TEST_BINARY_DIR}/mutant")
    file(MAKE_DIRECTORY "${_baseline_dir}" "${_mutant_dir}")
    file(COPY "${CURRENT_TEST_SOURCE_DIR}/." DESTINATION "${_baseline_dir}")
    file(COPY "${CURRENT_TEST_SOURCE_DIR}/." DESTINATION "${_mutant_dir}")

    set(_baseline_log "${_baseline_dir}/${TEST_NAME}_baseline.log")
    set(_mutant_log "${_mutant_dir}/${TEST_NAME}_mutant.log")
    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${NP}
        -DTEST_EXE=${TEST_EXE}
        -DBASELINE_INPUT=${_baseline_dir}/SHOC_Stable_Clear.i
        -DMUTANT_INPUT=${_mutant_dir}/SHOC_Stable_Clear.i
        -DBASELINE_OPTIONS=
        -DMUTANT_OPTIONS=${MUTATION_OPTION}
        -DBASELINE_WORKING_DIRECTORY=${_baseline_dir}
        -DMUTANT_WORKING_DIRECTORY=${_mutant_dir}
        -DBASELINE_LOG=${_baseline_log}
        -DMUTANT_LOG=${_mutant_log}
        -DCHECKER=${SHOC_MUTATION_DIFFERENTIAL}
        -DTARGET_FIELD=${TARGET_FIELD}
        -DMIN_FINAL_DIFFERENCE=${MIN_FINAL_DIFFERENCE}
        -DMIN_BASELINE_EVOLUTION=${MIN_BASELINE_EVOLUTION}
        -DMAX_MUTANT_TO_BASELINE_EVOLUTION_RATIO=${MAX_MUTANT_TO_BASELINE_EVOLUTION_RATIO}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunShocMutationRegression.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 900
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;shoc;mutation"
        ATTACHED_FILES_ON_FAIL "${_baseline_log};${_mutant_log}")
endfunction(add_test_shoc_mutation)

# Negative-control regression: the true fixed_dt > dt_wall guard must fire
# and expose stable diagnostic fields for automated CI forensics.
add_test_cloud_chamber_fixed_dt_guard(CloudChamber_Bulk_FixedDtGuard)

if(ERF_ENABLE_MPI)
add_test_anelastic_wall_diffusion(AnelasticWallDiffusion_X 0)
add_test_anelastic_wall_diffusion(AnelasticWallDiffusion_Y 1)
add_test_anelastic_wall_diffusion(AnelasticWallDiffusion_Z 2)
# Same stationary state as the _X case, but with erf.anelastic_type = MidPoint so the
# vertical implicit diffusion stays on (the _X/_Y/_Z cases opt out with vert_implicit).
add_test_anelastic_wall_diffusion(AnelasticWallDiffusion_X_MidPoint 0)
add_test_cloud_chamber(CloudChamber_Dry dry)
add_test_cloud_chamber_legacy_config(CloudChamber_Legacy_Config)
add_test_cloud_chamber_neutral_momentum(CloudChamber_Dry_NeutralMomentumActivation)
add_test_cloud_chamber_fixed_momentum(CloudChamber_Dry_FixedMomentumActivation)
add_test_cloud_chamber(CloudChamber_SatAdj cloudy)
add_test_cloud_chamber_parity(CloudChamber_SatAdj_Parity)
add_test_cloud_chamber_budget(CloudChamber_SatAdj_AllDry all_dry CloudChamber_SatAdj_AllDry)
add_test_cloud_chamber_budget(CloudChamber_SatAdj_WetBudget wet_budget CloudChamber_SatAdj_WetBudget)
add_test_cloud_chamber_budget(CloudChamber_SatAdj_BulkMixedWet bulk_wet CloudChamber_SatAdj_BulkMixedWet)
add_test_cloud_chamber_budget(CloudChamber_SatAdj_NeutralWetBudget neutral_wet CloudChamber_SatAdj_NeutralWet)
add_test_cloud_chamber_budget(CloudChamber_SatAdj_MOSTWetBudget most_wet CloudChamber_SatAdj_MOSTWetBudget)
add_test_cloud_chamber_most(CloudChamber_SatAdj_MOSTMixedWalls)
if(ERF_ENABLE_OPENMP)
add_test_cloud_chamber_openmp(CloudChamber_SatAdj_OpenMP)
endif()
add_test(SHOC_Unstable_Cloud_SatAdj_vs_NoCond
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    ${SHOC_MICROPHYSICS_DIFFERENTIAL}
    ${CMAKE_CURRENT_BINARY_DIR}/test_files/SHOC_Unstable_Cloud_SatAdj_Property/plt00020
    ${CMAKE_CURRENT_BINARY_DIR}/test_files/SHOC_Unstable_Cloud_NoCond_Property/plt00020)
set_tests_properties(SHOC_Unstable_Cloud_SatAdj_vs_NoCond
    PROPERTIES
    DEPENDS "SHOC_Unstable_Cloud_SatAdj_Property;SHOC_Unstable_Cloud_NoCond_Property"
    TIMEOUT 120
    PROCESSORS 1
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
    LABELS "regression;shoc;microphysics")
# execute_process needs mpiexec, and does not expand the executable globs used on Windows
if(NOT WIN32)
add_test_terrain_zsplit_parity(Terrain2Lev_BTF_ZSplit "plt00000")
add_test_at_rest_terrain_outflow(AtRestTerrainOutflow "plt00400" 1.0e-8 0.1)
endif()
endif()

# Debug regression test with lower tolerance
function(add_test_d TEST_NAME TEST_DIR TEST_EXE PLTFILE)
    set(options )
    set(oneValueArgs "INPUT_SOUNDING" "RUNTIME_OPTIONS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_D "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    
    setup_test()

    set(RUNTIME_OPTIONS "${ADD_TEST_D_RUNTIME_OPTIONS}")
    if(NOT "${ADD_TEST_D_INPUT_SOUNDING}" STREQUAL "")
      string(APPEND RUNTIME_OPTIONS "erf.input_sounding_file=${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_D_INPUT_SOUNDING}")
    endif()

    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)
    set(FCOMPARE_TOLERANCE "-r 3.0e-9 --abs_tol 3.0e-9")
    set(FCOMPARE_FLAGS "--abort_if_not_all_found -a ${FCOMPARE_TOLERANCE}")
    set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i ${RUNTIME_OPTIONS} > ${TEST_NAME}.log && ${MPI_FCOMP_COMMANDS} ${FCOMPARE_EXE} ${FCOMPARE_FLAGS} ${PLOT_GOLD} ${CURRENT_TEST_BINARY_DIR}/${PLTFILE}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 5400
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log"
    )
endfunction(add_test_d)

# Fire smoke test: one deck of a fire suite under Exec/RegTests, run for NSTEPS
# steps on the regression rank count, passing when the run exits cleanly and
# the fire plotfile of the last step is written. The suites carry no gold
# files: their physics checks live in the run_*.sh scripts beside the decks,
# which run every variant to its stop time and are too long for CI. The whole
# suite directory is copied so a deck finds its inputs_base, sounding, fuel
# map, building list and schedules.
function(add_test_fire TEST_NAME SUITE_DIR INPUT_FILE NSTEPS)
    set(options )
    set(oneValueArgs "RUNTIME_OPTIONS" "NRANKS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_FIRE "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(CURRENT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/Exec/RegTests/${SUITE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    # NRANKS overrides the regression rank count: a 20-cell coarse deck has no
    # two-rank decomposition whose box edges divide by the fire grid ratio
    if(ERF_ENABLE_MPI)
        if("${ADD_TEST_FIRE_NRANKS}" STREQUAL "")
            set(NP ${ERF_TEST_NRANKS})
        else()
            set(NP ${ADD_TEST_FIRE_NRANKS})
        endif()
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    # fire plotfile names carry the step number padded to five digits
    set(_step "0000${NSTEPS}")
    string(LENGTH "${_step}" _len)
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_step}" ${_start} 5 _step)
    set(PLTFILE "plt_fire_${_step}")

    set(RUNTIME_OPTIONS "max_step=${NSTEPS} erf.fire_plot_int=${NSTEPS} erf.fire_plot_file=plt_fire_ erf.plot_int_1=-1 erf.plot_int_2=-1 erf.check_int=-1 ${ADD_TEST_FIRE_RUNTIME_OPTIONS}")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    # the fire and dust CSVs are appended to (a restart keeps its earlier leg), so a
    # rerun in the same directory starts from a clean slate
    set(test_command sh -c "cd ${CURRENT_TEST_BINARY_DIR} && rm -rf plt_fire_* plt_dust_* fire_stats* dust_diag* && ${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE} ${RUNTIME_OPTIONS} > ${test_log} 2>&1 && test -f ${CURRENT_TEST_BINARY_DIR}/${PLTFILE}/Header")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;fire"
        ATTACHED_FILES_ON_FAIL "${test_log}"
    )
endfunction(add_test_fire)

# Start-up check: copy SOURCE_DIR, run INPUT_FILE on one rank with RUNTIME_OPTIONS
# that break a start-up requirement, and pass when the run stops with
# EXPECTED_MESSAGE in its output. The run is meant to abort, so its exit status is
# dropped by the pipe into tee (a ';' here would split the CMake command list).
function(add_test_abort TEST_NAME SOURCE_DIR INPUT_FILE EXPECTED_MESSAGE RUNTIME_OPTIONS)
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}")
    else()
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE} max_step=1 erf.plot_int_1=-1 erf.plot_int_2=-1 erf.check_int=-1 ${RUNTIME_OPTIONS} 2>&1 | tee ${test_log}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 600
        PROCESSORS 1
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        PASS_REGULAR_EXPRESSION "${EXPECTED_MESSAGE}"
        ATTACHED_FILES_ON_FAIL "${test_log}"
    )
endfunction(add_test_abort)

# Fire suite deck followed by its check script (pure Python; the plotfile reader
# erf_plotfile.py from Canonical_RANS is copied next to it). The script's exit
# code is the verdict and its table is echoed into the ctest output.
function(add_test_fire_check TEST_NAME SUITE_DIR INPUT_FILE NSTEPS CHECK_SCRIPT)
    set(options )
    set(oneValueArgs "RUNTIME_OPTIONS" "NRANKS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_FIRE_CHECK "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(CURRENT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/Exec/RegTests/${SUITE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(COPY ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS/erf_plotfile.py
         DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        if("${ADD_TEST_FIRE_CHECK_NRANKS}" STREQUAL "")
            set(NP ${ERF_TEST_NRANKS})
        else()
            set(NP ${ADD_TEST_FIRE_CHECK_NRANKS})
        endif()
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(RUNTIME_OPTIONS "max_step=${NSTEPS} erf.plot_int_1=-1 erf.plot_int_2=-1 erf.check_int=-1 ${ADD_TEST_FIRE_CHECK_RUNTIME_OPTIONS}")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(check_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.check.log")
    # stale plotfiles and diagnostics of an earlier run are removed first: the
    # dust and fire CSVs are appended to (a restart keeps its earlier leg)
    set(test_command sh -c "cd ${CURRENT_TEST_BINARY_DIR} && rm -rf plt_fire_* plt_dust_* dust_diag* fire_stats* CHECK_FAILED && ${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE} ${RUNTIME_OPTIONS} > ${test_log} 2>&1 || ( tail -n 60 ${test_log} && false ) && ( ${ERF_RANS_PYTHON} ${CURRENT_TEST_BINARY_DIR}/${CHECK_SCRIPT} > ${check_log} 2>&1 || touch ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED ) && cat ${check_log} && test ! -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;fire"
        ATTACHED_FILES_ON_FAIL "${test_log};${check_log}"
    )
endfunction(add_test_fire_check)

# Fire suite driven by its own run script: copy SUITE_DIR (and erf_plotfile.py from
# Canonical_RANS) and run SCRIPT with the executable, MPIRUN set for NRANKS ranks. The
# script's exit status is the verdict; it runs the variants and their checks itself.
function(add_test_fire_script TEST_NAME SUITE_DIR SCRIPT)
    set(options )
    set(oneValueArgs "NRANKS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_FIRE_SCRIPT "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(CURRENT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/Exec/RegTests/${SUITE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(COPY ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS/erf_plotfile.py
         DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if("${ADD_TEST_FIRE_SCRIPT_NRANKS}" STREQUAL "")
        set(NP ${ERF_TEST_NRANKS})
    else()
        set(NP ${ADD_TEST_FIRE_SCRIPT_NRANKS})
    endif()
    set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")

    resolve_test_exe("" "erf_exec" TEST_EXE)
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    # the script removes its own earlier output before running
    set(test_command sh -c "cd ${CURRENT_TEST_BINARY_DIR} && MPIRUN='${MPI_COMMANDS}' PYTHON=${ERF_RANS_PYTHON} sh ${CURRENT_TEST_BINARY_DIR}/${SCRIPT} ${TEST_EXE} > ${test_log} 2>&1 || ( cat ${test_log} && false ) && cat ${test_log}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;fire"
        ATTACHED_FILES_ON_FAIL "${test_log}"
    )
endfunction(add_test_fire_script)

# Fire start-up check: add_test_abort on one deck of a fire suite under Exec/RegTests
function(add_test_fire_abort TEST_NAME SUITE_DIR INPUT_FILE EXPECTED_MESSAGE RUNTIME_OPTIONS)
    add_test_abort(${TEST_NAME} ${PROJECT_SOURCE_DIR}/Exec/RegTests/${SUITE_DIR} ${INPUT_FILE}
                   "${EXPECTED_MESSAGE}" "${RUNTIME_OPTIONS}")
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "regression;fire")
endfunction(add_test_fire_abort)

# Fire fuel map row order: run INPUT_FILE of a fire suite under Exec/RegTests to
# its step-0 fire plotfile and check fire_fuel_load at the south and north edges
# along x = X with amrex_fextract (Tests/RunFireFuelMapRows.cmake). The deck must
# load its ESRI ASCII map with erf.fire.fuel_map.load_from_map.
function(add_test_fire_fuel_map_rows TEST_NAME SUITE_DIR INPUT_FILE X SOUTH_MIN SOUTH_MAX NORTH_MIN NORTH_MAX)
    set(CURRENT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/Exec/RegTests/${SUITE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    resolve_test_exe("" "erf_exec" TEST_EXE)
    string(REPLACE "amrex_fcompare" "amrex_fextract" FEXTRACT_EXE "${FCOMPARE_EXE}")

    add_test(${TEST_NAME} ${CMAKE_COMMAND}
        -DMPIEXEC=${MPIEXEC_EXECUTABLE}
        -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
        -DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}
        -DNRANKS=${ERF_TEST_NRANKS}
        -DTEST_EXE=${TEST_EXE}
        -DFEXTRACT=${FEXTRACT_EXE}
        -DINPUT=${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE}
        -DWORKING_DIRECTORY=${CURRENT_TEST_BINARY_DIR}
        -DX=${X}
        -DSOUTH_MIN=${SOUTH_MIN}
        -DSOUTH_MAX=${SOUTH_MAX}
        -DNORTH_MIN=${NORTH_MIN}
        -DNORTH_MAX=${NORTH_MAX}
        -P ${PROJECT_SOURCE_DIR}/Tests/RunFireFuelMapRows.cmake)
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 600
        PROCESSORS ${ERF_TEST_NRANKS}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;fire"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/run.log")
endfunction(add_test_fire_fuel_map_rows)

# Stationary test -- compare with time 0
function(add_test_0 TEST_NAME TEST_DIR TEST_EXE PLTFILE)
    set(options )
    set(oneValueArgs "INPUT_SOUNDING" "RUNTIME_OPTIONS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_0 "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    
    setup_test()

    set(RUNTIME_OPTIONS "${ADD_TEST_0_RUNTIME_OPTIONS}")
    if(NOT "${ADD_TEST_0_INPUT_SOUNDING}" STREQUAL "")
      string(APPEND RUNTIME_OPTIONS "erf.input_sounding_file=${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_0_INPUT_SOUNDING}")
    endif()

    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)
    set(FCOMPARE_TOLERANCE "-r 1e-14 --abs_tol 1.0e-14")
    set(FCOMPARE_FLAGS "-a ${FCOMPARE_TOLERANCE}")
    set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i erf.input_sounding_file=${CURRENT_TEST_BINARY_DIR}/input_sounding ${RUNTIME_OPTIONS} > ${TEST_NAME}.log && ${MPI_FCOMP_COMMANDS} ${FCOMPARE_EXE} ${FCOMPARE_FLAGS} ${CURRENT_TEST_BINARY_DIR}/plt00000 ${CURRENT_TEST_BINARY_DIR}/${PLTFILE}")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 5400
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log"
    )
endfunction(add_test_0)

# SDM regression test
function(add_test_sdm TEST_NAME TEST_DIR TEST_EXE PLTFILE TEST_RTOL TEST_ATOL)
    set(options )
    set(oneValueArgs "INPUT_SOUNDING" "RUNTIME_OPTIONS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_SDM "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    setup_test()

    set(RUNTIME_OPTIONS "${ADD_TEST_SDM_RUNTIME_OPTIONS}")
    if(NOT "${ADD_TEST_SDM_INPUT_SOUNDING}" STREQUAL "")
      string(APPEND RUNTIME_OPTIONS "erf.input_sounding_file=${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_SDM_INPUT_SOUNDING}")
    endif()

    resolve_test_exe("${TEST_DIR}" "${TEST_EXE}" TEST_EXE)

    if(ERF_SDM_SMOKE_ONLY)
        # No gold file is available for this case here, so run it to completion
        # and let assertions and aborts be the check.
        set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i ${RUNTIME_OPTIONS} > ${TEST_NAME}.log")
        set(TEST_LABELS "smoke")
    else()
        set(FCOMPARE_TOLERANCE "--rel_tol ${TEST_RTOL} --abs_tol ${TEST_ATOL}")
        set(FCOMPARE_FLAGS "--abort_if_not_all_found --allow_diff_grids ${FCOMPARE_TOLERANCE}")
        set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.i ${RUNTIME_OPTIONS} > ${TEST_NAME}.log && ${MPI_FCOMP_COMMANDS} ${FCOMPARE_EXE} ${FCOMPARE_FLAGS} ${PLOT_GOLD} ${CURRENT_TEST_BINARY_DIR}/${PLTFILE}")
        set(TEST_LABELS "regression")
    endif()

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 5400
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "${TEST_LABELS}"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log"
    )
endfunction(add_test_sdm)

#=============================================================================
# Regression tests
#=============================================================================

if(ERF_ENABLE_TESTS AND ERF_ENABLE_MPI)
    # The checker is a small AMReX PlotFileData consumer and is built only
    # when regression tests are enabled.  All SHOC cases use explicit
    # state-update ownership settings in their input fixture.
    # DOUBLE keeps the historical strict fcompare oracle.  SINGLE uses the
    # field-aware comparator so the shared gold remains the scientific
    # reference while represented-float roundoff is bounded per field.
    if(ERF_PRECISION STREQUAL "SINGLE")
        set(_shoc_clear_gold_comparison "field_aware")
    else()
        set(_shoc_clear_gold_comparison "fcompare")
    endif()

    add_test_shoc_r(SHOC_Stable_Clear "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Stable_Clear"
        CHECK_MODE "stable_clear"
        GOLD_COMPARISON "${_shoc_clear_gold_comparison}"
        GOLD_MODE "stable_clear"
        LABELS regression shoc
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Stable_Cloud "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Stable_Cloud"
        CHECK_MODE "stable_cloud"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "stable_cloud"
        LABELS regression shoc
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Clear_BOMEX "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Clear_BOMEX"
        CHECK_MODE "unstable_clear"
        GOLD_COMPARISON "${_shoc_clear_gold_comparison}"
        GOLD_MODE "unstable_clear"
        LABELS regression shoc
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_SatAdj "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud.i"
        CHECK_MODE "unstable_cloud"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud"
        RUNTIME_OPTIONS "erf.moisture_model=SatAdj erf.buoyancy_type=1 "
        LABELS regression shoc microphysics
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_NoCond "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud.i"
        CHECK_MODE "unstable_cloud_nocond"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud_nocond"
        RUNTIME_OPTIONS "erf.moisture_model=MoistNoCondensation erf.buoyancy_type=1 "
        LABELS regression shoc microphysics
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_SatAdj_Property "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud.i"
        CHECK_MODE "unstable_cloud"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud"
        RUNTIME_OPTIONS "erf.moisture_model=SatAdj erf.buoyancy_type=1 "
        SKIP_GOLD
        LABELS regression shoc microphysics property
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_NoCond_Property "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud.i"
        CHECK_MODE "unstable_cloud_nocond"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud_nocond"
        RUNTIME_OPTIONS "erf.moisture_model=MoistNoCondensation erf.buoyancy_type=1 "
        SKIP_GOLD
        LABELS regression shoc microphysics property
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_Kessler "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud_Kessler.i"
        CHECK_MODE "unstable_cloud_kessler"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud_kessler"
        LABELS regression shoc microphysics
        TIMEOUT 900)
    add_test_shoc_r(SHOC_Unstable_Cloud_WSM6 "" "erf_exec" "plt00020"
        TEST_FILES_DIR "SHOC_Unstable_Cloud"
        INPUT_FILE "SHOC_Unstable_Cloud_WSM6.i"
        CHECK_MODE "unstable_cloud_wsm6"
        GOLD_COMPARISON "field_aware"
        GOLD_MODE "unstable_cloud_wsm6"
        LABELS regression shoc microphysics
        TIMEOUT 900)
    add_test_shoc_mutation(SHOC_Mutation_Disable_Tke_State_Update
        "erf.shoc.debug_disable_tke_state_update=true" rhoKE
        1.0e-3 1.0e-3 0.10)
    add_test_shoc_mutation(SHOC_Mutation_Disable_Theta_State_Update
        "erf.shoc.debug_disable_theta_state_update=true" theta
        1.0e-2 1.0e-2 0.05)
endif()

# These tests will all be built in Exec
add_test_plotfile_header(Plotfile3D_DryUnavailableSelection "" "erf_exec" "plt00000")
add_test_r(DensityCurrent                    ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(DensityCurrent_anelastic          ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(DensityCurrent_detJ2              ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(DensityCurrent_detJ2_nosub        ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(DensityCurrent_detJ2_MT           ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(EkmanSpiral                       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(IsentropicVortexStationary        ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(IsentropicVortexAdvecting         ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(IVA_NumDiff                       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(MovingTerrain_nosub               ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(MovingTerrain_sub                 ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(Terrain2Lev_STF_interp            ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(Terrain2Lev_STF_transform         ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(RayleighDamping                   ""  "erf_exec" "plt00100" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvectionUniformU           ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvectionShearedU           ""  "erf_exec" "plt00080" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_order2              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_order3              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_order4              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_order5              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_order6              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_weno3               ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_d(ScalarAdvDiff_weno3z              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_weno5               ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_d(ScalarAdvDiff_weno5z              ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarAdvDiff_wenomzq3            ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarDiffusionGaussian           ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ScalarDiffusionSine               ""  "erf_exec" "plt00020" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(TaylorGreenAdvecting              ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(TaylorGreenAdvectingDiffusing     ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(MSF_NoSub_IsentropicVortexAdv     ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(MSF_Sub_IsentropicVortexAdv       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
#add_test_r(FlowInABox                       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ABL_MOST                          ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
# A terrain-fitted mesh whose BoxArray is split in z (amr.max_grid_size below the number of
# cells in z), under a MOST surface layer. The anelastic case covers the projection as well,
# but its terrain Poisson solve is the FFT-preconditioned GMRES, so it needs the FFT build.
# The compressible case without acoustic substepping runs in every build.
if(ERF_ENABLE_FFT)
add_test_r(ABL_MOST_WOA_ZSplit               ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
endif()
add_test_r(ABL_MOST_WOA_ZSplit_NoSub         ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
# The ZSplit decks on one box against their 12 boxes: the base state and the projection used to
# depend on the split in z (u 0.01 m/s and theta 0.18 K apart after 10 steps in the anelastic
# case). The runner is a cmake -P script, so it needs MPI and cannot expand the Windows exe glob.
if(ERF_ENABLE_MPI AND NOT WIN32)
if(ERF_ENABLE_FFT)
add_test_box_parity(ABL_MOST_WOA_ZSplit_BoxParity ABL_MOST_WOA_ZSplit "plt00010"
    COMMON_OPTIONS "erf.vert_implicit=false erf.input_sounding_file=${CMAKE_CURRENT_BINARY_DIR}/test_files/ABL_MOST_WOA_ZSplit_BoxParity/input_sounding"
    REFERENCE_OPTIONS "amr.max_grid_size=64"
    FCOMPARE_RTOL "1.0e-9")
endif()
add_test_box_parity(ABL_MOST_WOA_ZSplit_NoSub_BoxParity ABL_MOST_WOA_ZSplit_NoSub "plt00010"
    COMMON_OPTIONS "erf.vert_implicit=false erf.input_sounding_file=${CMAKE_CURRENT_BINARY_DIR}/test_files/ABL_MOST_WOA_ZSplit_NoSub_BoxParity/input_sounding"
    REFERENCE_OPTIONS "amr.max_grid_size=64"
    FCOMPARE_RTOL "1.0e-9")
endif()
add_test_r(ABL_MOST_IMP_DIFF                 ""  "erf_exec" "plt00010")
add_test_r(ABL_MOST_IMP_DIFF_WOA             ""  "erf_exec" "plt00010")
add_test_r(ABL_MOST_IMP_DIFF_TKE
    ""
    "erf_exec"
    "plt00010"
    FCOMPARE_ATOL "4.0e-10")
if(ERF_ENABLE_FFT)
    add_test_r(ABL_MOST_Cloudchamber         ""  "erf_exec" "plt00010")
endif()
add_test_r(ABL_MOST_SFC                      ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ABL_MOST_SST                      ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(ABL_MYNN_PBL                      ""  "erf_exec" "plt00100" INPUT_SOUNDING "input_sounding_GABLS1" RUNTIME_OPTIONS "erf.vert_implicit=false " )
# RunTilingParity.cmake calls mpiexec and fcompare through execute_process,
# which neither drops an empty MPIEXEC nor expands the Windows exe globs.
if(ERF_ENABLE_MPI AND NOT WIN32)
  # pblh (2D) and Lturb (3D) are the per-tile PBL height copied out of the
  # scheme; the deck is set up so they differ from column to column.
  add_test_tiling_parity(ABL_MRF_Tiling      ABL_MRF_Tiling "00010" "00010"
      VARYING_3D "Lturb Kmv" VARYING_2D "pblh u_star")
  add_test_tiling_parity(ABL_YSUNew_Tiling   ABL_MRF_Tiling "00010" "00010"
      RUNTIME_OPTIONS "erf.pbl_type=YSUNew erf.most.pblh_calc=YSU"
      VARYING_3D "Lturb Kmv" VARYING_2D "pblh u_star")
  # Legacy YSU aborts in unstable conditions, so cool the surface (a stronger
  # cooling than -0.02 with the 5 m/s wind stops the MOST iteration converging).
  # It covers the full-column assert only: legacy YSU never calls set_pblh, so
  # pblh is left out of the 2D plotfile rather than compared as a constant.
  add_test_tiling_parity(ABL_YSU_Tiling      ABL_MRF_Tiling "00010" "00010"
      RUNTIME_OPTIONS "erf.pbl_type=YSU erf.most.pblh_calc=YSU erf.most.surf_temp_flux=-0.02 'erf.plot2d_vars_1=u_star t_star Olen'"
      VARYING_3D "Lturb Kmv" VARYING_2D "u_star")
  # The PBLH smoothing stencil reads a column its own tile does not own, so it
  # needs its own coverage: with the stencil reading off the end of the array the
  # MRF deck differed by 24.5 m in Lturb (12%) between the tiled and untiled runs.
  # MRF and YSUNew size and fill that halo separately, so both are registered.
  add_test_tiling_parity(ABL_MRF_Tiling_Smooth    ABL_MRF_Tiling "00010" "00010"
      RUNTIME_OPTIONS "erf.enable_pblh_smoothing=true"
      VARYING_3D "Lturb Kmv" VARYING_2D "pblh u_star")
  add_test_tiling_parity(ABL_YSUNew_Tiling_Smooth ABL_MRF_Tiling "00010" "00010"
      RUNTIME_OPTIONS "erf.pbl_type=YSUNew erf.most.pblh_calc=YSU erf.enable_pblh_smoothing=true"
      VARYING_3D "Lturb Kmv" VARYING_2D "pblh u_star")
  # The immersed-boundary-aware MRF and YSUNew (erf.pbl_ib_aware) build their
  # per-column surface and work arrays on the tile work box; a cube by
  # immersed forcing makes them differ from column to column.
  add_test_tiling_parity(PBL_IBAware_MRF_Tiling    PBL_IBAware_Tiling "00010" "00010"
      VARYING_3D "Kmv" VARYING_2D "pblh u_star")
  add_test_tiling_parity(PBL_IBAware_YSUNew_Tiling PBL_IBAware_Tiling "00010" "00010"
      RUNTIME_OPTIONS "erf.pbl_type=YSUNew erf.most.pblh_calc=YSU"
      VARYING_3D "Kmv" VARYING_2D "pblh u_star")
endif()
# A column PBL scheme on boxes split in z must stop at start-up, not at the kernel assert in step 1
add_test_abort(ABL_MRF_ZSplit_abort ${PROJECT_SOURCE_DIR}/Tests/test_files/ABL_MRF_Tiling ABL_MRF_Tiling.i
    "every box on level 0 must span the vertical domain" "amr.max_grid_size_z=16")
# Boxes stacked in z must stop at start-up with the implicit acoustic substep, the implicit
# vertical diffusion or a surface layer, which all work on whole columns inside one box
add_test_abort(ABL_ZSplit_ImplicitSubstep_abort ${PROJECT_SOURCE_DIR}/Tests/test_files/ABL_MRF_Tiling ABL_MRF_Tiling.i
    "split in z, the implicit acoustic substep and the implicit vertical diffusion give"
    "zlo.type=SlipWall erf.pbl_type=None erf.most.pblh_calc=None erf.les_type=Smagorinsky erf.Cs=0.1 amr.max_grid_size_z=16")
add_test_abort(ABL_ZSplit_ImplicitDiffusion_abort ${PROJECT_SOURCE_DIR}/Tests/test_files/ABL_MRF_Tiling ABL_MRF_Tiling.i
    "split in z, the implicit vertical diffusion gives"
    "zlo.type=SlipWall erf.pbl_type=None erf.most.pblh_calc=None erf.les_type=Smagorinsky erf.Cs=0.1 erf.substepping_type=None erf.fixed_dt=0.05 amr.max_grid_size_z=16")
add_test_r(ABL_InflowFile                    ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(MoistBubble                       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(SquallLine_2D                     ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_r(SuperCell_3D                      ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
if(ERF_ENABLE_NETCDF)
  # Distributed terrain ownership and gridded forest interpolation are both
  # exercised by this one-step, two-rank regression.  The NetCDF files are
  # static fixtures so CI does not require ncgen.
  add_test_r(BellForest                       ""  "erf_exec" "plt00001"
      FCOMPARE_RTOL "2.0e-9" FCOMPARE_ATOL "2.0e-9")
endif()
if(ERF_ENABLE_PARTICLES)
  # Production regression: protect the fixed SuperDroplets water-field
  # contract against confusing the constructor sentinel with state width.
  add_test_plotfile_header(Plotfile3D_SuperDropletsSelection "" "erf_exec" "plt00000")
  add_test_r(ParticleAdvect                  ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
  add_test_r(ParticleWoA                     ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
  add_test_r(ParticleAdvect_AMR1_box         ""  "erf_exec" "plt00050" RUNTIME_OPTIONS "erf.vert_implicit=false ")
  add_test_sdm(ParticleAdvect_AMR1_pcount      ""  "erf_exec" "plt00050" 2e-8 3e-9 RUNTIME_OPTIONS "erf.vert_implicit=false ")
  # Skip AMR2_pcount for Debug/RelWithDebInfo builds with AMD GPUs (it freezes!)
  if((CMAKE_BUILD_TYPE STREQUAL "Release") OR (NOT ERF_ENABLE_HIP))
    add_test_sdm(ParticleAdvect_AMR2_pcount    ""  "erf_exec" "plt00050" 1e-7 5e-9 RUNTIME_OPTIONS "erf.vert_implicit=false ")
  endif()
endif( )
# The option name used to be misspelled (ERF_ENABLE_RRGMTP), which kept this
# test unregistered; Tests/test_files/Radiation has never existed, so it is
# registered only once someone adds the inputs.
if(ERF_ENABLE_RRTMGP AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_files/Radiation")
  add_test_r(Radiation                       ""  "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
endif()

# TwoStream radiation needs no external library and no gold plotfiles: the
# column-physics checker verifies the vertical structure of the heating
# rates, and the header test verifies that qsrc_sw/qsrc_lw are written.
# The column test runs through cmake -P and execute_process, which needs a
# launcher and a resolved executable path; the Windows job builds without
# MPI and resolves test executables through sh -c globs, so it is skipped
# there like the other script-driven tests.
if(ERF_ENABLE_MPI AND NOT WIN32)
  add_test_two_stream_radiation(TwoStream_ColumnHeating "plt00002")
  # Same column over a Witch-of-Agnesi hill on a terrain-fitted mesh: the
  # layer thicknesses come from the nodal heights, every column differs, and
  # the runner's 1-rank vs NRANKS comparison of the diagnostics CSV has a
  # real signal (rank-local means fail it).
  add_test_two_stream_radiation(TwoStream_ColumnHeating_Terrain "plt00002")
endif()
add_test_plotfile_header(Plotfile3D_TwoStreamHeatingSelection "" "erf_exec" "plt00000")

add_test_0(CouetteFlow_x                     "" "erf_exec" "plt00050" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_0(CouetteFlow_y                     "" "erf_exec" "plt00050" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_0(PoiseuilleFlow_x                  "" "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_0(PoiseuilleFlow_y                  "" "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_0(InitSoundingIdeal_stationary      "" "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")
add_test_0(Deardorff_stationary              "" "erf_exec" "plt00010" RUNTIME_OPTIONS "erf.vert_implicit=false ")

if(ERF_ENABLE_PARTICLES)
    # These tests require machine-specific gold files due to platform-dependent initial sampling.
    # Without those gold files they can still be run to completion as smoke tests, which is the
    # only coverage they get on a machine that builds with assertions enabled.
    if(ERF_TEST_ENABLE_EXTRA_SDM_TESTS OR ERF_TEST_SDM_SMOKE_GATED)
        if(NOT ERF_TEST_ENABLE_EXTRA_SDM_TESTS)
            set(ERF_SDM_SMOKE_ONLY TRUE)
        endif()
        # log-normal distribution for radius
        add_test_sdm(SDM_RICO3D_InitSampling         ""  "erf_exec"   "plt00000" 1e-14 2e-13 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # mass-exponential distribution for mass
        add_test_sdm(SDM_Bubble2D_Adv_InitSampling   ""  "erf_exec"   "plt00000" 1e-14 1e-14 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # per-box high-multiplicity injection (stochastic cell scatter -> platform-specific gold)
        add_test_sdm(SDM_Bubble2D_PerBoxInjection    ""  "erf_exec"   "plt00050" 5e-12 5e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # INAS sampled initialization for freezing temperature
        add_test_sdm(SDM_Bubble2D_Adv_TfzINAS        ""  "erf_exec"   "plt00000" 1e-14 1e-14 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # column case to test condensation
        add_test_sdm(SDM_SineMassFlux                "" "erf_exec" "plt00050" 1e-14 1e-14 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # recycling
        add_test_sdm(SDM_Box3D_Recycling             "" "erf_exec"  "plt00060" 5e-13 1e-14 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # INAS immersion freezing in a 1D cooling column (Tfz sampling -> platform-specific gold)
        add_test_sdm(SDM_FreezingShaft               "" "erf_exec"  "plt00001" 1e-12 1e-12 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
        # Collision processes: stochastic pair sampling and skipped on GPU (RNG/reduction ordering differs).
        if(NOT (ERF_ENABLE_CUDA OR ERF_ENABLE_HIP OR ERF_ENABLE_SYCL))
            # ice-ice aggregation (0D box)
            add_test_sdm(SDM_Box3D_IceAgg            "" "erf_exec"  "plt04500" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
            # warm-rain coalescence (0D box) -- one test per collection kernel.
            # These dominate the runtime and the kernels they cover already have
            # unit tests, so they are left out of the smoke pass.
            if(NOT ERF_SDM_SMOKE_ONLY)
                add_test_sdm(SDM_Box3D_Coal_Golovin      "" "erf_exec"  "plt04000" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
                add_test_sdm(SDM_Box3D_Coal_Halls        "" "erf_exec"  "plt04000" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
                add_test_sdm(SDM_Box3D_Coal_Longs        "" "erf_exec"  "plt04000" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
                add_test_sdm(SDM_Box3D_Coal_Sedimentation "" "erf_exec" "plt04000" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
            endif()
            # riming (ice collecting cloud droplets, 1D shaft)
            add_test_sdm(SDM_RimingShaft             "" "erf_exec"  "plt00400" 1e-12 1e-12 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
        endif()
        unset(ERF_SDM_SMOKE_ONLY)
    endif()

    # passive advection of particles
    add_test_sdm(SDM_Bubble2D_Adv                "" "erf_exec"  "plt00050" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # super-droplets on a terrain-fitted mesh: covers the pos(2) zeta convention
    add_test_sdm(SDM_Bubble2D_WoA                "" "erf_exec"  "plt00050" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # same case with MFIter tiling forced on: in-place kernels written over
    # grown tiles must still reproduce the untiled answer
    add_test_sdm(SDM_Bubble2D_Tiled              "" "erf_exec"  "plt00050" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false fabarray.mfiter_tile_size=8 8 8 ")
    add_test_sdm(SDM_Bubble2D_Adv_AMR1           "" "erf_exec"  "plt00050" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    add_test_sdm(SDM_Bubble2D_Adv_AMR2           "" "erf_exec"  "plt00025" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    add_test_sdm(SDM_Bubble3D_Adv                "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    add_test_sdm(SDM_Bubble3D_Adv_AMR1           "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    add_test_sdm(SDM_Bubble3D_Adv_AMR2           "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # Gold files are MPI-rank-specific (particle-to-mesh FP ordering).
    if(ERF_ENABLE_MPI)
        add_test_sdm(SDM_MoistBubble2D_AMR1      "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        #add_test_sdm(SDM_MoistBubble2D_AMR2      "" "erf_exec" "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        add_test_sdm(SDM_MoistBubble3D_AMR1      "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
        add_test_sdm(SDM_MoistBubble3D_AMR2      "" "erf_exec"  "plt00020" 1e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    endif()
    # passive advection of particles with injection
    add_test_sdm(SDM_Bubble2D_Adv_wInjection     "" "erf_exec"  "plt00050" 5e-12 5e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # fractional injection (sub-unity per-step multiplicity accumulates to one)
    add_test_sdm(SDM_Bubble2D_FracInjection      "" "erf_exec"  "plt00050" 5e-12 5e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # condensation/evaporation
    add_test_sdm(SDM_Box3D_Cond                  "" "erf_exec"  "plt00010" 2e-12 3e-13 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # ice freezing + deposition
    add_test_sdm(SDM_Box3D_IceFrzDep             "" "erf_exec"  "plt00010" 1e-14 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    if(NOT (ERF_ENABLE_HIP OR ERF_ENABLE_SYCL))
        # 1D sublimation shaft: monodisperse ice in a subsaturated column (supersedes the 0D box sublimation test)
        add_test_sdm(SDM_SublimationShaft            "" "erf_exec"  "plt00100" 1e-12 1e-12 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
    endif()
    # 1D melting layer: melting + mixed-phase fall as ice flakes descend into warmer air (supersedes the 0D box melting test)
    add_test_sdm(SDM_MeltingLayer                "" "erf_exec"  "plt00300" 1e-12 1e-12 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # terminal velocity
    add_test_sdm(SDM_Box3D_VTerm                 "" "erf_exec"  "plt00001" 5e-13 1e-14 RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # Congestus case
    add_test_sdm(SDM_Congestus3D                 "" "erf_exec"  "plt00020" 5e-13 5e-13 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # RICO case
    add_test_sdm(SDM_RICO3D                      "" "erf_exec"  "plt00010" 5e-13 5e-13 INPUT_SOUNDING "input_sounding" RUNTIME_OPTIONS "erf.vert_implicit=false ")
    # multispecies setup with dummy water species
    add_test_sdm(SDM_MultiSpecies_Bubble2D       "" "erf_exec"  "plt00001" 5e-12 1e-12 RUNTIME_OPTIONS "erf.vert_implicit=false ")
endif()

# Python for the fire, dust and RANS check scripts
#=============================================================================
# Canonical RANS cases (Exec/CanonicalTests/Canonical_RANS)
#
# Each case runs a short smoke deck and then its Python check script, which
# compares planar-averaged numbers against stated targets with tolerances.
# A clean exit alone is never the pass criterion.
#
# The decks run the anelastic projection with the FFT solver (erf.use_fft),
# which no CI configuration builds. The flat decks are therefore run here
# with the MLMG projection (erf.use_fft=false), and the terrain-fitted decks,
# whose general-terrain projection has no non-FFT path, are registered only
# when the build enables FFT (ERF_ENABLE_FFT).
#=============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(ERF_RANS_PYTHON "${Python3_EXECUTABLE}")
else()
    set(ERF_RANS_PYTHON "python3")
endif()

#=============================================================================
# Fire and dust smoke tests: one deck per suite under Exec/RegTests, a few
# steps each (ctest -L fire, or -R Fire)
#=============================================================================
if(ERF_ENABLE_FIRE)
add_test_fire(FireAccelerationClock_levelset_front FireAccelerationClock inputs_levelset_front 40)
add_test_fire(FireBurnout_base              FireBurnout           inputs_base                40)
add_test_fire(FireDirectionalShape_ellipse  FireDirectionalShape  inputs_ellipse             40)
add_test_fire(FireEmcModel_van_wagner       FireEmcModel          inputs_van_wagner          40)
add_test_fire(FireExposure_noib             FireExposure          inputs_noib                40)
add_test_fire(FireFbp_c2                    FireFbp               inputs_fbp_c2              40)
add_test_fire(FireFluxPartition_cfbm        FireFluxPartition     inputs_cfbm                40)
add_test_fire(FireHeatPlacement_add_noib    FireHeatPlacement     inputs_add_noib            40)
add_test_fire(FireHybridObstacles_noib      FireHybridObstacles   inputs_hybrid_noib         40)
add_test_fire(FireLevelSetEllipse_ellipse   FireLevelSetEllipse   inputs_ellipse             40)
add_test_fire(FireNearWall_noib_mask_wall   FireNearWall          inputs_noib_mask_wall      40)
add_test_fire(FirePerimeterIgnition_t0      FirePerimeterIgnition inputs_t0                  40)
add_test_fire(FireRestart_levelset_straight FireRestart           inputs_levelset_straight   40 NRANKS 1)
# the MRF fire thermal excess reads the lagged fire flux in the halo columns of every
# tile; the coupled restart deck with MRF and the option on stopped on an out-of-bound
# read in the first step before the flux carried ghost columns
add_test_fire(FireMrfThermalExcess          FireRestart           inputs_coupled_straight    40 NRANKS 1
    RUNTIME_OPTIONS "erf.pbl_type=MRF erf.pbl_mrf_fire_thermal_excess=true")
add_test_fire(FireRosComparison_rothermel   FireRosComparison     inputs_rothermel_isotropic 40 NRANKS 1)
add_test_fire(FireScottBurgan_gr2           FireScottBurgan       inputs_sb_gr2              40)
add_test_fire(FireCustomFuel_uniform        FireCustomFuel        inputs_custom_grass        40)
add_test_fire(FireCustomFuel_map            FireCustomFuel        inputs_custom_map          40)
add_test_fire(FireStickMoisture_stick       FireStickMoisture     inputs_stick               40)
add_test_fire(FireStructureIgnition_on      FireStructureIgnition inputs_on                  40 NRANKS 1)
add_test_fire(FirePrecipSource_atmosphere   FirePrecipSource      inputs_atmosphere          40 NRANKS 1)
add_test_fire(FireLiveMoisture_fixed        FireLiveMoisture      inputs_fixed               40)
add_test_fire(FireWindSampling_sample20     FireWindSampling      inputs_sample20            40)
add_test_fire(FirePrescribed_ros_circle     FirePrescribed        inputs_ros_circle          40)
add_test_fire(FirePrescribed_heat_patch     FirePrescribed        inputs_heat_patch          40)
add_test_fire(FireFarsiteDefault            FarsiteDefault        inputs                     40)
add_test_fire(FireLevelSetPropagation       LevelSetPropagation   inputs                     40)
# fire without a surface layer at zlo must stop at start-up, not crash in the first step
add_test_fire_abort(FireNoSurfaceLayer_abort  FireRestart           inputs_levelset_straight
    "The fire module requires a surface layer" "zlo.type=SlipWall")
# a deck-defined fuel model has to be described completely and in range: a code
# outside 1000-1015, a missing property, a bed depth under the floor where the
# Balbi models silently return zero spread, a heat content left in BTU/lb, and a
# raster code no block defines all have to stop the run at start-up
add_test_fire_abort(FireCustomFuelBadCode_abort     FireCustomFuel inputs_bad_code
    "outside the custom range 1000-1015" "max_step=1")
add_test_fire_abort(FireCustomFuelMissing_abort     FireCustomFuel inputs_bad_missing
    "erf.fire.custom_fuel.1000.sav_1h_1_m is required" "max_step=1")
add_test_fire_abort(FireCustomFuelBadDepth_abort    FireCustomFuel inputs_bad_depth
    "erf.fire.custom_fuel.1000.depth_m must be in" "max_step=1")
add_test_fire_abort(FireCustomFuelBadHeat_abort     FireCustomFuel inputs_bad_heat
    "erf.fire.custom_fuel.1000.heat_content_J_kg must be in" "max_step=1")
add_test_fire_abort(FireCustomFuelBurnout_abort     FireCustomFuel inputs_bad_burnout
    "needs erf.fire.custom_fuel.1000.burnout_time_s" "max_step=1")
add_test_fire_abort(FireCustomFuelUndeclared_abort  FireCustomFuel inputs_bad_undeclared
    "the fuel map holds code 1007" "max_step=1")
# a fuel map is placed by cell index, so it must have the fire grid's size
add_test_fire_abort(FireFuelMapSize_abort     FireScottBurgan       inputs_sb_map
    "has 256 x 128 cells but the fire grid has 128 x 64" "erf.fire.grid_ratio=2")
# the first data row of an ESRI ASCII fuel map is the north edge: the sb_map deck's
# map puts TL3 (1.2329 kg/m2) along the north and NB8 water (no fuel) along the south
# execute_process needs mpiexec, and does not expand the executable globs used on Windows
if(ERF_ENABLE_MPI AND NOT WIN32)
add_test_fire_fuel_map_rows(FireScottBurgan_map_rows FireScottBurgan inputs_sb_map 100
    -0.000001 0.000001 1.2329 1.2330)
endif()
# a misspelt selector or a value the kernels cannot use stops at start-up
add_test_fire_abort(FireBadRosModel_abort     FireRestart           inputs_levelset_straight
    "erf.fire.ros_model = \"rothermal\" is not one of" "erf.fire.ros_model=rothermal")
add_test_fire_abort(FireBadCoupling_abort     FireRestart           inputs_levelset_straight
    "erf.fire.coupling_type = \"laged\" is not one of" "erf.fire.coupling_type=laged")
# the fire grid on a refined level (erf.fire.anchor_level, the finest level by default):
# the same front as a single-level run at that resolution, the level-0 heat budget after
# average-down (and its loss with the fire on the coarser level), restart, and a restart
# that would move the fire grid stopping at start-up
if(ERF_ENABLE_MPI AND NOT WIN32)
add_test_fire_script(FireAnchorLevel          FireAnchorLevel       run_anchor_level.sh NRANKS 2)
endif()
# where the rain that wets the dead fuel comes from (erf.fire.precip_source): the
# atmosphere's rain per column wets the fuel under the raining columns only and the
# fire-grid rate equals the change of the Kessler surface accumulation over the step,
# the uniform rate wets every cell, and the restart carries the accumulation snapshot
if(ERF_ENABLE_MPI AND NOT WIN32)
add_test_fire_script(FirePrecipSource         FirePrecipSource      run_precip.sh NRANKS 1)
endif()
# its start-up checks: a scheme without precipitation accumulators, static moisture,
# and both rain sources named at once
add_test_fire_abort(FirePrecipSource_norain_abort  FirePrecipSource inputs_atmosphere
    "provides none" "erf.moisture_model=Kessler_NoRain")
add_test_fire_abort(FirePrecipSource_static_abort  FirePrecipSource inputs_atmosphere
    "holds them fixed" "erf.fire.moisture_dynamic=false")
add_test_fire_abort(FirePrecipSource_two_sources_abort FirePrecipSource inputs_atmosphere
    "the rain has one source" "erf.fire.precip_rate_mm_hr=1.0")
# its start-up checks: a level above the finest, a regridding level, a refinement box
# short of the domain top, two separate patches, and the dust layer (level 0 only)
add_test_fire_abort(FireAnchorLevel_above_finest_abort FireAnchorLevel inputs_base
    "is above the finest level of this run" "erf.fire.anchor_level=2")
add_test_fire_abort(FireAnchorLevel_regrid_abort       FireAnchorLevel inputs_base
    "regrids it" "erf.regrid_int=10")
add_test_fire_abort(FireAnchorLevel_partial_height_abort FireAnchorLevel inputs_partial_height
    "Cannot decompose in z direction" "")
add_test_fire_abort(FireAnchorLevel_two_patches_abort  FireAnchorLevel inputs_two_patches
    "but the fire grid needs one rectangle" "")
if(ERF_ENABLE_DUST)
add_test_fire_abort(FireAnchorLevel_dust_abort         FireAnchorLevel inputs_base
    "The dust layer and the fire-dust coupling run on level 0" "erf.dust.enable=true")
endif()
# the fire at the edge of the fire grid: the guard band records the first contact in the
# statistics CSV and warns (warn), never fires on a fire far from every wall (far), or
# stops the run on a disc that starts inside the band (abort)
# Suppression (erf.fire.suppression.*): each scenario on the level-set and the
# FARSITE path, checked from the last fire plotfile and the suppression log
add_test_fire_check(FireSuppression_line_early_levelset FireSuppression inputs_line_early 40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_line_early_farsite  FireSuppression inputs_line_early 40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_check(FireSuppression_line_late_levelset  FireSuppression inputs_line_late  40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_line_late_farsite   FireSuppression inputs_line_late  40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_check(FireSuppression_drop_hold_levelset  FireSuppression inputs_drop_hold  40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_drop_hold_farsite   FireSuppression inputs_drop_hold  40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_check(FireSuppression_drop_slow_levelset  FireSuppression inputs_drop_slow  40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_drop_slow_farsite   FireSuppression inputs_drop_slow  40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_check(FireSuppression_hold_levelset       FireSuppression inputs_hold       40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_hold_farsite        FireSuppression inputs_hold       40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_check(FireSuppression_burnout_levelset    FireSuppression inputs_burnout    40 check_suppression.py NRANKS 1)
add_test_fire_check(FireSuppression_burnout_farsite     FireSuppression inputs_burnout    40 check_suppression.py NRANKS 1
                    RUNTIME_OPTIONS "erf.fire.propagation_method=farsite")
add_test_fire_abort(FireSuppression_bad_line_abort      FireSuppression inputs_line_early
                    "suppression file 'actions_bad.txt' line 2: rate must be a number > 0"
                    "erf.fire.suppression.file=actions_bad.txt")
add_test_fire_abort(FireSuppression_duplicate_id_abort  FireSuppression inputs_line_early
                    "suppression file 'actions_duplicate.txt' line 3: duplicate id"
                    "erf.fire.suppression.file=actions_duplicate.txt")
add_test_fire_abort(FireSuppression_no_file_abort       FireSuppression inputs_base
                    "erf.fire.suppression.enable needs erf.fire.suppression.file"
                    "erf.v=0")
# The watched file (a writer process appends the line while the run polls, on
# one rank and on two, where the re-read is broadcast) and the restart with a
# line under construction, both on both propagation paths
if(ERF_ENABLE_MPI AND NOT WIN32)
add_test_fire_script(FireSuppression_poll_levelset    FireSuppression run_poll.sh NRANKS 1)
add_test_fire_script(FireSuppression_poll_farsite     FireSuppression run_poll_farsite.sh NRANKS 1)
add_test_fire_script(FireSuppression_poll_2ranks      FireSuppression run_poll.sh NRANKS 2)
add_test_fire_script(FireSuppression_restart_levelset FireSuppression run_restart.sh NRANKS 1)
add_test_fire_script(FireSuppression_restart_farsite  FireSuppression run_restart_farsite.sh NRANKS 1)
endif()

add_test_fire_check(FireBoundaryGuard_warn    FireBoundaryGuard     inputs_warn  40 check_guard.py NRANKS 1)
add_test_fire_check(FireBoundaryGuard_far     FireBoundaryGuard     inputs_far   40 check_guard.py NRANKS 1)
add_test_fire_abort(FireBoundaryGuard_abort   FireBoundaryGuard     inputs_abort
    "reached the boundary guard band" "")
# structure ignition and house-to-house spread (erf.fire.structures.ignition.*): the
# old and the new path on the same three houses, a radiation-only variant in which
# the second house can only ignite from the first, a restart, and the checker on the
# old path, where it must fail
if(ERF_ENABLE_MPI AND NOT WIN32)
add_test_fire_script(FireStructureIgnition   FireStructureIgnition run_structure_ignition.sh NRANKS 1)
endif()
# its start-up checks: ignition without the exposure accumulators it reads, and a
# burn curve whose growth phase alone would release more than the EN 1991-1-2 curve allows
add_test_fire_abort(FireStructureIgnition_no_exposure_abort FireStructureIgnition inputs_on
    "needs erf.fire.exposure.enable" "erf.fire.exposure.enable=false")
add_test_fire_abort(FireStructureIgnition_curve_abort FireStructureIgnition inputs_on
    "exceeds 70 % of fuel_load_J_m2" "erf.fire.structures.ignition.growth_time_s=100.0")
# every documented fire/dust key is read, every read key is documented, no deck sets an unread key
add_test(FireDustInputsDocs ${ERF_RANS_PYTHON} ${PROJECT_SOURCE_DIR}/Tests/check_fire_dust_inputs.py ${PROJECT_SOURCE_DIR})
set_tests_properties(FireDustInputsDocs PROPERTIES LABELS "docs;fire" TIMEOUT 120)
if(ERF_ENABLE_DUST)
add_test_fire(FireRestart_dust_straight     FireRestart           inputs_dust_straight       40 NRANKS 1)
# the three fire-dust couplings applied once per step, in the right order
add_test_fire_check(FireDustCoupling_check  FireDustCoupling      inputs                     40 check_firedust.py NRANKS 1)
# dust inputs the kernels cannot use stop at start-up
add_test_fire_abort(DustBadBins_abort         FireRestart           inputs_dust_straight
    "erf.dust.n_size_bins must be >= 1" "erf.dust.n_size_bins=0")
add_test_fire_abort(DustZrefMismatch_abort    FireRestart           inputs_dust_straight
    "must equal erf.most.zref" "erf.most.zref=12.0")
endif()
endif()

#=============================================================================
# Canonical RANS cases (Exec/CanonicalTests/Canonical_RANS)
#
# Each case runs a short smoke deck and then its Python check script, which
# compares planar-averaged numbers against stated targets with tolerances.
# A clean exit alone is never the pass criterion.
#
# The decks run the anelastic projection with the FFT solver (erf.use_fft),
# which no CI configuration builds. The flat decks are therefore run here
# with the MLMG projection (erf.use_fft=false), and the terrain-fitted decks,
# whose general-terrain projection has no non-FFT path, are registered only
# when the build enables FFT (ERF_ENABLE_FFT).
#=============================================================================
# (Python3 for the check scripts is found above the fire tests)

#=============================================================================
# Anelastic slow-scalar advection (Exec/RegTests/AnelasticScalarAdvection)
#
# A deck in Exec/RegTests/<CASE_DIR> runs and its Python check script, whose
# exit code is the verdict, reads the plotfile. With REFERENCE_INPUT a second
# deck runs first (plotfile prefix ref_plt) and the check script gets both
# plotfiles, the tested one first. The check table is echoed into the ctest
# output, and the tail of the run log when a run itself fails.
#=============================================================================
function(add_test_checked TEST_NAME CASE_DIR INPUT_FILE NSTEPS CHECK_SCRIPT)
    set(options )
    set(oneValueArgs "REFERENCE_INPUT")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_CHECKED "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(CURRENT_TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/Exec/RegTests/${CASE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(COPY ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS/erf_plotfile.py DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        set(NP ${ERF_TEST_NRANKS})
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    # plotfile names carry the step number padded to five digits
    set(_step "0000${NSTEPS}")
    string(LENGTH "${_step}" _len)
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_step}" ${_start} 5 _step)

    set(RUNTIME_OPTIONS "max_step=${NSTEPS} erf.plot_int_1=${NSTEPS} erf.check_int=-1")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(check_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.check.log")
    set(run_test "( ${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE} ${RUNTIME_OPTIONS} erf.plot_file_1=plt > ${test_log} 2>&1 || ( tail -n 60 ${test_log} && false ) )")
    set(check_args "${CURRENT_TEST_BINARY_DIR}/plt${_step}")
    if(NOT "${ADD_TEST_CHECKED_REFERENCE_INPUT}" STREQUAL "")
        set(ref_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.reference.log")
        set(run_ref "( ${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${ADD_TEST_CHECKED_REFERENCE_INPUT} ${RUNTIME_OPTIONS} erf.plot_file_1=ref_plt > ${ref_log} 2>&1 || ( tail -n 60 ${ref_log} && false ) )")
        set(run_test "${run_ref} && ${run_test}")
        set(check_args "${check_args} ${CURRENT_TEST_BINARY_DIR}/ref_plt${_step}")
    endif()
    # remove the plotfiles of an earlier pass first, so a run that ends before
    # NSTEPS cannot be checked against them
    set(test_command sh -c "rm -rf ${check_args} && ${run_test} && rm -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED && ( ${ERF_RANS_PYTHON} ${CURRENT_TEST_BINARY_DIR}/${CHECK_SCRIPT} ${check_args} > ${check_log} 2>&1 || touch ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED ) && cat ${check_log} && test ! -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 900
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression;anelastic"
        ATTACHED_FILES_ON_FAIL "${test_log};${check_log}"
    )
endfunction(add_test_checked)

# Map factors on a uniform mesh (MLMG projection): runs in every build
add_test_checked(AnelasticScalarAdvection_MapFactor AnelasticScalarAdvection inputs_mapfac_anelastic 60 check_mapfac_parity.py
                 REFERENCE_INPUT inputs_mapfac_compressible)
# Stretched and terrain-fitted meshes: their anelastic projection needs FFT
if(ERF_ENABLE_FFT)
    add_test_checked(AnelasticScalarAdvection_Fitted    AnelasticScalarAdvection inputs_fitted    50 check_scalar_centroid.py)
    add_test_checked(AnelasticScalarAdvection_Stretched AnelasticScalarAdvection inputs_stretched 50 check_scalar_centroid.py)
endif()

function(add_test_rans TEST_NAME CASE_DIR INPUT_FILE NSTEPS CHECK_SCRIPT)
    set(options )
    set(oneValueArgs "RUNTIME_OPTIONS" "NRANKS")
    set(multiValueArgs )
    cmake_parse_arguments(ADD_TEST_RANS "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    set(_rans_root ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS)
    set(CURRENT_TEST_SOURCE_DIR ${_rans_root}/${CASE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    # shared plotfile reader and check helpers used by every check script
    file(GLOB _rans_py "${_rans_root}/*.py")
    file(COPY ${_rans_py} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        if("${ADD_TEST_RANS_NRANKS}" STREQUAL "")
            set(NP ${ERF_TEST_NRANKS})
        else()
            set(NP ${ADD_TEST_RANS_NRANKS})
        endif()
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    # plotfile names carry the step number padded to five digits
    set(_step "0000${NSTEPS}")
    string(LENGTH "${_step}" _len)
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_step}" ${_start} 5 _step)
    set(PLTFILE "plt${_step}")

    set(RUNTIME_OPTIONS "max_step=${NSTEPS} erf.plot_int_1=${NSTEPS} erf.check_int=-1 ${ADD_TEST_RANS_RUNTIME_OPTIONS}")
    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(check_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.check.log")
    # The check script's exit code is the verdict; its table is echoed into
    # the ctest output so a failure shows the measured numbers, and the tail
    # of the run log is echoed when the executable itself exits non-zero.
    set(test_command sh -c "${MPI_COMMANDS} ${TEST_EXE} ${CURRENT_TEST_BINARY_DIR}/${INPUT_FILE} ${RUNTIME_OPTIONS} > ${test_log} 2>&1 || ( tail -n 60 ${test_log} && false ) && rm -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED && ( ${ERF_RANS_PYTHON} ${CURRENT_TEST_BINARY_DIR}/${CHECK_SCRIPT} --smoke ${CURRENT_TEST_BINARY_DIR}/${PLTFILE} > ${check_log} 2>&1 || touch ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED ) && cat ${check_log} && test ! -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "rans;regression"
        ATTACHED_FILES_ON_FAIL "${test_log};${check_log}"
    )
endfunction(add_test_rans)

# flat meshes: MLMG projection so the tests run in every build
add_test_rans(RANS_Neutral_ABL_Flat     Neutral_ABL_Flat     inputs_neutral     40  check_neutral.py    RUNTIME_OPTIONS "erf.use_fft=false")
add_test_rans(RANS_Neutral_ABL_Flat_Implicit Neutral_ABL_Flat  inputs_neutral     40  check_neutral.py    RUNTIME_OPTIONS "erf.use_fft=false erf.vert_implicit=true erf.anelastic_type=MidPoint erf.fixed_dt=10")
add_test_rans(RANS_Stable_ABL_Flat      Stable_ABL_Flat      inputs_stable      40  check_stable.py     RUNTIME_OPTIONS "erf.use_fft=false")
add_test_rans(RANS_Convective_ABL_Flat  Convective_ABL_Flat  inputs_convective  40  check_convective.py RUNTIME_OPTIONS "erf.use_fft=false")

# The checkers' own pass/fail logic. kind = "range" used to accept half a band
# width outside the band (erf-model/ERF#4027), so every band check was looser
# than it reads. Pure Python, no ERF run.
add_test(RANS_Checks_SelfTest ${ERF_RANS_PYTHON}
    ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS/test_rans_checks.py)
set_tests_properties(RANS_Checks_SelfTest
    PROPERTIES
    TIMEOUT 60
    PROCESSORS 1
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS"
    LABELS "rans;unit")

# Terrain-following inflow profiles on flat ground: one deck run with
# xlo.dirichlet_file and again with the equivalent xlo.inflow_profile file must
# give identical plotfiles, since the level-indexed lookup and the
# height-above-ground lookup coincide when the ground is on the floor.
function(add_test_inflow_profile_parity TEST_NAME CASE_DIR INPUT_FILE NSTEPS OPTIONS_DIRICHLET OPTIONS_PROFILE)
    set(_rans_root ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS)
    set(CURRENT_TEST_SOURCE_DIR ${_rans_root}/${CASE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    if(ERF_ENABLE_MPI)
        set(NP ${ERF_TEST_NRANKS})
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${MPIEXEC_PREFLAGS}")
    else()
        set(NP 1)
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(_step "0000${NSTEPS}")
    string(LENGTH "${_step}" _len)
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_step}" ${_start} 5 _step)

    set(_dir ${CURRENT_TEST_BINARY_DIR})
    set(_common "max_step=${NSTEPS} erf.plot_int_1=${NSTEPS} erf.check_int=-1")
    # the plotfiles of an earlier pass are removed first; a failing run or
    # comparison prints the tail of its log
    set(test_command sh -c "rm -rf ${_dir}/plt_dirichlet${_step} ${_dir}/plt_profile${_step} && ( ${MPI_COMMANDS} ${TEST_EXE} ${_dir}/${INPUT_FILE} ${_common} erf.plot_file_1=plt_dirichlet ${OPTIONS_DIRICHLET} > ${_dir}/dirichlet_file.log 2>&1 || ( tail -n 40 ${_dir}/dirichlet_file.log && false ) ) && ( ${MPI_COMMANDS} ${TEST_EXE} ${_dir}/${INPUT_FILE} ${_common} erf.plot_file_1=plt_profile ${OPTIONS_PROFILE} > ${_dir}/inflow_profile.log 2>&1 || ( tail -n 40 ${_dir}/inflow_profile.log && false ) ) && ( ${FCOMPARE_EXE} --abort_if_not_all_found -r 0.0 --abs_tol 0.0 ${_dir}/plt_dirichlet${_step} ${_dir}/plt_profile${_step} > ${_dir}/fcompare.log 2>&1 || ( cat ${_dir}/fcompare.log && false ) )")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS ${NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "rans;regression"
        ATTACHED_FILES_ON_FAIL "${_dir}/dirichlet_file.log;${_dir}/inflow_profile.log;${_dir}/fcompare.log"
    )
endfunction(add_test_inflow_profile_parity)

# erf.input_sounding_theta_above_ground over the incline (compressible, so in
# every build): theta must start above the local ground as on the inflow face,
# and the same deck with the flag off must not pass that check
add_test_rans(InflowProfile_ThetaAboveGround         Terrain_Inflow_Profile inputs_theta 4 check_theta_above_ground.py)
add_test_rans(InflowProfile_ThetaAboveGround_FlagOff Terrain_Inflow_Profile inputs_theta 4 check_theta_flag_off.py
              RUNTIME_OPTIONS "erf.input_sounding_theta_above_ground=false")

if(ERF_ENABLE_FFT)
    # terrain-fitted mesh (FFT-preconditioned projection): wall distance against
    # the exact ridge distance, and the same deck flattened (prob.hmax = 1e-6)
    # against the analytic height, each with the terrain_height and Poisson paths
    add_test_rans(RANS_Neutral_Hill_2D        Neutral_Hill_2D      inputs_hill        40  check_hill.py)
    add_test_rans(RANS_Neutral_Hill_2D_Poisson Neutral_Hill_2D     inputs_hill        40  check_hill.py RUNTIME_OPTIONS "erf.wall_dist_type=poisson")
    add_test_rans(RANS_Flat_Fitted_2D         Neutral_Hill_2D      inputs_hill        40  check_flat_fitted.py RUNTIME_OPTIONS "prob.hmax=1e-6")
    add_test_rans(RANS_Flat_Fitted_2D_Poisson Neutral_Hill_2D      inputs_hill        40  check_flat_fitted.py RUNTIME_OPTIONS "prob.hmax=1e-6 erf.wall_dist_type=poisson")
    add_test_rans(RANS_Neutral_Hill_3D        Neutral_Hill_3D      inputs_hill3d      40  check_hill3d.py)
    add_test_rans(RANS_Neutral_Hill_3D_Poisson Neutral_Hill_3D     inputs_hill3d      40  check_hill3d.py RUNTIME_OPTIONS "erf.wall_dist_type=poisson")
endif()

# Largest stable time step of one closure under explicit anelastic, implicit
# anelastic and implicit compressible integration (Timestep_Limits). The
# sweep spins the closure up and climbs a ladder of steps, about 30 short ERF
# runs and 60-90 s on one rank in Release, so it is labelled dt_sweep and kept
# out of the regression label that the CI runs in Debug.
function(add_test_rans_dt TEST_NAME CLOSURE)
    set(_rans_root ${PROJECT_SOURCE_DIR}/Exec/CanonicalTests/Canonical_RANS)
    set(CURRENT_TEST_SOURCE_DIR ${_rans_root}/Timestep_Limits)
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(GLOB _rans_py "${_rans_root}/*.py")
    file(COPY ${_rans_py} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    # a 4 x 4 x 200 column: one rank
    if(ERF_ENABLE_MPI)
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}")
    else()
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    # The sweep's exit code is the verdict; its tables are echoed into the
    # ctest output so a failure shows every rung.
    set(test_command sh -c "rm -f ${CURRENT_TEST_BINARY_DIR}/SWEEP_FAILED && ( ${ERF_RANS_PYTHON} ${CURRENT_TEST_BINARY_DIR}/sweep_dt.py --exe ${TEST_EXE} --closure ${CLOSURE} --mpi-cmd \"${MPI_COMMANDS}\" --workdir ${CURRENT_TEST_BINARY_DIR}/dt_runs > ${test_log} 2>&1 || touch ${CURRENT_TEST_BINARY_DIR}/SWEEP_FAILED ) && cat ${test_log} && test ! -f ${CURRENT_TEST_BINARY_DIR}/SWEEP_FAILED")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 3600
        PROCESSORS 1
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "rans;dt_sweep"
        ATTACHED_FILES_ON_FAIL "${test_log}"
    )
endfunction(add_test_rans_dt)

# The sweep's spin-up deck uses the FFT projection (the limits in its README
# were measured with it), so the entries exist only in FFT builds; the command
# runs through sh, so not on Windows.
if(ERF_ENABLE_FFT AND NOT WIN32)
    add_test_rans_dt(RANS_Timestep_Limits_kEqn      kEqn)
    add_test_rans_dt(RANS_Timestep_Limits_Deardorff Deardorff)
    add_test_rans_dt(RANS_Timestep_Limits_MRF       MRF)
endif()

#=============================================================================
# MOST reference height on flat stretched meshes
#
# run_most_zref.py runs one flat stretched column through the terrain-fitted,
# interpolated and no-terrain MOST lookups plus a uniform 10 m column, and
# checks u* against the log law at the reported reference height and the
# stretched column against the uniform one.  Its exit code is the verdict.
#=============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(ERF_MOST_ZREF_PYTHON "${Python3_EXECUTABLE}")
else()
    set(ERF_MOST_ZREF_PYTHON "python3")
endif()

function(add_test_most_zref TEST_NAME)
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${TEST_NAME})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")

    # 4x4 columns: one rank
    if(ERF_ENABLE_MPI)
        set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}")
    else()
        unset(MPI_COMMANDS)
    endif()

    resolve_test_exe("" "erf_exec" TEST_EXE)

    set(test_log "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    set(test_command sh -c "rm -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED && ( ${ERF_MOST_ZREF_PYTHON} ${CURRENT_TEST_BINARY_DIR}/run_most_zref.py --exe ${TEST_EXE} --mpi-cmd \"${MPI_COMMANDS}\" --workdir ${CURRENT_TEST_BINARY_DIR}/runs > ${test_log} 2>&1 || touch ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED ) && cat ${test_log} && test ! -f ${CURRENT_TEST_BINARY_DIR}/CHECK_FAILED")

    add_test(${TEST_NAME} ${test_command})
    set_tests_properties(${TEST_NAME}
        PROPERTIES
        TIMEOUT 1800
        PROCESSORS 1
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${test_log}"
    )
endfunction(add_test_most_zref)

add_test_most_zref(MOST_Zref_Stretched)

# Immersed-boundary surface energy balance on the faces of a height-map cube
# (prognostic skin, slab conduction, heat flux into the air), 40 steps.
add_test_r(IBSEB_Cube                        ""  "erf_exec" "plt00040")
add_test_r(PBL_IBAware_MRF_Smoothing         ""  "erf_exec" "plt00010")

#=============================================================================
# Performance tests
#=============================================================================
