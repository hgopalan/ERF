# Run one moving-bodies deck twice, with OPTIONS_A and with OPTIONS_B (space-separated ERF inputs,
# after COMMON_OPTIONS), and require every file in FILES (space-separated, relative to the run
# directory) to exist after both runs and to hold the same bytes. For settings that change only how
# the files are written (erf.moving_bodies.max_open_logs, stats_int), never what they hold at the end.
# EXPECT_B, when given, is a regular expression the second run's log must match (to show that
# OPTIONS_B reached the run).
# -DX= defines X as empty, so test for a value, not for DEFINED
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")

foreach(arg NRANKS TEST_EXE INPUT WORKING_DIRECTORY RUN_TIMEOUT OPTIONS_B FILES)
    if("${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunSameLogs.cmake: ${arg} must be given and non-empty")
    endif()
endforeach()
if(NOT "${MPIEXEC}" STREQUAL "" AND "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
    message(FATAL_ERROR "RunSameLogs.cmake: MPIEXEC_NUMPROC_FLAG must be given with MPIEXEC")
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunSameLogs.cmake: ERF executable")
separate_arguments(common_options UNIX_COMMAND "${COMMON_OPTIONS}")
separate_arguments(options_a UNIX_COMMAND "${OPTIONS_A}")
separate_arguments(options_b UNIX_COMMAND "${OPTIONS_B}")
separate_arguments(files UNIX_COMMAND "${FILES}")

set(A_DIR "${WORKING_DIRECTORY}/run_a")
set(B_DIR "${WORKING_DIRECTORY}/run_b")
file(REMOVE_RECURSE "${A_DIR}" "${B_DIR}")
file(MAKE_DIRECTORY "${A_DIR}" "${B_DIR}")

# Stage the deck's auxiliary inputs (turbine files, ...) into each run directory, since the runs
# read them relative to their own working directory
file(GLOB _sl_aux LIST_DIRECTORIES false "${WORKING_DIRECTORY}/*")
list(FILTER _sl_aux EXCLUDE REGEX "\\.(i|log|log\\..*|txt)$")
foreach(_sl_f IN LISTS _sl_aux)
    file(COPY "${_sl_f}" DESTINATION "${A_DIR}")
    file(COPY "${_sl_f}" DESTINATION "${B_DIR}")
endforeach()

erf_mpi_launcher_command(launch
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunSameLogs.cmake")

function(run_leg dir)
    execute_process(
        COMMAND ${launch} ${TEST_EXE} ${INPUT} ${common_options} ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_FILE "${dir}/simulation.log"
        ERROR_FILE "${dir}/simulation.log"
        TIMEOUT ${RUN_TIMEOUT}
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "RunSameLogs.cmake: the run in ${dir} failed or exceeded ${RUN_TIMEOUT} s: ${_result}")
    endif()
endfunction()

run_leg("${A_DIR}" ${options_a})
run_leg("${B_DIR}" ${options_b})
if(NOT "${EXPECT_B}" STREQUAL "")
    file(READ "${B_DIR}/simulation.log" _log_b)
    if(NOT _log_b MATCHES "${EXPECT_B}")
        message(FATAL_ERROR "RunSameLogs.cmake: the run with '${OPTIONS_B}' does not print '${EXPECT_B}'")
    endif()
endif()

set(_differ "")
foreach(_f IN LISTS files)
    foreach(_d "${A_DIR}" "${B_DIR}")
        if(NOT EXISTS "${_d}/${_f}")
            message(FATAL_ERROR "RunSameLogs.cmake: ${_d}/${_f} was not written")
        endif()
    endforeach()
    file(SHA256 "${A_DIR}/${_f}" _ha)
    file(SHA256 "${B_DIR}/${_f}" _hb)
    if(NOT _ha STREQUAL _hb)
        list(APPEND _differ "${_f}")
    endif()
endforeach()
if(_differ)
    message(FATAL_ERROR "RunSameLogs.cmake: with '${OPTIONS_B}' these files differ from the run with '${OPTIONS_A}': ${_differ}")
endif()
list(LENGTH files _n)
message(STATUS "RunSameLogs.cmake: ${_n} file(s) identical with '${OPTIONS_A}' and '${OPTIONS_B}'")
