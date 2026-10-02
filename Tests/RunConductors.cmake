# Run a conductor-span deck and check it two ways: the plotfile against its gold with fcompare,
# and the span's log (<output_root>.dat, written every step) against the committed gold log,
# field by field to SIGDIGITS significant digits.
#
# With TOTALS (a file such as conductors/total_load.dat, columns time, drag_x..z, force_on_air_x..z,
# source_x..z), every row must also have the integrated momentum source equal to the force the lines
# put into the air, to SIGDIGITS digits: the spreading's normalisation is exact.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, FCOMPARE, PLTFILE, PLOT_GOLD, RTOL, ATOL, LOG, GOLD, SIGDIGITS, TOTALS.

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY FCOMPARE PLTFILE PLOT_GOLD LOG GOLD)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunConductors.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()
if(NOT DEFINED SIGDIGITS OR "${SIGDIGITS}" STREQUAL "")
    set(SIGDIGITS 8)
endif()
if(NOT DEFINED RTOL OR "${RTOL}" STREQUAL "")
    set(RTOL 2.0e-10)
endif()
if(NOT DEFINED ATOL OR "${ATOL}" STREQUAL "")
    set(ATOL 2.0e-10)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}" CONTEXT "RunConductors.cmake: ERF executable")
erf_resolve_executable(FCOMPARE "${FCOMPARE}" CONFIG "${CONFIG}" CONTEXT "RunConductors.cmake: fcompare")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}" NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}" CONTEXT "RunConductors.cmake")
erf_mpi_launcher_command(launch_one
    LAUNCHER "${MPIEXEC}" NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}" CONTEXT "RunConductors.cmake")

set(log "${WORKING_DIRECTORY}/simulation.log")
file(REMOVE "${WORKING_DIRECTORY}/${LOG}")
file(REMOVE_RECURSE "${WORKING_DIRECTORY}/${PLTFILE}")
execute_process(
    COMMAND ${launcher} ${TEST_EXE} ${INPUT} amrex.call_addr2line=0
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${log}"
    ERROR_FILE "${log}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    file(READ "${log}" contents)
    message(FATAL_ERROR "RunConductors.cmake: the simulation failed (${result}):\n${contents}")
endif()

# ---- plotfile against the gold ----
execute_process(
    COMMAND ${launch_one} ${FCOMPARE} --abort_if_not_all_found -a -r ${RTOL} --abs_tol ${ATOL}
            ${PLOT_GOLD} ${WORKING_DIRECTORY}/${PLTFILE}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE fc_out ERROR_VARIABLE fc_out
    RESULT_VARIABLE fc_result)
if(NOT fc_result EQUAL 0)
    message(FATAL_ERROR "RunConductors.cmake: the plotfile differs from its gold (${fc_result}):\n${fc_out}")
endif()

# ---- the span's log against the gold log ----
if(NOT EXISTS "${WORKING_DIRECTORY}/${LOG}")
    message(FATAL_ERROR "RunConductors.cmake: the run wrote no ${LOG}")
endif()
erf_compare_data_logs("${WORKING_DIRECTORY}/${LOG}" "${GOLD}" ${SIGDIGITS} 2 logs_agree log_message)
if(NOT logs_agree)
    message(FATAL_ERROR "RunConductors.cmake: ${LOG} differs from its gold: ${log_message}")
endif()
file(STRINGS "${WORKING_DIRECTORY}/${LOG}" rows)
list(LENGTH rows nrows)
message(STATUS "RunConductors: ${LOG} agrees with its gold (${nrows} rows), plotfile agrees")

# ---- the spread source integrates to the force on the air ----
if(DEFINED TOTALS AND NOT "${TOTALS}" STREQUAL "")
    if(NOT EXISTS "${WORKING_DIRECTORY}/${TOTALS}")
        message(FATAL_ERROR "RunConductors.cmake: the run wrote no ${TOTALS}")
    endif()
    file(STRINGS "${WORKING_DIRECTORY}/${TOTALS}" total_rows)
    list(LENGTH total_rows ntotal)
    if(ntotal LESS 2)
        message(FATAL_ERROR "RunConductors.cmake: ${TOTALS} holds no data rows")
    endif()
    set(nonzero FALSE)
    foreach(row IN LISTS total_rows)
        if(row MATCHES "^time")
            continue()
        endif()
        string(REGEX MATCHALL "[^ \t]+" fields "${row}")
        list(LENGTH fields nfields)
        if(NOT nfields EQUAL 10)
            message(FATAL_ERROR "RunConductors.cmake: ${TOTALS} row '${row}' does not have 10 columns")
        endif()
        foreach(d 0 1 2)
            math(EXPR ia "4 + ${d}")
            math(EXPR ib "7 + ${d}")
            list(GET fields ${ia} force)
            list(GET fields ${ib} source)
            erf_numbers_close("${force}" "${source}" ${SIGDIGITS} 2 close)
            if(NOT close)
                message(FATAL_ERROR "RunConductors.cmake: ${TOTALS}: the integrated source ${source} N differs from the "
                                    "force on the air ${force} N in row '${row}'")
            endif()
            if(NOT "${force}" STREQUAL "0")
                set(nonzero TRUE)
            endif()
        endforeach()
    endforeach()
    if(NOT nonzero)
        message(FATAL_ERROR "RunConductors.cmake: ${TOTALS}: every force on the air is zero; nothing was checked")
    endif()
    message(STATUS "RunConductors: the integrated source equals the force on the air in every row of ${TOTALS}")
endif()
