# Run the Actuator_UniformCtDisk deck and check it three ways: the plotfile against its gold
# with fcompare, the disk's log (<output_root>_disk.csv) against the committed gold log, and,
# in every row, that the column COL_B (the integrated momentum source, projected on the disk
# normal) equals the column COL_A (the disk's thrust): the discrete normalisation of the force
# spreading is exact, so the two must agree to roundoff whatever the mesh.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, FCOMPARE, PLTFILE, PLOT_GOLD, RTOL, ATOL, CSV, GOLD, SIGDIGITS, COL_A, COL_B.

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY FCOMPARE PLTFILE PLOT_GOLD CSV GOLD COL_A COL_B)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunCtDisk.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()
if(NOT DEFINED SIGDIGITS OR "${SIGDIGITS}" STREQUAL "")
    set(SIGDIGITS 10)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunCtDisk.cmake: ERF executable")
erf_resolve_executable(FCOMPARE "${FCOMPARE}" CONFIG "${CONFIG}"
    CONTEXT "RunCtDisk.cmake: fcompare")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunCtDisk.cmake")
erf_mpi_launcher_command(launch_one
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunCtDisk.cmake")

set(log "${WORKING_DIRECTORY}/simulation.log")
file(REMOVE "${WORKING_DIRECTORY}/${CSV}")
file(REMOVE_RECURSE "${WORKING_DIRECTORY}/${PLTFILE}")
execute_process(
    COMMAND ${launcher} ${TEST_EXE} ${INPUT} amrex.call_addr2line=0
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${log}"
    ERROR_FILE "${log}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    file(READ "${log}" contents)
    message(FATAL_ERROR "RunCtDisk.cmake: the simulation failed (${result}):\n${contents}")
endif()

# ---- plotfile against the gold ----
execute_process(
    COMMAND ${launch_one} ${FCOMPARE} --abort_if_not_all_found -a -r ${RTOL} --abs_tol ${ATOL}
            ${PLOT_GOLD} ${WORKING_DIRECTORY}/${PLTFILE}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE fc_out
    ERROR_VARIABLE fc_out
    RESULT_VARIABLE fc_result)
if(NOT fc_result EQUAL 0)
    message(FATAL_ERROR "RunCtDisk.cmake: fcompare failed (${fc_result}):\n${fc_out}")
endif()

# ---- the disk log against its gold ----
if(NOT EXISTS "${WORKING_DIRECTORY}/${CSV}")
    message(FATAL_ERROR "RunCtDisk.cmake: the run wrote no ${CSV}")
endif()
erf_compare_data_logs("${GOLD}" "${WORKING_DIRECTORY}/${CSV}" ${SIGDIGITS} 2 logs_agree log_message)
if(NOT logs_agree)
    file(READ "${WORKING_DIRECTORY}/${CSV}" contents)
    message(FATAL_ERROR "RunCtDisk.cmake: ${CSV} differs from the gold log: ${log_message}\n${contents}")
endif()

# ---- integrated source equals the thrust, every row ----
file(STRINGS "${WORKING_DIRECTORY}/${CSV}" rows)
list(GET rows 0 header)
string(REPLACE "," ";" header_fields "${header}")
list(FIND header_fields "${COL_A}" ia)
list(FIND header_fields "${COL_B}" ib)
if(ia LESS 0 OR ib LESS 0)
    message(FATAL_ERROR "RunCtDisk.cmake: ${CSV} header lacks ${COL_A} or ${COL_B}: ${header}")
endif()
list(LENGTH rows nrows)
if(nrows LESS 3)
    message(FATAL_ERROR "RunCtDisk.cmake: ${CSV} has only ${nrows} lines; the check would be trivial")
endif()
math(EXPR last "${nrows} - 1")
foreach(r RANGE 1 ${last})
    list(GET rows ${r} row)
    string(REPLACE "," ";" fields "${row}")
    list(GET fields ${ia} a)
    list(GET fields ${ib} b)
    erf_read_decimal("${a}" ok sign digits exp)
    if(NOT ok OR "${digits}" STREQUAL "0")
        message(FATAL_ERROR "RunCtDisk.cmake: row ${r}: ${COL_A} = '${a}' is zero or not a number; the disk carried no force")
    endif()
    erf_numbers_close("${a}" "${b}" 8 2 close)
    if(NOT close)
        message(FATAL_ERROR "RunCtDisk.cmake: row ${r}: ${COL_B} = ${b} differs from ${COL_A} = ${a}; the spread force does not integrate back to the thrust")
    endif()
endforeach()
message(STATUS "RunCtDisk.cmake: plotfile matches the gold, ${CSV} matches its gold, and ${COL_B} equals ${COL_A} in ${last} rows")
