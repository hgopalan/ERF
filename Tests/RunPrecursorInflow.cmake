# Run a precursor deck that writes boundary planes, then the main deck that reads them at its
# inflow, and check the main run three ways: the plotfile against its gold with fcompare, the
# turbine's statistics file (STATS_CSV) against its gold, and, row by row, that the integrated
# momentum source (fx in SOURCE_CSV) equals minus the turbine's thrust (thrust_x in
# TURBINE_CSV); turbine row r + 1 pairs with source row r as in RunOpenFASTADM.cmake.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG,
# PRECURSOR_INPUT, INPUT, WORKING_DIRECTORY, FCOMPARE, PLTFILE, PLOT_GOLD, RTOL, ATOL,
# STATS_CSV, STATS_GOLD, SIGDIGITS, TURBINE_CSV, SOURCE_CSV, BNDRY_DIR.

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg TEST_EXE PRECURSOR_INPUT INPUT WORKING_DIRECTORY FCOMPARE PLTFILE PLOT_GOLD STATS_CSV STATS_GOLD TURBINE_CSV SOURCE_CSV BNDRY_DIR)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunPrecursorInflow.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()
if(NOT DEFINED SIGDIGITS OR "${SIGDIGITS}" STREQUAL "")
    set(SIGDIGITS 10)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunPrecursorInflow.cmake: ERF executable")
erf_resolve_executable(FCOMPARE "${FCOMPARE}" CONFIG "${CONFIG}"
    CONTEXT "RunPrecursorInflow.cmake: fcompare")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunPrecursorInflow.cmake")
erf_mpi_launcher_command(launch_one
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunPrecursorInflow.cmake")

function(run_erf input log)
    execute_process(
        COMMAND ${launcher} ${TEST_EXE} ${input} amrex.call_addr2line=0
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${WORKING_DIRECTORY}/${log}"
        ERROR_FILE "${WORKING_DIRECTORY}/${log}"
        RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        file(READ "${WORKING_DIRECTORY}/${log}" contents)
        message(FATAL_ERROR "RunPrecursorInflow.cmake: the run of ${input} failed (${result}):\n${contents}")
    endif()
endfunction()

# ---- the precursor writes the planes ----
file(REMOVE_RECURSE "${WORKING_DIRECTORY}/${BNDRY_DIR}" "${WORKING_DIRECTORY}/${PLTFILE}")
foreach(f ${STATS_CSV} ${TURBINE_CSV} ${SOURCE_CSV})
    file(REMOVE "${WORKING_DIRECTORY}/${f}")
endforeach()
run_erf("${PRECURSOR_INPUT}" "precursor.log")
if(NOT EXISTS "${WORKING_DIRECTORY}/${BNDRY_DIR}/time.dat")
    message(FATAL_ERROR "RunPrecursorInflow.cmake: the precursor wrote no ${BNDRY_DIR}/time.dat")
endif()

# ---- the main run reads them ----
run_erf("${INPUT}" "simulation.log")

execute_process(
    COMMAND ${launch_one} ${FCOMPARE} --abort_if_not_all_found -a -r ${RTOL} --abs_tol ${ATOL}
            ${PLOT_GOLD} ${WORKING_DIRECTORY}/${PLTFILE}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE fc_out
    ERROR_VARIABLE fc_out
    RESULT_VARIABLE fc_result)
if(NOT fc_result EQUAL 0)
    message(FATAL_ERROR "RunPrecursorInflow.cmake: fcompare failed (${fc_result}):\n${fc_out}")
endif()

foreach(f ${STATS_CSV} ${TURBINE_CSV} ${SOURCE_CSV})
    if(NOT EXISTS "${WORKING_DIRECTORY}/${f}")
        message(FATAL_ERROR "RunPrecursorInflow.cmake: the run wrote no ${f}")
    endif()
endforeach()
erf_compare_data_logs("${STATS_GOLD}" "${WORKING_DIRECTORY}/${STATS_CSV}" ${SIGDIGITS} 2 logs_agree log_message)
if(NOT logs_agree)
    file(READ "${WORKING_DIRECTORY}/${STATS_CSV}" contents)
    message(FATAL_ERROR "RunPrecursorInflow.cmake: ${STATS_CSV} differs from the gold: ${log_message}\n${contents}")
endif()

# ---- integrated source equals minus the thrust, every row ----
function(column_index csv_rows name out_var)
    list(GET csv_rows 0 header)
    string(REPLACE "," ";" fields "${header}")
    list(FIND fields "${name}" idx)
    if(idx LESS 0)
        message(FATAL_ERROR "RunPrecursorInflow.cmake: header lacks ${name}: ${header}")
    endif()
    set(${out_var} ${idx} PARENT_SCOPE)
endfunction()
file(STRINGS "${WORKING_DIRECTORY}/${TURBINE_CSV}" turb_rows)
file(STRINGS "${WORKING_DIRECTORY}/${SOURCE_CSV}" src_rows)
column_index("${turb_rows}" "thrust_x" it)
column_index("${src_rows}" "fx" is)
list(LENGTH turb_rows nturb)
list(LENGTH src_rows nsrc)
math(EXPR nturb_data "${nturb} - 2")
math(EXPR nsrc_data "${nsrc} - 1")
if(NOT nturb_data EQUAL nsrc_data)
    message(FATAL_ERROR "RunPrecursorInflow.cmake: ${TURBINE_CSV} has ${nturb_data} stepped rows but ${SOURCE_CSV} has ${nsrc_data}")
endif()
if(nsrc_data LESS 2)
    message(FATAL_ERROR "RunPrecursorInflow.cmake: only ${nsrc_data} rows; the check would be trivial")
endif()
foreach(r RANGE 1 ${nsrc_data})
    math(EXPR rt "${r} + 1")
    list(GET turb_rows ${rt} trow)
    list(GET src_rows ${r} srow)
    string(REPLACE "," ";" tfields "${trow}")
    string(REPLACE "," ";" sfields "${srow}")
    list(GET tfields ${it} thrust)
    list(GET sfields ${is} fx)
    erf_read_decimal("${thrust}" ok sign digits exp)
    if(NOT ok OR "${digits}" STREQUAL "0")
        message(FATAL_ERROR "RunPrecursorInflow.cmake: row ${r}: thrust_x = '${thrust}' is zero or not a number")
    endif()
    if("${sign}" STREQUAL "-")
        set(minus_thrust "${digits}e${exp}")
    else()
        set(minus_thrust "-${digits}e${exp}")
    endif()
    erf_numbers_close("${minus_thrust}" "${fx}" 8 2 close)
    if(NOT close)
        message(FATAL_ERROR "RunPrecursorInflow.cmake: row ${r}: fx = ${fx} differs from -thrust_x = ${minus_thrust}")
    endif()
endforeach()
message(STATUS "RunPrecursorInflow.cmake: plotfile matches the gold, ${STATS_CSV} matches its gold, and fx equals -thrust_x in ${nsrc_data} rows")
