# Run the OpenFAST_ADM_Uniform deck and check it three ways: the plotfile against its gold with
# fcompare; a log of the turbine (FLOW_CSV: by default <output_root>_flow.csv, the sampled hub and blade-mean
# velocities, which the disk's induction lowers) against the committed gold log; and, row by
# row, that the integrated momentum source (fx in SOURCE_CSV) equals minus the turbine's thrust
# (thrust_x in TURBINE_CSV): the rings preserve the rotor's force and the spreading is
# normalised exactly, so the two must agree to roundoff. The turbine log's first row is the
# initial solution before any step; the source log's first row belongs to the state after the
# first step, so turbine row r + 1 pairs with source row r.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, FCOMPARE, PLTFILE, PLOT_GOLD, RTOL, ATOL, FLOW_CSV, FLOW_GOLD, SIGDIGITS,
# TURBINE_CSV, SOURCE_CSV.

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY FCOMPARE PLTFILE PLOT_GOLD FLOW_CSV FLOW_GOLD TURBINE_CSV SOURCE_CSV)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunOpenFASTADM.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()
if(NOT DEFINED SIGDIGITS OR "${SIGDIGITS}" STREQUAL "")
    set(SIGDIGITS 10)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunOpenFASTADM.cmake: ERF executable")
erf_resolve_executable(FCOMPARE "${FCOMPARE}" CONFIG "${CONFIG}"
    CONTEXT "RunOpenFASTADM.cmake: fcompare")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunOpenFASTADM.cmake")
erf_mpi_launcher_command(launch_one
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunOpenFASTADM.cmake")

set(log "${WORKING_DIRECTORY}/simulation.log")
foreach(f ${FLOW_CSV} ${TURBINE_CSV} ${SOURCE_CSV})
    file(REMOVE "${WORKING_DIRECTORY}/${f}")
endforeach()
file(REMOVE_RECURSE "${WORKING_DIRECTORY}/${PLTFILE}")
execute_process(
    COMMAND ${launcher} ${TEST_EXE} ${INPUT} amrex.call_addr2line=0
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${log}"
    ERROR_FILE "${log}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    file(READ "${log}" contents)
    message(FATAL_ERROR "RunOpenFASTADM.cmake: the simulation failed (${result}):\n${contents}")
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
    message(FATAL_ERROR "RunOpenFASTADM.cmake: fcompare failed (${fc_result}):\n${fc_out}")
endif()

# ---- the flow log against its gold ----
foreach(f ${FLOW_CSV} ${TURBINE_CSV} ${SOURCE_CSV})
    if(NOT EXISTS "${WORKING_DIRECTORY}/${f}")
        message(FATAL_ERROR "RunOpenFASTADM.cmake: the run wrote no ${f}")
    endif()
endforeach()
erf_compare_data_logs("${FLOW_GOLD}" "${WORKING_DIRECTORY}/${FLOW_CSV}" ${SIGDIGITS} 2 logs_agree log_message)
if(NOT logs_agree)
    file(READ "${WORKING_DIRECTORY}/${FLOW_CSV}" contents)
    message(FATAL_ERROR "RunOpenFASTADM.cmake: ${FLOW_CSV} differs from the gold log: ${log_message}\n${contents}")
endif()

# ---- integrated source equals minus the thrust, every row ----
function(column_index csv_rows name out_var)
    list(GET csv_rows 0 header)
    string(REPLACE "," ";" fields "${header}")
    list(FIND fields "${name}" idx)
    if(idx LESS 0)
        message(FATAL_ERROR "RunOpenFASTADM.cmake: header lacks ${name}: ${header}")
    endif()
    set(${out_var} ${idx} PARENT_SCOPE)
endfunction()
file(STRINGS "${WORKING_DIRECTORY}/${TURBINE_CSV}" turb_rows)
file(STRINGS "${WORKING_DIRECTORY}/${SOURCE_CSV}" src_rows)
column_index("${turb_rows}" "thrust_x" it)
column_index("${src_rows}" "fx" is)
list(LENGTH turb_rows nturb)
list(LENGTH src_rows nsrc)
math(EXPR nturb_data "${nturb} - 2")   # header and the initial-solution row
math(EXPR nsrc_data "${nsrc} - 1")
if(NOT nturb_data EQUAL nsrc_data)
    message(FATAL_ERROR "RunOpenFASTADM.cmake: ${TURBINE_CSV} has ${nturb_data} stepped rows but ${SOURCE_CSV} has ${nsrc_data}")
endif()
if(nsrc_data LESS 2)
    message(FATAL_ERROR "RunOpenFASTADM.cmake: only ${nsrc_data} rows; the check would be trivial")
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
        message(FATAL_ERROR "RunOpenFASTADM.cmake: row ${r}: thrust_x = '${thrust}' is zero or not a number; the rotor carried no load")
    endif()
    if("${sign}" STREQUAL "-")
        set(minus_thrust "${digits}e${exp}")
    else()
        set(minus_thrust "-${digits}e${exp}")
    endif()
    erf_numbers_close("${minus_thrust}" "${fx}" 8 2 close)
    if(NOT close)
        message(FATAL_ERROR "RunOpenFASTADM.cmake: row ${r}: fx = ${fx} differs from -thrust_x = ${minus_thrust}; the spread rotor force does not integrate back to the thrust")
    endif()
endforeach()
message(STATUS "RunOpenFASTADM.cmake: plotfile matches the gold, ${FLOW_CSV} matches its gold, and fx equals -thrust_x in ${nsrc_data} rows")
