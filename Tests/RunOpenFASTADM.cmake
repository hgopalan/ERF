# Run an OpenFAST turbine deck (OpenFAST_ADM_Uniform, OpenFAST_ALM_Uniform and their variants)
# and check it three ways: the plotfile against its gold with fcompare; a log of the turbine
# (FLOW_CSV: by default <output_root>_flow.csv, the sampled hub and blade-mean velocities,
# which the rotor's induction lowers) against the committed gold log; and, row by row, that
# the integrated momentum source (fx in SOURCE_CSV) equals minus the turbine's thrust
# (thrust_x in TURBINE_CSV, or the FORCE_COLUMN given: load_x, the sum of thrust, tower and
# nacelle forces, for a turbine with a forced tower and nacelle): the disk rings and the line
# points preserve the rotor's force and the spreading is normalised exactly, so the two must
# agree to roundoff. The turbine log's first row is the
# initial solution before any step; the source log's first row belongs to the state after the
# first step, so turbine row r + 1 pairs with source row r.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, FCOMPARE, PLTFILE, PLOT_GOLD, RTOL, ATOL, FLOW_CSV, FLOW_GOLD, SIGDIGITS,
# TURBINE_CSV, SOURCE_CSV, and optionally FORCE_COLUMN (default thrust_x) and LOG_COLUMNS (the
# columns of FLOW_CSV compared with the gold, a ;-list; default all). FORCE_COLUMN = none is for a
# turbine that adds no forcing (mode = none): SOURCE_CSV must then not be written, and thrust_x in
# TURBINE_CSV must be non-zero in every row, so the turbine was stepped and carried a load.

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
if(NOT DEFINED FORCE_COLUMN OR "${FORCE_COLUMN}" STREQUAL "")
    set(FORCE_COLUMN "thrust_x")
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
file(REMOVE_RECURSE "${WORKING_DIRECTORY}/moving_bodies")
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
set(_required ${FLOW_CSV} ${TURBINE_CSV})
if(NOT FORCE_COLUMN STREQUAL "none")
    list(APPEND _required ${SOURCE_CSV})
endif()
foreach(f ${_required})
    if(NOT EXISTS "${WORKING_DIRECTORY}/${f}")
        message(FATAL_ERROR "RunOpenFASTADM.cmake: the run wrote no ${f}")
    endif()
endforeach()
# Keep only LOG_COLUMNS of a comma-separated table, written whitespace-separated to out_file
function(select_columns in_file out_file)
    file(STRINGS "${in_file}" rows)
    list(GET rows 0 header)
    string(REPLACE "," ";" names "${header}")
    set(idx "")
    foreach(c IN LISTS LOG_COLUMNS)
        list(FIND names "${c}" i)
        if(i LESS 0)
            message(FATAL_ERROR "RunOpenFASTADM.cmake: ${in_file} has no column ${c}: ${header}")
        endif()
        list(APPEND idx ${i})
    endforeach()
    set(text "")
    foreach(row IN LISTS rows)
        string(REPLACE "," ";" fields "${row}")
        set(kept "")
        foreach(i IN LISTS idx)
            list(GET fields ${i} v)
            list(APPEND kept "${v}")
        endforeach()
        string(JOIN " " line ${kept})
        string(APPEND text "${line}\n")
    endforeach()
    file(WRITE "${out_file}" "${text}")
endfunction()
set(_gold "${FLOW_GOLD}")
set(_run "${WORKING_DIRECTORY}/${FLOW_CSV}")
if(NOT "${LOG_COLUMNS}" STREQUAL "")
    select_columns("${FLOW_GOLD}" "${WORKING_DIRECTORY}/log_gold_columns.txt")
    select_columns("${WORKING_DIRECTORY}/${FLOW_CSV}" "${WORKING_DIRECTORY}/log_run_columns.txt")
    set(_gold "${WORKING_DIRECTORY}/log_gold_columns.txt")
    set(_run "${WORKING_DIRECTORY}/log_run_columns.txt")
endif()
erf_compare_data_logs("${_gold}" "${_run}" ${SIGDIGITS} 2 logs_agree log_message)
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
if(FORCE_COLUMN STREQUAL "none")
    if(EXISTS "${WORKING_DIRECTORY}/${SOURCE_CSV}")
        message(FATAL_ERROR "RunOpenFASTADM.cmake: ${SOURCE_CSV} was written, but this turbine adds no forcing")
    endif()
    column_index("${turb_rows}" "thrust_x" it)
    list(LENGTH turb_rows nturb)
    math(EXPR nlast "${nturb} - 1")
    if(nlast LESS 2)
        message(FATAL_ERROR "RunOpenFASTADM.cmake: ${TURBINE_CSV} has only ${nlast} rows; the check would be trivial")
    endif()
    foreach(r RANGE 1 ${nlast})
        list(GET turb_rows ${r} trow)
        string(REPLACE "," ";" tfields "${trow}")
        list(GET tfields ${it} thrust)
        erf_read_decimal("${thrust}" ok sign digits exp)
        if(NOT ok OR "${digits}" STREQUAL "0")
            message(FATAL_ERROR "RunOpenFASTADM.cmake: row ${r}: thrust_x = '${thrust}' is zero or not a number; the turbine carried no load")
        endif()
    endforeach()
    message(STATUS "RunOpenFASTADM.cmake: plotfile matches the gold, ${FLOW_CSV} matches its gold, no momentum source was written and the turbine carried a load in ${nlast} rows")
    return()
endif()
file(STRINGS "${WORKING_DIRECTORY}/${SOURCE_CSV}" src_rows)
column_index("${turb_rows}" "${FORCE_COLUMN}" it)
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
        message(FATAL_ERROR "RunOpenFASTADM.cmake: row ${r}: ${FORCE_COLUMN} = '${thrust}' is zero or not a number; the bodies carried no load")
    endif()
    if("${sign}" STREQUAL "-")
        set(minus_thrust "${digits}e${exp}")
    else()
        set(minus_thrust "-${digits}e${exp}")
    endif()
    erf_numbers_close("${minus_thrust}" "${fx}" 8 2 close)
    if(NOT close)
        message(FATAL_ERROR "RunOpenFASTADM.cmake: row ${r}: fx = ${fx} differs from -${FORCE_COLUMN} = ${minus_thrust}; the spread forces do not integrate back to the bodies' loads")
    endif()
endforeach()
message(STATUS "RunOpenFASTADM.cmake: plotfile matches the gold, ${FLOW_CSV} matches its gold, and fx equals -${FORCE_COLUMN} in ${nsrc_data} rows")
