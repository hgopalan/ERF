# Run one moving-bodies deck twice, once with the bodies' logs in the per_turbine format (one file
# per body and log, <output_root>_<log>.csv) and once in FORMAT (erf.moving_bodies.
# diagnostics_format = per_rank or farm), and require the second to hold the same rows: the rows
# of each body, picked out of the rank or farm files by their body column and with that column
# removed, must equal the body's own file, and the run's own files (EXTRA_LOGS, e.g.
# moving_bodies/total_load.csv) must be the same in both runs.
#
# LOGS lists body/log pairs, e.g. "T1/erf T1/flow T2/stats": the per_turbine file is
# <body>_<log>.csv in the run directory (the decks set output_root to the body's name); the FORMAT
# files are <DIAG_DIR>/farm_<log>.csv or <DIAG_DIR>/rank<r>_<log>.csv. The body column is the
# second of a time series (time,body,...) and the first of the statistics (body,...).
# -DX= defines X as empty, so test for a value, not for DEFINED
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg NRANKS TEST_EXE INPUT WORKING_DIRECTORY RUN_TIMEOUT FORMAT LOGS DIAG_DIR)
    if("${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: ${arg} must be given and non-empty")
    endif()
endforeach()
if(NOT FORMAT STREQUAL "per_rank" AND NOT FORMAT STREQUAL "farm")
    message(FATAL_ERROR "RunDiagnosticsFormat.cmake: FORMAT must be per_rank or farm, not '${FORMAT}'")
endif()
if(NOT "${MPIEXEC}" STREQUAL "" AND "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
    message(FATAL_ERROR "RunDiagnosticsFormat.cmake: MPIEXEC_NUMPROC_FLAG must be given with MPIEXEC")
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunDiagnosticsFormat.cmake: ERF executable")
separate_arguments(common_options UNIX_COMMAND "${COMMON_OPTIONS}")

set(REF_DIR "${WORKING_DIRECTORY}/per_turbine")
set(FMT_DIR "${WORKING_DIRECTORY}/${FORMAT}")
file(REMOVE_RECURSE "${REF_DIR}" "${FMT_DIR}")
file(MAKE_DIRECTORY "${REF_DIR}" "${FMT_DIR}")

# Stage the deck's auxiliary inputs (turbine files, ...) into each run directory, since the runs
# read them relative to their own working directory
file(GLOB _df_aux LIST_DIRECTORIES false "${WORKING_DIRECTORY}/*")
list(FILTER _df_aux EXCLUDE REGEX "\\.(i|log|log\\..*|txt)$")
foreach(_df_f IN LISTS _df_aux)
    file(COPY "${_df_f}" DESTINATION "${REF_DIR}")
    file(COPY "${_df_f}" DESTINATION "${FMT_DIR}")
endforeach()

erf_mpi_launcher_command(launch
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunDiagnosticsFormat.cmake")

function(run_leg dir)
    execute_process(
        COMMAND ${launch} ${TEST_EXE} ${INPUT} ${common_options} ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_FILE "${dir}/simulation.log"
        ERROR_FILE "${dir}/simulation.log"
        TIMEOUT ${RUN_TIMEOUT}
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: the run in ${dir} failed or exceeded ${RUN_TIMEOUT} s: ${_result}")
    endif()
endfunction()

run_leg("${REF_DIR}" "erf.moving_bodies.diagnostics_format=per_turbine")
run_leg("${FMT_DIR}" "erf.moving_bodies.diagnostics_format=${FORMAT}")

# a table with its commas as spaces, as erf_compare_data_logs reads it
function(write_table out_file lines)
    set(_text "")
    foreach(_line IN LISTS ${lines})
        string(REPLACE "," " " _line "${_line}")
        string(APPEND _text "${_line}\n")
    endforeach()
    file(WRITE "${out_file}" "${_text}")
endfunction()

function(compare_tables a b what)
    erf_compare_data_logs("${a}" "${b}" 10 0 _agree _message)
    if(NOT _agree)
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: ${what}: ${_message}")
    endif()
endfunction()

separate_arguments(_pairs UNIX_COMMAND "${LOGS}")
set(_ilog 0)
foreach(_pair IN LISTS _pairs)
    if(NOT _pair MATCHES "^([^/]+)/([^/]+)$")
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: '${_pair}' in LOGS is not body/log")
    endif()
    set(_body "${CMAKE_MATCH_1}")
    set(_log "${CMAKE_MATCH_2}")
    set(_ref "${REF_DIR}/${_body}_${_log}.csv")
    if(NOT EXISTS "${_ref}")
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: the per_turbine run wrote no ${_body}_${_log}.csv")
    endif()
    if(EXISTS "${FMT_DIR}/${_body}_${_log}.csv")
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: the ${FORMAT} run wrote ${_body}_${_log}.csv, a per_turbine file")
    endif()
    if(FORMAT STREQUAL "farm")
        set(_files "${FMT_DIR}/${DIAG_DIR}/farm_${_log}.csv")
    else()
        file(GLOB _files "${FMT_DIR}/${DIAG_DIR}/rank*_${_log}.csv")
        list(SORT _files)
    endif()
    if(_log STREQUAL "stats")
        set(_col 0)
    else()
        set(_col 1)
    endif()

    # the body's rows, with the body column removed; the header from the first file
    set(_picked "")
    set(_header "")
    foreach(_f IN LISTS _files)
        if(NOT EXISTS "${_f}")
            message(FATAL_ERROR "RunDiagnosticsFormat.cmake: the ${FORMAT} run wrote no ${_f}")
        endif()
        file(STRINGS "${_f}" _lines)
        list(POP_FRONT _lines _first)
        string(REPLACE "," ";" _fields "${_first}")
        list(GET _fields ${_col} _name)
        if(NOT _name STREQUAL "body")
            message(FATAL_ERROR "RunDiagnosticsFormat.cmake: ${_f} has no body column: '${_first}'")
        endif()
        if(_header STREQUAL "")
            list(REMOVE_AT _fields ${_col})
            list(JOIN _fields "," _header)
        endif()
        foreach(_line IN LISTS _lines)
            string(REPLACE "," ";" _fields "${_line}")
            list(GET _fields ${_col} _name)
            if(_name STREQUAL _body)
                list(REMOVE_AT _fields ${_col})
                list(JOIN _fields "," _row)
                list(APPEND _picked "${_row}")
            endif()
        endforeach()
    endforeach()
    list(LENGTH _picked _rows)
    if(_rows LESS 1)
        message(FATAL_ERROR "RunDiagnosticsFormat.cmake: no row of ${_body} in the ${FORMAT} ${_log} files")
    endif()
    list(PREPEND _picked "${_header}")
    file(STRINGS "${_ref}" _ref_lines)
    write_table("${WORKING_DIRECTORY}/format_ref_${_ilog}.txt" _ref_lines)
    write_table("${WORKING_DIRECTORY}/format_split_${_ilog}.txt" _picked)
    compare_tables("${WORKING_DIRECTORY}/format_ref_${_ilog}.txt" "${WORKING_DIRECTORY}/format_split_${_ilog}.txt"
                   "${_body}'s ${_log} rows in the ${FORMAT} files differ from ${_body}_${_log}.csv")
    message(STATUS "RunDiagnosticsFormat: ${_body}'s ${_log} rows agree (${_rows} rows)")
    math(EXPR _ilog "${_ilog} + 1")
endforeach()

separate_arguments(_extra UNIX_COMMAND "${EXTRA_LOGS}")
foreach(_log IN LISTS _extra)
    foreach(dir "${REF_DIR}" "${FMT_DIR}")
        if(NOT EXISTS "${dir}/${_log}")
            message(FATAL_ERROR "RunDiagnosticsFormat.cmake: no ${dir}/${_log}")
        endif()
    endforeach()
    file(STRINGS "${REF_DIR}/${_log}" _a)
    file(STRINGS "${FMT_DIR}/${_log}" _b)
    write_table("${WORKING_DIRECTORY}/format_ref_${_ilog}.txt" _a)
    write_table("${WORKING_DIRECTORY}/format_split_${_ilog}.txt" _b)
    compare_tables("${WORKING_DIRECTORY}/format_ref_${_ilog}.txt" "${WORKING_DIRECTORY}/format_split_${_ilog}.txt"
                   "${_log} differs between the per_turbine and ${FORMAT} runs")
    message(STATUS "RunDiagnosticsFormat: ${_log} agrees")
    math(EXPR _ilog "${_ilog} + 1")
endforeach()
