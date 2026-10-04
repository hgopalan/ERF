# Run a deck straight to STEP_END, run it again to STEP_CHK with a checkpoint there,
# restart from that checkpoint to STEP_END, and require the restarted run's plotfile
# at STEP_END to equal the straight run's with fcompare. With OVERRUN set, the second run
# goes on past its checkpoint to STEP_END before the restart, so the logs it appended to
# hold rows from after the checkpoint, which the restarted run must drop. Each leg uses the forwarded
# RUN_TIMEOUT, and the enclosing CTest timeout is sized separately by the caller.
# -DX= defines X as empty, so test for a value, not for DEFINED
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")

foreach(arg NRANKS TEST_EXE INPUT WORKING_DIRECTORY FCOMPARE STEP_CHK STEP_END RTOL ATOL RUN_TIMEOUT)
    if("${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunRestartParity.cmake: ${arg} must be given and non-empty")
    endif()
endforeach()
if(NOT "${MPIEXEC}" STREQUAL "" AND "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
    message(FATAL_ERROR "RunRestartParity.cmake: MPIEXEC_NUMPROC_FLAG must be given with MPIEXEC")
endif()

# On Windows the executables are named with a wildcard for the config subdirectory
# a multi-config generator picks; execute_process does not expand it.
erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunRestartParity.cmake: ERF executable")
erf_resolve_executable(FCOMPARE "${FCOMPARE}" CONFIG "${CONFIG}"
    CONTEXT "RunRestartParity.cmake: fcompare")

separate_arguments(common_options   UNIX_COMMAND "${COMMON_OPTIONS}")

set(STRAIGHT_DIR "${WORKING_DIRECTORY}/straight")
set(RESTART_DIR  "${WORKING_DIRECTORY}/restart")
file(REMOVE_RECURSE "${STRAIGHT_DIR}" "${RESTART_DIR}")
file(MAKE_DIRECTORY "${STRAIGHT_DIR}" "${RESTART_DIR}")

# A deck may need auxiliary inputs sitting beside it -- an input_sounding, a table, a
# terrain file -- which it names relatively. Each leg runs in its own subdirectory, so
# those files have to be there too; otherwise the run aborts at start-up reading them.
# Directories are skipped: the plotfiles and checkpoints of an earlier run are not inputs.
file(GLOB _rp_aux "${WORKING_DIRECTORY}/*")
foreach(_rp_f IN LISTS _rp_aux)
    if(NOT IS_DIRECTORY "${_rp_f}")
        file(COPY "${_rp_f}" DESTINATION "${STRAIGHT_DIR}")
        file(COPY "${_rp_f}" DESTINATION "${RESTART_DIR}")
    endif()
endforeach()

# MPIEXEC may be a multi-word command such as "flux run"; the helper splits
# it, validates the program and applies MPIEXEC_PREFLAGS. An empty MPIEXEC
# yields an empty prefix, so the runs stay serial.
erf_mpi_launcher_command(launch
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunRestartParity.cmake")
erf_mpi_launcher_command(launch_one
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunRestartParity.cmake")

# plotfile and checkpoint names carry the step padded to five digits
function(padded step out_var)
    set(_s "0000${step}")
    string(LENGTH "${_s}" _len)
    math(EXPR _start "${_len} - 5")
    string(SUBSTRING "${_s}" ${_start} 5 _s)
    set(${out_var} "${_s}" PARENT_SCOPE)
endfunction()
padded(${STEP_CHK} chk_step)
padded(${STEP_END} end_step)
set(PLTFILE "plt${end_step}")
set(CHKFILE "chk${chk_step}")

function(run_erf dir log timeout_s)
    execute_process(
        COMMAND ${launch} ${TEST_EXE} ${INPUT} ${common_options} ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_FILE "${dir}/${log}"
        ERROR_FILE "${dir}/${log}"
        TIMEOUT ${timeout_s}
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "RunRestartParity.cmake: the run in ${dir} (${log}) failed or exceeded ${timeout_s} s: ${_result}")
    endif()
endfunction()

# Optional second comparison: a 2D plotfile. The 3D plotfile carries no surface
# fields, so state that is checkpointed but never restored -- a surface temperature
# reloading as its scalar default, say -- can leave plt identical while plt2d is
# wrong. Drive its cadence exactly as the 3D one is driven.
set(plot2d_end "")
set(plot2d_off "")
if(NOT "${PLT2DFILE}" STREQUAL "")
    set(plot2d_end "erf.plot2d_int_1=${STEP_END}")
    set(plot2d_off "erf.plot2d_int_1=-1")
endif()

# straight to the end, no checkpoint
run_erf("${STRAIGHT_DIR}" "simulation.log" ${RUN_TIMEOUT}
        "max_step=${STEP_END}" "erf.check_int=-1" "erf.plot_int_1=${STEP_END}"
        ${plot2d_end})
# to the checkpoint step, writing it there (on to the end with OVERRUN)
set(_chk_run_end ${STEP_CHK})
if(OVERRUN)
    set(_chk_run_end ${STEP_END})
endif()
run_erf("${RESTART_DIR}" "checkpoint.log" ${RUN_TIMEOUT}
        "max_step=${_chk_run_end}" "erf.check_int=${STEP_CHK}" "erf.plot_int_1=-1"
        ${plot2d_off})
if(NOT EXISTS "${RESTART_DIR}/${CHKFILE}/Header")
    message(FATAL_ERROR "RunRestartParity.cmake: no ${CHKFILE} written by the checkpoint run")
endif()
# from the checkpoint to the end
run_erf("${RESTART_DIR}" "restart.log" ${RUN_TIMEOUT}
        "erf.restart=${CHKFILE}" "max_step=${STEP_END}" "erf.check_int=-1" "erf.plot_int_1=${STEP_END}"
        ${plot2d_end})

foreach(dir "${STRAIGHT_DIR}" "${RESTART_DIR}")
    if(NOT EXISTS "${dir}/${PLTFILE}/Header")
        message(FATAL_ERROR "RunRestartParity.cmake: no ${PLTFILE} in ${dir}")
    endif()
endforeach()

execute_process(
    COMMAND ${launch_one} ${FCOMPARE} --abort_if_not_all_found
            --rel_tol ${RTOL} --abs_tol ${ATOL}
            ${STRAIGHT_DIR}/${PLTFILE} ${RESTART_DIR}/${PLTFILE}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${WORKING_DIRECTORY}/parity.log"
    ERROR_FILE "${WORKING_DIRECTORY}/parity.log"
    RESULT_VARIABLE parity_result)
if(NOT parity_result EQUAL 0)
    message(FATAL_ERROR "RunRestartParity.cmake: the restarted run's ${PLTFILE} differs from the straight run's: ${parity_result} (see parity.log)")
endif()
message(STATUS "RunRestartParity: restart from ${CHKFILE} reproduces ${PLTFILE}")

if(NOT "${PLT2DFILE}" STREQUAL "")
    foreach(dir "${STRAIGHT_DIR}" "${RESTART_DIR}")
        if(NOT EXISTS "${dir}/${PLT2DFILE}/Header")
            message(FATAL_ERROR
                "RunRestartParity.cmake: no ${PLT2DFILE} in ${dir}; the deck must select 2D "
                "output with erf.plot2d_vars_1 for the 2D comparison to mean anything")
        endif()
    endforeach()
    execute_process(
        COMMAND ${launch_one} ${FCOMPARE} --abort_if_not_all_found
                --rel_tol ${RTOL} --abs_tol ${ATOL}
                ${STRAIGHT_DIR}/${PLT2DFILE} ${RESTART_DIR}/${PLT2DFILE}
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${WORKING_DIRECTORY}/parity2d.log"
        ERROR_FILE "${WORKING_DIRECTORY}/parity2d.log"
        RESULT_VARIABLE parity2d_result)
    if(NOT parity2d_result EQUAL 0)
        message(FATAL_ERROR
            "RunRestartParity.cmake: the restarted run's ${PLT2DFILE} differs from the straight "
            "run's: ${parity2d_result} (see parity2d.log). A surface field that is written to the "
            "checkpoint but not read back looks exactly like this.")
    endif()
    message(STATUS "RunRestartParity: restart also reproduces ${PLT2DFILE}")
endif()

# Optional: time series that the run appends to, such as a station file written by
# erf.station_names, must come out the same whether they were written in one run or in two.
# DATALOG names one file or several separated by spaces; comma-separated tables compare as
# whitespace-separated ones. The restarted run marks the restart with a comment line the
# straight run does not have, so the comparison is of the data lines only.
if(NOT "${DATALOG}" STREQUAL "")
    function(strip_comments in_file out_file out_count)
        file(STRINGS "${in_file}" _lines)
        set(_kept "")
        foreach(_line IN LISTS _lines)
            if(NOT _line MATCHES "^#")
                string(REPLACE "," " " _line "${_line}")
                list(APPEND _kept "${_line}")
            endif()
        endforeach()
        list(LENGTH _kept _n)
        string(JOIN "\n" _text ${_kept})
        file(WRITE "${out_file}" "${_text}\n")
        set(${out_count} ${_n} PARENT_SCOPE)
    endfunction()

    if("${DATALOG_SIGDIGITS}" STREQUAL "")
        set(DATALOG_SIGDIGITS 6)
    endif()
    include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

    separate_arguments(_datalogs UNIX_COMMAND "${DATALOG}")
    set(_ilog 0)
    foreach(_log IN LISTS _datalogs)
        foreach(dir "${STRAIGHT_DIR}" "${RESTART_DIR}")
            if(NOT EXISTS "${dir}/${_log}")
                message(FATAL_ERROR "RunRestartParity.cmake: no time series ${dir}/${_log}")
            endif()
        endforeach()

        set(_straight "${WORKING_DIRECTORY}/datalog_straight_${_ilog}.txt")
        set(_restart  "${WORKING_DIRECTORY}/datalog_restart_${_ilog}.txt")
        strip_comments("${STRAIGHT_DIR}/${_log}" "${_straight}" straight_rows)
        strip_comments("${RESTART_DIR}/${_log}"  "${_restart}"  restart_rows)
        if(straight_rows LESS 2)
            message(FATAL_ERROR "RunRestartParity.cmake: ${_log} has ${straight_rows} data rows; the comparison would be trivial")
        endif()

        erf_compare_data_logs("${_straight}" "${_restart}" ${DATALOG_SIGDIGITS} 2 logs_agree log_message)
        if(NOT logs_agree)
            message(FATAL_ERROR "RunRestartParity.cmake: ${_log} differs between the straight run "
                                "and the restarted run: ${log_message}")
        endif()
        message(STATUS "RunRestartParity: ${_log} agrees (${straight_rows} rows)")
        math(EXPR _ilog "${_ilog} + 1")
    endforeach()
endif()
