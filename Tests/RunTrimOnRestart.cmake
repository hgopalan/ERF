# A restart cuts every log back to the checkpoint, each on the rank that writes it: run a deck straight
# to STEP_CHK on CHK_NRANKS ranks; run it again on CHK_NRANKS ranks to CHK_LEG_END with a checkpoint at
# STEP_CHK; restart that run from the checkpoint on RESTART_NRANKS ranks (at most CHK_NRANKS) with no step
# left (max_step = STEP_CHK), and require every file in FILES (space-separated, relative to the run
# directory) to hold the straight run's bytes, which only the cut brings back. TRIMMED_BY, when given,
# lists file:rank pairs: the restart's log must report exactly one cut of each file, by that rank.
# -DX= defines X as empty, so test for a value, not for DEFINED
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY RUN_TIMEOUT STEP_CHK CHK_LEG_END CHK_NRANKS RESTART_NRANKS FILES)
    if("${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunTrimOnRestart.cmake: ${arg} must be given and non-empty")
    endif()
endforeach()
if(RESTART_NRANKS GREATER CHK_NRANKS)
    message(FATAL_ERROR "RunTrimOnRestart.cmake: RESTART_NRANKS (${RESTART_NRANKS}) must not exceed CHK_NRANKS (${CHK_NRANKS})")
endif()
if(NOT "${MPIEXEC}" STREQUAL "" AND "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
    message(FATAL_ERROR "RunTrimOnRestart.cmake: MPIEXEC_NUMPROC_FLAG must be given with MPIEXEC")
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunTrimOnRestart.cmake: ERF executable")
separate_arguments(common_options UNIX_COMMAND "${COMMON_OPTIONS}")
separate_arguments(files UNIX_COMMAND "${FILES}")

set(STRAIGHT_DIR "${WORKING_DIRECTORY}/straight")
set(RUN_DIR "${WORKING_DIRECTORY}/run")
file(REMOVE_RECURSE "${STRAIGHT_DIR}" "${RUN_DIR}")
file(MAKE_DIRECTORY "${STRAIGHT_DIR}" "${RUN_DIR}")
file(GLOB _tf_aux LIST_DIRECTORIES false "${WORKING_DIRECTORY}/*")
list(FILTER _tf_aux EXCLUDE REGEX "\\.(i|log|log\\..*|txt)$")
foreach(_tf_f IN LISTS _tf_aux)
    file(COPY "${_tf_f}" DESTINATION "${STRAIGHT_DIR}")
    file(COPY "${_tf_f}" DESTINATION "${RUN_DIR}")
endforeach()

function(run_leg dir nranks logname)
    erf_mpi_launcher_command(launch
        LAUNCHER "${MPIEXEC}"
        NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
        NRANKS ${nranks}
        PREFLAGS "${MPIEXEC_PREFLAGS}"
        CONTEXT "RunTrimOnRestart.cmake")
    execute_process(
        COMMAND ${launch} ${TEST_EXE} ${INPUT} ${common_options} ${ARGN}
        WORKING_DIRECTORY "${dir}"
        OUTPUT_FILE "${dir}/${logname}"
        ERROR_FILE "${dir}/${logname}"
        TIMEOUT ${RUN_TIMEOUT}
        RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "RunTrimOnRestart.cmake: the run in ${dir} (${logname}) failed or exceeded ${RUN_TIMEOUT} s: ${_result}")
    endif()
endfunction()

string(LENGTH "${STEP_CHK}" _len)
set(_chk "${STEP_CHK}")
while(_len LESS 5)
    set(_chk "0${_chk}")
    math(EXPR _len "${_len} + 1")
endwhile()

run_leg("${STRAIGHT_DIR}" ${CHK_NRANKS} simulation.log "max_step=${STEP_CHK}" "erf.check_int=-1" "erf.plot_int_1=-1")
run_leg("${RUN_DIR}" ${CHK_NRANKS} simulation.log "max_step=${CHK_LEG_END}" "erf.check_int=${STEP_CHK}" "erf.plot_int_1=-1")
run_leg("${RUN_DIR}" ${RESTART_NRANKS} restart.log "erf.restart=chk${_chk}" "max_step=${STEP_CHK}" "erf.check_int=-1" "erf.plot_int_1=-1")

set(_differ "")
foreach(_f IN LISTS files)
    foreach(_d "${STRAIGHT_DIR}" "${RUN_DIR}")
        if(NOT EXISTS "${_d}/${_f}")
            message(FATAL_ERROR "RunTrimOnRestart.cmake: ${_d}/${_f} was not written")
        endif()
    endforeach()
    file(SHA256 "${STRAIGHT_DIR}/${_f}" _ha)
    file(SHA256 "${RUN_DIR}/${_f}" _hb)
    if(NOT _ha STREQUAL _hb)
        list(APPEND _differ "${_f}")
    endif()
endforeach()
if(_differ)
    message(FATAL_ERROR "RunTrimOnRestart.cmake: after the restart on ${RESTART_NRANKS} rank(s) these files differ from the straight run's: ${_differ}")
endif()
separate_arguments(trimmed_by UNIX_COMMAND "${TRIMMED_BY}")
file(STRINGS "${RUN_DIR}/restart.log" _cuts REGEX "Moving bodies: rank [0-9]+ dropped")
foreach(_pair IN LISTS trimmed_by)
    string(REPLACE ":" ";" _pr "${_pair}")
    list(GET _pr 0 _f)
    list(GET _pr 1 _rank)
    set(_by "")
    foreach(_c IN LISTS _cuts)
        if(_c MATCHES "rank ([0-9]+) dropped .* from (.*)$" AND CMAKE_MATCH_2 STREQUAL "${_f}")
            list(APPEND _by "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(NOT "${_by}" STREQUAL "${_rank}")
        message(FATAL_ERROR "RunTrimOnRestart.cmake: ${_f} must be cut once, by rank ${_rank}; the restart's log reports cuts by rank(s) '${_by}'")
    endif()
endforeach()
list(LENGTH files _n)
message(STATUS "RunTrimOnRestart.cmake: ${_n} file(s) cut back to the straight run's")
