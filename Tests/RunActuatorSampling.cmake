# Run the Actuator_Sampling deck and check the turbine's flow file (<output_root>_flow.csv: the
# velocities sampled at its nodes) two ways: against the committed gold log (every column, to
# SIGDIGITS significant digits), and against the analytic values the linear shear profile gives
# at the hub and as the blade mean, so a change that moved the gold along with a wrong answer
# would still be caught.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, CSV (the diagnostics file, relative to WORKING_DIRECTORY), GOLD (its
# committed reference), SIGDIGITS, HUB_U, HUB_V (the analytic hub velocities; w must be 0).

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY CSV GOLD HUB_U HUB_V)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunActuatorSampling.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()
if(NOT DEFINED SIGDIGITS OR "${SIGDIGITS}" STREQUAL "")
    set(SIGDIGITS 10)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}"
    CONTEXT "RunActuatorSampling.cmake: ERF executable")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunActuatorSampling.cmake")

set(log "${WORKING_DIRECTORY}/simulation.log")
file(REMOVE "${WORKING_DIRECTORY}/${CSV}")
execute_process(
    COMMAND ${launcher} ${TEST_EXE} ${INPUT} amrex.call_addr2line=0
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${log}"
    ERROR_FILE "${log}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    file(READ "${log}" contents)
    message(FATAL_ERROR "RunActuatorSampling.cmake: the simulation failed (${result}):\n${contents}")
endif()
if(NOT EXISTS "${WORKING_DIRECTORY}/${CSV}")
    message(FATAL_ERROR "RunActuatorSampling.cmake: the run wrote no ${CSV}")
endif()

# ---- the whole log against the gold ----
erf_compare_data_logs("${GOLD}" "${WORKING_DIRECTORY}/${CSV}" ${SIGDIGITS} 2 logs_agree log_message)
if(NOT logs_agree)
    file(READ "${WORKING_DIRECTORY}/${CSV}" contents)
    message(FATAL_ERROR "RunActuatorSampling.cmake: ${CSV} differs from the gold log: ${log_message}\n${contents}")
endif()

# ---- the sampled velocities against the analytic profile, every row ----
file(STRINGS "${WORKING_DIRECTORY}/${CSV}" rows)
list(GET rows 0 header)
string(REPLACE "," ";" header_fields "${header}")
list(FIND header_fields "hub_u" i_hub_u)
list(FIND header_fields "blade_mean_v" i_blade_v)
if(i_hub_u LESS 0 OR i_blade_v LESS 0)
    message(FATAL_ERROR "RunActuatorSampling.cmake: ${CSV} header lacks hub_u or blade_mean_v: ${header}")
endif()
list(LENGTH rows nrows)
if(nrows LESS 3)
    message(FATAL_ERROR "RunActuatorSampling.cmake: ${CSV} has only ${nrows} lines; the check would be trivial")
endif()
set(expected_hub "${HUB_U};${HUB_V};0")            # hub_u hub_v hub_w
set(expected_blade "${HUB_U};${HUB_V};0")          # the rotor is symmetric about the hub
math(EXPR last "${nrows} - 1")
foreach(r RANGE 1 ${last})
    list(GET rows ${r} row)
    string(REPLACE "," ";" fields "${row}")
    foreach(c RANGE 0 2)
        math(EXPR ih "${i_hub_u} + ${c}")
        math(EXPR ib "${i_blade_v} - 1 + ${c}")
        list(GET fields ${ih} got_hub)
        list(GET fields ${ib} got_blade)
        list(GET expected_hub ${c} want)
        # hub_w and blade_mean_w are zero up to roundoff of a sum of terms of order 1e-15
        if("${want}" STREQUAL "0")
            foreach(got IN ITEMS ${got_hub} ${got_blade})
                erf_read_decimal("${got}" ok sign digits exp)
                if(NOT ok)
                    message(FATAL_ERROR "RunActuatorSampling.cmake: row ${r}: '${got}' is not a number")
                endif()
                string(LENGTH "${digits}" ndig)
                math(EXPR lead "${exp} + ${ndig} - 1")
                if(NOT "${digits}" STREQUAL "0" AND lead GREATER -9)
                    message(FATAL_ERROR "RunActuatorSampling.cmake: row ${r}: vertical velocity ${got} is not zero to 1e-9")
                endif()
            endforeach()
        else()
            erf_numbers_close("${got_hub}" "${want}" 8 2 hub_ok)
            erf_numbers_close("${got_blade}" "${want}" 8 2 blade_ok)
            if(NOT hub_ok OR NOT blade_ok)
                message(FATAL_ERROR "RunActuatorSampling.cmake: row ${r}: hub ${got_hub} / blade mean ${got_blade} differ from the analytic ${want}")
            endif()
        endif()
    endforeach()
endforeach()
message(STATUS "RunActuatorSampling.cmake: ${CSV} matches the gold log and the analytic profile in ${last} rows")
