# Run a conductor deck whose towers move twice, with the coupling steps ERF chooses and with
# erf.conductors.substeps = SUBSTEPS (more), and check that the columns COLUMNS (space-separated
# header names) of the log LOG agree row by row to SIGDIGITS significant digits (ULPS units of the
# last one): the answer must not depend on the coupling step once it resolves the towers' sway.
#
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS, NRANKS, TEST_EXE, CONFIG, INPUT,
# WORKING_DIRECTORY, LOG, COLUMNS, SUBSTEPS, SIGDIGITS, ULPS.

cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")
set(ERF_DATALOG_ZERO_EXPONENT -4)

foreach(arg TEST_EXE INPUT WORKING_DIRECTORY LOG COLUMNS SUBSTEPS SIGDIGITS ULPS)
    if(NOT DEFINED ${arg} OR "${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunCouplingConvergence.cmake: ${arg} must be given")
    endif()
endforeach()
if(NOT DEFINED NRANKS OR "${NRANKS}" STREQUAL "")
    set(NRANKS 1)
endif()

erf_resolve_executable(TEST_EXE "${TEST_EXE}" CONFIG "${CONFIG}" CONTEXT "RunCouplingConvergence.cmake: ERF executable")
erf_mpi_launcher_command(launcher
    LAUNCHER "${MPIEXEC}" NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}" CONTEXT "RunCouplingConvergence.cmake")

# the two runs, each with its own simulation_<run>.log
foreach(run default fine)
    set(log "${WORKING_DIRECTORY}/simulation_${run}.log")
    file(REMOVE "${WORKING_DIRECTORY}/${LOG}")
    set(options "")
    if(run STREQUAL "fine")
        set(options "erf.conductors.substeps=${SUBSTEPS}")
    endif()
    execute_process(
        COMMAND ${launcher} ${TEST_EXE} ${INPUT} amrex.call_addr2line=0 ${options}
        WORKING_DIRECTORY "${WORKING_DIRECTORY}"
        OUTPUT_FILE "${log}"
        ERROR_FILE "${log}"
        RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        file(READ "${log}" contents)
        message(FATAL_ERROR "RunCouplingConvergence.cmake: the ${run} run failed (${result}):\n${contents}")
    endif()
    if(NOT EXISTS "${WORKING_DIRECTORY}/${LOG}")
        message(FATAL_ERROR "RunCouplingConvergence.cmake: the ${run} run wrote no ${LOG}")
    endif()
    file(STRINGS "${WORKING_DIRECTORY}/${LOG}" lines_${run})
endforeach()

list(LENGTH lines_default n_default)
list(LENGTH lines_fine n_fine)
if(NOT n_default EQUAL n_fine OR n_default LESS 3)
    message(FATAL_ERROR "RunCouplingConvergence.cmake: the runs wrote ${n_default} and ${n_fine} lines of ${LOG}")
endif()
list(GET lines_default 0 header)
string(REGEX REPLACE "[ \t]+" ";" header "${header}")
separate_arguments(columns UNIX_COMMAND "${COLUMNS}")
set(indices "")
foreach(c IN LISTS columns)
    list(FIND header "${c}" i)
    if(i LESS 0)
        message(FATAL_ERROR "RunCouplingConvergence.cmake: ${LOG} has no column ${c}")
    endif()
    list(APPEND indices ${i})
endforeach()

math(EXPR last "${n_default} - 1")
set(failures "")
foreach(row RANGE 1 ${last})
    list(GET lines_default ${row} a)
    list(GET lines_fine ${row} b)
    string(STRIP "${a}" a)
    string(STRIP "${b}" b)
    string(REGEX REPLACE "[ \t]+" ";" a "${a}")
    string(REGEX REPLACE "[ \t]+" ";" b "${b}")
    list(GET a 0 time)
    foreach(c i IN ZIP_LISTS columns indices)
        list(GET a ${i} va)
        list(GET b ${i} vb)
        erf_numbers_close("${va}" "${vb}" ${SIGDIGITS} ${ULPS} close)
        if(NOT close)
            string(APPEND failures "  t = ${time}: ${c} ${va} with the coupling steps ERF chose, ${vb} with ${SUBSTEPS}\n")
        endif()
    endforeach()
endforeach()
if(NOT failures STREQUAL "")
    message(FATAL_ERROR "RunCouplingConvergence.cmake: ${LOG} depends on the coupling step:\n${failures}")
endif()
message(STATUS "RunCouplingConvergence.cmake: ${COLUMNS} of ${LOG} agree with ${SUBSTEPS} coupling steps to ${SIGDIGITS} digits")
