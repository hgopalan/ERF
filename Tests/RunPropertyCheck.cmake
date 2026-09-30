include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")

# Run a case, then check a property of its plotfile with Tests/MultiLevelPropertyCheck.cpp.
if(NOT DEFINED MPIEXEC OR NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   NOT DEFINED NRANKS OR NOT DEFINED TEST_EXE OR NOT DEFINED INPUT OR
   NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED SIMULATION_LOG OR
   NOT DEFINED CHECKER_LOG OR NOT DEFINED CHECKER OR NOT DEFINED MODE OR
   NOT DEFINED PLOTFILE OR NOT DEFINED ARG_A OR NOT DEFINED ARG_B)
    message(FATAL_ERROR "RunPropertyCheck.cmake missing required argument")
endif()

erf_mpi_launcher_command(mpi_launch
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS ${NRANKS}
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunPropertyCheck.cmake")
erf_mpi_launcher_command(mpi_launch_one
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunPropertyCheck.cmake")

separate_arguments(runtime_options UNIX_COMMAND "${RUNTIME_OPTIONS}")
execute_process(
    COMMAND ${mpi_launch} ${TEST_EXE} ${INPUT} ${runtime_options}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${SIMULATION_LOG}"
    ERROR_FILE "${SIMULATION_LOG}"
    RESULT_VARIABLE simulation_result)
if(NOT simulation_result EQUAL 0)
    message(FATAL_ERROR "simulation failed: ${simulation_result} (see ${SIMULATION_LOG})")
endif()

execute_process(
    COMMAND ${mpi_launch_one} ${CHECKER} ${MODE} ${PLOTFILE} ${ARG_A} ${ARG_B}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${CHECKER_LOG}"
    ERROR_FILE "${CHECKER_LOG}"
    RESULT_VARIABLE checker_result)
if(NOT checker_result EQUAL 0)
    file(READ "${CHECKER_LOG}" checker_output)
    message(FATAL_ERROR "property check '${MODE}' failed: ${checker_result}\n${checker_output}")
endif()
file(READ "${CHECKER_LOG}" checker_output)
message(STATUS "${checker_output}")
