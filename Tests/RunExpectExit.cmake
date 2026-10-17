# Run a command and require both its exit code and its output: EXPECTED_RESULT (the exit code) and
# EXPECTED_REGEX (a regular expression the combined stdout and stderr must match). For checkers whose
# callers act on the exit code alone, so a negative test must prove the verdict, not just the message
# (CTest's PASS_REGULAR_EXPRESSION ignores the exit code).
#
# Variables: CHECK_COMMAND (a ;-list: the program and its arguments), EXPECTED_RESULT (an exit code, or nonzero
# for any failure), EXPECTED_REGEX.
foreach(arg CHECK_COMMAND EXPECTED_RESULT EXPECTED_REGEX)
    if("${${arg}}" STREQUAL "")
        message(FATAL_ERROR "RunExpectExit.cmake: ${arg} must be given and non-empty")
    endif()
endforeach()
execute_process(
    COMMAND ${CHECK_COMMAND}
    OUTPUT_VARIABLE out
    ERROR_VARIABLE out
    RESULT_VARIABLE result)
# EXPECTED_RESULT = nonzero: any failing exit (an abort under mpiexec, whose code the MPI library chooses)
if("${EXPECTED_RESULT}" STREQUAL "nonzero")
    if("${result}" STREQUAL "0")
        message(FATAL_ERROR "RunExpectExit.cmake: the command exited with 0, expected a failure:\n${out}")
    endif()
elseif(NOT "${result}" STREQUAL "${EXPECTED_RESULT}")
    message(FATAL_ERROR "RunExpectExit.cmake: the command exited with ${result}, expected ${EXPECTED_RESULT}:\n${out}")
endif()
if(NOT out MATCHES "${EXPECTED_REGEX}")
    message(FATAL_ERROR "RunExpectExit.cmake: the output does not match '${EXPECTED_REGEX}':\n${out}")
endif()
message(STATUS "RunExpectExit.cmake: exit code ${result} and the expected message:\n${out}")
