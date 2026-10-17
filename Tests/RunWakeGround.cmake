# Run a moving-bodies deck with wake lines on terrain, then check_wake_ground.py on two rotors' lines: each must
# start on the ground under its centre, where another body's base stands. The checker is also run with the
# rotor's own base named instead, which must fail (its ground differs), so the check cannot pass vacuously.
# Variables: MPIEXEC, MPIEXEC_NUMPROC_FLAG, NRANKS, TEST_EXE, INPUT, OPTIONS (the run's options joined by |, since
# a ; would split them on the way here), PYTHON_EXE, CHECKER, CHECKS (a ;-list of rotor:at:xD), WORKING_DIRECTORY.
string(REPLACE "|" ";" OPTIONS "${OPTIONS}")
execute_process(
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS} ${TEST_EXE} ${INPUT} ${OPTIONS}
    WORKING_DIRECTORY ${WORKING_DIRECTORY}
    OUTPUT_FILE ${WORKING_DIRECTORY}/simulation.log ERROR_FILE ${WORKING_DIRECTORY}/simulation.log
    RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "RunWakeGround.cmake: the run exited with ${rc} (see ${WORKING_DIRECTORY}/simulation.log)")
endif()
foreach(check ${CHECKS})
    string(REPLACE ":" ";" parts "${check}")
    list(GET parts 0 rotor)
    list(GET parts 1 at)
    list(GET parts 2 xD)
    execute_process(
        COMMAND ${PYTHON_EXE} ${CHECKER} --wake ${rotor}_wake.csv --ground moving_bodies/ground.csv --rotor ${rotor} --at ${at} --xD ${xD}
        WORKING_DIRECTORY ${WORKING_DIRECTORY} OUTPUT_VARIABLE out ERROR_VARIABLE out RESULT_VARIABLE r)
    message(STATUS "${out}")
    if(NOT r EQUAL 0)
        message(FATAL_ERROR "RunWakeGround.cmake: ${rotor}'s wake line does not start on the ground under ${at}:\n${out}")
    endif()
    # the negative case: the rotor's own base is not at the line's centre
    execute_process(
        COMMAND ${PYTHON_EXE} ${CHECKER} --wake ${rotor}_wake.csv --ground moving_bodies/ground.csv --rotor ${at} --at ${rotor} --xD ${xD}
        WORKING_DIRECTORY ${WORKING_DIRECTORY} OUTPUT_VARIABLE out ERROR_VARIABLE out RESULT_VARIABLE r)
    if(r EQUAL 0)
        message(FATAL_ERROR "RunWakeGround.cmake: the checker accepted ${rotor}'s line on the ground under ${rotor} itself:\n${out}")
    endif()
endforeach()
