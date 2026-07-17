add_test(
  NAME LNG_SpillSchedule
  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1
          ${CMAKE_BINARY_DIR}/Exec/ERF inputs_lng_spillschedule
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
