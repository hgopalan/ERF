if (ERF_ENABLE_LNG)

  add_test(
    NAME LNG_PoolEvap
    COMMAND bash -c "
      cd ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_BINARY_DIR}/Exec/ERF3d ${CMAKE_CURRENT_SOURCE_DIR}/inputs_lng_poolevap > run.log 2>&1
      EXIT_CODE=$?
      
      # Check exit code
      if [ $EXIT_CODE -ne 0 ]; then
        echo 'FAILED: ERF3d exited with code '$EXIT_CODE
        exit 1
      fi
      
      # Check for initialization message
      if ! grep -q 'Phase 2: pool evaporation model initialized' run.log; then
        echo 'FAILED: Initialization message not found'
        exit 1
      fi
      
      # Check for 5 advance steps
      STEP_COUNT=\$(grep -c 'advance: step=' run.log)
      if [ \"\$STEP_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 advance steps, got '$STEP_COUNT
        exit 1
      fi
      
      # Check for 5 NaN checks PASSED
      NAN_COUNT=\$(grep -c 'NaN check PASSED' run.log)
      if [ \"\$NAN_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 NaN checks, got '$NAN_COUNT
        exit 1
      fi
      
      # Check CSV file exists
      if [ ! -f lng_diag.csv ]; then
        echo 'FAILED: lng_diag.csv not created'
        exit 1
      fi
      
      # Check CSV has header + 5 data rows (6 total lines)
      LINE_COUNT=\$(wc -l < lng_diag.csv)
      if [ \"\$LINE_COUNT\" -ne 6 ]; then
        echo 'FAILED: CSV should have 6 lines (header + 5 data), got '$LINE_COUNT
        exit 1
      fi
      
      echo 'PASSED: LNG_PoolEvap test'
      exit 0
    "
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(LNG_PoolEvap PROPERTIES
    TIMEOUT 300
    LABELS "LNG;Phase2"
  )

endif()
