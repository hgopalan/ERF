if (ERF_ENABLE_LNG)

  add_test(
    NAME LNG_ScalarInjection
    COMMAND bash -c "
      cd ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_BINARY_DIR}/Exec/ERF3d ${CMAKE_CURRENT_SOURCE_DIR}/inputs_lng_scalarinjection > run.log 2>&1
      EXIT_CODE=\$?
      
      # Check exit code
      if [ \$EXIT_CODE -ne 0 ]; then
        echo 'FAILED: ERF3d exited with code '\$EXIT_CODE
        exit 1
      fi
      
      # Check for Phase 3 apply_to_cc_source (5 times for 5 steps)
      COUPLING_COUNT=\$(grep -c 'Phase 3: apply_to_cc_source' run.log)
      if [ \"\$COUPLING_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 apply_to_cc_source calls, got '\$COUPLING_COUNT
        exit 1
      fi
      
      # Check for coupling injection messages
      INJECT_COUNT=\$(grep -c 'LNG COUPLING' run.log)
      if [ \"\$INJECT_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 LNG COUPLING messages, got '\$INJECT_COUNT
        exit 1
      fi
      
      # Check that F_evap_max > 0 in coupling messages
      if ! grep 'LNG COUPLING' run.log | grep -q 'F_evap_max=[0-9]*\.[0-9]*[1-9]'; then
        echo 'FAILED: F_evap_max should be > 0'
        exit 1
      fi
      
      # Check that RhoLNG_tend_sum > 0
      if ! grep 'LNG COUPLING' run.log | grep -q 'sum=[0-9]*\.[0-9]*[1-9]'; then
        echo 'FAILED: RhoLNG_tend_sum should be > 0'
        exit 1
      fi
      
      # Check for 5 NaN checks PASSED
      NAN_COUNT=\$(grep -c 'NaN check PASSED' run.log)
      if [ \"\$NAN_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 NaN checks, got '\$NAN_COUNT
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
        echo 'FAILED: CSV should have 6 lines (header + 5 data), got '\$LINE_COUNT
        exit 1
      fi
      
      echo 'PASSED: LNG_ScalarInjection test'
      exit 0
    "
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(LNG_ScalarInjection PROPERTIES
    TIMEOUT 300
    LABELS "LNG;Phase3"
  )

endif()
