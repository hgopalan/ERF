if (ERF_ENABLE_LNG)

  add_test(
    NAME LNG_WindExtraction
    COMMAND bash -c "
      cd ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_BINARY_DIR}/Exec/ERF3d ${CMAKE_CURRENT_SOURCE_DIR}/inputs_lng_windextraction > run.log 2>&1
      EXIT_CODE=\$?
      
      # Check exit code
      if [ \$EXIT_CODE -ne 0 ]; then
        echo 'FAILED: ERF3d exited with code '\$EXIT_CODE
        exit 1
      fi
      
      # Check for Phase 4 live extraction (5 times for 5 steps)
      EXTRACTION_COUNT=\$(grep -c 'Phase 4: live ATM extraction active' run.log)
      if [ \"\$EXTRACTION_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 Phase 4 live extraction messages, got '\$EXTRACTION_COUNT
        exit 1
      fi
      
      # Check for u* extraction messages
      USTAR_COUNT=\$(grep -c 'Phase 4: u* extracted' run.log)
      if [ \"\$USTAR_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 u* extraction messages, got '\$USTAR_COUNT
        exit 1
      fi
      
      # Check for wind extraction messages
      WIND_COUNT=\$(grep -c 'Phase 4: wind extracted' run.log)
      if [ \"\$WIND_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 wind extraction messages, got '\$WIND_COUNT
        exit 1
      fi
      
      # Check that ustar_max > 0
      if ! grep 'Phase 4: u* extracted' run.log | grep -q 'ustar_max=[0-9]*\.[0-9]*[1-9]'; then
        echo 'FAILED: ustar_max should be > 0'
        exit 1
      fi
      
      # Check that u_max > 0 in wind extraction
      if ! grep 'Phase 4: wind extracted' run.log | grep -q 'u_max=[0-9]*\.[0-9]*[1-9]'; then
        echo 'FAILED: u_max should be > 0'
        exit 1
      fi
      
      # Check that Phase 3 apply_to_cc_source still appears 5 times (no regression)
      COUPLING_COUNT=\$(grep -c 'Phase 3: apply_to_cc_source' run.log)
      if [ \"\$COUPLING_COUNT\" -ne 5 ]; then
        echo 'FAILED: Expected 5 Phase 3 apply_to_cc_source calls, got '\$COUPLING_COUNT
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
      
      echo 'PASSED: LNG_WindExtraction test'
      exit 0
    "
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(LNG_WindExtraction PROPERTIES
    TIMEOUT 300
    LABELS "LNG;Phase4"
  )

endif()
