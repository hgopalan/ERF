# Self-test of the normwise column groups of Tests/CompareDataLogs.cmake (ERF_DATALOG_GROUPS) with
# the groups the conductor tests use (Tests/ConductorLogGroups.cmake).
#
# The rows are two rows of the real-MoorDyn gold Conductors_ImmersedHills/towers.dat.moordyn.gold
# (one tower, times 0 and 1.8 s). Compared field by field to four significant digits, the
# along-line pull at time 0 (-2.5 N, the near-cancelling sum of span pulls of 10 to 20 kN) fails on
# a difference of 1e-7 of those pulls, and a cross-arm displacement below 1e-4 m is not compared
# at all. Compared normwise, the first agrees and the second is checked, while a real change of a
# large component still fails.
#
# Pure CMake, no ERF run.  Run with
#   cmake -DWORK_DIR=<scratch directory> -P Tests/CompareDataLogsGroupsSelfTest.cmake

if("${WORK_DIR}" STREQUAL "")
    message(FATAL_ERROR "CompareDataLogsGroupsSelfTest.cmake: WORK_DIR must be given and non-empty")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CompareDataLogs.cmake")
# as Tests/RunConductors.cmake sets it
set(ERF_DATALOG_ZERO_EXPONENT -4)
include("${CMAKE_CURRENT_LIST_DIR}/ConductorLogGroups.cmake")
set(conductor_groups "${ERF_DATALOG_GROUPS}")

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

set(selftest_failures 0)
set(selftest_cases 0)

# Conductors_ImmersedHills_MoorDyn compares its logs to four significant digits, two units of
# the last
set(SIGDIGITS 4)
set(ULPS 2)

# Compare two logs given as strings, with the conductor groups (GROUPED TRUE) or without, and
# check the verdict against the one expected
function(expect_logs name grouped expected_agree text_a text_b)
    math(EXPR _cases "${selftest_cases} + 1")
    set(selftest_cases ${_cases} PARENT_SCOPE)
    if(grouped)
        set(ERF_DATALOG_GROUPS "${conductor_groups}")
    else()
        set(ERF_DATALOG_GROUPS "")
    endif()
    set(_file_a "${WORK_DIR}/${name}_a.dat")
    set(_file_b "${WORK_DIR}/${name}_b.dat")
    file(WRITE "${_file_a}" "${text_a}")
    file(WRITE "${_file_b}" "${text_b}")
    erf_compare_data_logs("${_file_a}" "${_file_b}" ${SIGDIGITS} ${ULPS} _agree _message)
    if(_agree AND NOT expected_agree)
        message(SEND_ERROR "${name}: the logs were accepted but must differ")
        math(EXPR _failures "${selftest_failures} + 1")
        set(selftest_failures ${_failures} PARENT_SCOPE)
    elseif(NOT _agree AND expected_agree)
        message(SEND_ERROR "${name}: the logs were rejected but must agree: ${_message}")
        math(EXPR _failures "${selftest_failures} + 1")
        set(selftest_failures ${_failures} PARENT_SCOPE)
    endif()
endfunction()

# Columns: the members' drag and the lines' pull on the tower (N), its foundation loads (N, N m),
# the over-allowable flag and the cross-arm displacement (m)
set(header "time L1b_t1_drag_Fx L1b_t1_drag_Fy L1b_t1_drag_Fz L1b_t1_line_Fx L1b_t1_line_Fy L1b_t1_line_Fz L1b_t1_shear L1b_t1_overturning L1b_t1_vertical L1b_t1_max_compression L1b_t1_max_uplift L1b_t1_over L1b_t1_arm_dx L1b_t1_arm_dy\n")
set(row0 "0 22387.06002 27.02611911 -80.8228954 -2.515519815 1.429554688 -48283.14603 22384.56258 364474.291 108363.9689 58990.62791 4808.643449 0 0 0\n")
set(row1 "1.8 20363.93794 232.9283784 -168.8124999 -4854.084697 375.255438 -50576.81546 19879.35142 317804.7616 110745.628 55686.8173 314.0033193 0 0.02120279188 3.391081724e-05\n")
set(gold "${header}${row0}${row1}")

# Identical logs agree either way
expect_logs(identical_plain FALSE TRUE "${gold}" "${gold}")
expect_logs(identical_grouped TRUE TRUE "${gold}" "${gold}")

# The Linux CI value of the along-line pull at time 0 against the gold made on macOS: a difference
# of 0.0137 N, fatal to four digits of -2.5 N, nothing beside the 48 kN pull it belongs to
string(REPLACE " -2.515519815 " " -2.529222277 " linux_row0 "${row0}")
expect_logs(near_cancelling_pull_plain FALSE FALSE "${gold}" "${header}${linux_row0}${row1}")
expect_logs(near_cancelling_pull_grouped TRUE TRUE "${gold}" "${header}${linux_row0}${row1}")

# A real change of the pull: 80 N on the 48 kN vertical component
string(REPLACE " -48283.14603 " " -48203.14603 " pull_row0 "${row0}")
expect_logs(vertical_pull_grouped TRUE FALSE "${gold}" "${header}${pull_row0}${row1}")

# A leg's uplift lost: 314 N of uplift against none, beside a 111 kN vertical load
string(REPLACE " 314.0033193 " " 0 " uplift_row1 "${row1}")
expect_logs(uplift_lost_grouped TRUE FALSE "${gold}" "${header}${row0}${uplift_row1}")

# A cross-arm displacement 2.7 times too large: below 1e-4 m both count as zero field by field,
# while against the 0.021 m displacement of the same cross-arm it is a difference
string(REPLACE " 3.391081724e-05" " 9e-05" arm_row1 "${row1}")
expect_logs(arm_displacement_plain FALSE TRUE "${gold}" "${header}${row0}${arm_row1}")
expect_logs(arm_displacement_grouped TRUE FALSE "${gold}" "${header}${row0}${arm_row1}")

# A small component may change sign within the tolerance of its vector
string(REPLACE " 1.429554688 " " -1.2 " flip_row0 "${row0}")
expect_logs(small_component_sign_grouped TRUE TRUE "${gold}" "${header}${flip_row0}${row1}")

# A column in no group is still compared on its own digits
string(REPLACE " 364474.291 " " 364900.291 " moment_row0 "${row0}")
expect_logs(ungrouped_moment_grouped TRUE FALSE "${gold}" "${header}${moment_row0}${row1}")

if(selftest_failures GREATER 0)
    message(FATAL_ERROR "CompareDataLogsGroupsSelfTest: ${selftest_failures} of ${selftest_cases} cases failed")
endif()
message(STATUS "CompareDataLogsGroupsSelfTest: ${selftest_cases} cases passed")
