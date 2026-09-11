# Fuel map row order: run a fire deck whose ESRI ASCII fuel map is loaded with
# erf.fire.fuel_map.load_from_map to its step-0 fire plotfile, take the y profile
# of fire_fuel_load at x = X with amrex_fextract, and require the load of the
# southernmost fire cell in [SOUTH_MIN, SOUTH_MAX] and of the northernmost in
# [NORTH_MIN, NORTH_MAX]. The two ranges must not overlap, so a map placed upside
# down fails.
#
# Required: TEST_EXE, FEXTRACT, INPUT, WORKING_DIRECTORY, X, SOUTH_MIN, SOUTH_MAX,
# NORTH_MIN, NORTH_MAX. Optional: MPIEXEC, MPIEXEC_NUMPROC_FLAG, MPIEXEC_PREFLAGS,
# NRANKS, RUNTIME_OPTIONS.

foreach(_var TEST_EXE FEXTRACT INPUT WORKING_DIRECTORY X SOUTH_MIN SOUTH_MAX NORTH_MIN NORTH_MAX)
    if("${${_var}}" STREQUAL "")
        message(FATAL_ERROR "RunFireFuelMapRows.cmake: ${_var} is not set")
    endif()
endforeach()
if(NOT (NORTH_MIN GREATER SOUTH_MAX OR NORTH_MAX LESS SOUTH_MIN))
    message(FATAL_ERROR "RunFireFuelMapRows.cmake: the south [${SOUTH_MIN}, ${SOUTH_MAX}] and north "
                        "[${NORTH_MIN}, ${NORTH_MAX}] ranges overlap, so a flipped map would pass")
endif()

set(_plt "${WORKING_DIRECTORY}/plt_fire_00000")
set(_slice "${WORKING_DIRECTORY}/fuel_load_y.txt")
file(REMOVE_RECURSE "${_plt}")
file(REMOVE "${_slice}")

set(_launch)
if(NOT "${MPIEXEC}" STREQUAL "")
    if("${NRANKS}" STREQUAL "")
        set(NRANKS 1)
    endif()
    set(_launch ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS})
    if(NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
        separate_arguments(_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
        list(APPEND _launch ${_preflags})
    endif()
endif()
separate_arguments(_options UNIX_COMMAND "${RUNTIME_OPTIONS}")

execute_process(
    COMMAND ${_launch} ${TEST_EXE} ${INPUT} max_step=0 erf.fire_plot_int=1
            erf.fire_plot_file=plt_fire_ erf.plot_int_1=-1 erf.plot_int_2=-1 erf.check_int=-1
            ${_options}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${WORKING_DIRECTORY}/run.log"
    ERROR_FILE "${WORKING_DIRECTORY}/run.log"
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR NOT EXISTS "${_plt}/Header")
    message(FATAL_ERROR "the run exited with '${_rc}' and wrote no ${_plt}; see run.log")
endif()

execute_process(
    COMMAND ${FEXTRACT} -d 1 -v fire_fuel_load -x ${X} -s ${_slice} ${_plt}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0 OR NOT EXISTS "${_slice}")
    message(FATAL_ERROR "amrex_fextract failed (${_rc}):\n${_out}")
endif()

# fextract writes '#' comment lines, then one "y value" line per fire cell, south first
file(STRINGS "${_slice}" _rows REGEX "^[ \t]*[-+0-9.]")
list(LENGTH _rows _n)
if(_n LESS 2)
    message(FATAL_ERROR "no profile in ${_slice}")
endif()

function(_edge ROW Y_OUT V_OUT)
    string(STRIP "${ROW}" _row)
    string(REGEX REPLACE "[ \t]+" ";" _fields "${_row}")
    list(GET _fields 0 _y)
    list(GET _fields 1 _v)
    set(${Y_OUT} "${_y}" PARENT_SCOPE)
    set(${V_OUT} "${_v}" PARENT_SCOPE)
endfunction()
list(GET _rows 0 _south_row)
list(GET _rows -1 _north_row)
_edge("${_south_row}" _south_y _south)
_edge("${_north_row}" _north_y _north)

message(STATUS "fire_fuel_load at x = ${X} over ${_n} fire cells:")
message(STATUS "  south edge y = ${_south_y}: ${_south} kg/m2, expected [${SOUTH_MIN}, ${SOUTH_MAX}]")
message(STATUS "  north edge y = ${_north_y}: ${_north} kg/m2, expected [${NORTH_MIN}, ${NORTH_MAX}]")
set(_failed FALSE)
if(_south LESS SOUTH_MIN OR _south GREATER SOUTH_MAX)
    message(SEND_ERROR "the south edge carries ${_south} kg/m2")
    set(_failed TRUE)
endif()
if(_north LESS NORTH_MIN OR _north GREATER NORTH_MAX)
    message(SEND_ERROR "the north edge carries ${_north} kg/m2")
    set(_failed TRUE)
endif()
if(_failed)
    message(FATAL_ERROR "the fuel map is not placed with its first data row on the north edge")
endif()
