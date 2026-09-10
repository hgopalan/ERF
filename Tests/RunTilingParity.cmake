cmake_minimum_required(VERSION 3.24)

# Run one deck twice over two decompositions of the same grid, then require the
# two plotfiles (3D and 2D) to agree.  The second run is the reference, so no
# gold file is needed.
#
# SPLIT tiles (the default): the CPU default MFIter tiling against tiling off.
# PBL schemes build per-tile planar work arrays; a kernel that loops over the
# valid box instead of the tile reads outside them.  Boxes wider than the tile
# size in y are the case that exposes this.
#
# SPLIT boxes: the deck's own boxes, tiled, against a single box, untiled.  A
# horizontal stencil that stops at box edges gives a different answer on the
# single box.
#
# SPLIT fine_z: level 1 split in z (amr.max_grid_size_z = unlimited 16) against
# level 1 left whole, both untiled.  Where a column solve is on, ERF joins fine
# boxes stacked in z into whole columns, so both runs must give the same answer.
#
# PLT2DFILE may be empty for a deck that writes no 2D plotfile.

foreach(_required MPIEXEC MPIEXEC_NUMPROC_FLAG NRANKS TEST_EXE INPUT
                  WORKING_DIRECTORY FCOMPARE RTOL ATOL PLTFILE PLT2DFILE)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "RunTilingParity.cmake requires ${_required}")
  endif()
endforeach()

set(_mpi_run "${MPIEXEC}" "${MPIEXEC_NUMPROC_FLAG}" "${NRANKS}")
set(_mpi_one "${MPIEXEC}" "${MPIEXEC_NUMPROC_FLAG}" "1")
if(DEFINED MPIEXEC_PREFLAGS AND NOT "${MPIEXEC_PREFLAGS}" STREQUAL "")
  separate_arguments(_mpi_preflags UNIX_COMMAND "${MPIEXEC_PREFLAGS}")
  list(APPEND _mpi_run ${_mpi_preflags})
  list(APPEND _mpi_one ${_mpi_preflags})
endif()

set(_runtime_options)
if(DEFINED RUNTIME_OPTIONS AND NOT "${RUNTIME_OPTIONS}" STREQUAL "")
  separate_arguments(_runtime_options UNIX_COMMAND "${RUNTIME_OPTIONS}")
endif()

# The tiled runs spell out the AMReX CPU default (1024000 8 8) so the test does
# not depend on it; the untiled runs make every tile its whole box.  The single
# box keeps the deck's max_grid_size_z, so the columns stay whole, and runs on
# one rank: on more, amr.refine_grid_layout chops it into a box per rank.
set(_tiled   "fabarray.mfiter_tile_size=1024000 8 8")
set(_untiled "fabarray.mfiter_tile_size=1024000 1024000 1024000")
if(NOT DEFINED SPLIT OR "${SPLIT}" STREQUAL "" OR "${SPLIT}" STREQUAL "tiles")
  set(_runs tiled untiled)
  set(_tiled_options   "${_tiled}")
  set(_untiled_options "${_untiled}")
elseif("${SPLIT}" STREQUAL "boxes")
  set(_runs multibox singlebox)
  set(_multibox_options  "${_tiled}")
  set(_singlebox_options "${_untiled}" "amr.max_grid_size_x=1048576" "amr.max_grid_size_y=1048576")
  set(_singlebox_one_rank TRUE)
elseif("${SPLIT}" STREQUAL "fine_z")
  set(_runs finesplit finewhole)
  set(_finesplit_options "${_untiled}" "amr.max_grid_size_z=1048576 16")
  set(_finewhole_options "${_untiled}" "amr.max_grid_size_z=1048576 1048576")
else()
  message(FATAL_ERROR "RunTilingParity.cmake: SPLIT must be tiles, boxes or fine_z, not ${SPLIT}")
endif()
list(GET _runs 0 _candidate)
list(GET _runs 1 _reference)

foreach(_run IN LISTS _runs)
  set(_run_options ${_${_run}_options})
  if(_${_run}_one_rank)
    set(_mpi ${_mpi_one})
  else()
    set(_mpi ${_mpi_run})
  endif()
  file(REMOVE_RECURSE "${WORKING_DIRECTORY}/${_run}_plt" "${WORKING_DIRECTORY}/${_run}_plt2d")
  execute_process(
    COMMAND ${_mpi} "${TEST_EXE}" "${INPUT}" ${_runtime_options}
            ${_run_options}
            "erf.plot_file_1=${_run}_plt"
            "erf.plot2d_file_1=${_run}_plt2d"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${WORKING_DIRECTORY}/${_run}.log"
    ERROR_FILE "${WORKING_DIRECTORY}/${_run}.log"
    RESULT_VARIABLE _result)
  if(NOT _result EQUAL 0)
    string(REPLACE ";" " " _run_options_text "${_run_options}")
    message(FATAL_ERROR "${_run} run (${_run_options_text}) failed "
                        "with exit code ${_result}; see ${WORKING_DIRECTORY}/${_run}.log")
  endif()
endforeach()

set(_kinds plt)
if(NOT "${PLT2DFILE}" STREQUAL "")
  list(APPEND _kinds plt2d)
endif()
foreach(_kind IN LISTS _kinds)
  if(_kind STREQUAL "plt")
    set(_step "${PLTFILE}")
  else()
    set(_step "${PLT2DFILE}")
  endif()
  execute_process(
    COMMAND ${_mpi_one} "${FCOMPARE}" --abort_if_not_all_found -a
            -r "${RTOL}" --abs_tol "${ATOL}"
            "${WORKING_DIRECTORY}/${_reference}_${_kind}${_step}"
            "${WORKING_DIRECTORY}/${_candidate}_${_kind}${_step}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_FILE "${WORKING_DIRECTORY}/fcompare_${_kind}.log"
    ERROR_FILE "${WORKING_DIRECTORY}/fcompare_${_kind}.log"
    RESULT_VARIABLE _result)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${_candidate} and ${_reference} ${_kind}${_step} differ; "
                        "see ${WORKING_DIRECTORY}/fcompare_${_kind}.log")
  endif()
endforeach()
