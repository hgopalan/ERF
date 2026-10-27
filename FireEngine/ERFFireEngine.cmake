# The native fire engine directory, for CMake: ERF includes this file when
# ERF_FIRE_ENGINE_DIRS lists this directory and calls erf_fire_engine_add with
# its object library (ERF_FIRE_ENGINE_DIR is this directory during the call).
function(erf_fire_engine_add target)
  target_sources(${target} PRIVATE
    ${ERF_FIRE_ENGINE_DIR}/ERF_NativeFireEngine.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireEnginePrerequisites.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireWindAdjust.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireTerrainReader.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_TerrainSlope.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_Rothermel.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireLayer.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireCheckpoint.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireSuppression.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_Fire.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FirePlotfile.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_StructureIgnition.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_CustomFuel.cpp
  )
  target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${ERF_FIRE_ENGINE_DIR}>)
endfunction()
