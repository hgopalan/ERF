#ifndef ERF_FIRE_H
#define ERF_FIRE_H

/**
 * @file ERF_Fire.H
 * @brief Main include file for ERF fire module
 *
 * Includes all fire module headers for integration with ERF atmospheric solver.
 * The shared coupling layer (Source/FireCoupling: FireCoupler, FireEngine)
 * places a 2D fire grid on the level it anchors to and couples a fire engine
 * to the atmosphere through a lagged surface heat flux. The native engine
 * (Source/Fire: FireLayer) advances a fire front (level set or FARSITE) with
 * eight rate-of-spread models, fuel moisture, spotting, crown fire,
 * suppression and structure ignition.
 */

#ifdef ERF_ENABLE_FIRE

#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <ERF_FirePrerequisites.H>
#include <ERF_FireWindExtract.H>
#include <ERF_FireSurfaceHeight.H>
#include <ERF_FireEngine.H>
#include <ERF_FireCoupler.H>
#include <ERF_TerrainSlope.H>
#include <ERF_Rothermel.H>
#include <ERF_FarsiteEllipse.H>
#include <ERF_FireIgnition.H>
#include <ERF_FireLayer.H>

#endif  // ERF_ENABLE_FIRE

#endif  // ERF_FIRE_H
