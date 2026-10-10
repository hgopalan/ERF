#ifndef ERF_FIRE_H
#define ERF_FIRE_H

/**
 * @file ERF_Fire.H
 * @brief Main include file for ERF fire module
 *
 * Includes all fire module headers for integration with ERF atmospheric solver.
 * The fire module advances a fire front (level set or FARSITE) on a 2D grid
 * refined from the level it anchors to, with eight rate-of-spread models,
 * fuel moisture, spotting, crown fire, suppression and structure ignition,
 * coupled to the atmosphere through a lagged surface heat flux.
 */

#ifdef ERF_ENABLE_FIRE

#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <ERF_FirePrerequisites.H>
#include <ERF_FireWindExtract.H>
#include <ERF_TerrainSlope.H>
#include <ERF_Rothermel.H>
#include <ERF_FarsiteEllipse.H>
#include <ERF_FireIgnition.H>
#include <ERF_FireLayer.H>

#endif  // ERF_ENABLE_FIRE

#endif  // ERF_FIRE_H
