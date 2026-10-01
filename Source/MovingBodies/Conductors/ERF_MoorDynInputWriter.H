#ifndef ERF_MOORDYN_INPUT_WRITER_H
#define ERF_MOORDYN_INPUT_WRITER_H

#include <array>
#include <string>

#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"

namespace erf_conductors {

/**
 * The MoorDyn-C input file of one span, generated from the ERF inputs.
 *
 * MoorDyn applies its fluid loads to nodes below its free surface at z = 0 and puts a flat
 * bottom at -WtrDpth, so the span is written in MoorDyn's frame, every height lowered by
 * surface_offset (z_md = z_erf - surface_offset); the bottom is placed another surface_offset
 * lower, where nothing reaches it. The fluid density is the air density, the fluid kinematics
 * are external (WaveKin = 1, the wind ERF samples at the nodes), both attachments are fixed,
 * the initial shape comes from MoorDyn's stationary solver, and MoorDyn's own outputs are off
 * (ERF writes the diagnostics).
 *
 * @param s the span; end_a and end_b already at their absolute heights (the terrain added)
 * @param in the shared settings (air density, surface offset, MoorDyn step, CFL)
 * @param gravity ERF's gravity (m/s^2)
 */
std::string moordyn_input_text (const SpanInputs& s, const ConductorInputs& in, amrex::Real gravity);

/** Write moordyn_input_text() to a file (creating its directory); aborts when the file cannot be written. */
void write_moordyn_input (const std::string& fname, const SpanInputs& s, const ConductorInputs& in, amrex::Real gravity);

/** A position in ERF's frame from MoorDyn's (the surface offset added back). */
inline std::array<amrex::Real,3> to_erf_frame (const std::array<double,3>& p, amrex::Real surface_offset)
{
    return {{static_cast<amrex::Real>(p[0]), static_cast<amrex::Real>(p[1]), static_cast<amrex::Real>(p[2]) + surface_offset}};
}

} // namespace erf_conductors

#endif
