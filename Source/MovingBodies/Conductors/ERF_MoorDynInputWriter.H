#ifndef ERF_MOORDYN_INPUT_WRITER_H
#define ERF_MOORDYN_INPUT_WRITER_H

/** The MoorDyn-C input file of one conductor line, generated from the ERF inputs, and the MoorDyn -> ERF frame conversion. */

#include <array>
#include <string>

#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"

namespace erf_conductors {

/**
 * The MoorDyn-C input file of one conductor line, generated from the ERF inputs.
 *
 * With N = s.num_spans(), points 1 .. N+1 are the attachment points in order (end_a, the towers,
 * end_b). end_a and end_b are Fixed; the towers are Fixed, or Coupled when the line's towers move
 * (ConductorInputs::towers_move), and ERF then drives them. With insulator strings, points
 * N+2 .. 2N are the Free points at the lower end of the string at each tower, where the conductor
 * hangs. MoorDyn lines 1 .. N are the spans, from end_a to end_b; with insulator strings, MoorDyn
 * lines N+1 .. 2N-1 are the strings, each from its tower down to its free point.
 *
 * MoorDyn applies its fluid loads to nodes below its free surface at z = 0 and puts a flat
 * bottom at -WtrDpth, so the line is written in MoorDyn's frame, every height lowered by
 * surface_offset (z_md = z_erf - surface_offset); the bottom is placed another surface_offset
 * lower, at ERF z = -surface_offset, where nothing reaches it. The fluid density is the air
 * density, the fluid kinematics are external (WaveKin = 1, the wind ERF samples at the nodes),
 * the initial shape comes from MoorDyn's stationary solver, and MoorDyn's own outputs are off
 * (ERF writes the diagnostics).
 *
 * @param s the line; its attachment points already at their absolute heights (the terrain added)
 * @param in the shared settings (air density, surface offset, moordyn_dt, and moordyn_cfl, MoorDyn's
 *           Courant (CFL) factor) and the tower types, which decide whether the towers are Coupled points
 * @param gravity ERF's gravity (m/s^2)
 */
std::string moordyn_input_text (const LineInputs& s, const ConductorInputs& in, amrex::Real gravity);

/** Write moordyn_input_text() to a file (creating its directory); aborts when the file cannot be written. */
void write_moordyn_input (const std::string& fname, const LineInputs& s, const ConductorInputs& in, amrex::Real gravity);

/** A position (m) in ERF's frame from MoorDyn's: z_erf = z_md + surface_offset. */
inline std::array<amrex::Real,3> to_erf_frame (const std::array<double,3>& p, amrex::Real surface_offset)
{
    // the offset added in double: in a single-precision build MoorDyn's z, near -surface_offset, holds only centimetres
    return {{static_cast<amrex::Real>(p[0]), static_cast<amrex::Real>(p[1]),
             static_cast<amrex::Real>(p[2] + static_cast<double>(surface_offset))}};
}

} // namespace erf_conductors

#endif
