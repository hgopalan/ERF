#include "ERF_MoorDynInputWriter.H"

#include <fstream>
#include <iomanip>
#include <sstream>

#include <AMReX.H>
#include <AMReX_Utility.H>

using namespace amrex;

namespace erf_conductors {

std::string moordyn_input_text (const SpanInputs& s, const ConductorInputs& in, Real gravity)
{
    std::ostringstream out;
    out << std::setprecision(12);
    out << "MoorDyn-C input written by ERF for conductor span " << s.name << " (erf.conductors." << s.name << ".*)\n"
        << "----------------------- LINE TYPES ------------------------------------------\n"
        << "TypeName   Diam     Mass/m     EA         BA/-zeta    EI         Cd     Ca     CdAx    CaAx\n"
        << "(name)     (m)      (kg/m)     (N)        (N-s/-)     (N-m^2)    (-)    (-)    (-)     (-)\n"
        << s.name << "   " << s.diameter << "   " << s.mass_per_length << "   " << s.axial_stiffness
        << "   " << -s.damping_ratio << "   0   " << s.drag_coefficient << "   0.0   0.0   0.0\n"
        << "---------------------- POINT PROPERTIES --------------------------------\n"
        << "ID    Type      X       Y       Z       Mass   Volume  CdA    Ca\n"
        << "(#)   (-)       (m)     (m)     (m)     (kg)   (m^3)   (m^2)  (-)\n"
        << "1     Fixed     " << s.end_a[0] << "   " << s.end_a[1] << "   " << s.end_a[2] - in.surface_offset << "   0   0   0   0\n"
        << "2     Fixed     " << s.end_b[0] << "   " << s.end_b[1] << "   " << s.end_b[2] - in.surface_offset << "   0   0   0   0\n"
        << "---------------------- LINES ----------------------------------------\n"
        << "ID   LineType   AttachA  AttachB  UnstrLen  NumSegs  LineOutputs\n"
        << "(#)   (name)     (#)      (#)       (m)       (-)     (-)\n"
        << "1     " << s.name << "      1        2         " << s.length << "   " << s.segments << "   -\n"
        << "---------------------- OPTIONS -----------------------------------------\n"
        << "0             writeLog      ERF writes the diagnostics\n";
    if (in.moordyn_dt > 0.0) {
        out << in.moordyn_dt << "   dtM           upper bound on MoorDyn's internal step (s, erf.conductors.moordyn_dt)\n";
    }
    out << in.moordyn_cfl << "   CFL           Courant factor that sets MoorDyn's internal step (erf.conductors.moordyn_cfl)\n"
        << gravity << "   g             gravity (m/s^2)\n"
        << in.air_density << "   WtrDnsty      the fluid is air (kg/m^3)\n"
        << 2.0 * in.surface_offset << "   WtrDpth       flat bottom, far below the ground (m)\n"
        << "1             WaveKin       the fluid kinematics come through the API\n"
        << "0             ICgenDynamic  stationary initial-condition solver\n"
        << "1             disableOutput\n"
        << "1             disableOutTime\n"
        << "------------------------- need this line --------------------------------------\n";
    return out.str();
}

void write_moordyn_input (const std::string& fname, const SpanInputs& s, const ConductorInputs& in, Real gravity)
{
    const auto slash = fname.rfind('/');
    if (slash != std::string::npos) { UtilCreateDirectory(fname.substr(0, slash), 0755); }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { Abort("erf.conductors." + s.name + ": cannot write the MoorDyn input file '" + fname + "'"); }
    out << moordyn_input_text(s, in, gravity);
}

} // namespace erf_conductors
