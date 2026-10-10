#include "ERF_MoorDynInputWriter.H"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <AMReX.H>
#include <AMReX_Utility.H>

using namespace amrex;

namespace erf_conductors {

std::string moordyn_input_text (const LineInputs& s, const ConductorInputs& in, Real gravity)
{
    const int N = s.num_spans();
    const bool strings = s.has_insulators();
    const std::string ins_type = s.name + "_insulator";
    // the point a span ends on at attachment k: the fixed point itself, or the free point under the
    // tower's string
    auto hang = [&] (int k) { return (strings && k > 0 && k < N) ? N + 1 + k : k + 1; };
    std::ostringstream out;
    // material constants to twelve digits, or as many as a Real holds: a single-precision build writes
    // 0.0281, not 0.0280999993
    out << std::setprecision(std::min(12, std::numeric_limits<Real>::digits10));
    // positions and lengths in double to twelve digits, so a single-precision build keeps every bit of
    // them (at 30 km six digits would move a point by 5 cm, and 300.0104 m would become 300.01)
    auto exact = [] (double v) {
        std::ostringstream o;
        o << std::setprecision(12) << v;
        return o.str();
    };
    const double offset = static_cast<double>(in.surface_offset);
    out << "MoorDyn-C input written by ERF for conductor line " << s.name << " (erf.conductors." << s.name << ".*)\n"
        << "----------------------- LINE TYPES ------------------------------------------\n"
        << "TypeName   Diam     Mass/m     EA         BA/-zeta    EI         Cd     Ca     CdAx    CaAx\n"
        << "(name)     (m)      (kg/m)     (N)        (N-s/-)     (N-m^2)    (-)    (-)    (-)     (-)\n"
        << s.name << "   " << s.diameter << "   " << s.mass_per_length << "   " << s.axial_stiffness
        << "   " << -s.damping_ratio << "   0   " << s.drag_coefficient << "   0.0   0.0   0.0\n";
    if (strings) {
        out << ins_type << "   " << s.insulator_diameter << "   " << s.insulator_mass / s.insulator_length << "   "
            << LineInputs::insulator_axial_stiffness << "   " << -s.damping_ratio << "   0   "
            << LineInputs::insulator_drag_coefficient << "   0.0   0.0   0.0\n";
    }
    out << "---------------------- POINT PROPERTIES --------------------------------\n"
        << "ID    Type      X       Y       Z       Mass   Volume  CdA    Ca\n"
        << "(#)   (-)       (m)     (m)     (m)     (kg)   (m^3)   (m^2)  (-)\n";
    // the towers' cross-arms are coupled points when the towers move: ERF drives them and takes their pull
    const bool moving = in.towers_move(s);
    for (int k = 0; k <= N; ++k) {
        const auto& p = s.point(k);
        const char* type = (moving && k > 0 && k < N) ? "Coupled" : "Fixed  ";
        out << k + 1 << "     " << type << "   " << exact(p[0]) << "   " << exact(p[1]) << "   "
            << exact(static_cast<double>(p[2]) - offset) << "   0   0   0   0\n";
    }
    if (strings) {
        for (int k = 1; k < N; ++k) {
            const auto& p = s.point(k);
            out << N + 1 + k << "     Free      " << exact(p[0]) << "   " << exact(p[1]) << "   "
                << exact(static_cast<double>(p[2]) - static_cast<double>(s.insulator_length) - offset) << "   0   0   0   0\n";
        }
    }
    out << "---------------------- LINES ----------------------------------------\n"
        << "ID   LineType   AttachA  AttachB  UnstrLen  NumSegs  LineOutputs\n"
        << "(#)   (name)     (#)      (#)       (m)       (-)     (-)\n";
    for (int k = 0; k < N; ++k) {
        out << k + 1 << "     " << s.name << "      " << hang(k) << "        " << hang(k + 1) << "         "
            << exact(s.lengths[static_cast<std::size_t>(k)]) << "   " << s.segments << "   -\n";
    }
    if (strings) {
        for (int k = 1; k < N; ++k) {
            out << N + k << "     " << ins_type << "      " << k + 1 << "        " << N + 1 + k << "         "
                << s.insulator_length << "   " << LineInputs::insulator_segments << "   -\n";
        }
    }
    out << "---------------------- OPTIONS -----------------------------------------\n"
        << "0             writeLog      ERF writes the diagnostics\n";
    if (in.moordyn_dt > 0.0) {
        out << in.moordyn_dt << "   dtM           upper bound on MoorDyn's internal step (s, erf.conductors.moordyn_dt)\n";
    }
    out << in.moordyn_cfl << "   CFL           Courant factor that sets MoorDyn's internal step (erf.conductors.moordyn_cfl)\n"
        << gravity << "   g             gravity (m/s^2)\n"
        << in.air_density << "   WtrDnsty      the fluid is air (kg/m^3)\n"
        << exact(2.0 * offset) << "   WtrDpth       flat bottom, far below the ground (m)\n"
        << "1             WaveKin       the fluid kinematics come through the API\n"
        << "0             ICgenDynamic  stationary initial-condition solver\n"
        << "1             disableOutput\n"
        << "1             disableOutTime\n"
        << "------------------------- need this line --------------------------------------\n";
    return out.str();
}

void write_moordyn_input (const std::string& fname, const LineInputs& s, const ConductorInputs& in, Real gravity)
{
    const auto slash = fname.rfind('/');
    if (slash != std::string::npos) { UtilCreateDirectory(fname.substr(0, slash), 0755); }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { Abort("erf.conductors." + s.name + ": cannot write the MoorDyn input file '" + fname + "'"); }
    out << moordyn_input_text(s, in, gravity);
}

} // namespace erf_conductors
