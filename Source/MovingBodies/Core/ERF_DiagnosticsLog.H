#ifndef ERF_DIAGNOSTICS_LOG_H
#define ERF_DIAGNOSTICS_LOG_H

#include <fstream>
#include <string>

#include <AMReX.H>
#include <AMReX_Utility.H>

namespace erf_actuator {

/**
 * Open a CSV diagnostics file for writing, creating its directory. A fresh run truncates
 * the file; a restarted run appends to what the run it continues wrote. Returns whether the
 * caller must write the header: always after truncating, and on append only when the file is
 * new or empty (a restart in a clean directory).
 */
inline bool open_log (std::ofstream& out, const std::string& fname, bool truncate)
{
    const auto slash = fname.rfind('/');
    if (slash != std::string::npos) { amrex::UtilCreateDirectory(fname.substr(0, slash), 0755); }
    bool need_header = truncate;
    if (!truncate) {
        std::ifstream probe(fname, std::ios::ate);
        need_header = !probe || probe.tellg() <= 0;
    }
    out.open(fname, truncate ? std::ios::trunc : std::ios::app);
    if (!out) { amrex::Abort("cannot open moving-bodies diagnostics file '" + fname + "'"); }
    return need_header;
}

} // namespace erf_actuator

#endif
