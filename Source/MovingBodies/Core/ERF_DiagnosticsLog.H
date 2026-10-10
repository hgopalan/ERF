#ifndef ERF_DIAGNOSTICS_LOG_H
#define ERF_DIAGNOSTICS_LOG_H

/** Opening of the moving-bodies diagnostics files for a fresh or a restarted run. */

#include <fstream>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

namespace erf_actuator {

/**
 * Open a diagnostics text file for writing (the caller writes CSV or space-separated rows),
 * creating its directory. With truncate (a fresh run) the file starts empty; without it (a
 * restarted run) the rows are appended to what the run it continues wrote. Returns whether the
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

/**
 * Whether fname exists, as the I/O rank sees it, on every rank: a checkpoint read that every rank
 * enters (ParallelDescriptor::ReadAndBcastFile) must be decided alike everywhere, whatever a
 * parallel file system shows each node.
 */
inline bool file_exists_everywhere (const std::string& fname)
{
    int exists = amrex::ParallelDescriptor::IOProcessor() ? (amrex::FileExists(fname) ? 1 : 0) : 0;
    amrex::ParallelDescriptor::Bcast(&exists, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    return exists != 0;
}

/**
 * On a restart, drop the rows a log gained after the checkpoint (a run that went on past it before
 * being restarted from it), so the appended rows continue the series once. The first column is
 * the row's time (s). keep_at_chk says whether a row stamped at the checkpoint time belongs to the
 * run before it (rows stamped at the end of their step) or after it (rows stamped at its start).
 * Done by the I/O rank alone; a missing file is left alone.
 */
inline void trim_log_after (const std::string& fname, double t_chk, bool keep_at_chk, double tol)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ifstream in(fname);
    if (!in) { return; }
    std::vector<std::string> keep;
    std::string line;
    bool header = true;
    std::size_t dropped = 0;
    while (std::getline(in, line)) {
        if (header) { keep.push_back(line); header = false; continue; }
        if (line.empty()) { continue; }
        double t = 0.0;
        try { t = std::stod(line.substr(0, line.find(','))); } catch (...) { keep.push_back(line); continue; }
        const bool before = keep_at_chk ? (t <= t_chk + tol) : (t < t_chk - tol);
        if (before) { keep.push_back(line); } else { ++dropped; }
    }
    in.close();
    if (dropped == 0) { return; }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { amrex::Abort("cannot rewrite moving-bodies diagnostics file '" + fname + "'"); }
    for (const auto& l : keep) { out << l << "\n"; }
    amrex::Print() << "Moving bodies: dropped " << dropped << " row(s) after t = " << t_chk << " s from " << fname << "\n";
}

} // namespace erf_actuator

#endif
