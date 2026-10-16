#ifndef ERF_DIAGNOSTICS_LOG_H
#define ERF_DIAGNOSTICS_LOG_H

/** Opening of the moving-bodies diagnostics files for a fresh or a restarted run. */

#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
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
 * One row of a time-series log, comma separated, with 10 significant digits: the time, then the body's
 * name when body is given (the per_rank and farm formats, which hold several bodies), then the n values.
 */
inline void write_log_row (std::ostream& out, double time, const std::string* body, const amrex::Real* v, int n)
{
    out << std::setprecision(10) << time;
    if (body != nullptr) { out << "," << *body; }
    for (int k = 0; k < n; ++k) { out << "," << v[k]; }
    out << "\n";
}

/**
 * The appended diagnostics files of one writer, each opened once and kept open: a farm writes a row
 * to thousands of files every diagnostics_int steps, and opening and closing each of them every time
 * costs far more than the row. A file is opened at its first row, by open_log() (truncated on a fresh
 * run, appended to after a restart, with the header when open_log() asks for one); the rows are
 * buffered, flushed by flush() (at every checkpoint, so a restart's trim sees every row written
 * before it) and when the writer is destroyed, at the end of the run.
 */
class LogStreams
{
public:
    /**
     * The stream of fname, opened at the first call: truncate and header as for open_log() (the
     * header, without its newline, is written when open_log() asks for one). Later calls return the
     * open stream and ignore truncate and header.
     */
    std::ofstream& get (const std::string& fname, bool truncate, const std::string& header)
    {
        auto it = m_files.find(fname);
        if (it != m_files.end()) { return *it->second; }
        auto out = std::make_unique<std::ofstream>();
        if (open_log(*out, fname, truncate) && !header.empty()) { *out << header << "\n"; }
        return *m_files.emplace(fname, std::move(out)).first->second;
    }
    bool is_open (const std::string& fname) const { return m_files.count(fname) > 0; }
    std::size_t size () const { return m_files.size(); }
    /** Write every buffered row to its file; aborts when a write failed. */
    void flush ()
    {
        for (auto& f : m_files) {
            f.second->flush();
            if (!*f.second) { amrex::Abort("cannot write the moving-bodies diagnostics file '" + f.first + "'"); }
        }
    }
    /** Flush and close every file: the next get() opens it again. */
    void close ()
    {
        flush();
        m_files.clear();
    }

private:
    std::map<std::string, std::unique_ptr<std::ofstream>> m_files;
};

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
