#ifndef ERF_DIAGNOSTICS_LOG_H
#define ERF_DIAGNOSTICS_LOG_H

/** Opening of the moving-bodies diagnostics files for a fresh or a restarted run. */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#include <cstdio>
#else
#include <sys/resource.h>
#endif

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

namespace erf_actuator {

/**
 * Make dir (and its parents) the first time this process asks for it; later calls return at once.
 * A farm rewrites or opens thousands of files under a few directories, and making the directory
 * each time costs one mkdir per path component on the file system's metadata server. A directory
 * removed after its first use is not made again: the next open of a file in it stops the run.
 */
inline void make_directory_once (const std::string& dir)
{
    static std::set<std::string> made;
    if (dir.empty() || made.count(dir) > 0) { return; }
    if (!amrex::UtilCreateDirectory(dir, 0755)) { amrex::Abort("cannot create the moving-bodies diagnostics directory '" + dir + "'"); }
    made.insert(dir);
}

/**
 * Open a diagnostics text file for writing (the caller writes CSV or space-separated rows),
 * creating its directory (once per process, make_directory_once()). With truncate (a fresh run)
 * the file starts empty; without it (a restarted run) the rows are appended to what the run it
 * continues wrote. Returns whether the caller must write the header: always after truncating, and
 * on append only when the file is new or empty (a restart in a clean directory).
 */
inline bool open_log (std::ofstream& out, const std::string& fname, bool truncate)
{
    const auto slash = fname.rfind('/');
    if (slash != std::string::npos) { make_directory_once(fname.substr(0, slash)); }
    bool need_header = truncate;
    if (!truncate) {
        std::ifstream probe(fname, std::ios::ate);
        need_header = !probe || probe.tellg() <= 0;
    }
    out.open(fname, truncate ? std::ios::trunc : std::ios::app);
    if (!out) {
        amrex::Abort("cannot open moving-bodies diagnostics file '" + fname +
                     "' (with many bodies per rank: raise the open-file limit, ulimit -n, or lower erf.moving_bodies.max_open_logs)");
    }
    return need_header;
}

/** The process's limit on open files: the soft RLIMIT_NOFILE (_getmaxstdio() on Windows); 0 when it has none. */
inline double open_file_limit ()
{
#ifdef _WIN32
    return static_cast<double>(_getmaxstdio());
#else
    rlimit rl{};
    return (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) ? static_cast<double>(rl.rlim_cur) : 0.0;
#endif
}

/**
 * How many logs one LogStreams keeps open by default: a quarter of the process's limit on open
 * files, so that MPI, OpenFAST's own output and the plotfile and checkpoint writers keep the rest,
 * and between 16 and 4096 (4096 with no limit).
 */
inline std::size_t default_max_open_logs ()
{
    const double limit = open_file_limit();
    if (limit <= 0.0) { return 4096; }
    return static_cast<std::size_t>(std::min(4096.0, std::max(16.0, limit / 4.0)));
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
 * before it, and at the end of the run) and when the writer is destroyed.
 *
 * At most max_open files are open at once (an owner of many bodies writes several logs for each,
 * which would pass the process's limit on open files). Opening one more first closes the file
 * written to most recently: the logs are written in the same order every step, so on average
 * max_open - 1 of them are still open at their next row (closing the least recent file instead
 * would close and open every file at every row). A closed file's next
 * row opens it again and appends, without truncating it or writing its header again, so every file
 * holds the same bytes as with no limit.
 */
class LogStreams
{
public:
    explicit LogStreams (std::size_t max_open = default_max_open_logs ()) : m_max_open(std::max<std::size_t>(max_open, 1)) {}

    /**
     * The stream of fname, opened at the first call: truncate and header as for open_log() (the
     * header, without its newline, is written when open_log() asks for one). Later calls return the
     * open stream, or open the file again to append when it was closed to keep within max_open,
     * and ignore truncate and header.
     */
    std::ofstream& get (const std::string& fname, bool truncate, const std::string& header)
    {
        auto it = m_files.find(fname);
        if (it != m_files.end()) {
            m_order.splice(m_order.end(), m_order, it->second.at);
            return *it->second.out;
        }
        while (m_files.size() >= m_max_open) { close_one(m_order.back()); }
        ++m_opens;
        auto out = std::make_unique<std::ofstream>();
        if (m_started.count(fname) > 0) {
            out->open(fname, std::ios::app);
            if (!*out) {
                amrex::Abort("cannot open moving-bodies diagnostics file '" + fname +
                             "' again (raise the open-file limit, ulimit -n, or lower erf.moving_bodies.max_open_logs)");
            }
        } else {
            if (open_log(*out, fname, truncate) && !header.empty()) { *out << header << "\n"; }
            m_started.insert(fname);
        }
        m_order.push_back(fname);
        Entry& e = m_files[fname];
        e.out = std::move(out);
        e.at = std::prev(m_order.end());
        return *e.out;
    }
    bool is_open (const std::string& fname) const { return m_files.count(fname) > 0; }
    /** The number of files open now (at most max_open()). */
    std::size_t size () const { return m_files.size(); }
    std::size_t max_open () const { return m_max_open; }
    /** How many times a file was opened (first opens and reopens) since construction. */
    std::size_t opens () const { return m_opens; }
    /** Write every buffered row to its file; aborts when a write failed. */
    void flush ()
    {
        for (auto& f : m_files) {
            f.second.out->flush();
            if (!*f.second.out) { amrex::Abort("cannot write the moving-bodies diagnostics file '" + f.first + "'"); }
        }
    }
    /** Flush and close every file: the next get() opens it as at its first call. */
    void close ()
    {
        flush();
        m_files.clear();
        m_order.clear();
        m_started.clear();
    }

private:
    struct Entry
    {
        std::unique_ptr<std::ofstream> out;
        std::list<std::string>::iterator at;   //!< the file's place in m_order
    };
    /** Flush and close fname, which stays started: its next get() appends. By value: the caller passes an element of m_order, erased here. */
    void close_one (const std::string fname)
    {
        auto it = m_files.find(fname);
        it->second.out->close();
        if (!*it->second.out) { amrex::Abort("cannot write the moving-bodies diagnostics file '" + fname + "'"); }
        m_order.erase(it->second.at);
        m_files.erase(it);
    }

    std::size_t m_max_open;
    std::size_t m_opens{0};
    std::map<std::string, Entry> m_files;   //!< the open files
    std::list<std::string> m_order;         //!< the open files, written to most recently last
    std::set<std::string> m_started;        //!< every file opened since construction or close()
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
