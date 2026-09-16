#ifndef ERF_FIRE_SUPPRESSION_H
#define ERF_FIRE_SUPPRESSION_H

/**
 * @file ERF_FireSuppression.H
 * @brief Suppression actions on the fire grid: fire lines built at a rate,
 *        retardant drops, a hold test on the flame length, and burnout
 *        ignitions along a built line, all read from a watched action file.
 *
 * The action file is a whitespace-separated table with one action per line
 * and `#` comments (see parse_suppression_line for the format). It is read on
 * the IO rank at start-up and again, every erf.fire.suppression.poll_interval
 * fire steps, whenever its modification time changed; the text is broadcast
 * and every rank parses the same bytes. Ids already known are skipped, so an
 * action is never applied twice and an edited line has no effect: to change an
 * action, add a new id. Every applied, completed, expired, hold-failed and
 * burnout event goes to erf.fire.suppression.log with its time and step, so a
 * run steered live can be reproduced from the log.
 *
 * Both propagation paths see the actions through the non-burnable mask
 * (fire_nonburnable, an OR of the static mask and the suppression mask) and,
 * for a drop with 0 < ros_factor < 1, through fire_ros_factor multiplied into
 * the rate of spread before the front update.
 */

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <ERF_IgnitionSchedule.H>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace fire_suppression {

/// Kind of suppression action.
enum ActionType : int { line = 0, drop = 1, burnout = 2 };

/// One action of the file with its runtime state (the state is checkpointed).
struct SuppressionAction
{
    std::string id;
    int         type          = line;
    amrex::Real start_s       = 0.0;    ///< Time the action starts [s]
    amrex::Real expiry_s      = -1.0;   ///< Duration after start_s [s]; -1 = permanent
    amrex::Real rate          = 0.0;    ///< line: construction rate [m/s]
    amrex::Real ros_factor    = 0.0;    ///< drop: multiplier on the rate of spread [0,1)
    amrex::Real offset        = 0.0;    ///< burnout: distance from the line on the fire side [m]
    amrex::Real flame_limit_m = -1.0;   ///< line: hold test limit [m]; -1 = always holds
    std::string ref;                    ///< burnout: id of the line it follows
    std::vector<amrex::Real> verts;     ///< x0 y0 x1 y1 ... (polyline or polygon) [m]

    // Runtime state
    bool        applied    = false;     ///< start_s has passed and the action is (or was) active
    bool        expired    = false;     ///< removed after start_s + expiry_s
    bool        completed  = false;     ///< line fully built (logged once)
    amrex::Real built_m    = 0.0;       ///< line: length built so far [m]
    amrex::Real burnout_m  = 0.0;       ///< burnout: length of the reference line already fired
    amrex::Long cells_last = 0;         ///< cells the action covered at its last count

    /// Polyline length [m].
    amrex::Real length () const
    {
        amrex::Real L = 0.0;
        for (std::size_t k = 0; k + 3 < verts.size(); k += 2) {
            const amrex::Real dx = verts[k+2] - verts[k];
            const amrex::Real dy = verts[k+3] - verts[k+1];
            L += std::sqrt(dx*dx + dy*dy);
        }
        return L;
    }
    int nverts () const { return static_cast<int>(verts.size() / 2); }
};

/// The action type's name in the file and the log.
inline const char* type_name (int t)
{
    return (t == line) ? "line" : (t == drop) ? "drop" : "burnout";
}

/**
 * @brief Parse one non-comment line of the action file.
 *
 * Format (whitespace separated):
 * @code
 *   id  type     start_s  geometry                param           expiry_s  flame_limit_m
 *   L1  line     3600     500 200 900 600 [...]   rate=0.5        -1        2.4
 *   D1  drop     5400     poly:x1,y1;x2,y2;...    ros_factor=0.0  1800      -
 *   B1  burnout  4000     ref=L1                  offset=20       -         -
 * @endcode
 * A line's geometry is a flat list of vertex coordinates (at least two
 * vertices), a drop's a single `poly:` token with at least three vertices, a
 * burnout's the id of the line it follows. `-` stands for "none" in the last
 * two columns; a drop's expiry may be -1 (permanent); a burnout has neither.
 *
 * @param[in]  text  The line without its comment
 * @param[out] a     The parsed action
 * @param[out] err   Reason on failure
 * @return true on success
 */
inline bool parse_suppression_line (const std::string& text, SuppressionAction& a, std::string& err)
{
    std::istringstream iss(text);
    std::vector<std::string> tok;
    for (std::string t; iss >> t;) { tok.push_back(t); }
    if (tok.size() < 6) { err = "fewer than six fields"; return false; }

    auto to_real = [] (const std::string& s, amrex::Real& v) -> bool {
        char* end = nullptr;
        const double d = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0' || !std::isfinite(d)) { return false; }
        v = static_cast<amrex::Real>(d);
        return true;
    };

    a = SuppressionAction();
    a.id = tok[0];
    if (tok[1] == "line")         { a.type = line; }
    else if (tok[1] == "drop")    { a.type = drop; }
    else if (tok[1] == "burnout") { a.type = burnout; }
    else { err = "unknown type \"" + tok[1] + "\" (line, drop or burnout)"; return false; }
    if (!to_real(tok[2], a.start_s) || a.start_s < 0.0) { err = "start_s must be a number >= 0"; return false; }

    std::size_t k = 3;
    if (a.type == line) {
        // Vertices up to the first token with an '='.
        while (k < tok.size() && tok[k].find('=') == std::string::npos) {
            amrex::Real v;
            if (!to_real(tok[k], v)) { err = "vertex coordinate \"" + tok[k] + "\" is not a number"; return false; }
            a.verts.push_back(v);
            ++k;
        }
        if (a.verts.size() < 4 || a.verts.size() % 2 != 0) {
            err = "a line needs an even number of coordinates for at least two vertices"; return false;
        }
        if (k >= tok.size() || tok[k].rfind("rate=", 0) != 0) { err = "a line needs rate=<m/s> after its vertices"; return false; }
        if (!to_real(tok[k].substr(5), a.rate) || a.rate <= 0.0) { err = "rate must be a number > 0"; return false; }
        ++k;
    } else if (a.type == drop) {
        const std::string& g = tok[k];
        if (g.rfind("poly:", 0) != 0) { err = "a drop needs poly:x1,y1;x2,y2;... as its geometry"; return false; }
        std::string body = g.substr(5);
        for (auto& c : body) { if (c == ';' || c == ',') { c = ' '; } }
        std::istringstream vs(body);
        for (std::string t; vs >> t;) {
            amrex::Real v;
            if (!to_real(t, v)) { err = "polygon coordinate \"" + t + "\" is not a number"; return false; }
            a.verts.push_back(v);
        }
        if (a.verts.size() < 6 || a.verts.size() % 2 != 0) { err = "a drop polygon needs at least three x,y vertices"; return false; }
        ++k;
        if (k >= tok.size() || tok[k].rfind("ros_factor=", 0) != 0) { err = "a drop needs ros_factor=<0..1> after its polygon"; return false; }
        if (!to_real(tok[k].substr(11), a.ros_factor) || a.ros_factor < 0.0 || a.ros_factor >= 1.0) {
            err = "ros_factor must be a number in [0, 1)"; return false;
        }
        ++k;
    } else {
        const std::string& g = tok[k];
        if (g.rfind("ref=", 0) != 0 || g.size() == 4) { err = "a burnout needs ref=<line id> as its geometry"; return false; }
        a.ref = g.substr(4);
        ++k;
        if (k >= tok.size() || tok[k].rfind("offset=", 0) != 0) { err = "a burnout needs offset=<m> after ref="; return false; }
        if (!to_real(tok[k].substr(7), a.offset) || a.offset < 0.0) { err = "offset must be a number >= 0"; return false; }
        ++k;
    }

    if (k + 2 != tok.size()) { err = "expected exactly two fields (expiry_s, flame_limit_m) after the parameter"; return false; }
    const std::string& ex = tok[k];
    const std::string& fl = tok[k+1];
    if (a.type == burnout) {
        if (ex != "-" || fl != "-") { err = "a burnout takes - for expiry_s and flame_limit_m"; return false; }
    } else {
        if (ex == "-") { a.expiry_s = -1.0; }
        else if (!to_real(ex, a.expiry_s) || !(a.expiry_s == -1.0 || a.expiry_s > 0.0)) {
            err = "expiry_s must be -1 (permanent) or a number > 0"; return false;
        }
        if (fl == "-") { a.flame_limit_m = -1.0; }
        else if (a.type == drop) { err = "flame_limit_m applies to lines only; use - for a drop"; return false; }
        else if (!to_real(fl, a.flame_limit_m) || a.flame_limit_m <= 0.0) {
            err = "flame_limit_m must be - or a number > 0"; return false;
        }
    }
    return true;
}

/**
 * @brief Parse the whole action file text.
 *
 * Comments start at `#`; blank lines are skipped. Ids must be unique within
 * the text and against `known_ids` (actions read earlier); a burnout must
 * reference a line id in the text or in `known_line_ids`.
 *
 * @param[in]  text            File contents
 * @param[in]  known_ids       Ids read earlier (duplicates are rejected)
 * @param[in]  known_line_ids  Line ids read earlier (valid burnout references)
 * @param[out] actions         Parsed actions in file order
 * @param[out] err             "line <n>: <reason>: \"<text>\"" on failure
 * @return true on success
 */
inline bool parse_suppression_text (const std::string& text,
                                    const std::vector<std::string>& known_ids,
                                    const std::vector<std::string>& known_line_ids,
                                    std::vector<SuppressionAction>& actions,
                                    std::string& err)
{
    actions.clear();
    std::istringstream file(text);
    std::string raw;
    int n = 0;
    std::vector<std::string> ids(known_ids);
    std::vector<std::string> line_ids(known_line_ids);
    auto has = [] (const std::vector<std::string>& v, const std::string& s) {
        for (const auto& x : v) { if (x == s) { return true; } }
        return false;
    };
    auto fail = [&] (const std::string& why, const std::string& what) {
        err = "line " + std::to_string(n) + ": " + why + ": \"" + what + "\"";
        return false;
    };
    while (std::getline(file, raw)) {
        ++n;
        std::string t = raw;
        const std::size_t hash = t.find('#');
        if (hash != std::string::npos) { t = t.substr(0, hash); }
        if (t.find_first_not_of(" \t\r") == std::string::npos) { continue; }
        SuppressionAction a;
        std::string why;
        if (!parse_suppression_line(t, a, why)) { return fail(why, raw); }
        if (has(ids, a.id)) { return fail("duplicate id \"" + a.id + "\"", raw); }
        ids.push_back(a.id);
        if (a.type == line) { line_ids.push_back(a.id); }
        actions.push_back(a);
    }
    for (const auto& a : actions) {
        if (a.type == burnout && !has(line_ids, a.ref)) {
            err = "burnout \"" + a.id + "\" references unknown line id \"" + a.ref + "\"";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Geometry kernels, host and device (the gtest calls them on the host).
// ---------------------------------------------------------------------------

/// Ray-casting point-in-polygon test; v holds nv (x, y) pairs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool point_in_polygon (amrex::Real x, amrex::Real y, const amrex::Real* v, int nv)
{
    bool inside = false;
    for (int i = 0, j = nv - 1; i < nv; j = i++) {
        const amrex::Real xi = v[2*i], yi = v[2*i+1];
        const amrex::Real xj = v[2*j], yj = v[2*j+1];
        const bool straddles = (yi > y) != (yj > y);
        // The division is only meaningful when the edge straddles y (then
        // yj != yi); the select keeps a speculated evaluation trap-free.
        const amrex::Real dy = (yj - yi != amrex::Real(0.0)) ? (yj - yi) : amrex::Real(1.0);
        const amrex::Real xcross = (xj - xi) * (y - yi) / dy + xi;
        if (straddles && x < xcross) { inside = !inside; }
    }
    return inside;
}

/// Distance from (px, py) to the segment (x0, y0)-(x1, y1).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real segment_distance (amrex::Real px, amrex::Real py,
                              amrex::Real x0, amrex::Real y0,
                              amrex::Real x1, amrex::Real y1)
{
    const amrex::Real dx = x1 - x0, dy = y1 - y0;
    const amrex::Real L2 = dx*dx + dy*dy;
    const amrex::Real den = (L2 > amrex::Real(0.0)) ? L2 : amrex::Real(1.0);
    amrex::Real t = ((px - x0)*dx + (py - y0)*dy) / den;
    t = amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), t));
    const amrex::Real cx = x0 + t*dx, cy = y0 + t*dy;
    return std::sqrt((px - cx)*(px - cx) + (py - cy)*(py - cy));
}

/// Liang-Barsky: does the segment intersect the closed box [xlo,xhi]x[ylo,yhi]?
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool segment_crosses_box (amrex::Real x0, amrex::Real y0, amrex::Real x1, amrex::Real y1,
                          amrex::Real xlo, amrex::Real ylo, amrex::Real xhi, amrex::Real yhi)
{
    amrex::Real t0 = 0.0, t1 = 1.0;
    const amrex::Real dx = x1 - x0, dy = y1 - y0;
    const amrex::Real p[4] = {-dx, dx, -dy, dy};
    const amrex::Real q[4] = {x0 - xlo, xhi - x0, y0 - ylo, yhi - y0};
    for (int k = 0; k < 4; ++k) {
        if (p[k] == amrex::Real(0.0)) {
            if (q[k] < amrex::Real(0.0)) { return false; }
        } else {
            const amrex::Real r = q[k] / p[k];
            if (p[k] < amrex::Real(0.0)) {
                if (r > t1) { return false; }
                if (r > t0) { t0 = r; }
            } else {
                if (r < t0) { return false; }
                if (r < t1) { t1 = r; }
            }
        }
    }
    return t0 <= t1;
}

/**
 * @brief Does the built part of a polyline cover the cell centred at (xc, yc)?
 *
 * The first `built` metres of the polyline (nv vertices in v) cover a cell
 * when they come within `half` of its centre or pass through the interior of
 * the box of half sizes (hx, hy) around it (a touch of its edge or corner
 * does not count). A 4-connected path from one side of the line to the other
 * crosses it inside a cell's interior, or on a cell edge the line lies along,
 * whose two cells the distance test covers; so the stamped cells block every
 * such path and the band is one cell wide, whichever way the line runs.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool polyline_covers_cell (amrex::Real xc, amrex::Real yc, amrex::Real hx, amrex::Real hy,
                           amrex::Real half, const amrex::Real* v, int nv, amrex::Real built)
{
    amrex::Real s = 0.0;
    for (int k = 0; k + 1 < nv; ++k) {
        if (s >= built) { break; }
        const amrex::Real x0 = v[2*k], y0 = v[2*k+1];
        const amrex::Real x1 = v[2*k+2], y1 = v[2*k+3];
        const amrex::Real len = std::sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0));
        amrex::Real xe = x1, ye = y1;
        if (s + len > built) {
            const amrex::Real f = (len > amrex::Real(0.0)) ? (built - s) / len : amrex::Real(0.0);
            xe = x0 + f*(x1 - x0);
            ye = y0 + f*(y1 - y0);
        }
        if (segment_distance(xc, yc, x0, y0, xe, ye) <= half) { return true; }
        const amrex::Real ex = amrex::Real(1.0e-6) * hx;
        const amrex::Real ey = amrex::Real(1.0e-6) * hy;
        if (segment_crosses_box(x0, y0, xe, ye, xc - hx + ex, yc - hy + ey, xc + hx - ex, yc + hy - ey)) { return true; }
        s += len;
    }
    return false;
}

} // namespace fire_suppression

/**
 * @class FireSuppression
 * @brief Owns the action list, the per-cell suppression fields and their
 *        checkpoint and log, and applies the actions once per fire step.
 *
 * Fields on the fire grid (all one component, no ghost cells):
 *  - mask:     1 in cells held non-burnable this step (built line cells and
 *              ros_factor = 0 drops), 0 elsewhere
 *  - factor:   multiplier on the rate of spread, 1 outside 0 < ros_factor < 1 drops
 *  - progress: ordinal (1-based, among the lines in file order) of the line a
 *              built cell belongs to, 0 elsewhere
 *  - failed:   1 where a line cell failed its hold test (never rebuilt)
 *
 * The fields are rebuilt from the active actions every step, so an expired
 * action's cells revert by themselves; the fields are still checkpointed so a
 * restart reproduces them before its first step.
 */
class FireSuppression
{
public:
    FireSuppression () = default;
    ~FireSuppression () = default;

    /**
     * @brief Allocate the fields and read the action file once.
     * @param[in] file           erf.fire.suppression.file
     * @param[in] poll_interval  erf.fire.suppression.poll_interval
     * @param[in] log_file       erf.fire.suppression.log
     * @param[in] ba, dm, geom   Fire grid
     * @param[in] debug          erf.fire.fire_debug
     */
    void initialize (const std::string& file, int poll_interval, const std::string& log_file,
                     const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                     const amrex::Geometry& geom, bool debug);

    /**
     * @brief Apply the actions for the fire step ending at `time`.
     *
     * Polls the file when due, starts and expires actions, runs the hold test
     * on the previous step's built cells against `flame_length`, rebuilds the
     * fields, and pushes burnout ignitions for the newly built part of a
     * referenced line into `schedule` as events at `time` (the caller's
     * scheduled-ignition pass stamps them this step).
     *
     * @param[in]     time          Time at the end of the step [s]
     * @param[in]     dt            Fire step [s]
     * @param[in]     step          Fire step counter (1 on the first step of a fresh run)
     * @param[in]     phi           Level set (burned cells are never stamped)
     * @param[in]     arrival_time  Arrival time (>= 0 burned)
     * @param[in]     flame_length  Flame length of the previous step [m], or nullptr
     * @param[in,out] schedule      Ignition schedule the burnout events are appended to
     * @param[in,out] has_schedule  Set when an event was appended
     */
    void update (amrex::Real time, amrex::Real dt, int step,
                 const amrex::MultiFab& phi, const amrex::MultiFab& arrival_time,
                 const amrex::MultiFab* flame_length,
                 IgnitionSchedule& schedule, bool& has_schedule);

    const amrex::MultiFab* mask () const     { return m_mask.get(); }
    const amrex::MultiFab* ros_factor () const { return m_factor.get(); }
    const amrex::MultiFab* progress () const { return m_progress.get(); }
    const amrex::MultiFab* failed () const   { return m_failed.get(); }
    /// A drop with 0 < ros_factor < 1 is active: the factor field is not all ones.
    bool factor_active () const { return m_factor_active; }

    const std::vector<fire_suppression::SuppressionAction>& actions () const { return m_actions; }

    /// Write the fields and the action state under `dir` (MultiFabs on level `lev`).
    void write_checkpoint (const std::string& dir, int lev) const;
    /// Restore them; actions of the file not in the checkpoint are appended.
    void read_checkpoint (const std::string& dir, int lev);

    /// The plotted mask: 1 suppressed, -1 hold failed and not suppressed, 0 else.
    void fill_plot_mask (amrex::MultiFab& out, int comp) const;

    // nvcc forbids an extended device lambda inside a private member; the
    // kernels below are public for that reason only.
public:
    /// Stamp (or only count) one action's cells; `built` is the line length used.
    amrex::Long stamp_action (const fire_suppression::SuppressionAction& a, int ordinal,
                              amrex::Real built, const amrex::Real* d_verts,
                              const amrex::MultiFab& phi, const amrex::MultiFab& arrival_time,
                              bool write);
    /// Hold test of one line: mark built cells next to a flame longer than the limit.
    amrex::Long hold_test (int ordinal, amrex::Real limit, const amrex::MultiFab& flame_grown);
    /// Which side of a segment the nearest burning cell is on: +1 left, -1 right, 0 none.
    int fire_side (amrex::Real x0, amrex::Real y0, amrex::Real x1, amrex::Real y1,
                   const amrex::MultiFab& phi, const amrex::MultiFab& arrival_time) const;

private:
    /// Read the file on the IO rank (if `force` or its mtime changed), broadcast, parse, append new ids.
    void read_file (bool force, amrex::Real time, int step);
    /// Rebuild the device copies of the vertex lists.
    void upload_vertices ();
    /// Append one event line to the log (IO rank).
    void log_event (amrex::Real time, int step, const std::string& id, int type,
                    const std::string& event, amrex::Long cells, const std::string& detail = "");
    /// Ordinal of a line action (1-based among lines), 0 for other types.
    int line_ordinal (std::size_t index) const;
    /// Index of the action with this id, or -1.
    int find_action (const std::string& id) const;

    std::string m_file;
    std::string m_log_file;
    int         m_poll_interval = 10;
    bool        m_debug = false;
    bool        m_log_opened = false;
    std::vector<fire_suppression::SuppressionAction> m_actions;
    std::vector<amrex::Gpu::DeviceVector<amrex::Real>> m_d_verts;
    // The file as last seen (IO rank): modification time and size.
    long long   m_mtime_ns = -1;
    long long   m_size     = -1;

    amrex::Geometry m_geom;
    std::unique_ptr<amrex::MultiFab> m_mask;
    std::unique_ptr<amrex::MultiFab> m_factor;
    std::unique_ptr<amrex::MultiFab> m_progress;
    std::unique_ptr<amrex::MultiFab> m_failed;
    bool m_factor_active = false;
};

#endif // ERF_FIRE_SUPPRESSION_H
