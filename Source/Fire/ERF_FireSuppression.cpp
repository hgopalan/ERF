/**
 * @file ERF_FireSuppression.cpp
 * @brief Suppression actions on the fire grid (see ERF_FireSuppression.H).
 */

#include <ERF_FireSuppression.H>
#include <ERF_FireGrid.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_Print.H>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <limits>

using namespace amrex;
using namespace fire_suppression;

namespace {

/// Broadcast a string from the IO rank.
void broadcast_text (std::string& text)
{
    if (ParallelDescriptor::NProcs() == 1) { return; }
    int n = static_cast<int>(text.size());
    ParallelDescriptor::Bcast(&n, 1, ParallelDescriptor::IOProcessorNumber());
    std::vector<char> buf(static_cast<std::size_t>(n) + 1, '\0');
    if (ParallelDescriptor::IOProcessor()) {
        std::copy(text.begin(), text.end(), buf.begin());
    }
    if (n > 0) {
        ParallelDescriptor::Bcast(buf.data(), static_cast<std::size_t>(n), ParallelDescriptor::IOProcessorNumber());
    }
    text.assign(buf.data(), static_cast<std::size_t>(n));
}

/// Position and unit tangent at arc length s along a polyline (clamped to its ends).
void polyline_point (const std::vector<Real>& v, Real s, Real& x, Real& y, Real& tx, Real& ty, int& seg)
{
    const int nv = static_cast<int>(v.size() / 2);
    Real acc = 0.0;
    seg = 0;
    tx = 1.0; ty = 0.0;
    x = v[0]; y = v[1];
    for (int k = 0; k + 1 < nv; ++k) {
        const Real x0 = v[2*k], y0 = v[2*k+1], x1 = v[2*k+2], y1 = v[2*k+3];
        const Real len = std::sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0));
        if (len <= 0.0) { continue; }
        tx = (x1 - x0) / len; ty = (y1 - y0) / len;
        seg = k;
        if (s <= acc + len || k + 2 == nv) {
            const Real f = amrex::max(Real(0.0), amrex::min(Real(1.0), (s - acc) / len));
            x = x0 + f*(x1 - x0);
            y = y0 + f*(y1 - y0);
            return;
        }
        acc += len;
    }
}

} // namespace

void FireSuppression::initialize (const std::string& file, int poll_interval, const std::string& log_file,
                                  const BoxArray& ba, const DistributionMapping& dm,
                                  const Geometry& geom, bool debug)
{
    m_file          = file;
    m_log_file      = log_file;
    m_poll_interval = poll_interval;
    m_debug         = debug;
    m_geom          = geom;

    m_mask     = std::make_unique<MultiFab>(ba, dm, 1, 0);
    m_factor   = std::make_unique<MultiFab>(ba, dm, 1, 0);
    m_progress = std::make_unique<MultiFab>(ba, dm, 1, 0);
    m_failed   = std::make_unique<MultiFab>(ba, dm, 1, 0);
    m_mask->setVal(0.0_rt);
    m_factor->setVal(1.0_rt);
    m_progress->setVal(0.0_rt);
    m_failed->setVal(0.0_rt);

    read_file(/*force=*/true, 0.0, 0);
    amrex::Print() << "[FIRE] Suppression: read " << m_actions.size() << " action(s) from '"
                   << m_file << "'; the file is re-read "
                   << (m_poll_interval > 0 ? "every " + std::to_string(m_poll_interval) + " fire steps"
                                           : "never (poll_interval = 0)")
                   << ", events go to '" << m_log_file << "'\n";
}

int FireSuppression::find_action (const std::string& id) const
{
    for (std::size_t i = 0; i < m_actions.size(); ++i) {
        if (m_actions[i].id == id) { return static_cast<int>(i); }
    }
    return -1;
}

int FireSuppression::line_ordinal (std::size_t index) const
{
    if (index >= m_actions.size() || m_actions[index].type != line) { return 0; }
    int n = 0;
    for (std::size_t i = 0; i <= index; ++i) {
        if (m_actions[i].type == line) { ++n; }
    }
    return n;
}

void FireSuppression::upload_vertices ()
{
    m_d_verts.resize(m_actions.size());
    for (std::size_t i = 0; i < m_actions.size(); ++i) {
        const auto& v = m_actions[i].verts;
        if (m_d_verts[i].size() == v.size() && !v.empty()) { continue; }
        m_d_verts[i].resize(v.size());
        if (!v.empty()) {
            Gpu::copy(Gpu::hostToDevice, v.begin(), v.end(), m_d_verts[i].begin());
        }
    }
    Gpu::streamSynchronize();
}

void FireSuppression::read_file (bool force, Real time, int step)
{
    int changed = 0;
    std::string text;
    if (ParallelDescriptor::IOProcessor()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path path(m_file);
        const auto ft = fs::last_write_time(path, ec);
        if (ec) {
            if (force) {
                amrex::Abort("[FIRE] Cannot stat suppression file '" + m_file + "': " + ec.message());
            }
            amrex::Print() << "[FIRE] WARNING: suppression file '" << m_file
                           << "' cannot be read at step " << step << "; keeping the actions read so far\n";
        } else {
            const long long ns = static_cast<long long>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ft.time_since_epoch()).count());
            const long long sz = static_cast<long long>(fs::file_size(path, ec));
            if (force || ns != m_mtime_ns || sz != m_size) {
                std::ifstream in(m_file);
                if (!in) {
                    if (force) { amrex::Abort("[FIRE] Cannot open suppression file '" + m_file + "'"); }
                } else {
                    std::ostringstream ss;
                    ss << in.rdbuf();
                    text = ss.str();
                    m_mtime_ns = ns;
                    m_size     = sz;
                    changed    = 1;
                }
            }
        }
    }
    ParallelDescriptor::Bcast(&changed, 1, ParallelDescriptor::IOProcessorNumber());
    if (!changed) { return; }
    broadcast_text(text);

    std::vector<std::string> known_ids, known_lines;
    for (const auto& a : m_actions) {
        known_ids.push_back(a.id);
        if (a.type == line) { known_lines.push_back(a.id); }
    }
    // Ids already known are skipped below; the parser rejects them as duplicates
    // only within the text, so pass an empty known list and filter here.
    std::vector<SuppressionAction> parsed;
    std::string err;
    if (!parse_suppression_text(text, {}, known_lines, parsed, err)) {
        amrex::Abort("[FIRE] suppression file '" + m_file + "' " + err);
    }
    int n_new = 0;
    for (auto& a : parsed) {
        if (find_action(a.id) >= 0) { continue; }
        m_actions.push_back(a);
        ++n_new;
    }
    upload_vertices();
    if (!force) {
        log_event(time, step, "-", -1, "reread", n_new, "new actions from the file");
        amrex::Print() << "[FIRE] Suppression: file '" << m_file << "' changed at step " << step
                       << "; " << n_new << " new action(s)\n";
    }
}

void FireSuppression::log_event (Real time, int step, const std::string& id, int type,
                                 const std::string& event, Long cells, const std::string& detail)
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream f(m_log_file, std::ios::app);
    f << std::setprecision(10) << time << "," << step << "," << id << ","
      << (type < 0 ? "-" : type_name(type)) << "," << event << "," << cells << "," << detail << "\n";
}

Long FireSuppression::stamp_action (const SuppressionAction& a, int ordinal, Real built,
                                    const Real* d_verts,
                                    const MultiFab& phi, const MultiFab& arrival_time, bool write)
{
    const auto plo  = m_geom.ProbLoArray();
    const auto dx   = m_geom.CellSizeArray();
    const Real hx   = 0.5_rt * dx[0];
    const Real hy   = 0.5_rt * dx[1];
    const Real half = 0.5_rt * amrex::min(dx[0], dx[1]);
    const int  type = a.type;
    const int  nv   = a.nverts();
    const Real fac  = a.ros_factor;
    const Real ord  = static_cast<Real>(ordinal);

    ReduceOps<ReduceOpSum> rops;
    ReduceData<Long> rdata(rops);
    using RT = typename decltype(rdata)::Type;

    for (MFIter mfi(*m_mask, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& m  = m_mask->array(mfi);
        auto const& f  = m_factor->array(mfi);
        auto const& pr = m_progress->array(mfi);
        auto const& fl = m_failed->const_array(mfi);
        auto const& p  = phi.const_array(mfi);
        auto const& at = arrival_time.const_array(mfi);
        rops.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
        {
            const Real xc = plo[0] + (i + 0.5_rt) * dx[0];
            const Real yc = plo[1] + (j + 0.5_rt) * dx[1];
            bool hit;
            if (type == line) {
                hit = (fl(i, j, k) < 0.5_rt) && polyline_covers_cell(xc, yc, hx, hy, half, d_verts, nv, built);
            } else {
                hit = point_in_polygon(xc, yc, d_verts, nv);
            }
            if (!hit) { return {Long(0)}; }
            // A burned cell never enters the mask: a line through the fire and
            // retardant on burned ground do not hold the front. The rate factor
            // is written in burned cells too, because the FARSITE front-cell
            // update takes a cell's rate from its burned neighbours as well.
            const bool burned = (at(i, j, k) >= 0.0_rt) || (p(i, j, k) < 0.0_rt);
            if (write) {
                if (type == line) {
                    if (!burned) { m(i, j, k) = 1.0_rt; pr(i, j, k) = ord; }
                } else if (fac <= 0.0_rt) {
                    if (!burned) { m(i, j, k) = 1.0_rt; }
                } else {
                    f(i, j, k) = amrex::min(f(i, j, k), fac);
                }
            }
            return {burned ? Long(0) : Long(1)};
        });
    }
    Long n = amrex::get<0>(rdata.value(rops));
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

Long FireSuppression::hold_test (int ordinal, Real limit, const MultiFab& flame_grown)
{
    const Real ord = static_cast<Real>(ordinal);
    ReduceOps<ReduceOpSum> rops;
    ReduceData<Long> rdata(rops);
    using RT = typename decltype(rdata)::Type;
    for (MFIter mfi(*m_failed, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& fl = m_failed->array(mfi);
        auto const& pr = m_progress->const_array(mfi);
        auto const& L  = flame_grown.const_array(mfi);
        rops.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
        {
            if (pr(i, j, k) != ord || fl(i, j, k) > 0.5_rt) { return {Long(0)}; }
            const bool hot = (L(i-1, j, k) > limit) || (L(i+1, j, k) > limit)
                          || (L(i, j-1, k) > limit) || (L(i, j+1, k) > limit);
            if (!hot) { return {Long(0)}; }
            fl(i, j, k) = 1.0_rt;
            return {Long(1)};
        });
    }
    Long n = amrex::get<0>(rdata.value(rops));
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

int FireSuppression::fire_side (Real x0, Real y0, Real x1, Real y1,
                                const MultiFab& phi, const MultiFab& arrival_time) const
{
    const auto plo = m_geom.ProbLoArray();
    const auto dx  = m_geom.CellSizeArray();
    const Real big = std::numeric_limits<Real>::max();
    ReduceOps<ReduceOpMin, ReduceOpMin> rops;
    ReduceData<Real, Real> rdata(rops);
    using RT = typename decltype(rdata)::Type;
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& p  = phi.const_array(mfi);
        auto const& at = arrival_time.const_array(mfi);
        rops.eval(bx, rdata, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
        {
            const bool burned = (at(i, j, k) >= 0.0_rt) || (p(i, j, k) < 0.0_rt);
            if (!burned) { return {big, big}; }
            const Real xc = plo[0] + (i + 0.5_rt) * dx[0];
            const Real yc = plo[1] + (j + 0.5_rt) * dx[1];
            const Real d  = segment_distance(xc, yc, x0, y0, x1, y1);
            const Real cross = (x1 - x0) * (yc - y0) - (y1 - y0) * (xc - x0);
            return (cross > 0.0_rt) ? RT{d, big} : RT{big, d};
        });
    }
    auto hv = rdata.value(rops);
    Real lmin = amrex::get<0>(hv);
    Real rmin = amrex::get<1>(hv);
    ParallelDescriptor::ReduceRealMin(lmin);
    ParallelDescriptor::ReduceRealMin(rmin);
    if (lmin >= big && rmin >= big) { return 0; }
    return (lmin <= rmin) ? 1 : -1;
}

void FireSuppression::update (Real time, Real dt, int step,
                              const MultiFab& phi, const MultiFab& arrival_time,
                              const MultiFab* flame_length,
                              IgnitionSchedule& schedule, bool& has_schedule)
{
    amrex::ignore_unused(dt);

    // The log: a fresh run starts it over, a restart (step > 1 on the first
    // call) appends to the earlier leg.
    if (!m_log_opened) {
        m_log_opened = true;
        if (ParallelDescriptor::IOProcessor()) {
            const bool fresh = (step <= 1);
            bool need_header = fresh;
            if (!fresh) {
                std::ifstream probe(m_log_file);
                need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
            }
            std::ofstream f(m_log_file, fresh ? std::ios::trunc : std::ios::app);
            if (need_header) { f << "time_s,step,id,type,event,cells,detail\n"; }
        }
    }

    if (m_poll_interval > 0 && step % m_poll_interval == 0) {
        read_file(/*force=*/false, time, step);
    }

    // Expiry
    for (auto& a : m_actions) {
        if (a.applied && !a.expired && a.expiry_s > 0.0 && time >= a.start_s + a.expiry_s) {
            a.expired = true;
            log_event(time, step, a.id, a.type, "expired", a.cells_last, "cells reverted");
        }
    }

    // Hold test on the built cells of the previous step against its flame length.
    if (flame_length) {
        bool any = false;
        for (const auto& a : m_actions) {
            any = any || (a.type == line && a.applied && !a.expired && a.flame_limit_m > 0.0 && a.built_m > 0.0);
        }
        if (any) {
            MultiFab grown(m_failed->boxArray(), m_failed->DistributionMap(), 1, 1);
            grown.setVal(0.0_rt);
            MultiFab::Copy(grown, *flame_length, 0, 0, 1, 0);
            fire_fill_boundary(grown, m_geom);
            for (std::size_t i = 0; i < m_actions.size(); ++i) {
                auto& a = m_actions[i];
                if (a.type != line || !a.applied || a.expired || a.flame_limit_m <= 0.0 || a.built_m <= 0.0) { continue; }
                const Long n = hold_test(line_ordinal(i), a.flame_limit_m, grown);
                if (n > 0) {
                    std::ostringstream d;
                    d << std::setprecision(10) << "flame length above " << a.flame_limit_m << " m next to the line";
                    log_event(time, step, a.id, a.type, "hold_failed", n, d.str());
                }
            }
        }
    }

    // Start
    std::vector<std::size_t> newly;
    for (std::size_t i = 0; i < m_actions.size(); ++i) {
        auto& a = m_actions[i];
        if (!a.applied && !a.expired && a.start_s <= time) {
            a.applied = true;
            newly.push_back(i);
        }
    }

    // Line construction
    std::vector<std::size_t> just_completed;
    for (std::size_t i = 0; i < m_actions.size(); ++i) {
        auto& a = m_actions[i];
        if (a.type != line || !a.applied || a.expired) { continue; }
        const Real L = a.length();
        a.built_m = amrex::min(L, a.rate * (time - a.start_s));
        if (!a.completed && a.built_m >= L) {
            a.completed = true;
            just_completed.push_back(i);
        }
    }

    // Rebuild the fields from the active actions
    m_mask->setVal(0.0_rt);
    m_factor->setVal(1.0_rt);
    m_progress->setVal(0.0_rt);
    m_factor_active = false;
    for (std::size_t i = 0; i < m_actions.size(); ++i) {
        auto& a = m_actions[i];
        if (a.type == burnout || !a.applied || a.expired) { continue; }
        const int ordinal = line_ordinal(i);
        a.cells_last = stamp_action(a, ordinal, a.built_m, m_d_verts[i].data(), phi, arrival_time, true);
        if (a.type == drop && a.ros_factor > 0.0) { m_factor_active = true; }
    }
    Gpu::streamSynchronize();

    for (std::size_t i : newly) {
        const auto& a = m_actions[i];
        std::ostringstream d;
        d << std::setprecision(10);
        if (a.type == line) {
            d << "ordinal=" << line_ordinal(i) << " length=" << a.length() << " rate=" << a.rate
              << " expiry=" << a.expiry_s << " flame_limit=" << a.flame_limit_m;
        } else if (a.type == drop) {
            d << "ros_factor=" << a.ros_factor << " expiry=" << a.expiry_s;
        } else {
            d << "ref=" << a.ref << " offset=" << a.offset;
        }
        log_event(time, step, a.id, a.type, "applied", a.cells_last, d.str());
    }
    for (std::size_t i : just_completed) {
        const auto& a = m_actions[i];
        std::ostringstream d;
        d << std::setprecision(10) << "line fully built; " << a.length() << " m";
        log_event(time, step, a.id, a.type, "completed", a.cells_last, d.str());
    }

    // Burnout: ignition points along the newly built part of the reference
    // line, offset on the side of the nearest burning cell, one every fire cell.
    const Real spacing = amrex::min(m_geom.CellSize(0), m_geom.CellSize(1));
    const Real radius  = 0.75_rt * spacing;
    for (auto& a : m_actions) {
        if (a.type != burnout || !a.applied || a.expired) { continue; }
        const int r = find_action(a.ref);
        if (r < 0 || !m_actions[r].applied) { continue; }
        const auto& ref = m_actions[r];
        const Real b1 = ref.built_m;
        if (b1 <= a.burnout_m) { continue; }
        int n_pts = 0;
        int side_seg = -1, side = 0;
        bool stalled = false;
        const Real b0 = a.burnout_m;
        for (Long n = static_cast<Long>(std::floor(b0 / spacing)); ; ++n) {
            const Real s = (static_cast<Real>(n) + 0.5_rt) * spacing;
            if (s < b0) { continue; }
            if (s >= b1) { break; }
            Real x, y, tx, ty;
            int seg;
            polyline_point(ref.verts, s, x, y, tx, ty, seg);
            if (seg != side_seg) {
                const Real x0 = ref.verts[2*seg], y0 = ref.verts[2*seg+1];
                const Real x1 = ref.verts[2*seg+2], y1 = ref.verts[2*seg+3];
                side = fire_side(x0, y0, x1, y1, phi, arrival_time);
                side_seg = seg;
            }
            if (side == 0) { stalled = true; break; }   // no burning cell yet: retry next step
            IgnitionEvent ev;
            ev.time_s = time;
            ev.cx = x + static_cast<Real>(side) * (-ty) * a.offset;
            ev.cy = y + static_cast<Real>(side) * ( tx) * a.offset;
            ev.radius = radius;
            ev.priority = 5;
            ev.suppress_if_burning = true;
            ev.fired = false;
            schedule.events.push_back(ev);
            ++n_pts;
        }
        if (!stalled) { a.burnout_m = b1; }
        if (n_pts > 0) {
            has_schedule = true;
            std::ostringstream d;
            // No comma in a detail: the log is a CSV.
            d << std::setprecision(10) << "ignition points along " << a.ref << " from " << b0 << " to "
              << (stalled ? a.burnout_m : b1) << " m; side " << (side > 0 ? "left" : "right");
            log_event(time, step, a.id, a.type, "burnout", n_pts, d.str());
        }
    }
}

void FireSuppression::fill_plot_mask (MultiFab& out, int comp) const
{
    for (MFIter mfi(out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& o  = out.array(mfi, comp);
        auto const& m  = m_mask->const_array(mfi);
        auto const& fl = m_failed->const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            o(i, j, k) = (m(i, j, k) > 0.5_rt) ? 1.0_rt : ((fl(i, j, k) > 0.5_rt) ? -1.0_rt : 0.0_rt);
        });
    }
}

void FireSuppression::write_checkpoint (const std::string& dir, int lev) const
{
    VisMF::Write(*m_mask,     MultiFabFileFullPrefix(lev, dir, "Level_", "FireSuppressionMask"));
    VisMF::Write(*m_factor,   MultiFabFileFullPrefix(lev, dir, "Level_", "FireSuppressionFactor"));
    VisMF::Write(*m_progress, MultiFabFileFullPrefix(lev, dir, "Level_", "FireSuppressionProgress"));
    VisMF::Write(*m_failed,   MultiFabFileFullPrefix(lev, dir, "Level_", "FireSuppressionFailed"));
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream f(dir + "/FireSuppression");
        f << std::setprecision(17);
        f << "n_actions " << m_actions.size() << "\n";
        for (const auto& a : m_actions) {
            f << a.id << " " << a.type << " " << a.start_s << " " << a.expiry_s << " " << a.rate << " "
              << a.ros_factor << " " << a.offset << " " << a.flame_limit_m << " "
              << (a.ref.empty() ? "-" : a.ref) << " "
              << (a.applied ? 1 : 0) << " " << (a.expired ? 1 : 0) << " " << (a.completed ? 1 : 0) << " "
              << a.built_m << " " << a.burnout_m << " " << a.cells_last << " " << a.nverts();
            for (const auto v : a.verts) { f << " " << v; }
            f << "\n";
        }
    }
    amrex::Print() << "[FIRE] Suppression state written to checkpoint " << dir << "\n";
}

void FireSuppression::read_checkpoint (const std::string& dir, int lev)
{
    std::vector<SuppressionAction> restored;
    {
        std::ifstream f(dir + "/FireSuppression");
        if (!f) {
            amrex::Print() << "[FIRE] Checkpoint has no FireSuppression state; keeping the actions read from the file\n";
            return;
        }
        std::string key;
        std::size_t n = 0;
        f >> key >> n;
        if (key != "n_actions") { amrex::Abort("[FIRE] Malformed FireSuppression state in checkpoint " + dir); }
        for (std::size_t i = 0; i < n; ++i) {
            SuppressionAction a;
            int applied = 0, expired = 0, completed = 0, nv = 0;
            std::string ref;
            f >> a.id >> a.type >> a.start_s >> a.expiry_s >> a.rate >> a.ros_factor >> a.offset
              >> a.flame_limit_m >> ref >> applied >> expired >> completed >> a.built_m >> a.burnout_m
              >> a.cells_last >> nv;
            if (!f) { amrex::Abort("[FIRE] Malformed FireSuppression state in checkpoint " + dir); }
            a.ref = (ref == "-") ? std::string() : ref;
            a.applied   = (applied != 0);
            a.expired   = (expired != 0);
            a.completed = (completed != 0);
            a.verts.resize(static_cast<std::size_t>(2 * nv));
            for (auto& v : a.verts) { f >> v; }
            restored.push_back(a);
        }
    }
    // The checkpoint's actions carry their state and geometry; the file adds
    // only ids the checkpoint does not know.
    int n_new = 0;
    for (const auto& a : m_actions) {
        bool known = false;
        for (const auto& r : restored) { known = known || (r.id == a.id); }
        if (!known) { restored.push_back(a); ++n_new; }
    }
    m_actions = restored;
    m_d_verts.clear();
    upload_vertices();

    auto restore = [&] (MultiFab& mf, const char* name) {
        const std::string header = dir + "/Level_" + std::to_string(lev) + "/" + name + "_H";
        if (!amrex::FileExists(header)) { return; }
        VisMF::Read(mf, MultiFabFileFullPrefix(lev, dir, "Level_", name));
    };
    restore(*m_mask,     "FireSuppressionMask");
    restore(*m_factor,   "FireSuppressionFactor");
    restore(*m_progress, "FireSuppressionProgress");
    restore(*m_failed,   "FireSuppressionFailed");
    m_factor_active = (m_factor->min(0) < 1.0_rt);
    amrex::Print() << "[FIRE] Suppression: restored " << (m_actions.size() - static_cast<std::size_t>(n_new))
                   << " action(s) from the checkpoint, " << n_new << " new from the file\n";
}
