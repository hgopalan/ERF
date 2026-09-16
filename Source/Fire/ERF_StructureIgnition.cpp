#include <ERF_StructureIgnition.H>
#include <ERF_HostFabView.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <cmath>

using namespace amrex;
using namespace erf_structure_ignition;

StructureIgnition::StructureIgnition (const FireGrid& fg,
                                      const FireParams::StructureParams::IgnitionParams& params,
                                      int n_structures, int ring, bool debug)
    : m_fg(fg), m_p(params), m_n(n_structures), m_ring(ring), m_debug(debug)
{
    // init_params() has already rejected a curve that cannot be built; this
    // is the same test, so a caller that skipped the parameter check still stops.
    if (!make_burn_curve(m_p.peak_flux_W_m2, m_p.fuel_load_J_m2, m_p.growth_time_s, m_curve)) {
        Abort("[FIRE] erf.fire.structures.ignition: the growth phase (peak_flux_W_m2 * growth_time_s / 3) "
              "releases more than 70 % of fuel_load_J_m2; no EN 1991-1-2 curve exists");
    }
    m_state            = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, NComp, 0);
    m_rad_flux         = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    m_launch_intensity = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    m_state->setVal(0.0_rt);
    m_state->setVal(-1.0_rt, IgnitionTimeComp, 1, 0);
    m_rad_flux->setVal(0.0_rt);
    m_launch_intensity->setVal(0.0_rt);

    m_h_state.assign(m_n + 1, Unignited);
    m_h_t_ign.assign(m_n + 1, -1.0_rt);
    m_h_t_above.assign(m_n + 1, 0.0_rt);
    m_h_cause.assign(m_n + 1, None);
    m_foot_i.assign(m_n + 1, {});
    m_foot_j.assign(m_n + 1, {});

    Print() << "[FIRE] Structure ignition: " << m_n << " structures; thresholds heat load "
            << m_p.heat_load_J_m2 * 1.0e-6 << " MJ/m2, " << m_p.ember_count << " embers, intensity "
            << m_p.intensity_kW_m << " kW/m for " << m_p.residence_s << " s; burn curve peak "
            << m_curve.q_peak * 1.0e-3 << " kW/m2 reached at " << m_curve.t_growth << " s, plateau "
            << m_curve.t_plateau << " s, decay " << m_curve.t_decay << " s (" << m_curve.duration()
            << " s in all); radiant fraction " << m_p.rad_fraction << " within "
            << m_p.rad_radius_m << " m\n";
}

Real StructureIgnition::flux_of (int s, Real time) const
{
    if (s < 1 || s > m_n || m_h_state[s] != Burning) { return 0.0_rt; }
    return m_curve.flux_at(time - m_h_t_ign[s]);
}

int StructureIgnition::n_ignited () const
{
    return static_cast<int>(std::count_if(m_h_state.begin() + 1, m_h_state.end(),
                                          [](int s) { return s != Unignited; }));
}

int StructureIgnition::n_burning () const
{
    return static_cast<int>(std::count(m_h_state.begin() + 1, m_h_state.end(), static_cast<int>(Burning)));
}

int StructureIgnition::n_burned_out () const
{
    return static_cast<int>(std::count(m_h_state.begin() + 1, m_h_state.end(), static_cast<int>(BurnedOut)));
}

void StructureIgnition::build_footprints (const MultiFab& structure_id)
{
    // The id field is reduced to a global host copy once (the spotting model
    // does the same every step): each rank fills its own cells, the sum over
    // disjoint boxes gives every rank the whole field.
    const Box& dom = m_fg.geom.Domain();
    const int nx = dom.length(0);
    const int ny = dom.length(1);
    const int i0 = dom.smallEnd(0);
    const int j0 = dom.smallEnd(1);
    std::vector<Real> h_id(static_cast<std::size_t>(nx) * ny, 0.0_rt);
    for (MFIter mfi(structure_id); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const ERFHostFabView id_v(structure_id[mfi]);
        auto id = id_v.array();
        LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
            h_id[static_cast<std::size_t>(j - j0) * nx + (i - i0)] = id(i, j, 0);
        });
    }
    ParallelDescriptor::ReduceRealSum(h_id.data(), static_cast<int>(h_id.size()));

    for (int s = 1; s <= m_n; ++s) { m_foot_i[s].clear(); m_foot_j[s].clear(); }
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int sid = static_cast<int>(h_id[static_cast<std::size_t>(j) * nx + i] + 0.5_rt);
            if (sid >= 1 && sid <= m_n) {
                m_foot_i[sid].push_back(i + i0);
                m_foot_j[sid].push_back(j + j0);
            }
        }
    }
    if (m_debug) {
        std::size_t total = 0;
        for (int s = 1; s <= m_n; ++s) { total += m_foot_i[s].size(); }
        Print() << "[FIRE DEBUG] Structure ignition: " << total << " footprint cells over "
                << m_n << " structures\n";
    }
}

void StructureIgnition::restore_from_field (const MultiFab& structure_id)
{
    std::vector<Real> st(m_n + 1, 0.0_rt), ti(m_n + 1, -1.0_rt), ta(m_n + 1, 0.0_rt), ca(m_n + 1, 0.0_rt);
    for (MFIter mfi(*m_state); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const ERFHostFabView id_v(structure_id[mfi]);
        const ERFHostFabView st_v((*m_state)[mfi]);
        auto id = id_v.array();
        auto f  = st_v.array();
        LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
            const int sid = static_cast<int>(id(i, j, 0) + 0.5_rt);
            if (sid < 1 || sid > m_n) { return; }
            st[sid] = std::max(st[sid], f(i, j, 0, StateComp));
            ti[sid] = std::max(ti[sid], f(i, j, 0, IgnitionTimeComp));
            ta[sid] = std::max(ta[sid], f(i, j, 0, TimeAboveComp));
            ca[sid] = std::max(ca[sid], f(i, j, 0, CauseComp));
        });
    }
    ParallelDescriptor::ReduceRealMax(st.data(), m_n + 1);
    ParallelDescriptor::ReduceRealMax(ti.data(), m_n + 1);
    ParallelDescriptor::ReduceRealMax(ta.data(), m_n + 1);
    ParallelDescriptor::ReduceRealMax(ca.data(), m_n + 1);
    for (int s = 1; s <= m_n; ++s) {
        m_h_state[s]   = static_cast<int>(st[s] + 0.5_rt);
        m_h_t_ign[s]   = ti[s];
        m_h_t_above[s] = ta[s];
        m_h_cause[s]   = static_cast<int>(ca[s] + 0.5_rt);
    }
    Print() << "[FIRE] Structure ignition restored: " << n_burning() << " burning, "
            << n_burned_out() << " burned out of " << m_n << "\n";
}

void StructureIgnition::apply_heat_sources (Real time, MultiFab& heat_flux, const MultiFab& structure_id)
{
    // Per-structure release now, on the host, then onto the device by id.
    std::vector<Real> h_q(m_n + 1, 0.0_rt);
    bool any = false;
    for (int s = 1; s <= m_n; ++s) {
        h_q[s] = flux_of(s, time);
        any = any || (h_q[s] > 0.0_rt);
    }
    m_launch_intensity->setVal(0.0_rt);
    m_rad_flux->setVal(0.0_rt);
    if (!any) { return; }

    Gpu::DeviceVector<Real> d_q(m_n + 1);
    Gpu::copy(Gpu::hostToDevice, h_q.begin(), h_q.end(), d_q.begin());
    const Real* p_q  = d_q.data();
    const int   nmax = m_n;
    const Real  dx   = m_fg.geom.CellSize(0);
    // Equivalent fireline intensity for the lofting height: the release per
    // unit length across one fire cell of the footprint, W/m2 * m -> kW/m.
    const Real  to_kW_m = dx * 1.0e-3_rt;
    for (MFIter mfi(heat_flux, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& q  = heat_flux.array(mfi);
        auto const& li = m_launch_intensity->array(mfi);
        auto const& id = structure_id.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int sid = static_cast<int>(id(i, j, k) + 0.5_rt);
            if (sid < 1 || sid > nmax) { return; }
            const Real qs = p_q[sid];
            if (qs <= 0.0_rt) { return; }
            q(i, j, k)  += qs;
            li(i, j, k)  = qs * to_kW_m;
        });
    }

    // Incident radiant flux: every burning footprint cell is a ground point
    // source of radiant power chi_r * q * dA, seen by the cells within the
    // cutoff radius. Sources are listed on the host (every rank knows every
    // footprint); each box takes only the sources whose reach touches it.
    if (m_p.rad_radius_m <= 0.0_rt || m_p.rad_fraction <= 0.0_rt) { return; }
    const Real dy      = m_fg.geom.CellSize(1);
    const Real dA      = dx * dy;
    const Real R       = m_p.rad_radius_m;
    const Real R2      = R * R;
    const Real r2_min  = 0.25_rt * dx * dx;
    const int  reach_i = static_cast<int>(std::ceil(R / dx)) + 1;
    const int  reach_j = static_cast<int>(std::ceil(R / dy)) + 1;
    const auto prob_lo = m_fg.geom.ProbLoArray();

    std::vector<int>  src_i, src_j;
    std::vector<Real> src_P;
    for (int s = 1; s <= m_n; ++s) {
        if (h_q[s] <= 0.0_rt) { continue; }
        const Real P = m_p.rad_fraction * h_q[s] * dA;
        for (std::size_t c = 0; c < m_foot_i[s].size(); ++c) {
            src_i.push_back(m_foot_i[s][c]);
            src_j.push_back(m_foot_j[s][c]);
            src_P.push_back(P);
        }
    }
    if (src_i.empty()) { return; }

    for (MFIter mfi(*m_rad_flux, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box reach = amrex::grow(bx, IntVect(AMREX_D_DECL(reach_i, reach_j, 0)));
        std::vector<Real> h_sx, h_sy, h_sP;
        for (std::size_t n = 0; n < src_i.size(); ++n) {
            if (!reach.contains(IntVect(AMREX_D_DECL(src_i[n], src_j[n], bx.smallEnd(2))))) { continue; }
            h_sx.push_back(prob_lo[0] + (src_i[n] + 0.5_rt) * dx);
            h_sy.push_back(prob_lo[1] + (src_j[n] + 0.5_rt) * dy);
            h_sP.push_back(src_P[n]);
        }
        const int ns = static_cast<int>(h_sx.size());
        if (ns == 0) { continue; }
        Gpu::DeviceVector<Real> d_sx(ns), d_sy(ns), d_sP(ns);
        Gpu::copy(Gpu::hostToDevice, h_sx.begin(), h_sx.end(), d_sx.begin());
        Gpu::copy(Gpu::hostToDevice, h_sy.begin(), h_sy.end(), d_sy.begin());
        Gpu::copy(Gpu::hostToDevice, h_sP.begin(), h_sP.end(), d_sP.begin());
        const Real* p_sx = d_sx.data();
        const Real* p_sy = d_sy.data();
        const Real* p_sP = d_sP.data();
        auto const& rf = m_rad_flux->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real cx = prob_lo[0] + (i + 0.5_rt) * dx;
            const Real cy = prob_lo[1] + (j + 0.5_rt) * dy;
            Real sum = 0.0_rt;
            for (int n = 0; n < ns; ++n) {
                const Real ddx = cx - p_sx[n];
                const Real ddy = cy - p_sy[n];
                const Real r2  = ddx * ddx + ddy * ddy;
                if (r2 > R2) { continue; }
                sum += point_source_incident_flux(p_sP[n], r2, r2_min);
            }
            rf(i, j, k) = sum;
        });
        Gpu::streamSynchronize();   // the device vectors go out of scope here
    }
}

void StructureIgnition::update_state (Real time_new, Real dt,
                                      const MultiFab& structure_id,
                                      const MultiFab& nonburnable,
                                      const MultiFab& heat_load,
                                      const MultiFab& fireline_intensity,
                                      const MultiFab& ember_landings)
{
    // Per-structure partial reductions on this rank: the largest heat load
    // and the largest current intensity in the wall band (the burnable cells
    // within m_ring of the footprint, as report_exposure() defines it), and
    // the embers on the footprint. Cold band cells are skipped early.
    std::vector<Real> hl_max(m_n + 1, 0.0_rt), ib_max(m_n + 1, 0.0_rt), emb(m_n + 1, 0.0_rt);
    const int ring = m_ring;
    for (MFIter mfi(structure_id); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const ERFHostFabView id_v(structure_id[mfi]);
        const ERFHostFabView mk_v(nonburnable[mfi]);
        const ERFHostFabView hl_v(heat_load[mfi]);
        const ERFHostFabView ib_v(fireline_intensity[mfi]);
        const ERFHostFabView em_v(ember_landings[mfi]);
        auto id = id_v.array();  auto mk = mk_v.array();
        auto hl = hl_v.array();  auto ib = ib_v.array();  auto em = em_v.array();
        LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
            const int sid = static_cast<int>(id(i, j, 0) + 0.5_rt);
            if (sid > 0) {
                if (sid <= m_n) { emb[sid] += em(i, j, 0); }
                return;
            }
            if (mk(i, j, 0) > 0.5_rt) { return; }
            const Real h = hl(i, j, 0);
            const Real b = ib(i, j, 0);
            if (h <= 0.0_rt && b <= 0.0_rt) { return; }
            int seen[8]; int ns = 0;
            for (int dj = -ring; dj <= ring; ++dj) {
                for (int di = -ring; di <= ring; ++di) {
                    const int nid = static_cast<int>(id(i + di, j + dj, 0) + 0.5_rt);
                    if (nid <= 0 || nid > m_n) { continue; }
                    bool dup = false;
                    for (int n = 0; n < ns; ++n) { if (seen[n] == nid) { dup = true; break; } }
                    if (!dup && ns < 8) { seen[ns++] = nid; }
                }
            }
            for (int n = 0; n < ns; ++n) {
                hl_max[seen[n]] = std::max(hl_max[seen[n]], h);
                ib_max[seen[n]] = std::max(ib_max[seen[n]], b);
            }
        });
    }
    ParallelDescriptor::ReduceRealMax(hl_max.data(), m_n + 1);
    ParallelDescriptor::ReduceRealMax(ib_max.data(), m_n + 1);
    ParallelDescriptor::ReduceRealSum(emb.data(),    m_n + 1);

    // The same decision on every rank, from the same reduced numbers.
    bool changed = false;
    for (int s = 1; s <= m_n; ++s) {
        if (m_h_state[s] == Burning) {
            if (time_new - m_h_t_ign[s] >= m_curve.duration()) {
                m_h_state[s] = BurnedOut;
                changed = true;
                Print() << "[FIRE STRUCTURE] structure " << s << " burned out at t=" << time_new
                        << " s (ignited at " << m_h_t_ign[s] << " s)\n";
            }
            continue;
        }
        if (m_h_state[s] != Unignited) { continue; }
        if (m_p.intensity_kW_m > 0.0_rt && ib_max[s] >= m_p.intensity_kW_m) {
            m_h_t_above[s] += dt;
            changed = true;
        }
        const int cause = ignition_cause(hl_max[s], m_p.heat_load_J_m2,
                                         emb[s], m_p.ember_count,
                                         m_h_t_above[s], m_p.intensity_kW_m, m_p.residence_s);
        if (cause != None) {
            m_h_state[s] = Burning;
            m_h_t_ign[s] = time_new;
            m_h_cause[s] = cause;
            changed = true;
            Print() << "[FIRE STRUCTURE] structure " << s << " ignited at t=" << time_new
                    << " s cause=" << cause_name(cause)
                    << " heat_load_max_MJm2=" << hl_max[s] * 1.0e-6
                    << " embers=" << static_cast<long>(emb[s])
                    << " time_above_intensity_s=" << m_h_t_above[s] << "\n";
        }
    }
    if (changed) { write_state_field(structure_id); }
}

void StructureIgnition::write_state_field (const MultiFab& structure_id)
{
    std::vector<Real> h_st(m_n + 1), h_ti(m_n + 1), h_ta(m_n + 1), h_ca(m_n + 1);
    for (int s = 0; s <= m_n; ++s) {
        h_st[s] = static_cast<Real>(m_h_state[s]);
        h_ti[s] = m_h_t_ign[s];
        h_ta[s] = m_h_t_above[s];
        h_ca[s] = static_cast<Real>(m_h_cause[s]);
    }
    Gpu::DeviceVector<Real> d_st(m_n + 1), d_ti(m_n + 1), d_ta(m_n + 1), d_ca(m_n + 1);
    Gpu::copy(Gpu::hostToDevice, h_st.begin(), h_st.end(), d_st.begin());
    Gpu::copy(Gpu::hostToDevice, h_ti.begin(), h_ti.end(), d_ti.begin());
    Gpu::copy(Gpu::hostToDevice, h_ta.begin(), h_ta.end(), d_ta.begin());
    Gpu::copy(Gpu::hostToDevice, h_ca.begin(), h_ca.end(), d_ca.begin());
    const Real* p_st = d_st.data();
    const Real* p_ti = d_ti.data();
    const Real* p_ta = d_ta.data();
    const Real* p_ca = d_ca.data();
    const int nmax = m_n;
    for (MFIter mfi(*m_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& f  = m_state->array(mfi);
        auto const& id = structure_id.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int sid = static_cast<int>(id(i, j, k) + 0.5_rt);
            if (sid < 1 || sid > nmax) { return; }
            f(i, j, k, StateComp)        = p_st[sid];
            f(i, j, k, IgnitionTimeComp) = p_ti[sid];
            f(i, j, k, TimeAboveComp)    = p_ta[sid];
            f(i, j, k, CauseComp)        = p_ca[sid];
        });
    }
    Gpu::streamSynchronize();
}
