// The owners of the bodies and the point-to-point exchanges of their points with the ranks whose boxes they touch.

#include "ERF_BodyExchange.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_ActuatorSpreading.H"

using namespace amrex;

namespace erf_actuator {

namespace {

// a cell index from a coordinate, kept far inside the int range for points far outside the domain
int cell_index (Real x, Real lo, Real dxi)
{
    const double c = std::floor(static_cast<double>(x - lo) * static_cast<double>(dxi));
    return static_cast<int>(std::max(-1.0e9, std::min(1.0e9, c)));
}

std::array<Real,6> empty_envelope ()
{
    return {{Real(1.0), Real(1.0), Real(1.0), Real(-1.0), Real(-1.0), Real(-1.0)}};
}

bool is_empty (const std::array<Real,6>& e) { return e[0] > e[3] || e[1] > e[4] || e[2] > e[5]; }

bool inside (const std::array<Real,6>& inner, const std::array<Real,6>& outer)
{
    if (is_empty(inner)) { return true; }
    if (is_empty(outer)) { return false; }
    for (int d = 0; d < 3; ++d) {
        if (inner[d] < outer[d] || inner[d+3] > outer[d+3]) { return false; }
    }
    return true;
}

// grow the box e to hold the points pos (3n), each with its own reach (n values, or none for no reach)
void hold_points (std::array<Real,6>& e, const std::vector<Real>& pos, const std::vector<Real>* reach)
{
    for (std::size_t p = 0; p < pos.size() / 3; ++p) {
        const Real r = (reach == nullptr) ? Real(0.0) : (*reach)[p];
        const std::array<Real,6> q{{pos[3*p] - r, pos[3*p+1] - r, pos[3*p+2] - r, pos[3*p] + r, pos[3*p+1] + r, pos[3*p+2] + r}};
        if (is_empty(e)) { e = q; continue; }
        for (int d = 0; d < 3; ++d) {
            e[d] = std::min(e[d], q[d]);
            e[d+3] = std::max(e[d+3], q[d+3]);
        }
    }
}

// the bodies a sender owns that a receiver takes part in, in increasing order: the sections of their messages
std::vector<int> sections (const std::vector<int>& owner, const std::vector<std::vector<int>>& part, int sender, int receiver)
{
    std::vector<int> b_list;
    for (int b = 0; b < static_cast<int>(owner.size()); ++b) {
        if (owner[b] == sender && std::binary_search(part[b].begin(), part[b].end(), receiver)) { b_list.push_back(b); }
    }
    return b_list;
}

} // namespace

Box
cells_within (const std::array<Real,3>& lo, const std::array<Real,3>& hi, const Geometry& geom, int pad, bool fitted)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    IntVect l, h;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        l[d] = cell_index(lo[d], plo[d], dxi[d]) - pad;
        h[d] = cell_index(hi[d], plo[d], dxi[d]) + pad;
    }
    if (fitted) {
        l[2] = geom.Domain().smallEnd(2);
        h[2] = geom.Domain().bigEnd(2);
    }
    return Box(l, h);
}

std::vector<Box>
domain_images (const Box& cells, const Geometry& geom)
{
    std::vector<Box> out;
    if (!cells.ok()) { return out; }
    out.push_back(cells);
    const Box& dom = geom.Domain();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const int dlo = dom.smallEnd(d), dhi = dom.bigEnd(d), n = dhi - dlo + 1;
        std::vector<Box> next;
        for (Box b : out) {
            if (geom.isPeriodic(d)) {
                if (b.length(d) >= n) {
                    b.setRange(d, dlo, n);
                    next.push_back(b);
                    continue;
                }
                // shift by whole periods so that the range starts in the domain
                const int s = static_cast<int>(std::floor(static_cast<double>(b.smallEnd(d) - dlo) / n));
                b.shift(d, -s * n);
                if (b.bigEnd(d) <= dhi) {
                    next.push_back(b);
                } else {
                    Box lo_part = b, hi_part = b;
                    lo_part.setBig(d, dhi);
                    hi_part.setSmall(d, dlo);
                    hi_part.setBig(d, b.bigEnd(d) - n);
                    next.push_back(lo_part);
                    next.push_back(hi_part);
                }
            } else {
                b.setSmall(d, std::max(b.smallEnd(d), dlo));
                b.setBig(d, std::min(b.bigEnd(d), dhi));
                if (b.ok()) { next.push_back(b); }
            }
        }
        out.swap(next);
    }
    return out;
}

std::vector<int>
ranks_touching (const Box& cells, const BoxArray& ba, const DistributionMapping& dm, const Geometry& geom)
{
    std::set<int> ranks;
    std::vector<std::pair<int,Box>> isects;
    for (const Box& piece : domain_images(cells, geom)) {
        ba.intersections(piece, isects);
        for (const auto& is : isects) { ranks.insert(dm[is.first]); }
    }
    return std::vector<int>(ranks.begin(), ranks.end());
}

std::vector<int>
choose_owners (const std::vector<Box>& footprints, const std::vector<Real>& dt, const BoxArray& ba,
               const DistributionMapping& dm, const Geometry& geom, int nranks)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ba.ixType().cellCentered() && !ba.empty(), "choose_owners: the bodies' level needs cell-centred boxes");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt.size() == footprints.size(), "choose_owners: one time step per body");
    for (int i = 0; i < static_cast<int>(ba.size()); ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dm[i] >= 0 && dm[i] < nranks, "choose_owners: the distribution names a rank outside 0 .. nranks - 1");
    }
    // two time steps that OpenFAST can step together (an unknown one, <= 0, only with another unknown one)
    auto same_dt = [] (Real a, Real b) {
        if (a <= Real(0.0) || b <= Real(0.0)) { return a <= Real(0.0) && b <= Real(0.0); }
        return std::abs(a - b) <= Real(1.0e-9) * std::max(a, b);
    };
    std::vector<int> owned(nranks, 0);
    std::vector<Real> rank_dt(nranks, Real(0.0));   // the time step of the bodies a rank owns
    // the bodies' time-step classes, so that an empty rank is kept for each class that has none yet
    std::vector<int> cls(footprints.size(), -1);
    std::vector<Real> class_dt;
    for (std::size_t b = 0; b < footprints.size(); ++b) {
        for (std::size_t c = 0; c < class_dt.size() && cls[b] < 0; ++c) { if (same_dt(class_dt[c], dt[b])) { cls[b] = static_cast<int>(c); } }
        if (cls[b] < 0) { cls[b] = static_cast<int>(class_dt.size()); class_dt.push_back(dt[b]); }
    }
    std::vector<int> class_has_rank(class_dt.size(), 0);
    int empty_ranks = nranks;
    int classes_without_rank = static_cast<int>(class_dt.size());
    std::vector<int> owners;
    std::vector<std::pair<int,Box>> isects;
    for (std::size_t b = 0; b < footprints.size(); ++b) {
        const Box& fp = footprints[b];
        std::map<int, Long> overlap;   // rank: cells of the footprint its boxes hold
        for (const Box& piece : domain_images(fp, geom)) {
            ba.intersections(piece, isects);
            for (const auto& is : isects) { overlap[dm[is.first]] += is.second.numPts(); }
        }
        // a rank the body may join: one stepping its time step, or an empty one, unless the empty ranks left are
        // all kept for the time steps that have no rank yet
        const bool own_class_placed = (class_has_rank[cls[b]] != 0);
        auto fits = [&] (int r) {
            if (owned[r] > 0) { return same_dt(rank_dt[r], dt[b]); }
            return !own_class_placed || empty_ranks > classes_without_rank;
        };
        // the least load among the ranks the body may join (any rank, when its time step clashes with all of them)
        bool any_fit = false;
        int least = std::numeric_limits<int>::max();
        for (int r = 0; r < nranks; ++r) {
            if (fits(r)) { any_fit = true; least = std::min(least, owned[r]); }
        }
        int best = -1;
        if (any_fit) {
            // a rank it stands on, if one is among the least loaded that fit: the most cells, then the lowest rank
            for (const auto& kv : overlap) {
                if (!fits(kv.first) || owned[kv.first] != least) { continue; }
                if (best < 0 || kv.second > overlap[best]) { best = kv.first; }
            }
            if (best < 0) {
                // otherwise the least loaded fitting rank whose boxes come nearest to the footprint's centre cell
                IntVect c;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { c[d] = (fp.smallEnd(d) + fp.bigEnd(d)) / 2; }
                std::vector<Long> dist(nranks, std::numeric_limits<Long>::max());
                for (int i = 0; i < static_cast<int>(ba.size()); ++i) {
                    const Box& bx = ba[i];
                    Long d2 = 0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        const Long g = std::max<Long>(0, std::max<Long>(bx.smallEnd(d) - c[d], c[d] - bx.bigEnd(d)));
                        d2 += g * g;
                    }
                    dist[dm[i]] = std::min(dist[dm[i]], d2);
                }
                for (int r = 0; r < nranks; ++r) {
                    if (!fits(r) || owned[r] != least) { continue; }
                    if (best < 0 || dist[r] < dist[best]) { best = r; }
                }
            }
        } else {
            // every rank already steps bodies of another time step: the fewest owned of all ranks, then the most
            // cells of the footprint, then the lowest rank; the driver's lockstep check then stops the run, naming
            // the two time steps
            for (int r = 0; r < nranks; ++r) {
                const Long cells = overlap.count(r) ? overlap[r] : Long(0);
                const Long best_cells = (best >= 0 && overlap.count(best)) ? overlap[best] : Long(0);
                if (best < 0 || owned[r] < owned[best] || (owned[r] == owned[best] && cells > best_cells)) { best = r; }
            }
        }
        if (owned[best] == 0) {
            rank_dt[best] = dt[b];
            --empty_ranks;
            if (class_has_rank[cls[b]] == 0) { class_has_rank[cls[b]] = 1; --classes_without_rank; }
        }
        owners.push_back(best);
        ++owned[best];
    }
    return owners;
}

void
exchange_buffers (std::map<int, std::vector<Real>>& send, const std::vector<int>& from, std::map<int, std::vector<Real>>& recv)
{
    const int me = ParallelDescriptor::MyProc();
    recv.clear();
#ifdef AMREX_USE_MPI
    // one tag per exchange, the same on every rank since every rank makes every exchange
    const int tag = ParallelDescriptor::SeqNum();
    if (ParallelDescriptor::NProcs() > 1) {
        MPI_Comm comm = ParallelDescriptor::Communicator();
        MPI_Datatype type = ParallelDescriptor::Mpi_typemap<Real>::type();
        std::vector<MPI_Request> reqs;
        reqs.reserve(send.size());
        for (auto& kv : send) {
            if (kv.first == me) { continue; }
            reqs.push_back(MPI_REQUEST_NULL);
            MPI_Isend(kv.second.data(), static_cast<int>(kv.second.size()), type, kv.first, tag, comm, &reqs.back());
        }
        for (int src : from) {
            if (src == me) { continue; }
            MPI_Status st;
            MPI_Probe(src, tag, comm, &st);
            int n = 0;
            MPI_Get_count(&st, type, &n);
            std::vector<Real>& buf = recv[src];
            buf.resize(static_cast<std::size_t>(n));
            MPI_Recv(buf.data(), n, type, src, tag, comm, MPI_STATUS_IGNORE);
        }
        if (!reqs.empty()) { MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE); }
    }
#endif
    if (std::find(from.begin(), from.end(), me) != from.end()) {
        auto it = send.find(me);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(it != send.end(), "exchange_buffers: this rank expects a buffer from itself but sends none");
        recv[me] = std::move(it->second);
    }
}

BodyExchange::BodyExchange (std::vector<int> owners)
    : m_owner(std::move(owners)), m_env(m_owner.size(), empty_envelope()), m_part(m_owner.size())
{
    for (int r : m_owner) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(r >= 0 && r < ParallelDescriptor::NProcs(), "BodyExchange: an owner is not a rank of this run");
    }
}

bool
BodyExchange::owns (int b) const
{
    return m_owner[b] == ParallelDescriptor::MyProc();
}

std::vector<int>
BodyExchange::senders () const
{
    const int me = ParallelDescriptor::MyProc();
    std::set<int> s;
    for (int b = 0; b < num_bodies(); ++b) {
        if (std::binary_search(m_part[b].begin(), m_part[b].end(), me)) { s.insert(m_owner[b]); }
    }
    return std::vector<int>(s.begin(), s.end());
}

std::vector<int>
BodyExchange::receivers () const
{
    std::set<int> s;
    for (int b = 0; b < num_bodies(); ++b) {
        if (owns(b)) { s.insert(m_part[b].begin(), m_part[b].end()); }
    }
    return std::vector<int>(s.begin(), s.end());
}

void
BodyExchange::find_participants (const Geometry& geom)
{
    for (int b = 0; b < num_bodies(); ++b) {
        m_part[b].clear();
        if (is_empty(m_env[b])) { continue; }
        // the same padding as a point's own cells (sample_velocity, spread_forces below), so that the
        // ranks of every point inside the envelope are among these
        const Box cells = cells_within({{m_env[b][0], m_env[b][1], m_env[b][2]}}, {{m_env[b][3], m_env[b][4], m_env[b][5]}},
                                       geom, 2, m_fitted);
        m_part[b] = ranks_touching(cells, m_ba, m_dm, geom);
    }
}

void
BodyExchange::ensure_envelopes (const std::vector<std::array<Real,6>>& needed, const std::vector<Real>& margin,
                                const BoxArray& ba, const DistributionMapping& dm, const Geometry& geom, bool fitted)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(needed.size()) == num_bodies() && static_cast<int>(margin.size()) == num_bodies(),
                                     "BodyExchange: one needed box and one margin per body");
    int grow = 0;
    for (int b = 0; b < num_bodies(); ++b) {
        if (owns(b) && !inside(needed[b], m_env[b])) { grow = 1; }
    }
    // the one operation every rank takes part in: whether any envelope must grow
    ParallelDescriptor::ReduceIntMax(grow);
    if (grow != 0) {
        // each owner writes its bodies' envelopes, grown where needed; the sum hands every rank all of them
        std::vector<Real> env(6 * static_cast<std::size_t>(num_bodies()), Real(0.0));
        for (int b = 0; b < num_bodies(); ++b) {
            if (!owns(b)) { continue; }
            std::array<Real,6> e = m_env[b];
            if (!inside(needed[b], e)) {
                const Real m = std::max(margin[b], Real(0.0));
                const std::array<Real,6>& n = needed[b];
                const std::array<Real,6> grown{{n[0] - m, n[1] - m, n[2] - m, n[3] + m, n[4] + m, n[5] + m}};
                if (is_empty(e)) {
                    e = grown;
                } else {
                    for (int d = 0; d < 3; ++d) { e[d] = std::min(e[d], grown[d]); e[d+3] = std::max(e[d+3], grown[d+3]); }
                }
            }
            for (int k = 0; k < 6; ++k) { env[6*b+k] = e[k]; }
        }
        ParallelDescriptor::ReduceRealSum(env.data(), static_cast<int>(env.size()));
        // every row comes from its body's owner alone (an empty envelope too), so the sum is that row exactly
        for (int b = 0; b < num_bodies(); ++b) {
            for (int k = 0; k < 6; ++k) { m_env[b][k] = env[6*b+k]; }
        }
        ++m_num_updates;
    }
    const bool grids_changed = !(ba == m_ba) || !(dm == m_dm) || fitted != m_fitted;
    if (grow != 0 || grids_changed) {
        m_ba = ba;
        m_dm = dm;
        m_fitted = fitted;
        find_participants(geom);
    }
}

void
BodyExchange::sample_velocity (const MultiFab& U, const MultiFab& V, const MultiFab& W, const MultiFab* z_phys_nd,
                               const Geometry& geom, const std::vector<std::vector<Real>>& pts, const std::vector<Real>& margin,
                               std::vector<std::vector<Real>>& vel)
{
    const int nb = num_bodies();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(pts.size()) == nb, "BodyExchange::sample_velocity: one point list per body");
    const int me = ParallelDescriptor::MyProc();
    const bool fitted = (z_phys_nd != nullptr);
    const BoxArray ba = amrex::convert(U.boxArray(), IntVect::TheZeroVector());
    const DistributionMapping& dm = U.DistributionMap();

    // the box of space each owned body's points need: the points themselves (a point's own cells
    // are padded in cells_within)
    std::vector<std::array<Real,6>> needed(nb, empty_envelope());
    for (int b = 0; b < nb; ++b) {
        if (!owns(b)) { continue; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pts[b].size() % 3 == 0, "BodyExchange::sample_velocity: points come as x,y,z triples");
        hold_points(needed[b], pts[b], nullptr);
    }
    ensure_envelopes(needed, margin, ba, dm, geom, fitted);

    // each owned point, moved into the domain across a periodic seam, to the ranks owning a box over
    // its cell's column (with a cell of slack each way, which the box test below settles exactly)
    std::vector<std::vector<Real>> wpos(nb);
    std::vector<std::map<int, std::vector<int>>> to(nb);   // body: receiver -> its point indices
    for (int b = 0; b < nb; ++b) {
        if (!owns(b)) { continue; }
        wpos[b] = wrap_periodic(pts[b], geom);
        for (int p = 0; p < static_cast<int>(wpos[b].size() / 3); ++p) {
            const std::array<Real,3> x{{wpos[b][3*p], wpos[b][3*p+1], wpos[b][3*p+2]}};
            for (int r : ranks_touching(cells_within(x, x, geom, 1, fitted), ba, dm, geom)) {
                // the envelope holds the point, and its cells are padded more than the point's: cannot fail
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::binary_search(m_part[b].begin(), m_part[b].end(), r),
                                                 "BodyExchange::sample_velocity: a point reaches a rank its body's envelope does not touch");
                to[b][r].push_back(p);
            }
        }
    }
    const std::vector<int> recv_ranks = receivers();
    const std::vector<int> send_ranks = senders();
    std::map<int, std::vector<Real>> out, in;
    for (int r : recv_ranks) {
        std::vector<Real>& buf = out[r];
        for (int b : sections(m_owner, m_part, me, r)) {
            const std::vector<int>& idx = to[b][r];
            buf.push_back(static_cast<Real>(idx.size()));
            for (int p : idx) { buf.insert(buf.end(), {wpos[b][3*p], wpos[b][3*p+1], wpos[b][3*p+2]}); }
        }
    }
    exchange_buffers(out, send_ranks, in);

    // sample every point received, from every owner at once, on this rank's boxes
    std::vector<Real> all_pos;
    for (int s : send_ranks) {
        const std::vector<Real>& buf = in[s];
        std::size_t k = 0;
        for (int b : sections(m_owner, m_part, s, me)) {
            amrex::ignore_unused(b);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k < buf.size(), "BodyExchange::sample_velocity: a message is shorter than its sections");
            const auto n = static_cast<std::size_t>(buf[k++]);
            all_pos.insert(all_pos.end(), buf.begin() + k, buf.begin() + k + 3 * n);
            k += 3 * n;
            if (s != me) { m_points_received += static_cast<long>(n); }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k == buf.size(), "BodyExchange::sample_velocity: a message is longer than its sections");
    }
    std::vector<Real> all_vel;
    std::vector<int> all_cnt;
    sample_velocity_local(U, V, W, z_phys_nd, geom, all_pos, all_vel, all_cnt);

    // each sender gets, per point it sent, how many of this rank's boxes hold it and the velocity there
    std::map<int, std::vector<Real>> reply_out, reply_in;
    {
        std::size_t q = 0;
        for (int s : send_ranks) {
            std::vector<Real>& buf = reply_out[s];
            std::size_t k = 0;
            for (int b : sections(m_owner, m_part, s, me)) {
                amrex::ignore_unused(b);
                const auto m = static_cast<std::size_t>(in[s][k++]);
                for (std::size_t p = 0; p < m; ++p, ++q) {
                    buf.insert(buf.end(), {static_cast<Real>(all_cnt[q]), all_vel[3*q], all_vel[3*q+1], all_vel[3*q+2]});
                }
                k += 3 * m;
            }
        }
    }
    exchange_buffers(reply_out, recv_ranks, reply_in);

    // the owner sums the answers over the ranks it asked, in rank order: one holds the point
    vel.assign(nb, std::vector<Real>());
    std::vector<std::vector<int>> cnt(nb);
    for (int b = 0; b < nb; ++b) {
        if (!owns(b)) { continue; }
        vel[b].assign(pts[b].size(), Real(0.0));
        cnt[b].assign(pts[b].size() / 3, 0);
    }
    for (int r : recv_ranks) {
        const std::vector<Real>& buf = reply_in[r];
        std::size_t k = 0;
        for (int b : sections(m_owner, m_part, me, r)) {
            for (int p : to[b][r]) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k + 4 <= buf.size(), "BodyExchange::sample_velocity: a reply is too short");
                cnt[b][p] += static_cast<int>(buf[k]);
                for (int d = 0; d < 3; ++d) { vel[b][3*p+d] += buf[k+1+d]; }
                k += 4;
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k == buf.size(), "BodyExchange::sample_velocity: a reply is too long");
    }
    for (int b = 0; b < nb; ++b) {
        if (!owns(b)) { continue; }
        for (std::size_t p = 0; p < cnt[b].size(); ++p) {
            if (cnt[b][p] != 1) {
                Abort("sample_velocity: point (" + std::to_string(pts[b][3*p]) + ", " + std::to_string(pts[b][3*p+1]) +
                      ", " + std::to_string(pts[b][3*p+2]) + ") was sampled by " + std::to_string(cnt[b][p]) +
                      " boxes; actuator points must lie inside the domain");
            }
        }
    }
}

void
BodyExchange::spread_forces (const std::vector<std::vector<Real>>& pos, const std::vector<std::vector<Real>>& force,
                             const std::vector<std::vector<Real>>& eps, const std::vector<Real>& margin,
                             const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom,
                             MultiFab& src_x, MultiFab& src_y, MultiFab& src_z)
{
    const int nb = num_bodies();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(pos.size()) == nb && static_cast<int>(force.size()) == nb &&
                                     static_cast<int>(eps.size()) == nb, "BodyExchange::spread_forces: one point list per body");
    const int me = ParallelDescriptor::MyProc();
    const bool fitted = (z_phys_nd != nullptr);
    const BoxArray ba = amrex::convert(src_x.boxArray(), IntVect::TheZeroVector());
    const DistributionMapping& dm = src_x.DistributionMap();
    check_spread_layout(z_phys_nd, detJ_cc, src_x, src_y, src_z);

    // each owned body's kernels, with their reach of 3 epsilon. A body whose points, forces or widths are
    // unusable (a diverged model) takes part with no points, and stops the run after the last exchange, so
    // that no rank is left waiting for this one
    std::vector<std::array<Real,6>> needed(nb, empty_envelope());
    std::vector<std::vector<Real>> reach(nb);
    std::vector<char> usable(nb, 1);
    std::string failed;
    for (int b = 0; b < nb; ++b) {
        if (!owns(b)) { continue; }
        const std::string problem = spread_points_problem(pos[b], force[b], eps[b]);
        if (!problem.empty()) {
            usable[b] = 0;
            if (failed.empty()) { failed = problem; }
            continue;
        }
        reach[b].resize(eps[b].size());
        for (std::size_t p = 0; p < eps[b].size(); ++p) { reach[b][p] = Real(3.0) * eps[b][p]; }
        hold_points(needed[b], pos[b], &reach[b]);
    }
    ensure_envelopes(needed, margin, ba, dm, geom, fitted);

    // each owned point to the ranks owning a box within its kernel's reach (and periodic images)
    std::vector<std::map<int, std::vector<int>>> to(nb);
    for (int b = 0; b < nb; ++b) {
        if (!owns(b) || usable[b] == 0) { continue; }
        for (int p = 0; p < static_cast<int>(pos[b].size() / 3); ++p) {
            const Real r = reach[b][p];
            const std::array<Real,3> lo{{pos[b][3*p] - r, pos[b][3*p+1] - r, pos[b][3*p+2] - r}};
            const std::array<Real,3> hi{{pos[b][3*p] + r, pos[b][3*p+1] + r, pos[b][3*p+2] + r}};
            for (int rk : ranks_touching(cells_within(lo, hi, geom, 2, fitted), ba, dm, geom)) {
                // the envelope holds the kernel, with the same padding: cannot fail
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::binary_search(m_part[b].begin(), m_part[b].end(), rk),
                                                 "BodyExchange::spread_forces: a kernel reaches a rank its body's envelope does not touch");
                to[b][rk].push_back(p);
            }
        }
    }
    const std::vector<int> recv_ranks = receivers();
    const std::vector<int> send_ranks = senders();

    // 0. on a uniform mesh the owner can find a kernel's whole normalisation alone (uniform_mesh_norm), when
    // the level's boxes cover its reach; it then sends F/S with the point, and the point skips the two rounds
    // of partial sums and scales. The other points (terrain, stretched dz, a reach beyond a refined level's
    // boxes) take those rounds. Decided per point by the owner, and sent as a flag with it
    std::vector<std::vector<Real>> norm(nb);
    std::vector<std::vector<char>> local(nb);
    const bool uniform = (z_phys_nd == nullptr && detJ_cc == nullptr);
    for (int b = 0; b < nb; ++b) {
        if (!owns(b) || usable[b] == 0) { continue; }
        norm[b].assign(pos[b].size(), Real(0.0));
        local[b].assign(eps[b].size(), 0);
        if (!uniform) { continue; }
        for (std::size_t p = 0; p < eps[b].size(); ++p) {
            if (uniform_mesh_norm(pos[b][3*p], pos[b][3*p+1], pos[b][3*p+2], eps[b][p], geom, ba, &norm[b][3*p])) { local[b][p] = 1; }
        }
    }
    auto scale_of = [&] (int b, int p, int d) {
        return norm[b][3*p+d] > Real(0.0) ? force[b][3*p+d] / norm[b][3*p+d] : Real(0.0);
    };

    // 1. the points, widths, and (for a local normalisation) scales to the ranks they reach:
    // x, y, z, epsilon, local, then F/S per component when local
    std::map<int, std::vector<Real>> out, in;
    // per owned body and receiver: how many of the points sent to it need the partial sums
    std::map<int, std::size_t> n_remote_to;
    for (int r : recv_ranks) {
        std::vector<Real>& buf = out[r];
        for (int b : sections(m_owner, m_part, me, r)) {
            const std::vector<int>& idx = to[b][r];
            buf.push_back(static_cast<Real>(idx.size()));
            for (int p : idx) {
                buf.insert(buf.end(), {pos[b][3*p], pos[b][3*p+1], pos[b][3*p+2], eps[b][p], static_cast<Real>(local[b][p])});
                if (local[b][p] != 0) {
                    buf.insert(buf.end(), {scale_of(b, p, 0), scale_of(b, p, 1), scale_of(b, p, 2)});
                } else {
                    ++n_remote_to[r];
                    ++m_remote_norms;
                }
            }
        }
    }
    exchange_buffers(out, send_ranks, in);

    // the points received, per body (each body has one owner, so one sender): positions, widths, and the
    // scales of those normalised on their owner
    std::vector<std::vector<Real>> rpos(nb), reps(nb), rscale(nb);
    std::vector<std::vector<char>> rlocal(nb);
    std::map<int, std::size_t> n_remote_from;   // per sender: its points that need this rank's partial sums
    for (int s : send_ranks) {
        const std::vector<Real>& buf = in[s];
        std::size_t k = 0;
        for (int b : sections(m_owner, m_part, s, me)) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k < buf.size(), "BodyExchange::spread_forces: a message is shorter than its sections");
            const auto n = static_cast<std::size_t>(buf[k++]);
            for (std::size_t p = 0; p < n; ++p) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k + 5 <= buf.size(), "BodyExchange::spread_forces: a message is shorter than its points");
                rpos[b].insert(rpos[b].end(), {buf[k], buf[k+1], buf[k+2]});
                reps[b].push_back(buf[k+3]);
                const bool loc = (buf[k+4] != Real(0.0));
                rlocal[b].push_back(loc ? 1 : 0);
                k += 5;
                if (loc) {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k + 3 <= buf.size(), "BodyExchange::spread_forces: a message is shorter than its points");
                    rscale[b].insert(rscale[b].end(), {buf[k], buf[k+1], buf[k+2]});
                    k += 3;
                } else {
                    rscale[b].insert(rscale[b].end(), {Real(0.0), Real(0.0), Real(0.0)});
                    ++n_remote_from[s];
                }
            }
            if (s != me) { m_points_received += static_cast<long>(n); }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k == buf.size(), "BodyExchange::spread_forces: a message is longer than its sections");
    }
    // all of them in body order, as spread_forces() orders its points
    std::vector<Real> all_pos, all_eps;
    for (int b = 0; b < nb; ++b) {
        all_pos.insert(all_pos.end(), rpos[b].begin(), rpos[b].end());
        all_eps.insert(all_eps.end(), reps[b].begin(), reps[b].end());
    }

    // 2. this rank's part of the normalisation of every point not normalised on its owner, back to the owners
    // (only between the ranks that have such points: none at all on a uniform mesh whose boxes cover the kernels).
    // This rank's bounds of a fitted mesh's heights, once for both passes
    ZBounds zb;
    if (z_phys_nd != nullptr && !all_eps.empty()) { zb = mesh_z_bounds(*z_phys_nd, geom, false); }
    std::vector<Real> rem_pos, rem_eps, part_norm;
    for (int b = 0; b < nb; ++b) {
        for (std::size_t p = 0; p < reps[b].size(); ++p) {
            if (rlocal[b][p] != 0) { continue; }
            rem_pos.insert(rem_pos.end(), {rpos[b][3*p], rpos[b][3*p+1], rpos[b][3*p+2]});
            rem_eps.push_back(reps[b][p]);
        }
    }
    spread_partial_norms(rem_pos, rem_eps, z_phys_nd, detJ_cc, geom, src_x, part_norm, &zb);
    std::map<int, std::vector<Real>> norm_out, norm_in;
    std::vector<int> norm_from;
    {
        // the offset of each body's remote points in the body-ordered list
        std::vector<std::size_t> first(nb + 1, 0);
        for (int b = 0; b < nb; ++b) {
            first[b+1] = first[b] + static_cast<std::size_t>(std::count(rlocal[b].begin(), rlocal[b].end(), char(0)));
        }
        for (int s : send_ranks) {
            if (n_remote_from[s] == 0) { continue; }
            std::vector<Real>& buf = norm_out[s];
            for (int b : sections(m_owner, m_part, s, me)) {
                buf.insert(buf.end(), part_norm.begin() + 3 * first[b], part_norm.begin() + 3 * first[b+1]);
            }
        }
        for (int r : recv_ranks) {
            if (n_remote_to[r] == 0) { continue; }
            norm_from.push_back(r);
        }
    }
    exchange_buffers(norm_out, norm_from, norm_in);

    // 3. the owner sums the parts in rank order and sends back force over normalisation per remote point
    for (int r : norm_from) {
        const std::vector<Real>& buf = norm_in[r];
        std::size_t k = 0;
        for (int b : sections(m_owner, m_part, me, r)) {
            for (int p : to[b][r]) {
                if (local[b][p] != 0) { continue; }
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k + 3 <= buf.size(), "BodyExchange::spread_forces: a reply is too short");
                for (int d = 0; d < 3; ++d) { norm[b][3*p+d] += buf[k+d]; }
                k += 3;
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k == buf.size(), "BodyExchange::spread_forces: a reply is too long");
    }
    // a kernel that reaches no face stops the run, but only after the last exchange, so that no rank is
    // left waiting for this one (it sends zero scales for it meanwhile)
    for (int b = 0; b < nb && failed.empty(); ++b) {
        if (!owns(b) || usable[b] == 0) { continue; }
        for (std::size_t p = 0; p < eps[b].size() && failed.empty(); ++p) {
            for (int d = 0; d < 3; ++d) {
                if (!(norm[b][3*p+d] > Real(0.0))) {
                    failed = "spread_forces: point (" + std::to_string(pos[b][3*p]) + ", " + std::to_string(pos[b][3*p+1]) +
                             ", " + std::to_string(pos[b][3*p+2]) + ") reaches no face of the mesh within 3 epsilon = " +
                             std::to_string(3.0 * eps[b][p]) + " m; actuator points must lie inside the domain";
                    break;
                }
            }
        }
    }
    std::map<int, std::vector<Real>> scale_out, scale_in;
    std::vector<int> scale_from;
    for (int r : norm_from) {
        std::vector<Real>& buf = scale_out[r];
        for (int b : sections(m_owner, m_part, me, r)) {
            for (int p : to[b][r]) {
                if (local[b][p] != 0) { continue; }
                buf.insert(buf.end(), {scale_of(b, p, 0), scale_of(b, p, 1), scale_of(b, p, 2)});
            }
        }
    }
    for (int s : send_ranks) {
        if (n_remote_from[s] == 0) { continue; }
        scale_from.push_back(s);
    }
    exchange_buffers(scale_out, scale_from, scale_in);

    // 4. every kernel this rank's faces see, in body order
    for (int s : scale_from) {
        const std::vector<Real>& buf = scale_in[s];
        std::size_t k = 0;
        for (int b : sections(m_owner, m_part, s, me)) {
            for (std::size_t p = 0; p < reps[b].size(); ++p) {
                if (rlocal[b][p] != 0) { continue; }
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k + 3 <= buf.size(), "BodyExchange::spread_forces: a scale message is too short");
                for (int d = 0; d < 3; ++d) { rscale[b][3*p+d] = buf[k+d]; }
                k += 3;
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k == buf.size(), "BodyExchange::spread_forces: a scale message is too long");
    }
    std::vector<Real> all_scale;
    for (int b = 0; b < nb; ++b) { all_scale.insert(all_scale.end(), rscale[b].begin(), rscale[b].end()); }
    add_scaled_kernels(all_pos, all_scale, all_eps, z_phys_nd, detJ_cc, geom, src_x, src_y, src_z, &zb);
    if (!failed.empty()) { Abort(failed); }
}

} // namespace erf_actuator
