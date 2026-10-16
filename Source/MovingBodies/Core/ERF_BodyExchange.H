#ifndef ERF_BODY_EXCHANGE_H
#define ERF_BODY_EXCHANGE_H

/**
 * Which rank owns each body, and the point-to-point messages that sample the flow at, and spread
 * the forces from, the points of the bodies a rank owns, so that a body's data goes only to the
 * ranks whose boxes its points touch.
 */

#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_ActuatorDeviceBuffers.H"

namespace erf_actuator {

/**
 * The cells of geom's index space within a box of space from lo to hi (m), grown by pad cells each
 * way: floor((lo - prob_lo) / dx) - pad to floor((hi - prob_lo) / dx) + pad. In z the same on a
 * uniform-dz mesh, and the whole height of the domain on a terrain-following or stretched one
 * (fitted), where a height's cell is not its height over dz. Not clipped to the domain.
 */
amrex::Box cells_within (const std::array<amrex::Real,3>& lo, const std::array<amrex::Real,3>& hi,
                         const amrex::Geometry& geom, int pad, bool fitted);

/**
 * The parts of a range of cells inside the domain: in a periodic direction the cells beyond the
 * domain are moved by whole periods to its other side (all of the direction once the range is as
 * long as the domain), in a non-periodic one they are dropped. Empty when nothing is left.
 */
std::vector<amrex::Box> domain_images (const amrex::Box& cells, const amrex::Geometry& geom);

/**
 * The ranks owning a box of ba (cell-centred) that intersects the range of cells or one of its
 * periodic images (domain_images): sorted, each rank once.
 */
std::vector<int> ranks_touching (const amrex::Box& cells, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                                 const amrex::Geometry& geom);

/**
 * The owner rank of each body, taken in order. footprints[b] is the range of cells body b stands on
 * (its base column, widened by its rotor radius when that is known; see cells_within), in ba's index
 * space; dt[b] its model's time step (s; <= 0 when unknown), since a rank steps the bodies it owns
 * together and they must share it. The owner is one of the ranks whose boxes of ba (the bodies' level,
 * cell-centred) overlap the footprint or one of its periodic images, as long as one of them is among
 * the least loaded ranks the body may join: the one holding the most cells of the footprint, then the
 * lowest rank. Otherwise it is the least loaded such rank whose boxes come nearest to the footprint's
 * centre (then the lowest). A body may join a rank whose bodies share its time step, or an empty rank,
 * except that one empty rank is kept for each time step that has no rank yet. So the bodies spread
 * over the ranks as evenly as dealing them out would (one per rank while there are as many ranks), on
 * ranks they stand on where that costs no balance, and bodies of different time steps share a rank
 * only when there are more time steps than ranks. Then the body goes to the least loaded rank of all
 * (the most cells of its footprint, then the lowest rank), and the driver's lockstep check stops the
 * run. Every rank computes the same owners from the same boxes.
 * @param nranks the ranks the owners are chosen from (the run's: every rank in dm is below it)
 */
std::vector<int> choose_owners (const std::vector<amrex::Box>& footprints, const std::vector<amrex::Real>& dt,
                                const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                                const amrex::Geometry& geom, int nranks);

/**
 * The exchanges of the bodies' data between their owners and the ranks whose boxes their points
 * touch. Every body has one owner rank, which alone knows its points (and runs its model); each
 * step the owner sends each point only to the ranks whose boxes can hold it (to sample the flow)
 * or its kernel (to spread its force), and gets back only what those ranks computed. No rank
 * receives the points of a body whose envelope does not reach its boxes.
 *
 * A body's envelope is a box of space (m) that holds every point it samples and the reach of
 * every kernel it spreads, grown by a margin its owner chooses (a rotor radius, so that a blade
 * turning or a rotor yawing stays inside). The ranks whose boxes the envelope touches (its
 * participants) are the only ranks the owner exchanges with. An exchange may first check, with
 * one integer reduction over all ranks, whether some body's points have left its envelope; the
 * envelopes of those bodies then grow to hold them, and every rank learns the new ones. That
 * check is the only operation of an exchange in which every rank takes part. It runs on every
 * exchange by default; with set_check_interval(k) on every k-th sampling and every k-th spreading
 * exchange only (and whenever the grids change). An envelope must then hold the body's points
 * until the next check, wherever its blades turn: it is built to hold, besides the points grown by
 * the margin, the sphere about the body's hub (set_hub) of its sweep (a rotor's tip radius) plus
 * the margin. At a check, an owner whose body's hub (set_hub; else the centre of its points, with
 * no sweep) has moved by more than half the margin since its envelope was last built has the
 * envelope built afresh, so that the margin is restored ahead of a moving body. Between checks
 * every owner checks, on its own, that its points are inside the envelope, and aborts if not: the
 * hub must not drift by more than the margin less the kernels' reach (3 epsilon) between two
 * builds, half the margin of it within one interval.
 *
 * The results are those of sample_velocity() and spread_forces() over all the points, whatever
 * the owners: a sampled velocity is the one value of the one box holding the point. A kernel's
 * normalisation is, on a uniform mesh whose boxes cover its reach, the owner's own sum over the
 * faces it reaches (uniform_mesh_norm(), sent with the point, so that spreading takes one message
 * each way instead of three); otherwise the sum of the participants' partial sums taken in rank
 * order. Either way the faces are those spread_forces() sums, added in another order, so the two
 * can differ in the last bits.
 */
class BodyExchange
{
public:
    BodyExchange () = default;

    /** The owner rank of every body (the same on every rank). */
    explicit BodyExchange (std::vector<int> owners);

    int num_bodies () const { return static_cast<int>(m_owner.size()); }
    int owner (int b) const { return m_owner[b]; }
    bool owns (int b) const;
    /** The ranks that body b exchanges with (its participants): empty before its first exchange. */
    const std::vector<int>& participants (int b) const { return m_part[b]; }
    /** Body b's envelope (lo, hi in m); lo > hi before its first exchange. */
    const std::array<amrex::Real,6>& envelope (int b) const { return m_env[b]; }
    /** How often the envelopes were rebuilt (any body): for the diagnostics and the tests. */
    int num_envelope_updates () const { return m_num_updates; }
    /** The points this rank has received from other ranks, over every exchange so far: for the tests. */
    long points_received () const { return m_points_received; }
    /** The kernels this rank's bodies sent whose normalisation the ranks they reach had to sum (not normalised on the owner): for the tests. */
    long remote_normalisations () const { return m_remote_norms; }
    /** How many exchanges ran the envelope check (the one reduction): for the diagnostics and the tests. */
    int num_envelope_checks () const { return m_num_checks; }

    /**
     * Check the envelopes on every k-th sampling and every k-th spreading exchange only (k >= 1; 1, the
     * default, checks them on every exchange). The same k on every rank.
     */
    void set_check_interval (int k);
    int check_interval () const { return m_check_interval; }
    /**
     * On body b's owner: where its hub is now (m), whose drift rebuilds the envelope (see the class), and
     * how far from the hub its points turn (m; a rotor's tip radius), which a throttled envelope holds whole.
     */
    void set_hub (int b, const std::array<amrex::Real,3>& x, amrex::Real sweep = 0);
    /** The device memory its sampling and spreading kernels reuse from step to step (and their launch count): for the tests. */
    const ActuatorDeviceBuffers& device_buffers () const { return m_buffers; }

    /**
     * Collective. Sample the face-centred velocities at the points of every body this rank owns,
     * as sample_velocity() would: each point by the box holding its cell, an abort naming the point
     * if no box or more than one holds it.
     * @param pts per body: on its owner its 3n point coordinates (m), ignored on the other ranks
     * @param margin per body: on its owner, how far (m) a new envelope reaches beyond the points
     * @param vel per body: on its owner its 3n velocities (m/s) on return, empty on the other ranks
     */
    void sample_velocity (const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                          const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                          const std::vector<std::vector<amrex::Real>>& pts, const std::vector<amrex::Real>& margin,
                          std::vector<std::vector<amrex::Real>>& vel);

    /**
     * Collective. Add the forces of every body this rank owns to the face-centred sources, as
     * spread_forces() would spread them (each point with its own width and its normalisation over
     * every face of the level), but onto sources that may already hold other forces: they are
     * added to, not set. The points of body b, in order, are added after those of the bodies
     * before it, so that a face gets its terms in the order spread_forces() adds them.
     * @param pos,force,eps per body: on its owner its points (3n, m), forces on the fluid (3n, N)
     *        and widths (n, m), ignored on the other ranks
     * @param margin per body: on its owner, how far (m) a new envelope reaches beyond the kernels
     */
    void spread_forces (const std::vector<std::vector<amrex::Real>>& pos, const std::vector<std::vector<amrex::Real>>& force,
                        const std::vector<std::vector<amrex::Real>>& eps, const std::vector<amrex::Real>& margin,
                        const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom,
                        amrex::MultiFab& src_x, amrex::MultiFab& src_y, amrex::MultiFab& src_z);

private:
    std::vector<int> m_owner;
    std::vector<std::array<amrex::Real,6>> m_env;   //!< per body: lo x y z, hi x y z (m)
    std::vector<std::vector<int>> m_part;           //!< per body: its participants, sorted
    amrex::BoxArray m_ba;                           //!< the cell-centred grids the participants were found on
    amrex::DistributionMapping m_dm;
    bool m_fitted{false};
    int m_num_updates{0};
    long m_points_received{0};
    long m_remote_norms{0};
    int m_check_interval{1};
    int m_num_checks{0};
    std::array<long,2> m_calls{{0, 0}};             //!< sampling, spreading exchanges so far
    std::vector<std::array<amrex::Real,3>> m_hub;   //!< per owned body: its hub, when set_hub was called
    std::vector<char> m_has_hub;
    std::vector<amrex::Real> m_sweep;               //!< per owned body: how far its points turn about the hub
    std::vector<std::array<amrex::Real,3>> m_hub_at_build;   //!< per owned body: its hub when its envelope was last built
    std::vector<char> m_has_hub_at_build;
    ActuatorDeviceBuffers m_buffers;                //!< kept from step to step: grows to the largest exchange, never shrinks

    /**
     * Collective. Grow the envelope of every body whose needed box (lo, hi in m, on its owner; lo > hi
     * when it has no points) is not inside it, by the needed box grown by the margin, and find the
     * participants again when an envelope or the grids changed. kind: 0 sampling, 1 spreading (each has
     * its own count of exchanges for the check interval). Between checks, only the local check that
     * the needed boxes are inside their envelopes (an abort if not).
     */
    void ensure_envelopes (const std::vector<std::array<amrex::Real,6>>& needed, const std::vector<amrex::Real>& margin,
                           const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::Geometry& geom,
                           bool fitted, int kind);
    void find_participants (const amrex::Geometry& geom);
    /** The owners that send to this rank: of every body this rank takes part in (sorted, once each; this rank itself when it owns one). */
    std::vector<int> senders () const;
    /** The ranks this rank sends to: the participants of the bodies it owns (sorted, once each; this rank itself when it is one). */
    std::vector<int> receivers () const;
};

/**
 * Collective. Send send[r] to every rank r it names and receive one buffer from every rank in
 * from; a buffer for this rank itself is moved across without MPI. Every rank must call it, with
 * sends and receives that pair up: a rank that names r in send must be named in r's from. Uses one
 * MPI tag of the ParallelDescriptor sequence, so every rank calls it the same number of times.
 *
 * With sizes (the length of the buffer from each rank in from, which the caller knows when it
 * is a reply to its own message) every receive is posted at once with its exact size, before the
 * sends, and one wait completes them all; a message of another size aborts. Without, the sizes
 * are learned from the messages, which are taken in the order they arrive (matched probes).
 */
void exchange_buffers (std::map<int, std::vector<amrex::Real>>& send, const std::vector<int>& from,
                       std::map<int, std::vector<amrex::Real>>& recv,
                       const std::map<int, std::size_t>* sizes = nullptr);

} // namespace erf_actuator

#endif
