#ifndef ERF_PRESCRIBED_CT_DISK_H
#define ERF_PRESCRIBED_CT_DISK_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_MovingBodiesInputs.H"

namespace erf_actuator {

/**
 * A uniformly loaded actuator disk with a prescribed thrust coefficient: the classical
 * test rotor. The disk of radius R and normal n (in the horizontal plane, yawed by
 * yaw_deg from +x) is covered by a polar grid of points; the free-stream speed U is the
 * area-weighted mean of the normal velocity sampled on the same grid one or more diameters
 * upstream, the thrust is T = 1/2 rho Ct U^2 pi R^2, and each disk point carries the share
 * of -T n (the force on the fluid) proportional to its area. The disk-averaged normal
 * velocity U_d is sampled too, for the diagnostics (time, U, U_d, thrust, power = T U_d).
 */
class PrescribedCtDisk
{
public:
    explicit PrescribedCtDisk (const MovingBodyInputs& in);

    const std::string& name () const { return m_name; }

    /** The prefix of this disk's diagnostics files (erf.moving_bodies.<name>.output_root). */

    const std::string& output_root () const { return m_output_root; }
    int num_points () const { return static_cast<int>(m_area.size()); }

    /** 3*n positions of the disk points (the force points). */
    const std::vector<amrex::Real>& disk_points () const { return m_disk_pos; }
    /** 3*n positions of the upstream sampling points, in the same order. */
    const std::vector<amrex::Real>& sample_points () const { return m_sample_pos; }

    /**
     * Take the sampled velocities (3*n each, in point order) at the upstream and disk points,
     * update the free-stream and disk speeds and the thrust.
     */
    void update (const std::vector<amrex::Real>& sample_vel, const std::vector<amrex::Real>& disk_vel);

    /** 3*n forces on the fluid at the disk points (N), from the last update. */
    const std::vector<amrex::Real>& forces () const { return m_force; }

    amrex::Real free_stream_speed () const { return m_u_inf; }
    amrex::Real disk_speed () const { return m_u_disk; }
    amrex::Real thrust () const { return m_thrust; }
    amrex::Real power () const { return m_thrust * m_u_disk; }
    std::array<amrex::Real,3> normal () const { return m_normal; }
    std::array<amrex::Real,3> center () const { return m_center; }
    amrex::Real radius () const { return m_radius; }
    amrex::Real area () const { return m_area_total; }

    /** Raise the disk and its sampling points by dz (m): the terrain height under the base, at start-up. */
    void shift_z (amrex::Real dz);

    /** Open <output_root>_disk.csv: truncated for a fresh run, appended to after a restart. */
    void open_diagnostics (bool truncate) const;
    /** Append one row: time, u_inf, u_disk, thrust, power, plus the integrated source total. */
    void write_diagnostics (double time, amrex::Real spread_total) const;

private:
    std::string m_name;
    std::string m_output_root;
    std::array<amrex::Real,3> m_center{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> m_normal{{1.0, 0.0, 0.0}};
    amrex::Real m_radius{0.0};
    amrex::Real m_ct{0.0};
    amrex::Real m_rho{1.225};
    amrex::Real m_area_total{0.0};
    std::vector<amrex::Real> m_disk_pos, m_sample_pos, m_area, m_force;
    amrex::Real m_u_inf{0.0}, m_u_disk{0.0}, m_thrust{0.0};
};

} // namespace erf_actuator

#endif
