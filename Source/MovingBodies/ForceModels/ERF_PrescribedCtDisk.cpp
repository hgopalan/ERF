#include "ERF_PrescribedCtDisk.H"

#include "ERF_DiagnosticsLog.H"

#include <cmath>
#include <fstream>
#include <iomanip>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

using namespace amrex;

namespace erf_actuator {

namespace {
constexpr Real pi = 3.14159265358979323846;
}

PrescribedCtDisk::PrescribedCtDisk (const MovingBodyInputs& in)
    : m_name(in.name), m_output_root(in.output_root),
      m_radius(in.rotor_radius), m_ct(in.ct), m_rho(in.air_density)
{
    m_center = {{in.base_pos[0], in.base_pos[1], in.base_pos[2] + in.hub_height}};
    const Real yaw = in.yaw_deg * pi / 180.0;
    m_normal = {{std::cos(yaw), std::sin(yaw), 0.0}};
    // in-plane basis: e1 horizontal, e2 vertical
    const std::array<Real,3> e1{{-std::sin(yaw), std::cos(yaw), 0.0}};
    const std::array<Real,3> e2{{0.0, 0.0, 1.0}};

    // polar grid: ring i at radius (i + 1/2) dr with the area of the annulus split over the
    // azimuthal points, so the point areas sum to pi R^2 exactly
    const int nr = in.num_points_r;
    const int nt = in.num_points_t;
    const Real dr = m_radius / nr;
    const Real upstream = in.sample_diameters_upstream * 2.0 * m_radius;
    m_disk_pos.reserve(3 * nr * nt);
    m_sample_pos.reserve(3 * nr * nt);
    m_area.reserve(nr * nt);
    for (int i = 0; i < nr; ++i) {
        const Real r = (i + 0.5) * dr;
        const Real ring_area = pi * ((r + 0.5*dr)*(r + 0.5*dr) - (r - 0.5*dr)*(r - 0.5*dr));
        for (int j = 0; j < nt; ++j) {
            const Real th = 2.0 * pi * (j + 0.5) / nt;
            const Real c = std::cos(th), s = std::sin(th);
            for (int d = 0; d < 3; ++d) {
                const Real x = m_center[d] + r * (c * e1[d] + s * e2[d]);
                m_disk_pos.push_back(x);
                m_sample_pos.push_back(x - upstream * m_normal[d]);
            }
            m_area.push_back(ring_area / nt);
        }
    }
    m_area_total = 0.0;
    for (Real a : m_area) { m_area_total += a; }
    m_force.assign(m_disk_pos.size(), 0.0);
}

void
PrescribedCtDisk::update (const std::vector<Real>& sample_vel, const std::vector<Real>& disk_vel)
{
    const int n = num_points();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(sample_vel.size()) == 3 * n &&
                                     static_cast<int>(disk_vel.size()) == 3 * n,
                                     "PrescribedCtDisk::update: one u,v,w per disk point");
    Real u_inf = 0.0, u_disk = 0.0;
    for (int p = 0; p < n; ++p) {
        Real un_s = 0.0, un_d = 0.0;
        for (int d = 0; d < 3; ++d) {
            un_s += sample_vel[3*p+d] * m_normal[d];
            un_d += disk_vel[3*p+d] * m_normal[d];
        }
        u_inf  += m_area[p] * un_s;
        u_disk += m_area[p] * un_d;
    }
    m_u_inf = u_inf / m_area_total;
    m_u_disk = u_disk / m_area_total;
    // thrust along the normal, opposing the flow through the disk; the fluid gets -T n
    m_thrust = 0.5 * m_rho * m_ct * m_u_inf * m_u_inf * m_area_total;
    for (int p = 0; p < n; ++p) {
        const Real share = m_area[p] / m_area_total;
        for (int d = 0; d < 3; ++d) { m_force[3*p+d] = -m_thrust * share * m_normal[d]; }
    }
}

void
PrescribedCtDisk::open_diagnostics (bool truncate) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    if (erf_actuator::open_log(out, m_output_root + "_disk.csv", truncate)) {
        out << "time,u_inf,u_disk,thrust,power,spread_thrust\n";
    }
}

void
PrescribedCtDisk::write_diagnostics (Real time, Real spread_total) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out(m_output_root + "_disk.csv", std::ios::app);
    out << std::setprecision(10) << time << "," << m_u_inf << "," << m_u_disk << ","
        << m_thrust << "," << power() << "," << spread_total << "\n";
}

} // namespace erf_actuator
