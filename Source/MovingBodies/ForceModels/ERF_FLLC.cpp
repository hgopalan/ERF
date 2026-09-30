#include "ERF_FLLC.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>

#include <AMReX.H>
#include <AMReX_BLassert.H>

using namespace amrex;

namespace erf_actuator {

namespace {
constexpr Real pi = 3.14159265358979323846;

// linear interpolation of one scalar table (x increasing) at xd, end values held
Real interp1 (const std::vector<Real>& x, const std::vector<Real>& y, Real xd)
{
    const std::size_t n = x.size();
    if (n == 1 || xd <= x.front()) { return y.front(); }
    if (xd >= x.back()) { return y.back(); }
    const auto it = std::upper_bound(x.begin(), x.end(), xd);
    const std::size_t k = static_cast<std::size_t>(it - x.begin());   // x[k-1] <= xd < x[k]
    const Real w = (xd - x[k-1]) / (x[k] - x[k-1]);
    return (Real(1.0) - w) * y[k-1] + w * y[k];
}
} // namespace

void
interpolate_along (const std::vector<Real>& x_src, const std::vector<Real>& v_src,
                   const std::vector<Real>& x_dst, std::vector<Real>& v_dst)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!x_src.empty() && v_src.size() == 3 * x_src.size(),
                                     "interpolate_along: one 3-vector per source coordinate");
    for (std::size_t i = 1; i < x_src.size(); ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(x_src[i] > x_src[i-1], "interpolate_along: source coordinates must increase");
    }
    v_dst.assign(3 * x_dst.size(), 0.0);
    const std::size_t n = x_src.size();
    for (std::size_t p = 0; p < x_dst.size(); ++p) {
        const Real xd = x_dst[p];
        std::size_t k = 1;
        Real w = 0.0;
        if (n == 1 || xd <= x_src.front()) { k = 1; w = 0.0; }
        else if (xd >= x_src.back()) { k = n - 1; w = 1.0; }
        else {
            const auto it = std::upper_bound(x_src.begin(), x_src.end(), xd);
            k = static_cast<std::size_t>(it - x_src.begin());
            w = (xd - x_src[k-1]) / (x_src[k] - x_src[k-1]);
        }
        if (n == 1) { for (int d = 0; d < 3; ++d) { v_dst[3*p+d] = v_src[d]; } continue; }
        for (int d = 0; d < 3; ++d) {
            v_dst[3*p+d] = (Real(1.0) - w) * v_src[3*(k-1)+d] + w * v_src[3*k+d];
        }
    }
}

Real
FLLC::kernel (Real r, Real eps)
{
    const Real eps2 = eps * eps;
    if (r == Real(0.0)) { return Real(0.5) / eps2; }
    // expm1 keeps the second term exact when r is a rounding residue of a coincident point:
    // exp(-x) - 1 computed directly loses every digit there and the kernel explodes
    const Real x = r * r / eps2;
    return std::exp(-x) / eps2 + std::expm1(-x) / (Real(2.0) * r * r);
}

std::vector<Real>
FLLC::node_widths (const std::vector<Real>& r)
{
    const std::size_t n = r.size();
    std::vector<Real> dr(n, 0.0);
    if (n == 1) { dr[0] = 1.0; return dr; }
    for (std::size_t i = 1; i + 1 < n; ++i) { dr[i] = Real(0.5) * std::abs(r[i+1] - r[i-1]); }
    dr[0] = Real(0.5) * std::abs(r[1] - r[0]);
    dr[n-1] = Real(0.5) * std::abs(r[n-1] - r[n-2]);
    return dr;
}

std::vector<Real>
FLLC::fine_grid (const std::vector<Real>& r, const std::vector<Real>& eps_opt, Real eps_dr)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(eps_dr > Real(0.0), "FLLC::fine_grid: eps_dr must be positive");
    std::vector<Real> rf{r.front()};
    const Real r_end = r.back();
    while (rf.back() < r_end) {
        const Real r0 = rf.back();
        // the spacing the optimal kernel needs here and one step ahead; the smaller keeps
        // spacing <= epsilon_opt / eps_dr everywhere
        const Real d1 = interp1(r, eps_opt, r0) / eps_dr;
        const Real d2 = interp1(r, eps_opt, r0 + d1) / eps_dr;
        const Real d = std::min(d1, d2);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(d > std::numeric_limits<Real>::epsilon() * std::max(Real(1.0), std::abs(r_end)),
                                         "FLLC::fine_grid: the optimal kernel width is zero somewhere (a zero chord?)");
        rf.push_back(std::min(r0 + d, r_end));
    }
    return rf;
}

void
FLLC::induced_velocity (const std::vector<Real>& r_eval, const std::vector<Real>& r_src,
                        const std::vector<Real>& g_src, const std::vector<Real>& dr_src,
                        const std::vector<Real>& eps_src, std::vector<Real>& u)
{
    const std::size_t m = r_src.size();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(g_src.size() == 3 * m && dr_src.size() == m && eps_src.size() == m,
                                     "FLLC::induced_velocity: inconsistent source arrays");
    u.assign(3 * r_eval.size(), 0.0);
    const Real c = Real(1.0) / (Real(2.0) * pi);
    for (std::size_t i = 0; i < r_eval.size(); ++i) {
        Real s[3] = {0.0, 0.0, 0.0};
        for (std::size_t j = 0; j < m; ++j) {
            const Real k = kernel(std::abs(r_eval[i] - r_src[j]), eps_src[j]) * dr_src[j];
            for (int d = 0; d < 3; ++d) { s[d] += g_src[3*j+d] * k; }
        }
        for (int d = 0; d < 3; ++d) { u[3*i+d] = c * s[d]; }
    }
}

FLLC::FLLC (std::string name, std::vector<Real> r, std::vector<Real> chord,
            Real eps_les, Real eps_chord, Real eps_dr, Real relax)
    : m_name(std::move(name)), m_eps_chord(eps_chord), m_eps_dr(eps_dr), m_r(std::move(r)), m_chord(std::move(chord)),
      m_eps_les(eps_les), m_relax(relax)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_r.size() >= 2 && m_chord.size() == m_r.size(),
                                     "FLLC: at least two force nodes, one chord each, for " + m_name);
    for (std::size_t i = 1; i < m_r.size(); ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_r[i] > m_r[i-1], "FLLC: the span coordinates must increase root to tip for " + m_name);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(eps_les > Real(0.0) && eps_chord > Real(0.0) && eps_dr > Real(0.0) &&
                                     relax > Real(0.0) && relax <= Real(1.0),
                                     "FLLC: eps_les, eps_chord and eps_dr must be positive and 0 < relax <= 1 for " + m_name);
    build_grids(eps_chord, eps_dr);
    m_du.assign(3 * m_r.size(), 0.0);
    m_target.assign(3 * m_r.size(), 0.0);
}

void
FLLC::build_grids (Real eps_chord, Real eps_dr)
{
    m_eps_opt.resize(m_r.size());
    for (std::size_t i = 0; i < m_r.size(); ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_chord[i] > Real(0.0), "FLLC: every chord must be positive for " + m_name);
        m_eps_opt[i] = eps_chord * m_chord[i];
    }
    m_dr = node_widths(m_r);
    m_rf = fine_grid(m_r, m_eps_opt, eps_dr);
    m_dr_f = node_widths(m_rf);
    m_eps_opt_f.resize(m_rf.size());
    for (std::size_t j = 0; j < m_rf.size(); ++j) { m_eps_opt_f[j] = interp1(m_r, m_eps_opt, m_rf[j]); }
}

void
FLLC::update (const std::vector<Real>& force_over_rho, const std::vector<Real>& vel_rel)
{
    const std::size_t n = m_r.size();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(force_over_rho.size() == 3 * n && vel_rel.size() == 3 * n,
                                     "FLLC::update: one force and one velocity per force node for " + m_name);
    // the lift force per unit span (per unit density) over |v|: G_i / |v_i|
    std::vector<Real> g(3 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real* f = &force_over_rho[3*i];
        const Real* v = &vel_rel[3*i];
        const Real v2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        const Real vmag = std::max(std::sqrt(v2), std::numeric_limits<Real>::epsilon());
        const Real fv = f[0]*v[0] + f[1]*v[1] + f[2]*v[2];
        for (int d = 0; d < 3; ++d) {
            g[3*i+d] = (f[d] - v[d] * fv / (vmag * vmag)) / m_dr[i] / vmag;
        }
    }
    // onto the fine grid, then the induced velocities at the force nodes for both kernels
    std::vector<Real> gf, u_les, u_opt;
    interpolate_along(m_r, g, m_rf, gf);
    const std::vector<Real> eps_les_f(m_rf.size(), m_eps_les);
    induced_velocity(m_r, m_rf, gf, m_dr_f, eps_les_f, u_les);
    induced_velocity(m_r, m_rf, gf, m_dr_f, m_eps_opt_f, u_opt);
    for (std::size_t k = 0; k < 3 * n; ++k) {
        m_target[k] = u_opt[k] - u_les[k];
        m_du[k] = (Real(1.0) - m_relax) * m_du[k] + m_relax * m_target[k];
    }
}

Real
FLLC::max_correction () const
{
    Real m = 0.0;
    for (std::size_t i = 0; i < m_r.size(); ++i) {
        m = std::max(m, std::sqrt(m_du[3*i]*m_du[3*i] + m_du[3*i+1]*m_du[3*i+1] + m_du[3*i+2]*m_du[3*i+2]));
    }
    return m;
}

Real
FLLC::rms_correction () const
{
    if (m_r.empty()) { return 0.0; }
    Real s = 0.0;
    for (Real v : m_du) { s += v * v; }
    return std::sqrt(s / static_cast<Real>(m_r.size()));
}

void
FLLC::write_state (std::ostream& out) const
{
    out << std::setprecision(17) << "fllc " << m_name << " " << m_r.size() << "\n";
    for (std::size_t i = 0; i < m_r.size(); ++i) {
        out << m_r[i] << " " << m_chord[i] << " " << m_du[3*i] << " " << m_du[3*i+1] << " " << m_du[3*i+2] << "\n";
    }
}

bool
FLLC::read_state (std::istream& in)
{
    std::string key, name;
    std::size_t n = 0;
    if (!(in >> key >> name >> n) || key != "fllc") { return false; }
    if (name != m_name || n != m_r.size()) {
        amrex::Abort("FLLC state for '" + name + "' with " + std::to_string(n) + " nodes does not match blade '" +
                     m_name + "' with " + std::to_string(m_r.size()) + " nodes");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> m_r[i] >> m_chord[i] >> m_du[3*i] >> m_du[3*i+1] >> m_du[3*i+2])) {
            amrex::Abort("FLLC state for '" + name + "' is truncated");
        }
    }
    for (std::size_t i = 1; i < n; ++i) {
        if (!(m_r[i] > m_r[i-1])) { amrex::Abort("FLLC state for '" + name + "' has non-increasing span coordinates"); }
    }
    build_grids(m_eps_chord, m_eps_dr);
    return true;
}

} // namespace erf_actuator
