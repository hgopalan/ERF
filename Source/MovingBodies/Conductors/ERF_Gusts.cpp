// Gusts on the conductors from a RANS wind: a static gust factor per span from the mean wind and k, and the
// travelling and random gusts' formulas.

#include "ERF_Gusts.H"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <AMReX.H>
#include <AMReX_BLassert.H>

namespace erf_conductors {

namespace {
constexpr double pi = 3.14159265358979323846;
} // namespace

bool parse_gust_type (const std::string& name, GustType& g)
{
    std::string l = name;
    for (auto& ch : l) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
    if (l == "none")   { g = GustType::None;   return true; }
    if (l == "factor") { g = GustType::Factor; return true; }
    if (l == "event")  { g = GustType::Event;  return true; }
    if (l == "random") { g = GustType::Random; return true; }
    return false;
}

double default_gust_sigma_factor (double Cmu0)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Cmu0 > 0.0 && std::isfinite(Cmu0), "default_gust_sigma_factor: Cmu0 must be positive");
    return 2.5 * Cmu0;
}

double gust_background_factor (double span, double length_scale)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(span > 0.0 && std::isfinite(span), "gust_background_factor: the span must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(length_scale > 0.0 && std::isfinite(length_scale),
                                     "gust_background_factor: the length scale must be positive");
    return 1.0 / (1.0 + 0.8 * span / length_scale);
}

SpanGust span_gust (double wind, double normal_wind, double k, double span, double diameter, double cd, double air_density,
                    double sigma_factor, double peak_factor, double length_scale)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wind >= 0.0 && std::isfinite(wind), "span_gust: the wind must be finite and >= 0");
    // the normal component cannot exceed the speed, beyond roundoff
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(normal_wind >= 0.0 && normal_wind <= wind * (1.0 + 1.0e-9) + 1.0e-12,
                                     "span_gust: the normal wind must lie between 0 and the wind speed");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k >= 0.0 && std::isfinite(k), "span_gust: k must be finite and >= 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(diameter > 0.0 && cd >= 0.0 && air_density > 0.0,
                                     "span_gust: the diameter and air density must be positive and C_d >= 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sigma_factor > 0.0 && peak_factor > 0.0,
                                     "span_gust: the sigma and peak factors must be positive");
    const double B = gust_background_factor(span, length_scale);
    SpanGust s;
    s.sigma = sigma_factor * std::sqrt(k);
    s.gust_wind = wind + peak_factor * s.sigma;
    s.mean_load = 0.5 * air_density * cd * diameter * normal_wind * normal_wind;
    // the fluctuation normal to the span: the streamwise part along sin(phi), the lateral along cos(phi)
    const double sin_phi = (wind > 0.0) ? std::min(normal_wind / wind, 1.0) : 0.0;
    const double sigma_v = gust_sigma_v_ratio * s.sigma;
    s.normal_sigma = std::sqrt(s.sigma * s.sigma * sin_phi * sin_phi + sigma_v * sigma_v * (1.0 - sin_phi * sin_phi));
    // with no mean wind normal to the span there is no intensity and no mean load to scale: they stay 0
    if (normal_wind > 0.0) {
        s.intensity = s.normal_sigma / normal_wind;
        s.gust_response = 1.0 + 2.0 * peak_factor * s.intensity * std::sqrt(B);
        s.peak_load = s.gust_response * s.mean_load;
        s.linear_valid = (normal_wind >= s.normal_sigma);
    }
    return s;
}

double tower_background_factor (double height, double length_scale)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(height > 0.0 && std::isfinite(height), "tower_background_factor: the height must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(length_scale > 0.0 && std::isfinite(length_scale),
                                     "tower_background_factor: the length scale must be positive");
    return 1.0 / (1.0 + 0.375 * height / length_scale);
}

double gust_integral_length (double height)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(height > 0.0 && std::isfinite(height), "gust_integral_length: the height must be positive");
    return 8.1 * ((height < 60.0) ? 0.7 * height : 42.0);
}

double gust_event_shape (double s)
{
    if (!(s >= 0.0 && s <= 1.0)) { return 0.0; }
    return 0.5 * (1.0 - std::cos(2.0 * pi * s));
}

double gust_event_phase (const GustEvent& e, double x, double y, double t)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(e.speed > 0.0 && e.duration > 0.0, "gust_event_phase: the speed and duration must be positive");
    const double arrival = e.time + ((x - e.x0) * e.ex + (y - e.y0) * e.ey) / e.speed;
    return (t - arrival) / e.duration;
}

namespace {
// splitmix64's finaliser: a bijection of the 64-bit integers whose outputs pass the usual tests of independence
std::uint64_t mix64 (std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
} // namespace

double gust_normal (std::uint64_t seed, std::uint64_t stream, std::uint64_t step)
{
    const std::uint64_t key = mix64(mix64(mix64(step) + stream * 0xd1b54a32d192ed03ULL) + seed * 0x9e3779b97f4a7c15ULL);
    // u1 in (0, 1], so its logarithm is finite; u2 in [0, 1); 53 bits each
    const double u1 = static_cast<double>((mix64(key ^ 1ULL) >> 11) + 1) * 0x1.0p-53;
    const double u2 = static_cast<double>(mix64(key ^ 2ULL) >> 11) * 0x1.0p-53;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * pi * u2);
}

double gust_ou_step (double z, double dt, double T, double xi)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt >= 0.0 && std::isfinite(dt), "gust_ou_step: dt must be finite and >= 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(T > 0.0, "gust_ou_step: the time scale must be positive");
    const double a = std::exp(-dt / T);
    return a * z + std::sqrt(std::max(1.0 - a * a, 0.0)) * xi;
}

} // namespace erf_conductors
