// The quasi-static wind load on a conductor span by ASCE Manual of Practice 74, and the span's response.

#include "ERF_ASCE74.H"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <AMReX.H>
#include <AMReX_BLassert.H>

#include "ERF_ConductorInputs.H"

namespace erf_conductors {

namespace {
constexpr double foot = 0.3048;      // m
constexpr double kv = 1.43;          // the 3-second gust over the 10-minute mean at 33 ft, open country
}

ExposureConstants exposure_constants (Exposure e)
{
    ExposureConstants c;
    if (e == Exposure::B) { c.alpha = 7.0; c.zg = 1200.0 * foot; c.kappa = 0.010; c.Ls = 170.0 * foot; }
    else                  { c.alpha = 9.5; c.zg =  900.0 * foot; c.kappa = 0.005; c.Ls = 220.0 * foot; }
    return c;
}

bool parse_exposure (const std::string& name, Exposure& e)
{
    std::string u = name;
    for (auto& ch : u) { ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch))); }
    if (u == "B") { e = Exposure::B; return true; }
    if (u == "C") { e = Exposure::C; return true; }
    return false;
}

double exposure_factor (Exposure e, double z)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(z > 0.0, "exposure_factor: the height must be positive");
    const ExposureConstants c = exposure_constants(e);
    return 2.01 * std::pow(z / c.zg, 2.0 / c.alpha);
}

double wire_gust_response_factor (Exposure e, double z, double span)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(z > 0.0 && span > 0.0, "wire_gust_response_factor: the height and the span must be positive");
    const ExposureConstants c = exposure_constants(e);
    const double E = 4.9 * std::sqrt(c.kappa) * std::pow(33.0 * foot / z, 1.0 / c.alpha);
    const double Bw = 1.0 / (1.0 + 0.8 * span / c.Ls);
    return (1.0 + 2.7 * E * std::sqrt(Bw)) / (kv * kv);
}

WireWindLoad wire_wind_load (Exposure e, double gust, double z, double chord, double diameter, double cf, double weight,
                             double length, double EA, double air_density)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(gust >= 0.0 && z > 0.0 && chord > 0.0 && diameter > 0.0 && cf >= 0.0 && weight > 0.0 &&
                                     length > 0.0 && EA > 0.0 && air_density > 0.0,
                                     "wire_wind_load: an argument is out of range");
    WireWindLoad w;
    w.kz = exposure_factor(e, z);
    w.gust_response = wire_gust_response_factor(e, z, chord);
    w.pressure = 0.5 * air_density * w.kz * gust * gust;
    w.load = w.pressure * w.gust_response * cf * diameter;
    w.weight = weight;
    w.swing = std::atan2(w.load, weight);
    // the span hangs in the plane swung out by the wind, under the resultant of its weight and the wind
    const Catenary cat = elastic_catenary(static_cast<amrex::Real>(chord), static_cast<amrex::Real>(length),
                                          static_cast<amrex::Real>(std::hypot(w.load, weight)), static_cast<amrex::Real>(EA));
    w.sag = static_cast<double>(cat.sag);
    w.tension = static_cast<double>(cat.end_tension);
    w.blowout = w.sag * std::sin(w.swing);
    return w;
}

} // namespace erf_conductors
