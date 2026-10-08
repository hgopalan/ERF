#ifndef ERF_GUSTS_H
#define ERF_GUSTS_H

/**
 * Gusts on a conductor span from a RANS wind: the span's peak wind load from the mean wind and the
 * turbulent kinetic energy k of the k-equation RANS where the span hangs.
 *
 * The streamwise velocity fluctuation is sigma_u = c sqrt(k). In a neutral surface layer sigma_u is
 * about 2.5 u*, and the RANS model's equilibrium there is k = u*^2 / Cmu0^2, so the default is
 * c = 2.5 Cmu0; the RANS k is well below a measured one (about 3.2 u*^2 against 5.5 u*^2), so c must
 * not be taken from a measured sigma_u / sqrt(k). The lateral fluctuation is sigma_v = 0.8 sigma_u
 * (gust_sigma_v_ratio). A span sees the wind's component normal to it: with the mean horizontal wind U
 * at an angle phi to the span, U_n = U sin(phi), and the fluctuation normal to the span is
 *   sigma_n = sqrt(sigma_u^2 sin^2(phi) + sigma_v^2 cos^2(phi)),
 * so its turbulence intensity is I_n = sigma_n / U_n. The span's gust response factor takes ASCE
 * Manual of Practice 74's form with that intensity in place of the exposure's,
 *   G = 1 + 2 g I_n sqrt(B),  B = 1 / (1 + 0.8 L / L_s),
 * L the span (its chord), g a peak factor and L_s a length scale; the peak wind load per length is G
 * times the mean one, 0.5 rho C_d d U_n^2. ASCE 74 writes G_w kv^2 = 1 + 2.7 E sqrt(B_w) for wind normal
 * to the span, with E = 4.9 sqrt(kappa) (33 ft / z)^(1/alpha) twice its exposure's turbulence
 * intensity, so for wind normal to the span g = 2.7 and the exposure's L_s give ASCE 74's factor
 * exactly when I_n is that intensity; the default L_s, 220 ft, is exposure C's (170 ft for B). The form
 * is linear in the fluctuation: on a span the wind runs nearly along (U_n much below sigma_n) it misses
 * the load of the fluctuation itself, and the factor grows without bound as U_n falls; with U_n = 0 the
 * intensity, factor and loads are 0. SpanGust::linear_valid marks the spans where U_n >= sigma_n, those
 * on which the linear form holds. The peak wind at a point is U + g sigma_u.
 *
 * SI units (m, s, kg) throughout.
 */

#include <string>

namespace erf_conductors {

/** How a run treats gusts on the conductors: not at all, or as a static gust factor per span. */
enum class GustType { None, Factor };

/** sigma_v / sigma_u of the lateral fluctuation (the IEC 61400-1 Kaimal ratio; about 1.9 / 2.5 in a surface layer). */
constexpr double gust_sigma_v_ratio = 0.8;

/** Parse "none" or "factor" (case-insensitive) into g; false for anything else. */
bool parse_gust_type (const std::string& name, GustType& g);

/** The default sigma_u / sqrt(k): 2.5 Cmu0, Cmu0 the k-equation RANS constant (> 0). */
double default_gust_sigma_factor (double Cmu0);

/**
 * The background factor B = 1 / (1 + 0.8 L / L_s) of a span of length L (m, > 0) for a length
 * scale L_s (m, > 0): the part of a point gust's load the whole span takes.
 */
double gust_background_factor (double span, double length_scale);

/** A span's gust factor and peak wind load from the mean RANS wind and k. */
struct SpanGust
{
    double sigma{0.0};          //!< sigma_u = c sqrt(k) (m/s)
    double normal_sigma{0.0};   //!< sigma_n, the fluctuation normal to the span (m/s)
    double intensity{0.0};      //!< I_n = sigma_n / U_n (-); 0 when U_n = 0
    double gust_response{0.0};  //!< G = 1 + 2 g I_n sqrt(B) (-); 0 when U_n = 0
    double gust_wind{0.0};      //!< U + g sigma_u, the peak wind at a point (m/s)
    double mean_load{0.0};      //!< 0.5 rho C_d d U_n^2 (N/m)
    double peak_load{0.0};      //!< G times the mean load (N/m)
    bool linear_valid{false};   //!< U_n >= sigma_n: the linear form holds (it underestimates the peak load below)
};

/**
 * A span's gust factor and peak wind load.
 *
 * @param wind          U, the horizontal wind speed over the span's nodes (m/s, >= 0)
 * @param normal_wind   U_n, its component normal to the span's chord (m/s, 0 <= U_n <= U)
 * @param k             the turbulent kinetic energy over the span's nodes (m^2/s^2, >= 0)
 * @param span          the span's chord L (m, > 0)
 * @param diameter      d (m, > 0)
 * @param cd            the drag coefficient C_d (-, >= 0)
 * @param air_density   rho (kg/m^3, > 0)
 * @param sigma_factor  c in sigma_u = c sqrt(k) (-, > 0)
 * @param peak_factor   g (-, > 0)
 * @param length_scale  L_s (m, > 0)
 */
SpanGust span_gust (double wind, double normal_wind, double k, double span, double diameter, double cd, double air_density,
                    double sigma_factor, double peak_factor, double length_scale);

} // namespace erf_conductors

#endif
