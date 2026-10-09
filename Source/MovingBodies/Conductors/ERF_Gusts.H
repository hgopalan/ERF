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
 * Two simple time-dependent gusts add a fluctuation u' to the RANS wind the lines and towers take, along
 * the mean horizontal wind of the span, insulator string or tower it acts on:
 * - an event, one travelling 1 - cos gust: a plane front moving at speed c along the horizontal unit
 *   vector e crosses the point x0 at t0, so it reaches x at t_a = t0 + e.(x - x0) / c, and lasts D there:
 *     u' = g sigma_u (1 - cos(2 pi s)) / 2,  s = (t - t_a) / D,  for 0 <= s <= 1, else 0,
 *   a point peak on the whole front at once, not reduced for a span's length (conservative on long spans);
 * - a random gust, an Ornstein-Uhlenbeck process: u' = sigma_u sqrt(B) z, z of unit variance and integral
 *   time scale T = L_u / U, stepped exactly over any dt,
 *     z' = z exp(-dt / T) + sqrt(1 - exp(-2 dt / T)) xi,  xi standard normal,
 *   and L_u the integral length of IEC 61400-1's Kaimal spectrum, 8.1 Lambda_1 with Lambda_1 = 0.7 z below
 *   60 m and 42 m above. B takes the part of the point fluctuation's variance a whole span or tower feels:
 *   the span's B above, a tower's ASCE 74's 1 / (1 + 0.375 h / L_s). xi comes from (seed, stream, step)
 *   alone, so every rank draws the same numbers, and a restart, which carries z, continues the same series.
 *
 * SI units (m, s, kg) throughout.
 */

#include <cstdint>
#include <string>

namespace erf_conductors {

/**
 * How a run treats gusts on the conductors: not at all, as a static gust factor per span, as one travelling
 * 1 - cos gust, or as a random gust per span and tower.
 */
enum class GustType { None, Factor, Event, Random };

/** sigma_v / sigma_u of the lateral fluctuation (the IEC 61400-1 Kaimal ratio; about 1.9 / 2.5 in a surface layer). */
constexpr double gust_sigma_v_ratio = 0.8;

/** Parse "none", "factor", "event" or "random" (case-insensitive) into g; false for anything else. */
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

/**
 * ASCE 74's background factor B_t = 1 / (1 + 0.375 h / L_s) of a structure of height h (m, > 0) for the
 * length scale L_s (m, > 0).
 */
double tower_background_factor (double height, double length_scale);

/**
 * The integral length L_u = 8.1 Lambda_1 (m) of the streamwise fluctuation at height z (m, > 0) above the
 * ground: IEC 61400-1's turbulence scale parameter Lambda_1 = 0.7 z below 60 m and 42 m above.
 */
double gust_integral_length (double height);

/** The event gust's shape (1 - cos(2 pi s)) / 2 for 0 <= s <= 1, 0 outside: 1 at s = 1/2. */
double gust_event_shape (double s);

/** A travelling 1 - cos gust: a plane front crossing (x0, y0) at time, moving at speed along (ex, ey). */
struct GustEvent
{
    double time{0.0};       //!< t0, when the front crosses (x0, y0) (s)
    double x0{0.0};         //!< (m)
    double y0{0.0};         //!< (m)
    double ex{1.0};         //!< the horizontal unit vector the gust moves along
    double ey{0.0};
    double speed{1.0};      //!< c (m/s, > 0)
    double duration{1.0};   //!< D, how long the gust lasts at a point (s, > 0)
};

/** The event's phase s = (t - t_a) / D at (x, y) (m) and time t (s), t_a = t0 + ((x - x0) ex + (y - y0) ey) / c. */
double gust_event_phase (const GustEvent& e, double x, double y, double t);

/**
 * A standard normal number set by (seed, stream, step) alone: Box-Muller from two uniforms of a splitmix64
 * hash of the three. Different streams or steps give independent numbers.
 */
double gust_normal (std::uint64_t seed, std::uint64_t stream, std::uint64_t step);

/**
 * One exact step of the unit-variance Ornstein-Uhlenbeck process of integral time scale T (s, > 0; infinite
 * holds z): z exp(-dt / T) + sqrt(1 - exp(-2 dt / T)) xi, dt (s) >= 0 and xi standard normal.
 */
double gust_ou_step (double z, double dt, double T, double xi);

} // namespace erf_conductors

#endif
