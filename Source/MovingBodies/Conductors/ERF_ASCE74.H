#ifndef ERF_ASCE74_H
#define ERF_ASCE74_H

/**
 * The quasi-static wind load on a conductor span by ASCE Manual of Practice 74 (Guidelines for
 * Electrical Transmission Line Structural Loading), and the span's response to it: the design
 * values a run's conductor lines are compared with.
 *
 * The wire load per unit length is F = Q k_z V^2 G_w C_f d, with
 *   Q = rho / 2 (the air density's half; ASCE 74's 0.613 for 1.226 kg/m^3), V the 3-second gust at
 *     10 m (33 ft) over open country (exposure C), C_f the drag coefficient, d the diameter;
 *   k_z = 2.01 (z / z_g)^(2/alpha), the exposure factor at the wire's height z;
 *   G_w = (1 + 2.7 E sqrt(B_w)) / k_v^2, the wire's gust response factor, with
 *     E = 4.9 sqrt(kappa) (33 ft / z)^(1/alpha), B_w = 1 / (1 + 0.8 L / L_s) and k_v = 1.43, the ratio
 *     of the 3-second gust to the 10-minute mean at 33 ft over open country; L the span.
 * The constants of each exposure (ASCE 74's third edition; with them this reproduces the ASCE 2008 wire load
 * ratios of Peyrot, "Wind Loading: Uncertainties and Honesty Suggest Simplification", ASCE Electrical
 * Transmission and Substation Structures Conference 2009, Tables 1 and 2):
 *   B (suburban, wooded): alpha = 7.0, z_g = 1200 ft, kappa = 0.010, L_s = 170 ft;
 *   C (open country):     alpha = 9.5, z_g =  900 ft, kappa = 0.005, L_s = 220 ft.
 * The span then swings about its chord by atan(F / (W cos beta)), W its net weight per length and
 * beta the chord's angle to the horizontal (the weight's part along an inclined chord is carried by
 * the tension, not by the swing), and hangs as an elastic catenary under the resultant
 * sqrt(F^2 + (W cos beta)^2) across the chord, as a level span of the chord's length: its sag in
 * the swung plane, its mid-span's sideways blowout, the sag times sin(swing), and its upper end's
 * tension, the level span's plus W chord sin(beta) / 2 (the two ends' tensions differ by the weight
 * times the rise).
 *
 * SI units (m, N, Pa) throughout; the feet of ASCE 74's constants are converted.
 */

#include <string>

namespace erf_conductors {

/** ASCE 74's terrain exposure categories: B suburban or wooded, C open country. */
enum class Exposure { B, C };

/**
 * The constants of an exposure: power-law exponent alpha (-), gradient height z_g (m), surface drag coefficient
 * kappa (-), integral scale L_s (m).
 */
struct ExposureConstants
{
    double alpha{0.0};
    double zg{0.0};
    double kappa{0.0};
    double Ls{0.0};
};

/** The constants of exposure e. */
ExposureConstants exposure_constants (Exposure e);

/** Parse "B" or "C" (case-insensitive) into e; false for anything else. */
bool parse_exposure (const std::string& name, Exposure& e);

/** The exposure factor k_z = 2.01 (z / z_g)^(2/alpha) at height z (m, > 0) above the ground. */
double exposure_factor (Exposure e, double z);

/** The wire gust response factor G_w of a span of length span (m, > 0) at height z (m, > 0). */
double wire_gust_response_factor (Exposure e, double z, double span);

/** A span's ASCE 74 wind load and its quasi-static response. */
struct WireWindLoad
{
    double kz{0.0};           //!< exposure factor at the span's height (-)
    double gust_response{0.0};//!< G_w (-)
    double pressure{0.0};     //!< Q k_z V^2 (Pa)
    double load{0.0};         //!< the wind load per length, Q k_z V^2 G_w C_f d (N/m)
    double weight{0.0};       //!< the net weight per length (N/m)
    double swing{0.0};        //!< the span's swing about its chord, atan(load / (weight cos(incline))) (rad)
    double sag{0.0};          //!< mid-span sag below the chord in the swung plane (m)
    double tension{0.0};      //!< the upper end's tension under the resultant load, on the chord's incline given (N)
    double blowout{0.0};      //!< the mid-span's sideways displacement, sag sin(swing) (m)
    bool solved{false};       //!< the elastic catenary under the resultant load has a solution (Catenary::solved)
};

/**
 * The ASCE 74 wind load on a span and its quasi-static response.
 *
 * @param e            exposure
 * @param gust         V, the 3-second gust at 10 m over open country (m/s, >= 0)
 * @param z            the wire's height above the ground (m, > 0)
 * @param chord        the span's chord (m, > 0), also the span length L of B_w
 * @param diameter     d (m, > 0)
 * @param cf           the drag coefficient C_f (-, >= 0)
 * @param weight       the net weight per length W (N/m, > 0)
 * @param length       the span's unstretched length (m, > 0)
 * @param EA           axial stiffness (N, > 0)
 * @param air_density  rho (kg/m^3, > 0)
 * @param incline      beta, the chord's angle to the horizontal (rad, in (-pi/2, pi/2); 0: a level span)
 */
WireWindLoad wire_wind_load (Exposure e, double gust, double z, double chord, double diameter, double cf, double weight,
                             double length, double EA, double air_density, double incline = 0.0);

} // namespace erf_conductors

#endif
