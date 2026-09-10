#ifndef ERF_PRESCRIBED_FIRE_H
#define ERF_PRESCRIBED_FIRE_H

/**
 * @file ERF_PrescribedFire.H
 * @brief Prescribed rate of spread and prescribed surface heat flux, for
 *        verification cases whose answer is known in closed form.
 *
 * erf.fire.ros_model = "prescribed" replaces the fire behaviour model with
 *
 *   R(x, y) = max(min_ros, R_base + g_x (x - x_0) + g_y (y - y_0)),
 *
 * where R_base is erf.fire.prescribed.ros, or the rate listed for the cell's
 * fuel code in erf.fire.prescribed.by_fuel. The rate ignores wind, slope and
 * moisture and has no direction, so the level set solves the eikonal equation
 * |grad T| = 1 / R as posed: the arrival time is the shortest travel time from
 * the ignition, which geometry gives exactly for a constant rate (discs,
 * polygon offsets, paths around obstacles, refraction across a fuel boundary)
 * and in closed form for a linear gradient.
 *
 * erf.fire.prescribed_heat.flux adds a constant sensible heat flux over a disc
 * to fire_heat_flux between start_time and end_time, whatever the fuel and the
 * front are doing. The coupling then carries it into the atmosphere like any
 * fire heat, so a plume from a steady, known heat release can be compared with
 * plume theory.
 *
 * Both are off by default and change nothing unless selected.
 */

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>

/**
 * @brief Prescribed rate of spread at a point
 *
 * @param[in] R_base   Rate at (x0, y0) [m/s]
 * @param[in] gx, gy   Gradient of the rate [1/s]
 * @param[in] x0, y0   Point where the rate is R_base [m]
 * @param[in] min_ros  Floor on the rate [m/s]; the rate is never negative
 * @param[in] x, y     Evaluation point [m]
 * @return Rate of spread [m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real prescribed_ros_at(amrex::Real R_base, amrex::Real gx, amrex::Real gy,
                              amrex::Real x0, amrex::Real y0, amrex::Real min_ros,
                              amrex::Real x, amrex::Real y) noexcept
{
    const amrex::Real R     = R_base + gx * (x - x0) + gy * (y - y0);
    const amrex::Real floor = amrex::max(min_ros, amrex::Real(0.0));
    return amrex::max(R, floor);
}

/**
 * @brief Prescribed surface heat flux at a point and time
 *
 * @param[in] flux      Flux inside the disc [W/m2]; <= 0 is off
 * @param[in] cx, cy    Disc centre [m]
 * @param[in] radius    Disc radius [m]
 * @param[in] t_start   Time the flux switches on [s]
 * @param[in] t_end     Time it switches off [s]; negative = never
 * @param[in] x, y      Evaluation point [m]
 * @param[in] t         Time [s]
 * @return Heat flux [W/m2]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real prescribed_heat_flux_at(amrex::Real flux, amrex::Real cx, amrex::Real cy,
                                    amrex::Real radius, amrex::Real t_start, amrex::Real t_end,
                                    amrex::Real x, amrex::Real y, amrex::Real t) noexcept
{
    if (flux <= amrex::Real(0.0) || t < t_start || (t_end >= amrex::Real(0.0) && t >= t_end)) {
        return amrex::Real(0.0);
    }
    const amrex::Real dx = x - cx;
    const amrex::Real dy = y - cy;
    return (dx * dx + dy * dy <= radius * radius) ? flux : amrex::Real(0.0);
}

#endif
