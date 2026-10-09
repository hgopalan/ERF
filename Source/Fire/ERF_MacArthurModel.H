#ifndef ERF_MACARTHUR_MODEL_H
#define ERF_MACARTHUR_MODEL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>

/**
 * @file ERF_MacArthurModel.H
 * @brief McArthur (1966) Australian grassland fire spread formula
 *
 * Implements the grassland rate-of-spread fit of McArthur (1966), "Weather
 * and Grassland Fire Behaviour", Leaflet 100 (not the 1973 Mark 5 forest
 * meter), in the form WRF-SFIRE uses as ibeh=0 (Clark et al. 2004).
 *
 * Formula:
 *   R [m/s] = 0.18 * (exp(0.8424 * max(U, 0)) - 1) + 0.18
 *           = 0.18 * exp(0.8424 * max(U, 0))    [with wind]
 *   R_backing = 0.18 m/s                         [no-wind backing rate]
 *
 * This is a simple empirical wind-only formula suitable for Australian
 * grassland and open forest fuels. Not appropriate for complex terrain or
 * high-fuel-complexity environments. The rate is capped at
 * erf.fire.macarthur.ros_max (6 m/s by default, as WRF-Fire caps it), since
 * the exponential has no limit of its own.
 *
 * References:
 *   Clark, T.L., Coen, J.L. & Latham, D. (2004). Description of a coupled
 *     atmosphere-fire model. Int. J. Wildland Fire, 13, 49–63.
 *   McArthur, A.G. (1966). Weather and Grassland Fire Behaviour. Leaflet 100,
 *     Forestry and Timber Bureau, Canberra.
 */

/**
 * @brief MacArthur (1966) ROS formula for given wind speed
 *
 * Computes rate of spread using the simple exponential wind formula.
 *
 * @param[in] wind_speed_ms  Wind speed in the fire spread direction [m/s]
 * @param[in] ros_max        Cap on the rate [m/s] (erf.fire.macarthur.ros_max,
 *                           6 m/s by default, the cap WRF-Fire applies; <= 0 none).
 *                           Uncapped the exponential gives 12 m/s at a 5 m/s
 *                           midflame wind.
 * @return ROS [m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real macarthur_ros(amrex::Real wind_speed_ms, amrex::Real ros_max = amrex::Real(6.0)) noexcept
{
    constexpr amrex::Real backing = 0.18;
    const amrex::Real wind_contribution = backing * (std::exp(0.8424 * amrex::max(static_cast<amrex::Real>(wind_speed_ms), static_cast<amrex::Real>(0.0))) - 1.0);
    const amrex::Real ros = backing + wind_contribution;
    return (ros_max > amrex::Real(0.0)) ? amrex::min(ros, ros_max) : ros;
}

/**
 * @brief Fill fire_ros [m/s] using MacArthur (1966) formula
 *
 * Simple wind-driven ROS model for grassland and open forest.
 *
 * @param[out] fire_ros        Rate of spread [m/s]
 * @param[in]  fire_wind_eff   Effective wind after WAF + terrain [m/s]
 * @param[in]  ros_max         Cap on the rate [m/s]; <= 0 none
 */
inline void fill_macarthur_ros(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind_eff,
    amrex::Real ros_max = amrex::Real(6.0))
{
    for (amrex::MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto ros_arr = fire_ros.array(mfi);
        auto wind_arr = fire_wind_eff.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            // Get effective wind magnitude
            amrex::Real u = wind_arr(i, j, k, 0);
            amrex::Real v = wind_arr(i, j, k, 1);
            amrex::Real wind_mag = std::sqrt(u*u + v*v);

            // Compute MacArthur ROS
            amrex::Real ros = macarthur_ros(wind_mag, ros_max);

            ros_arr(i, j, k, 0) = ros;
        });
    }
}

#endif
