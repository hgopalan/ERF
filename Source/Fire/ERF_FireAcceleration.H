// ============================================================================
// Fire Acceleration Models (Phase 12)
// ============================================================================
// Small fires do not immediately achieve their quasi-steady-state (QSS)
// rate of spread. This module implements two optional acceleration models:
//
// 1. SIZE-BASED MODEL (Rothermel 1983 / Catchpole et al. 1992)
//    - Simple exponential approach based on current fire size
//    - Applies global scaling factor to ROS field
//    - α_acc = 1 − exp(−r_fire / L_acc)
//
// 2. FARSITE TEMPORAL MODEL (McAlpine & Wakimoto 1991 / VanWagner)
//    - R(t) = R_E × (1 − exp(−A × t))
//    - erf.fire.accel.clock = "legacy" (default): a clock per burned cell,
//      started when the cell burns and restarted on any change of its
//      equilibrium rate, written to burned cells only; optional wind-onset lag
//    - erf.fire.accel.clock = "front": the clock starts at ignition and is
//      carried with the front, and the accelerated rate is written wherever
//      the front can move next, so it slows the spread on every path
//
// Both models are disabled by default (enable = false) with zero runtime cost
// when disabled.
//
// References:
//   Rothermel, R.C. (1983). "How to predict the spread and intensity of forest
//       and range fires." USDA Forest Service GTR INT-143.
//   Catchpole, E.A., de Mestre, N.J. & Gill, A.M. (1992). "Intensity of fire
//       at its perimeter." Australian Journal of Ecology, 17(1), 1–4.
//   McAlpine, R.S. & Wakimoto, R.H. (1991). "The acceleration of fire from
//       point source to equilibrium spread." Forest Science, 37(5), 1314–1337.
//   Alexander, M.E., Stocks, B.J. & Lawson, B.D. (1992). "Fire behavior in
//       Black Spruce-lichen woodland." Info. Rep. NOR-X-310. USDA/CFS.
//   Finney, M.A. (1998/2004). "FARSITE: Fire Area Simulator." USDA Forest
//       Service RMRS-RP-4.
// ============================================================================

#ifndef ERF_FIRE_ACCELERATION_H
#define ERF_FIRE_ACCELERATION_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <ERF_FireGrid.H>
#include <cmath>
#include <utility>

/// Clock of the temporal model, erf.fire.accel.clock
namespace accel_clock {
    constexpr int legacy = 0;  ///< per burned cell, restarted when it burns (before 2026-09)
    constexpr int front  = 1;  ///< started at ignition and carried with the front
}

// ============================================================================
// AccelerationParams - Fire acceleration model parameters
// ============================================================================
struct AccelerationParams {
    bool          enable         = false;   ///< Master on/off switch (default: disabled)
    bool          use_temporal   = false;   ///< false = size-based, true = temporal per-cell
    amrex::Real   L_acc          = 50.0;    ///< Size-based: acceleration length scale [m]
    amrex::Real   A_point        = 0.115;   ///< Temporal: acceleration constant, point fire [1/min]
    amrex::Real   A_line         = 0.886;   ///< Temporal: acceleration constant, line fire [1/min]
    amrex::Real   perim_limit    = 500.0;   ///< Temporal: perimeter length [m] switching point vs line
    bool          enable_wind_lag = false;  ///< Temporal: apply exponential lag on wind increase
    amrex::Real   tau_wind       = 60.0;    ///< Temporal: wind-lag time constant [s]
    int           clock          = accel_clock::legacy;  ///< Temporal: accel_clock::legacy or accel_clock::front
};

// ============================================================================
// apply_fire_acceleration_size_based
//
// SIZE-BASED MODEL: Computes fire area from phi and scales R_mf by
// α_acc = 1 − exp(−r_fire / L_acc).
//
// When the fire is large (r_fire >> L_acc), α_acc → 1 and R_mf is unchanged.
// When the fire is small (r_fire << L_acc), α_acc → 0 and R_mf → 0.
//
// Parameters:
//   R_mf      – rate-of-spread field [m/s] (modified in place)
//   phi       – level-set field (phi < 0 = burned)
//   geom      – AMReX geometry (provides cell sizes)
//   L_acc     – acceleration length scale [m]
//   fire_debug – whether to print debug diagnostics
// ============================================================================
inline void apply_fire_acceleration_size_based(amrex::MultiFab&       R_mf,
                                               const amrex::MultiFab& phi,
                                               const amrex::Geometry& geom,
                                               const amrex::Real      L_acc,
                                               bool                   fire_debug)
{
    const auto dx = geom.CellSize();

    // Count burned cells. Device reduction, not a host loop: under CUDA phi's
    // data is device-only (the_arena_is_managed defaults to false), so reading
    // it from the host faults.
    amrex::Long n_burned = 0;
    {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<unsigned long long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const p         = phi.const_array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                return { (p(i, j, k) < 0.0) ? 1ull : 0ull };
            });
        }
        n_burned = static_cast<amrex::Long>(amrex::get<0>(reduce_data.value(reduce_op)));
    }
    amrex::ParallelDescriptor::ReduceLongSum(n_burned);

    if (n_burned == 0) return;  // no fire yet — skip

    // Burned area and effective fire radius
    const amrex::Real A_fire     = static_cast<amrex::Real>(n_burned) * dx[0] * dx[1];
    const amrex::Real r_fire     = std::sqrt(A_fire / amrex::Math::pi<amrex::Real>());
    const amrex::Real L_acc_safe = amrex::max(static_cast<amrex::Real>(L_acc), static_cast<amrex::Real>(1.0));  // floor prevents divide-by-zero
    const amrex::Real alpha      = 1.0 - std::exp(-r_fire / L_acc_safe);

    // Nothing to do when essentially at full speed (saves a kernel launch)
    if (alpha >= 1.0 - 1.0e-4) return;

    if (fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fire acceleration (size-based): r_fire=" << r_fire
                       << " m  L_acc=" << L_acc << " m  alpha=" << alpha << "\n";
    }

    // Scale R_mf by alpha
    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R               = R_mf.array(mfi);
        const amrex::Real a  = alpha;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            R(i, j, k) *= a;
        });
    }
}

// ============================================================================
// temporal_acceleration_constant
//
// Acceleration constant of the temporal model [1/s], shared by both clocks:
// A_point while the burned perimeter is at most perim_limit, A_line beyond it.
// The perimeter is the number of burned cells with an unburned 4-neighbour in
// the same box, times the cell diagonal.
// ============================================================================
inline amrex::Real temporal_acceleration_constant(const amrex::MultiFab& phi,
                                                  const amrex::Geometry& geom,
                                                  const AccelerationParams& params,
                                                  bool fire_debug)
{
    const auto dx = geom.CellSize();

    // Compute fire perimeter to determine acceleration constant
    // (point ignition: A_point; line ignition: A_line)
    // Device reduction for the same reason as the burned count above. Cell
    // semantics are unchanged, including the in-box-only neighbour guards.
    amrex::Long n_perimeter = 0;
    {
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<unsigned long long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const p         = phi.const_array(mfi);
            const auto lo        = amrex::lbound(bx);
            const auto hi        = amrex::ubound(bx);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (p(i, j, k) >= 0.0) { return {0ull}; }

                // Check if this is a perimeter cell (has unburned neighbor)
                // Guard all stencil accesses with validbox bounds
                bool is_perimeter = false;
                if (i+1 <= hi.x && p(i+1, j, k) >= 0.0) is_perimeter = true;
                if (i-1 >= lo.x && p(i-1, j, k) >= 0.0) is_perimeter = true;
                if (j+1 <= hi.y && p(i, j+1, k) >= 0.0) is_perimeter = true;
                if (j-1 >= lo.y && p(i, j-1, k) >= 0.0) is_perimeter = true;
                return { is_perimeter ? 1ull : 0ull };
            });
        }
        n_perimeter = static_cast<amrex::Long>(amrex::get<0>(reduce_data.value(reduce_op)));
    }
    amrex::ParallelDescriptor::ReduceLongSum(n_perimeter);

    // Estimate perimeter length in meters
    const amrex::Real perim_length = static_cast<amrex::Real>(n_perimeter) *
                                     std::sqrt(dx[0] * dx[0] + dx[1] * dx[1]);

    // Select acceleration constant based on perimeter size
    const amrex::Real A = (perim_length > params.perim_limit) ? params.A_line : params.A_point;

    if (fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fire acceleration (temporal): perim=" << perim_length
                       << " m  A=" << A << " 1/min  (using "
                       << (perim_length > params.perim_limit ? "line" : "point")
                       << " constant)\n";
    }

    // Convert A from [1/min] to [1/s]
    return A / 60.0;
}

// ============================================================================
// apply_fire_acceleration_temporal
//
// FARSITE TEMPORAL MODEL, erf.fire.accel.clock = "legacy": per-cell
// acceleration using VanWagner's equation R(t) = R_E × (1 − exp(−A × t)).
//
// Each cell tracks:
//   - ROS_current: current instantaneous ROS [m/s]
//   - elapsed_time: time since entering current state [s]
//   - ROS_equilibrium: target equilibrium ROS [m/s]
//
// When R_mf (quasi-steady Rothermel ROS) changes (e.g., due to wind change),
// the acceleration resets and ramps up exponentially.
//
// Only burned cells are written; unburned cells keep R_E and their clock is
// zero, so a cell starts again from R = 0 when it burns. The rate that moves
// the front is therefore mostly not the accelerated one: the FARSITE
// front_cell update takes the larger of a front cell's own rate and its
// burned neighbours', the legacy FARSITE update also advances the first
// unburned row, the level set reads the rate on both sides of phi = 0, and
// the directional level-set paths do not read R_mf at all
// (Exec/RegTests/FireAccelerationClock). Kept to reproduce results from
// before 2026-09; see apply_fire_acceleration_front.
//
// Parameters:
//   R_mf            – equilibrium ROS from Rothermel model [m/s] (read; output written to R_mf)
//   phi             – level-set field (phi < 0 = burned)
//   geom            – AMReX geometry
//   params          – acceleration parameters
//   dt              – timestep [s]
//   accel_state_mf  – 3-component state (ROS_current, elapsed_time, ROS_equilibrium)
//   fire_debug      – whether to print debug diagnostics
// ============================================================================
inline void apply_fire_acceleration_temporal(amrex::MultiFab&       R_mf,
                                             const amrex::MultiFab& phi,
                                             const amrex::Geometry& geom,
                                             const AccelerationParams& params,
                                             const amrex::Real      dt,
                                             amrex::MultiFab*       accel_state_mf,
                                             bool                   fire_debug)
{
    if (accel_state_mf == nullptr) {
        amrex::Abort("apply_fire_acceleration_temporal: accel_state_mf is null");
    }

    const amrex::Real A_per_sec = temporal_acceleration_constant(phi, geom, params, fire_debug);

    // Update acceleration state for each cell
    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R_eq            = R_mf.const_array(mfi);          // equilibrium ROS from Rothermel
        auto state           = accel_state_mf->array(mfi);     // [ROS_current, elapsed_time, ROS_equilibrium]
        auto p               = phi.const_array(mfi);           // level set
        auto R_out           = R_mf.array(mfi);                // output (will be modified)

        const amrex::Real dt_loc        = dt;
        const amrex::Real A_loc         = A_per_sec;
        const int  enable_wind_lag_loc  = params.enable_wind_lag ? 1 : 0;
        const amrex::Real tau_wind_loc  = params.tau_wind;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Only accelerate burned cells
            if (p(i, j, k) >= 0.0) {
                state(i, j, k, 0) = 0.0;  // ROS_current
                state(i, j, k, 1) = 0.0;  // elapsed_time
                state(i, j, k, 2) = 0.0;  // ROS_equilibrium
                return;
            }

            const amrex::Real R_E = R_eq(i, j, k);  // equilibrium ROS (target)
            amrex::Real R_curr    = state(i, j, k, 0);
            amrex::Real t_elapsed = state(i, j, k, 1);
            amrex::Real R_prev_eq = state(i, j, k, 2);

            // Detect equilibrium ROS change (wind change or fuel change)
            const amrex::Real EQUILIBRIUM_CHANGE_THRESHOLD = 1.0e-6;
            const amrex::Real dR_eq = amrex::Math::abs(R_E - R_prev_eq);
            const bool eq_changed = (dR_eq > EQUILIBRIUM_CHANGE_THRESHOLD *
                                     amrex::max(static_cast<amrex::Real>(R_E), static_cast<amrex::Real>(R_prev_eq), static_cast<amrex::Real>(1.0)));

            if (eq_changed) {
                // Equilibrium changed — apply wind-lag or reset
                if (enable_wind_lag_loc == 1 && R_E > R_prev_eq) {
                    // Wind increase: apply exponential lag
                    // New target approaches R_E with time constant tau_wind
                    // R_target(t) = R_prev + (R_E - R_prev) × (1 - exp(-dt/tau_wind))
                    const amrex::Real lag_factor = 1.0 - std::exp(-dt_loc / tau_wind_loc);
                    const amrex::Real R_target = R_prev_eq + (R_E - R_prev_eq) * lag_factor;

                    // Start new acceleration towards lagged target
                    R_prev_eq = R_target;
                    t_elapsed = 0.0;
                } else {
                    // Wind decrease or lag disabled: reset immediately to new equilibrium
                    R_prev_eq = R_E;
                    t_elapsed = 0.0;
                }
            }

            // VanWagner acceleration equation: R(t) = R_E × (1 - exp(-A × t))
            t_elapsed += dt_loc;
            R_curr = R_prev_eq * (1.0 - std::exp(-A_loc * t_elapsed));

            // Clamp to equilibrium (avoid overshoot due to numerical error)
            if (R_curr > R_prev_eq) R_curr = R_prev_eq;

            // Update state
            state(i, j, k, 0) = R_curr;
            state(i, j, k, 1) = t_elapsed;
            state(i, j, k, 2) = R_prev_eq;

            // Write accelerated ROS to output
            R_out(i, j, k) = R_curr;
        });
    }
}

// ============================================================================
// accel_step_mean_factor
//
// Mean of 1 − exp(−s) over a step in which s grows linearly from s0 to s0 + a:
// 1 − exp(−s0) (1 − exp(−a)) / a. A cell at a constant R_E then covers exactly
// R_E times the integral of 1 − exp(−A t) over the step, whatever its length.
// ============================================================================
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real accel_step_mean_factor(amrex::Real s0, amrex::Real a) noexcept
{
    using namespace amrex::literals;
    const amrex::Real e0 = std::exp(-s0);
    if (a <= 0.0_rt) { return 1.0_rt - e0; }
    return 1.0_rt - e0 * (-std::expm1(-a)) / a;
}

// ============================================================================
// apply_fire_acceleration_front
//
// FARSITE TEMPORAL MODEL, erf.fire.accel.clock = "front". McAlpine & Wakimoto
// (1991) describe a fire accelerating from its ignition, R = R_E (1 − exp(−A t))
// with t the time since the fire started, so the clock belongs to the fire and
// travels with its front rather than restarting in every cell it reaches.
//
// Each cell holds the progress s = ∫ A dt of the fire that burned it (state
// component 1; A t while A is constant):
//   - a burned cell advances its own s by A dt;
//   - an unburned cell within n_ext cells of the burned region takes the
//     largest s of its burned (or already reached) 4-neighbours, advanced by
//     A dt, so a cell the front reaches continues the clock of the cells that
//     ignited it;
//   - a cell further out stores s = 0, so an ignition there (a spot landing, a
//     scheduled or threshold ignition) starts a clock of its own;
//   - a change of R_E, in time or from cell to cell, leaves s alone and the
//     rate follows R_E at once. Under two-way coupling R_E changes every step;
//     restarting or re-accelerating on each rise would hold the fire far below
//     R_E (see the temporal-model section of fire_acceleration.rst).
// n_ext = 2 + ceil(max R_E dt / dx): the cells the front can reach within the
// step plus the neighbours the propagation stencils read.
//
// R_mf becomes R_E times the mean of 1 − exp(−s) over the step
// (accel_step_mean_factor) in the burned cells and the band, so the unburned
// front cells of the FARSITE front_cell update, both rows of the legacy FARSITE
// update and both sides of the level set all move at the accelerated rate.
// Beyond the band each cell takes the largest s over burned cells, so the level
// set away from the front translates with it; with no burned cell anywhere the
// factor is 0. The factor is kept in state component 2 for the level-set paths
// that rebuild the rate inside each stage, and the rate in component 0.
//
// A switches from A_point to A_line with the perimeter as for the legacy clock;
// s, and so the rate, is continuous across the switch.
//
// Parameters:
//   R_mf            – equilibrium ROS [m/s] on input, accelerated ROS on output
//   phi             – level-set field (phi < 0 = burned)
//   geom            – fire-grid geometry
//   params          – acceleration parameters
//   dt              – timestep [s]
//   state           – 3-component state (ROS_current, progress A t, factor)
//   fire_debug      – whether to print debug diagnostics
// ============================================================================
inline void apply_fire_acceleration_front(amrex::MultiFab&       R_mf,
                                          const amrex::MultiFab& phi,
                                          const amrex::Geometry& geom,
                                          const AccelerationParams& params,
                                          const amrex::Real      dt,
                                          amrex::MultiFab&       state,
                                          bool                   fire_debug)
{
    using namespace amrex;
    using namespace amrex::literals;

    const Real a = temporal_acceleration_constant(phi, geom, params, fire_debug) * dt;

    const auto dx    = geom.CellSizeArray();
    const Real h     = amrex::min(dx[0], dx[1]);
    const Real R_max = amrex::max(R_mf.max(0), 0.0_rt);
    const int  n_ext = 2 + static_cast<int>(std::ceil(R_max * dt / h));

    // Progress of the burned cells, -1 in unburned ones; one ghost cell for the
    // extension passes, which read the four neighbours across box boundaries.
    const IntVect ng(1, 1, 0);
    MultiFab s_ext(state.boxArray(), state.DistributionMap(), 1, ng);
    MultiFab s_tmp(state.boxArray(), state.DistributionMap(), 1, ng);

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(s_ext); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto se = s_ext.array(mfi);
        auto st = state.const_array(mfi);
        auto p  = phi.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            se(i, j, k) = (p(i, j, k) < 0.0_rt) ? st(i, j, k, 1) : -1.0_rt;
        });
        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
            return { (p(i, j, k) < 0.0_rt) ? st(i, j, k, 1) : -1.0_rt };
        });
    }
    Real s_far = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(s_far);
    const bool burning = (s_far >= 0.0_rt);

    // Carry the burned cells' progress into the unburned band, one cell a pass
    if (burning) {
        for (int pass = 0; pass < n_ext; ++pass) {
            fire_fill_boundary(s_ext, geom);
            for (MFIter mfi(s_tmp); mfi.isValid(); ++mfi) {
                auto src = s_ext.const_array(mfi);
                auto dst = s_tmp.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    Real s = src(i, j, k);
                    if (s < 0.0_rt) {
                        s = amrex::max(amrex::max(src(i-1, j, k), src(i+1, j, k)),
                                       amrex::max(src(i, j-1, k), src(i, j+1, k)));
                    }
                    dst(i, j, k) = s;
                });
            }
            std::swap(s_ext, s_tmp);
        }
    }

    const Real f_far = burning ? accel_step_mean_factor(s_far, a) : 0.0_rt;
    for (MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        auto R  = R_mf.array(mfi);
        auto st = state.array(mfi);
        auto se = s_ext.const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real s0      = se(i, j, k);
            const bool carried = (s0 >= 0.0_rt);
            const Real f       = carried ? accel_step_mean_factor(s0, a) : f_far;
            R(i, j, k)    *= f;
            st(i, j, k, 0) = R(i, j, k);
            st(i, j, k, 1) = carried ? s0 + a : 0.0_rt;
            st(i, j, k, 2) = f;
        });
    }
    // The kernels above read s_ext, which is freed on return.
    Gpu::streamSynchronize();

    if (fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fire acceleration (front clock): band " << n_ext
                       << " cells, largest A t " << (burning ? s_far + a : 0.0_rt)
                       << ", factor there " << f_far << "\n";
    }
}

// ============================================================================
// apply_fire_acceleration
//
// Main dispatch function: selects size-based or temporal model based on params.
// Returns immediately when enable == false (zero cost when disabled).
//
// Parameters:
//   R_mf            – rate-of-spread field [m/s] (modified in place)
//   phi             – level-set field (phi < 0 = burned)
//   geom            – AMReX geometry
//   params          – AccelerationParams from FireParams
//   dt              – timestep [s] (only used for temporal model)
//   accel_state_mf  – per-cell state for temporal model (nullptr for size-based)
//   fire_debug      – whether to print debug diagnostics
// ============================================================================
inline void apply_fire_acceleration(amrex::MultiFab&       R_mf,
                                    const amrex::MultiFab& phi,
                                    const amrex::Geometry& geom,
                                    const AccelerationParams& params,
                                    const amrex::Real      dt,
                                    amrex::MultiFab*       accel_state_mf,
                                    bool                   fire_debug)
{
    if (!params.enable) return;

    if (params.use_temporal) {
        // FARSITE temporal model
        if (params.clock == accel_clock::front) {
            if (accel_state_mf == nullptr) {
                amrex::Abort("apply_fire_acceleration: accel_state_mf is null");
            }
            apply_fire_acceleration_front(R_mf, phi, geom, params, dt, *accel_state_mf, fire_debug);
        } else {
            apply_fire_acceleration_temporal(R_mf, phi, geom, params, dt, accel_state_mf, fire_debug);
        }
    } else {
        // Simple size-based model
        apply_fire_acceleration_size_based(R_mf, phi, geom, params.L_acc, fire_debug);
    }
}

#endif // ERF_FIRE_ACCELERATION_H
