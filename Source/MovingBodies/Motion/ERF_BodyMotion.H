#ifndef ERF_BODY_MOTION_H
#define ERF_BODY_MOTION_H

#include <array>

#include <AMReX_REAL.H>

namespace erf_actuator {

/**
 * Where a body's reference frame is at a given time: the position of its base point and the
 * velocity of that point. A turbine stands still; bodies that translate implement this with
 * their own trajectory, and everything downstream (OpenFAST base position, sampling, spreading)
 * asks the motion rather than assuming a fixed place.
 */
class BodyMotion
{
public:
    virtual ~BodyMotion () = default;

    /** Position of the base point at time t (m). */
    virtual std::array<amrex::Real,3> position (amrex::Real t) const = 0;

    /** Velocity of the base point at time t (m/s). */
    virtual std::array<amrex::Real,3> velocity (amrex::Real t) const = 0;

    /** Whether the base point ever moves; a fixed body's derived data can be cached. */
    virtual bool is_fixed () const = 0;
};

/** A body that does not move: the position is its base position for all time. */
class FixedMotion : public BodyMotion
{
public:
    explicit FixedMotion (const std::array<amrex::Real,3>& base_pos) : m_base(base_pos) {}

    std::array<amrex::Real,3> position (amrex::Real /*t*/) const override { return m_base; }
    std::array<amrex::Real,3> velocity (amrex::Real /*t*/) const override { return {{0.0, 0.0, 0.0}}; }
    bool is_fixed () const override { return true; }

private:
    std::array<amrex::Real,3> m_base;
};

} // namespace erf_actuator

#endif
