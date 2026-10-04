// TowerType::force_coefficient() and the range checks of a tower type's inputs.

#include "ERF_TowerInputs.H"

#include <cmath>
#include <utility>

#include "ERF_MemberDrag.H"

using amrex::Real;

namespace erf_towers {

namespace {
// finite and > 0; finite and >= 0 (NaN and infinity fail both)
bool positive (Real x) { return std::isfinite(x) && x > 0.0; }
bool non_negative (Real x) { return std::isfinite(x) && x >= 0.0; }
}

Real TowerType::force_coefficient () const
{
    return drag_coefficient > 0.0 ? drag_coefficient : lattice_force_coefficient(solidity);
}

std::string TowerType::validate () const
{
    const std::string key = "erf.conductors." + name + ".";
    if (!positive(base_width)) { return key + "base_width must be finite and positive (m)"; }
    if (!(positive(top_width) && top_width <= base_width)) { return key + "top_width must be finite, positive and at most base_width (m)"; }
    if (!(solidity > 0.0 && solidity < 1.0)) { return key + "solidity must be in (0, 1): the members' area over a face's outline"; }
    if (!positive(arm_length)) { return key + "arm_length must be finite and positive (m)"; }
    if (!non_negative(arm_depth)) { return key + "arm_depth must be finite and >= 0 (m; 0: top_width)"; }
    if (!non_negative(peak)) { return key + "peak must be finite and >= 0 (m)"; }
    if (!non_negative(drag_coefficient)) { return key + "drag_coefficient must be finite and >= 0 (0: from the solidity)"; }
    if (segments < 1) { return key + "segments must be >= 1"; }
    if (!non_negative(weight)) { return key + "weight must be finite and >= 0 (N)"; }
    if (!non_negative(leg_spacing)) { return key + "leg_spacing must be finite and >= 0 (m; 0: base_width)"; }
    if (!non_negative(allowable_uplift)) { return key + "allowable_uplift must be finite and >= 0 (N; 0: not checked)"; }
    if (!non_negative(allowable_compression)) { return key + "allowable_compression must be finite and >= 0 (N; 0: not checked)"; }
    if (!non_negative(frequency)) { return key + "frequency must be finite and >= 0 (Hz; 0: the tower stands still)"; }
    if (!frame_file.empty()) {
        // the frame model gives the stiffness, the mass and the footings
        const std::pair<const char*, Real> from_frame[] = {
            {"frequency", frequency}, {"weight", weight}, {"foundation_rotational_stiffness", foundation_rotational_stiffness},
            {"foundation_lateral_stiffness", foundation_lateral_stiffness}};
        for (const auto& kv : from_frame) {
            if (kv.second != 0.0) {
                return key + kv.first + " is not given with " + key + "frame_file: the frame model sets the tower's stiffness, "
                       "mass and footings";
            }
        }
    }
    if (frequency > 0.0 && !(weight > 0.0)) { return key + "frequency needs the tower's weight (N), which sets its mass"; }
    if (!(damping_ratio >= 0.0 && damping_ratio < 1.0)) { return key + "damping_ratio must be in [0, 1) (fraction of critical)"; }
    if (!non_negative(foundation_rotational_stiffness)) {
        return key + "foundation_rotational_stiffness must be finite and >= 0 (N m/rad; 0: rigid)";
    }
    if (!non_negative(foundation_lateral_stiffness)) { return key + "foundation_lateral_stiffness must be finite and >= 0 (N/m; 0: rigid)"; }
    return std::string();
}

} // namespace erf_towers
