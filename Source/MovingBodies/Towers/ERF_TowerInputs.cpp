#include "ERF_TowerInputs.H"

#include "ERF_MemberDrag.H"

using amrex::Real;

namespace erf_towers {

Real TowerType::force_coefficient () const
{
    return drag_coefficient > 0.0 ? drag_coefficient : lattice_force_coefficient(solidity);
}

std::string TowerType::validate () const
{
    const std::string key = "erf.conductors." + name + ".";
    if (!(base_width > 0.0)) { return key + "base_width must be positive (m)"; }
    if (!(top_width > 0.0 && top_width <= base_width)) { return key + "top_width must be positive and at most base_width (m)"; }
    if (!(solidity > 0.0 && solidity < 1.0)) { return key + "solidity must be in (0, 1): the members' area over a face's outline"; }
    if (!(arm_length > 0.0)) { return key + "arm_length must be positive (m)"; }
    if (arm_depth < 0.0) { return key + "arm_depth must be >= 0 (m; 0: top_width)"; }
    if (peak < 0.0) { return key + "peak must be >= 0 (m)"; }
    if (drag_coefficient < 0.0) { return key + "drag_coefficient must be >= 0 (0: from the solidity)"; }
    if (segments < 1) { return key + "segments must be >= 1"; }
    if (weight < 0.0) { return key + "weight must be >= 0 (N)"; }
    if (leg_spacing < 0.0) { return key + "leg_spacing must be >= 0 (m; 0: base_width)"; }
    if (allowable_uplift < 0.0) { return key + "allowable_uplift must be >= 0 (N; 0: not checked)"; }
    if (allowable_compression < 0.0) { return key + "allowable_compression must be >= 0 (N; 0: not checked)"; }
    return std::string();
}

} // namespace erf_towers
