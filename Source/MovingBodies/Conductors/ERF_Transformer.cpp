#include "ERF_Transformer.H"

#include <cmath>

using amrex::Real;

namespace erf_conductors {

std::array<Real,3> Transformer::box_lo () const
{
    return {{m_in.position[0] - Real(0.5) * m_in.size[0], m_in.position[1] - Real(0.5) * m_in.size[1], m_base_z}};
}

std::array<Real,3> Transformer::box_hi () const
{
    return {{m_in.position[0] + Real(0.5) * m_in.size[0], m_in.position[1] + Real(0.5) * m_in.size[1], m_base_z + m_in.size[2]}};
}

TransformerLoad Transformer::load (const std::vector<std::array<Real,3>>& at, const std::vector<std::array<Real,3>>& force) const
{
    TransformerLoad L;
    const auto b = base();
    for (std::size_t i = 0; i < at.size() && i < force.size(); ++i) {
        const auto& F = force[i];
        const std::array<Real,3> r{{at[i][0] - b[0], at[i][1] - b[1], at[i][2] - b[2]}};
        for (int d = 0; d < 3; ++d) { L.force[d] += F[d]; }
        L.moment[0] += r[1] * F[2] - r[2] * F[1];
        L.moment[1] += r[2] * F[0] - r[0] * F[2];
        L.moment[2] += r[0] * F[1] - r[1] * F[0];
    }
    L.horizontal_force = std::sqrt(L.force[0] * L.force[0] + L.force[1] * L.force[1]);
    L.overturning_moment = std::sqrt(L.moment[0] * L.moment[0] + L.moment[1] * L.moment[1]);
    L.over_allowable = (m_in.allowable_force > 0.0 && L.horizontal_force > m_in.allowable_force) ||
                       (m_in.allowable_moment > 0.0 && L.overturning_moment > m_in.allowable_moment);
    return L;
}

std::string attach_line_ends (std::vector<Transformer>& transformers, const std::vector<SpanInputs>& placed)
{
    for (std::size_t i = 0; i < placed.size(); ++i) {
        const SpanInputs& s = placed[i];
        for (int end = 0; end < 2; ++end) {
            const auto& p = (end == 0) ? s.end_a : s.end_b;
            const std::string which = s.name + (end == 0 ? ".end_a" : ".end_b");
            Transformer* on = nullptr;
            for (Transformer& t : transformers) {
                if (!t.inputs().on_footprint(p[0], p[1])) { continue; }
                if (on != nullptr) {
                    return "erf.conductors." + which + " lies on the footprints of both " + on->name() + " and " + t.name() +
                           "; transformers must not overlap";
                }
                on = &t;
            }
            if (on == nullptr) { continue; }
            const Real top = on->box_hi()[2];
            if (!(p[2] > top)) {
                return "erf.conductors." + which + " ends on " + on->name() + " at " + std::to_string(p[2]) +
                       " m, not above the transformer's top at " + std::to_string(top) +
                       " m; raise the end's height above the terrain or lower " + on->name() + ".size";
            }
            on->add_end(LineEnd{i, end});
        }
    }
    for (const Transformer& t : transformers) {
        if (t.ends().empty()) {
            return "erf.conductors." + t.name() + ": no line ends on it (an end_a or end_b whose x, y lies on its footprint)";
        }
    }
    return std::string();
}

} // namespace erf_conductors
