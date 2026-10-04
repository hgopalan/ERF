#ifndef ERF_TRANSFORMER_H
#define ERF_TRANSFORMER_H

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"

namespace erf_conductors {

/** A line's end: the line (an index into erf.conductors.lines) and which end, 0 for end_a and 1 for end_b. */
struct LineEnd
{
    std::size_t line{0};
    int end{0};
};

/** The load the lines put on a transformer, about the centre of its base. */
struct TransformerLoad
{
    std::array<amrex::Real,3> force{{0.0, 0.0, 0.0}};    //!< the sum of the lines' pulls (N)
    std::array<amrex::Real,3> moment{{0.0, 0.0, 0.0}};   //!< their moment about the base centre (N m)
    amrex::Real horizontal_force{0.0};                   //!< the horizontal part of the force (N)
    amrex::Real overturning_moment{0.0};                 //!< the horizontal part of the moment, which tips the box (N m)
    bool over_allowable{false};                          //!< either exceeds its allowable value (where one is given)
};

/**
 * A transformer standing on the terrain (TransformerInputs): a box whose base is at the terrain
 * height under its centre, and the line ends dead-ended on it.
 */
class Transformer
{
public:
    Transformer (TransformerInputs in, amrex::Real base_z) : m_in(std::move(in)), m_base_z(base_z) {}

    const std::string& name () const { return m_in.name; }
    const TransformerInputs& inputs () const { return m_in; }
    /** The centre of the base, on the terrain (m). */
    std::array<amrex::Real,3> base () const { return {{m_in.position[0], m_in.position[1], m_base_z}}; }
    /** The box's lower and upper corners (m). */
    std::array<amrex::Real,3> box_lo () const;
    std::array<amrex::Real,3> box_hi () const;
    /** The line ends dead-ended on it. */
    const std::vector<LineEnd>& ends () const { return m_ends; }
    void add_end (LineEnd e) { m_ends.push_back(e); }

    /**
     * The load of the lines ending on it, given where each of its ends() is and the force the line
     * puts on it there (one entry per end, in the order of ends()): the sum of the forces, their
     * moment about the base centre, and whether the horizontal force or the overturning moment
     * exceeds its allowable value (a zero allowable is not checked).
     */
    TransformerLoad load (const std::vector<std::array<amrex::Real,3>>& at,
                          const std::vector<std::array<amrex::Real,3>>& force) const;

private:
    TransformerInputs m_in;
    amrex::Real m_base_z{0.0};
    std::vector<LineEnd> m_ends;
};

/**
 * Dead-end the lines on the transformers: every end of a line (placed at its absolute height)
 * whose x, y lies on a transformer's footprint is added to that transformer's ends. Returns an
 * empty string, or the message to abort with when an end lies on two footprints, when an end on a
 * footprint is not above the box's top, or when no line ends on a transformer.
 */
std::string attach_line_ends (std::vector<Transformer>& transformers, const std::vector<LineInputs>& placed);

} // namespace erf_conductors

#endif
