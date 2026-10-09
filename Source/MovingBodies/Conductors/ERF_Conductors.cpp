#include "ERF_Conductors.H"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ERF_ASCE74.H"
#include "ERF_ActuatorSampling.H"
#include "ERF_ActuatorSpreading.H"
#include "ERF_DiagnosticsLog.H"
#include "ERF_Constants.H"
#include "ERF_ConductorGeometry.H"
#include "ERF_Gusts.H"
#include "ERF_IndexDefines.H"
#include "ERF_LatticeFrame.H"
#include "ERF_MoorDynSystem.H"

using namespace amrex;
using erf_conductors::ConductorInputs;
using erf_conductors::ConductorLine;
using erf_conductors::LineInputs;
using erf_conductors::Transformer;

namespace {
constexpr Real rad2deg = Real(180.0 / 3.14159265358979323846);

// the input key of attachment point k of a line: end_a, end_b or one of its towers (k from 1)
std::string point_key (const LineInputs& s, int k)
{
    const std::string base = "erf.conductors." + s.name;
    if (k == 0) { return base + ".end_a"; }
    if (k == s.num_spans()) { return base + ".end_b"; }
    return base + ".towers (tower " + std::to_string(k) + ")";
}

// the largest utilisation of a tower's members, and the member it is in (0 and 0 without members)
std::pair<Real,int> governing (const std::vector<erf_towers::MemberCheck>& checks)
{
    std::pair<Real,int> g{Real(0.0), 0};
    for (std::size_t m = 0; m < checks.size(); ++m) {
        const Real u = static_cast<Real>(checks[m].utilisation);
        if (m == 0 || u > g.first) { g = {u, checks[m].member}; }
    }
    return g;
}
}

std::unique_ptr<Conductors>
Conductors::create (int max_level)
{
    ConductorInputs in = ConductorInputs::read();
    if (!in.active) { return nullptr; }
    const int anchor = ConductorInputs::resolve_anchor_level(in.anchor_level, max_level);
    const std::string err = ConductorInputs::validate_solver(max_level, anchor, erf_moordyn::fpe_traps_requested());
    if (!err.empty()) { Abort(err); }
    Print() << "erf.conductors: " << in.lines.size() << " line(s) on MoorDyn-C " << erf_moordyn::library_version()
            << (erf_moordyn::is_stub() ? " (the bundled stub stands in for MoorDyn)" : "") << ", anchor level " << anchor
            << ", wind " << (in.has_prescribed_velocity ? "prescribed" : "sampled from the flow at the line nodes")
            << ", drag " << (in.drag_on_flow ? "put back into the flow (epsilon " + std::to_string(in.epsilon) + " cells)" : "not put into the flow") << "\n";
    return std::unique_ptr<Conductors>(new Conductors(std::move(in), anchor));
}

void
Conductors::set_closure (bool keqn_rans, Real Cmu0)
{
    if (!m_in.gusts_on()) { m_closure_set = true; return; }
    if (!keqn_rans) {
        Abort("erf.conductors.gust_type = " + m_in.gust_type + " needs the k-equation RANS (erf.rans_type = kEqn) on the conductors' "
              "anchor level " + std::to_string(m_anchor) + ": the gusts come from its k");
    }
    m_gust_sigma = m_in.has_gust_sigma_factor ? static_cast<double>(m_in.gust_sigma_factor)
                                              : erf_conductors::default_gust_sigma_factor(static_cast<double>(Cmu0));
    m_closure_set = true;
    Print() << "erf.conductors: gusts from the RANS k: sigma_u = " << m_gust_sigma << " sqrt(k)"
            << (m_in.has_gust_sigma_factor ? "" : " (2.5 Cmu0)") << ", peak factor " << m_in.gust_peak_factor
            << ", span length scale " << m_in.gust_span_length_scale << " m; every span's static gust factor in "
            << m_in.diagnostics_dir << "/gusts.csv\n";
}

Conductors::Conductors (ConductorInputs_t in, int anchor)
    : m_in(std::move(in)), m_anchor(anchor) {}

void
Conductors::set_ground (const MultiFab* z_phys_nd, const Geometry& geom, const std::string& restart_chkdir)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_ground_set, "Conductors::set_ground: called twice");
    {
        // every node must lie in MoorDyn's fluid: below its free surface and above its bottom
        const std::string err = ConductorInputs::validate_frame(m_in.surface_offset, static_cast<Real>(geom.ProbLo(2)),
                                                                static_cast<Real>(geom.ProbHi(2)));
        if (!err.empty()) { Abort(err); }
    }
    m_ground_set = true;

    // the attachment points must lie inside the domain horizontally: the terrain height is read there
    std::vector<Real> pos;
    for (const LineInputs& s : m_in.lines) {
        for (int k = 0; k <= s.num_spans(); ++k) {
            const auto& e = s.point(k);
            for (int d = 0; d < 2; ++d) {
                if (e[d] < geom.ProbLo(d) || e[d] > geom.ProbHi(d)) {
                    Abort(point_key(s, k) + " at (" + std::to_string(e[0]) + ", " + std::to_string(e[1]) +
                          ") lies outside the domain horizontally");
                }
            }
            pos.insert(pos.end(), {e[0], e[1], e[2]});
        }
    }
    ground_heights(z_phys_nd, geom, pos, m_ground);
    // a line hanging from another line's towers stands its points on those towers: their heights
    // are above the tower's base, so that a cross-arm on sloping ground stays level
    {
        std::vector<std::size_t> first(m_in.lines.size() + 1, 0);
        for (std::size_t i = 0; i < m_in.lines.size(); ++i) { first[i+1] = first[i] + static_cast<std::size_t>(m_in.lines[i].num_spans() + 1); }
        for (std::size_t i = 0; i < m_in.lines.size(); ++i) {
            const LineInputs& s = m_in.lines[i];
            if (s.share_towers.empty()) { continue; }
            std::size_t owner = 0;
            while (owner < m_in.lines.size() && m_in.lines[owner].name != s.share_towers) { ++owner; }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owner < m_in.lines.size(),
                                             "Conductors::set_ground: share_towers names no line (validate_shared_towers did not run)");
            for (int k = 1; k < s.num_spans(); ++k) {
                const auto kk = static_cast<std::size_t>(k);
                m_ground[first[i] + kk] = m_ground[first[owner] + kk];
            }
        }
    }
    const Real floor_z = static_cast<Real>(geom.ProbLo(2));
    m_placed = m_in.lines;
    std::size_t ip = 0;
    // a misplaced line is reported after ground.dat is written, so that the placement can be looked at
    std::string misplaced;
    for (LineInputs& s : m_placed) {
        for (int k = 0; k <= s.num_spans(); ++k, ++ip) {
            auto& e = s.point(k);
            e[2] += m_ground[ip] - floor_z;
            if (e[2] < geom.ProbLo(2) || e[2] > geom.ProbHi(2)) {
                Abort(point_key(s, k) + ": the placed height " + std::to_string(e[2]) +
                      " m (z + the terrain height - geometry.prob_lo[2]) lies outside the domain");
            }
        }
        s.lengths_from_stringing_tension((s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) *
                                          s.diameter * s.diameter) * CONST_GRAV);
        const std::string slack = ConductorInputs::validate_slack(s, true);
        if (misplaced.empty()) { misplaced = slack; }
    }
    const std::string on_transformers = place_transformers(z_phys_nd, geom);
    if (misplaced.empty()) { misplaced = on_transformers; }
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(m_in.diagnostics_dir, 0755);
        std::ofstream out(m_in.diagnostics_dir + "/ground.dat", std::ios::trunc);
        out << "line point x y ground z\n" << std::setprecision(10);
        ip = 0;
        for (const LineInputs& s : m_placed) {
            for (int k = 0; k <= s.num_spans(); ++k, ++ip) {
                const auto& e = s.point(k);
                const std::string label = (k == 0) ? "a" : (k == s.num_spans() ? "b" : "t" + std::to_string(k));
                out << s.name << " " << label << " " << e[0] << " " << e[1] << " " << m_ground[ip] << " " << e[2] << "\n";
            }
        }
        // a transformer's row: the centre of its base on the terrain and the height of its top
        for (const Transformer& t : m_transformers) {
            const auto b = t.base();
            out << t.name() << " transformer " << b[0] << " " << b[1] << " " << b[2] << " " << t.box_hi()[2] << "\n";
        }
    }
    ParallelDescriptor::Barrier();
    if (!misplaced.empty()) { Abort(misplaced + " (the placement is in " + m_in.diagnostics_dir + "/ground.dat)"); }
    build_towers();
    write_asce74();
    setup_gusts(geom);

    if (!restart_chkdir.empty()) {
        const std::string dir = restart_chkdir + "/conductors";
        if (FileExists(dir + "/state")) {
            restore(dir);
            update_ground_under_nodes(z_phys_nd, geom);
            measure_separation();
            measure_transformers();
            return;
        }
        // a checkpoint written without conductors (a precursor, say): the lines start afresh here
        Print() << "erf.conductors: the checkpoint " << restart_chkdir << " holds no conductor state; the lines start at this time\n";
    }
    for (const LineInputs& s : m_placed) {
        const std::string file = m_in.diagnostics_dir + "/" + s.name + ".moordyn.txt";
        m_lines.push_back(std::make_unique<ConductorLine>(s, m_in, CONST_GRAV, file));
        const ConductorLine& c = *m_lines.back();
        // a string the line does not weigh on is pulled up by the spans either side (uplift)
        if (s.has_insulators()) {
            const Real w = (s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) * s.diameter * s.diameter) * CONST_GRAV;
            for (int j = 0; j < static_cast<int>(s.towers.size()); ++j) {
                if (m_lines.back()->string_in_uplift(j, w)) {
                    Print() << "erf.conductors." << s.name << ": WARNING the string at tower " << j + 1 << " carries "
                            << m_lines.back()->string_load(j) << " N of the conductor in still air: the spans either side "
                            << "rise away from the tower and pull it up (uplift), and the string flips over the cross-arm; "
                            << "raise that tower or make it a strain tower\n";
                }
            }
        }
        Print() << "erf.conductors." << s.name << ": " << s.num_spans() << " span(s)";
        if (s.has_insulators()) { Print() << ", hanging from insulator strings of " << s.insulator_length << " m at the towers"; }
        else if (!s.towers.empty()) { Print() << ", clamped at the towers"; }
        Print() << ", " << c.num_nodes() << " nodes, MoorDyn input " << file << "\n";
        for (int k = 0; k < s.num_spans(); ++k) {
            const std::string which = (s.num_spans() == 1) ? "" : " span " + std::to_string(k + 1);
            Print() << "  " << s.name << which << ": chord " << s.chord(k) << " m, length " << s.lengths[static_cast<std::size_t>(k)]
                    << " m, initial sag " << c.mid_sag(k) << " m, end tensions " << c.tension_a(k) << " and " << c.tension_b(k) << " N\n";
            if (!s.has_insulators() && std::abs(s.point(k + 1)[2] - s.point(k)[2]) < Real(1.0e-6) * s.chord(k)) {
                // a level span between fixed points: compare MoorDyn's still-air shape with the elastic catenary
                const Real w = (s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) * s.diameter * s.diameter) * CONST_GRAV;
                const erf_conductors::Catenary cat =
                    erf_conductors::elastic_catenary(s.chord(k), s.lengths[static_cast<std::size_t>(k)], w, s.axial_stiffness);
                Print() << "  " << s.name << which << ": elastic catenary sag " << cat.sag << " m, end tension "
                        << cat.end_tension << " N (MoorDyn's differ by " << 100.0 * (c.mid_sag(k) / cat.sag - 1.0) << " % and "
                        << 100.0 * (c.tension_a(k) / cat.end_tension - 1.0) << " %)\n";
            }
        }
        add_stats(s);
    }
    add_pair_stats();
    add_transformer_stats();
    add_tower_stats();
    update_ground_under_nodes(z_phys_nd, geom);
    measure_separation();
    measure_transformers();
    for (std::size_t p = 0; p < m_pairs.size(); ++p) {
        const auto& c = m_sep[p];
        Print() << "erf.conductors: " << m_lines[m_pairs[p].first]->name() << " and " << m_lines[m_pairs[p].second]->name()
                << " hang " << c.distance << " m apart at their closest"
                << (c.distance < m_in.flashover_distance ? ", already inside the flashover distance" : "") << "\n";
    }
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const Transformer& tr = m_transformers[t];
        const auto& L = m_tload[t];
        Print() << "erf.conductors." << tr.name() << ": base at " << tr.base()[2] << " m, ends of";
        for (const auto& e : tr.ends()) { Print() << " " << m_lines[e.line]->name() << (e.end == 0 ? ".end_a" : ".end_b"); }
        Print() << "; still-air pull " << L.horizontal_force << " N horizontal, " << L.force[2] << " N vertical, overturning moment "
                << L.overturning_moment << " N m" << (L.over_allowable ? ", already over its allowable" : "")
                << "; closest conductor " << m_tclear[t].distance << " m from the box"
                << (m_tclear[t].distance < m_in.flashover_distance ? ", inside the flashover distance" : "") << "\n";
    }
    if (!m_towers.empty()) {
        load_towers_with_lines();
        Print() << "erf.conductors: " << m_towers.size() << " lattice tower(s) loaded by the wind and the lines:\n";
        for (std::size_t ti = 0; ti < m_towers.size(); ++ti) {
            const auto& t = m_towers[ti];
            const auto L = tower_foundation(ti);
            Print() << "  " << t.name() << " (" << t.type().name << ", cross-arm " << t.arm_height() << " m): still-air line pull "
                    << std::hypot(t.line_force()[0], t.line_force()[1]) << " N horizontal, " << -t.line_force()[2]
                    << " N down; legs " << L.max_compression << " N compression, " << L.max_uplift << " N uplift at most"
                    << (L.over_allowable ? ", already over an allowable" : "") << "\n";
        }
        for (std::size_t t = 0; t < m_towers.size(); ++t) {
            if (const erf_towers::FrameTower* fr = frame_tower(t)) {
                Print() << "  " << m_towers[t].name() << " bends as " << fr->source() << ": " << fr->frame().num_nodes() << " nodes, "
                        << fr->frame().inputs().members.size() << " members, " << fr->frame().num_free_dofs()
                        << " free degrees of freedom, " << fr->mass() << " kg, first natural frequency " << fr->frequency() << " Hz";
                const Real theta = m_towers[t].type().steel_temperature;
                if (theta != Real(20.0)) { Print() << " with its steel at " << theta << " C"; }
                Print() << "; MoorDyn moves its cross-arm as a coupled point\n";
                const auto checks = fr->member_checks(m_towers[t]);
                if (!checks.empty()) {
                    std::size_t worst = 0;
                    int slender = 0, thin = 0;
                    for (std::size_t m = 0; m < checks.size(); ++m) {
                        if (checks[m].utilisation > checks[worst].utilisation) { worst = m; }
                        slender += checks[m].slender ? 1 : 0;
                        thin += checks[m].thin ? 1 : 0;
                    }
                    Print() << "    members (ASCE 10-15, in still air): utilisation " << checks[worst].utilisation << " at most, member "
                            << checks[worst].member << " (" << erf_towers::role_name(checks[worst].role) << "); " << slender
                            << " over their slenderness limit, " << thin << " with w/t over 25 (tower_" << m_towers[t].name()
                            << "_members.csv)\n";
                }
                continue;
            }
            const auto* m = dynamic_cast<const erf_towers::OneModeTower*>(m_models[t].get());
            if (m == nullptr) { continue; }
            Print() << "  " << m_towers[t].name() << " bends at " << m->frequency() << " Hz (" << m_towers[t].type().frequency
                    << " Hz on a rigid foundation): generalized mass " << m->generalized_mass() << " kg, stiffness "
                    << m->stiffness() << " N/m at the cross-arm; MoorDyn moves its cross-arm as a coupled point\n";
        }
        write_member_tables();
    }
}

void
Conductors::write_asce74 () const
{
    if (!(m_in.asce74_wind > 0.0)) { return; }
    erf_conductors::Exposure e = erf_conductors::Exposure::C;
    if (!m_in.asce74_exposure.empty()) { erf_conductors::parse_exposure(m_in.asce74_exposure, e); }
    const std::string file = m_in.diagnostics_dir + "/asce74.csv";
    Print() << "erf.conductors: ASCE 74 design check for a " << m_in.asce74_wind << " m/s 3-second gust at 10 m, exposure "
            << (e == erf_conductors::Exposure::B ? "B" : "C") << ", in " << file << "\n";
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out(file, std::ios::trunc);
    if (!out) { Abort("cannot write '" + file + "'"); }
    out << "line,span,height,chord,length,kz,gust_response,pressure,load,weight,swing_deg,sag,blowout,tension\n" << std::setprecision(10);
    for (std::size_t i = 0; i < m_placed.size(); ++i) {
        const LineInputs& s = m_placed[i];
        const double w = static_cast<double>((s.mass_per_length - m_in.air_density * Real(0.25) * Real(3.14159265358979323846) *
                                              s.diameter * s.diameter) * CONST_GRAV);
        for (int k = 0; k < s.num_spans(); ++k) {
            // the height of the conductor above the ground at each end of the span, averaged
            const double z = span_height(i, k);
            if (!(z > 0.0)) { Abort("erf.conductors." + s.name + ": span " + std::to_string(k + 1) + " is not above the ground"); }
            const auto L = erf_conductors::wire_wind_load(e, static_cast<double>(m_in.asce74_wind), z, static_cast<double>(s.chord(k)),
                                                          static_cast<double>(s.diameter), static_cast<double>(s.drag_coefficient), w,
                                                          static_cast<double>(s.lengths[static_cast<std::size_t>(k)]),
                                                          static_cast<double>(s.axial_stiffness), static_cast<double>(m_in.air_density));
            out << s.name << "," << k + 1 << "," << z << "," << s.chord(k) << "," << s.lengths[static_cast<std::size_t>(k)] << ","
                << L.kz << "," << L.gust_response << ","
                << L.pressure << "," << L.load << "," << L.weight << "," << L.swing * 180.0 / 3.14159265358979323846 << ","
                << L.sag << "," << L.blowout << "," << L.tension << "\n";
        }
    }
    if (!out) { Abort("cannot write '" + file + "'"); }
}

void
Conductors::build_towers ()
{
    m_towers.clear();
    m_tower_lines.clear();
    m_line_towers.assign(m_placed.size(), {});
    m_models.clear();
    m_aero = std::make_unique<erf_towers::MemberDrag>(m_in.air_density);
    std::vector<std::size_t> first(m_placed.size() + 1, 0);
    for (std::size_t i = 0; i < m_placed.size(); ++i) { first[i+1] = first[i] + static_cast<std::size_t>(m_placed[i].num_spans() + 1); }
    // the towers of the lines with a tower_type, each line hanging from its own at the centre of the cross-arm
    for (std::size_t line = 0; line < m_placed.size(); ++line) {
        const LineInputs& s = m_placed[line];
        if (s.tower_type.empty()) { continue; }
        const erf_towers::TowerType* type = m_in.tower_type(s);
        AMREX_ALWAYS_ASSERT(type != nullptr);   // checked when the inputs were read
        for (int k = 1; k < s.num_spans(); ++k) {
            // the cross-arm runs across the line: normal to the mean horizontal direction of the spans either side
            const auto& a = s.point(k - 1);
            const auto& p = s.point(k);
            const auto& b = s.point(k + 1);
            Real back[2] = {p[0] - a[0], p[1] - a[1]}, ahead[2] = {b[0] - p[0], b[1] - p[1]};
            const Real nback = std::hypot(back[0], back[1]), nahead = std::hypot(ahead[0], ahead[1]);
            Real along[2] = {back[0] / nback + ahead[0] / nahead, back[1] / nback + ahead[1] / nahead};
            const Real na = std::hypot(along[0], along[1]);
            if (!(na > Real(1.0e-6))) {
                Abort("erf.conductors." + s.name + ".towers: the line turns back on itself at tower " + std::to_string(k) +
                      "; a cross-arm across it is undefined");
            }
            const std::array<Real,3> across{{-along[1] / na, along[0] / na, 0.0}};
            const Real ground = m_ground[first[line] + static_cast<std::size_t>(k)];
            m_towers.emplace_back(s.name + "_t" + std::to_string(k), *type, std::array<Real,3>{{p[0], p[1], ground}},
                                  p[2] - ground, across);
            const std::size_t att = m_towers.back().add_attachment(p);
            m_tower_lines.push_back({{line, k - 1}});
            m_line_towers[line].emplace_back(m_towers.size() - 1, att);
        }
    }
    // the lines hanging from another line's towers, each at its own point on them
    for (std::size_t line = 0; line < m_placed.size(); ++line) {
        const LineInputs& s = m_placed[line];
        if (s.share_towers.empty()) { continue; }
        std::size_t owner = 0;
        while (owner < m_placed.size() && m_placed[owner].name != s.share_towers) { ++owner; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owner < m_placed.size() && m_line_towers[owner].size() == s.towers.size(),
                                         "Conductors::build_towers: share_towers owner missing or with another tower count");
        for (int k = 1; k < s.num_spans(); ++k) {
            const std::size_t t = m_line_towers[owner][static_cast<std::size_t>(k - 1)].first;
            erf_towers::Tower& tw = m_towers[t];
            const auto& p = s.point(k);
            const Real off = std::hypot(p[0] - tw.base()[0], p[1] - tw.base()[1]);
            const Real top = tw.arm_height() + tw.type().peak;
            const Real slack = Real(1.0) + Real(1.0e-6);
            if (off > Real(0.5) * tw.type().arm_length * slack || p[2] < tw.base()[2] || p[2] > tw.base()[2] + top * slack) {
                Abort("erf.conductors." + s.name + ".towers (tower " + std::to_string(k) + ", with erf.conductors." + s.name +
                      ".share_towers): the point on tower " + tw.name() + " lies " + std::to_string(off) + " m from the tower's axis and " +
                      std::to_string(p[2] - tw.base()[2]) + " m above its base, off the tower (half the cross-arm " +
                      std::to_string(Real(0.5) * tw.type().arm_length) + " m, the top " + std::to_string(top) + " m)");
            }
            const std::size_t att = tw.add_attachment(p);
            m_tower_lines[t].emplace_back(line, k - 1);
            m_line_towers[line].emplace_back(t, att);
        }
    }
    // the structural models once every line hangs from its towers: a frame model where the type gives a
    // frame file (one frame per type, shared by its towers) or generates one (one per type and cross-arm
    // height to the millimetre, the first such tower's), else one mode where it gives a frequency
    m_frames.clear();
    for (const auto& tw : m_towers) {
        const erf_towers::TowerType& type = tw.type();
        if (!type.has_frame()) {
            m_models.push_back(type.moves() ? std::make_unique<erf_towers::OneModeTower>(tw, CONST_GRAV) : nullptr);
            continue;
        }
        std::ostringstream fkey;
        fkey << type.name;
        if (type.frame_panels > 0) { fkey << " " << std::llround(1000.0 * static_cast<double>(tw.arm_height())); }
        if (m_frames.count(fkey.str()) == 0) { m_frames[fkey.str()] = make_frame(tw); }
        const TowerFrame& tf = m_frames[fkey.str()];
        m_models.push_back(std::make_unique<erf_towers::FrameTower>(tw, tf.frame, CONST_GRAV, tf.designs, tf.source, tf.link_nodes));
    }
    // one attachment per line hanging from a tower, in the same order on the tower and in its model
    for (std::size_t t = 0; t < m_towers.size(); ++t) {
        const std::size_t n = m_tower_lines[t].size();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_towers[t].attachments().size() == n &&
                                         (!m_models[t] || m_models[t]->num_attachments() == std::max<std::size_t>(n, 1)),
                                         "Conductors::build_towers: a tower's lines, attachments and model attachments differ in number");
    }
    // the lines that step together: those joined by the towers they share
    m_groups.clear();
    std::vector<int> group(m_placed.size(), -1);
    for (std::size_t line = 0; line < m_placed.size(); ++line) {
        if (group[line] >= 0) { continue; }
        group[line] = static_cast<int>(m_groups.size());
        m_groups.push_back({line});
        for (std::size_t g = 0; g < m_groups.back().size(); ++g) {
            for (const auto& ta : m_line_towers[m_groups.back()[g]]) {
                for (const auto& lj : m_tower_lines[ta.first]) {
                    if (group[lj.first] < 0) { group[lj.first] = group[line]; m_groups.back().push_back(lj.first); }
                }
            }
        }
    }
}

Conductors::TowerFrame
Conductors::make_frame (const erf_towers::Tower& tw) const
{
    const erf_towers::TowerType& type = tw.type();
    const std::string key = "erf.conductors." + type.name;
    TowerFrame tf;
    erf_towers::FrameInputs fin;
    std::vector<erf_towers::MemberDesign> designs;
    std::string err;
    std::string written;
    std::vector<int> load_joints;
    if (!type.frame_file.empty()) {
        tf.source = key + ".frame_file = " + type.frame_file;
        err = erf_towers::read_subdyn(type.frame_file, fin);
        if (err.empty() && !type.member_file.empty()) {
            const std::string mkey = key + ".member_file = " + type.member_file;
            err = erf_towers::read_member_designs(type.member_file, designs);
            if (err.empty()) { err = erf_towers::match_designs(fin, designs, type.member_file); }
            if (!err.empty()) { Abort(mkey + ": " + err); }
        }
    } else {
        written = m_in.diagnostics_dir + "/frame_" + tw.name() + ".dat";
        tf.source = "the frame generated for " + key + " (frame_panels = " + std::to_string(type.frame_panels) + ", written to " +
                    written + ")";
        erf_towers::LatticeSpec spec;
        spec.base_width = static_cast<double>(type.base_width);
        spec.top_width = static_cast<double>(type.top_width);
        spec.arm_height = static_cast<double>(tw.arm_height());
        spec.arm_length = static_cast<double>(type.arm_length);
        spec.arm_depth = static_cast<double>(type.arm_face());
        spec.peak = static_cast<double>(type.peak);
        spec.panels = type.frame_panels;
        spec.crossed = (type.bracing != "single");
        spec.leg_b = static_cast<double>(type.leg_angle[0]);
        spec.leg_t = static_cast<double>(type.leg_angle[1]);
        spec.brace_b = static_cast<double>(type.brace_angle[0]);
        spec.brace_t = static_cast<double>(type.brace_angle[1]);
        spec.yield = static_cast<double>(type.yield());
        err = erf_towers::lattice_frame(spec, fin, designs, &load_joints);
        if (!err.empty()) { err = "tower " + tw.name() + " (cross-arm " + std::to_string(spec.arm_height) + " m): " + err; }
    }
    if (!err.empty()) { Abort(tf.source + ": " + err); }
    if (type.steel_temperature != Real(20.0)) { fin.temperature.assign(fin.members.size(), static_cast<double>(type.steel_temperature)); }
    std::unique_ptr<erf_towers::Frame> frame = erf_towers::Frame::create(fin, err);
    if (!frame) { Abort(tf.source + ": " + err); }
    tf.frame = std::move(frame);
    for (const int id : load_joints) { tf.link_nodes.push_back(tf.frame->node_of_joint(id)); }
    if (!designs.empty()) { tf.designs = std::make_shared<const std::vector<erf_towers::MemberDesign>>(std::move(designs)); }
    // a generated frame is written out, to read or to run through SubDyn
    if (!written.empty() && ParallelDescriptor::IOProcessor()) {
        std::ostringstream title;
        title << "Lattice tower generated by ERF for " << key << " with its cross-arm at " << tw.arm_height()
              << " m (tower-local axes: x along the line, y along the cross-arm, z up)";
        err = erf_towers::write_subdyn(fin, written, title.str());
        if (!err.empty()) { Abort(tf.source + ": " + err); }
        const std::string mfile = m_in.diagnostics_dir + "/frame_" + tw.name() + "_members.dat";
        if (!erf_towers::write_member_designs(mfile, *tf.designs, "Member design data of " + written)) {
            Abort(tf.source + ": cannot write '" + mfile + "'");
        }
    }
    return tf;
}

const erf_towers::FrameTower*
Conductors::frame_tower (std::size_t t) const
{
    return dynamic_cast<const erf_towers::FrameTower*>(m_models[t].get());
}

std::vector<erf_towers::MemberCheck>
Conductors::member_checks (std::size_t t) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(t < m_towers.size(), "Conductors::member_checks: no such tower");
    const erf_towers::FrameTower* fr = frame_tower(t);
    return fr ? fr->member_checks(m_towers[t]) : std::vector<erf_towers::MemberCheck>();
}

void
Conductors::add_tower_stats ()
{
    m_tower_stats.clear();
    m_member_stats.clear();
    for (std::size_t t = 0; t < m_towers.size(); ++t) {
        const std::string name = "tower_" + m_towers[t].name();
        std::vector<std::string> q{"drag_h", "line_h", "shear", "overturning", "max_compression", "max_uplift", "over_allowable"};
        if (m_towers[t].type().moves()) { q.emplace_back("arm_displacement"); }
        const erf_towers::FrameTower* fr = frame_tower(t);
        const bool checked = fr && fr->has_member_checks();
        if (checked) { q.emplace_back("max_utilisation"); }
        m_tower_stats.emplace_back(name, m_in.diagnostics_dir + "/" + name, q);
        m_member_stats.emplace_back();
        if (checked) {
            // per member: its axial force (tension positive, the larger of its tension and compression) and its utilisation
            std::vector<std::string> mq;
            for (const auto& m : fr->frame().inputs().members) {
                mq.push_back("m" + std::to_string(m.id) + "_axial");
                mq.push_back("m" + std::to_string(m.id) + "_utilisation");
            }
            m_member_stats.back() = std::make_unique<erf_actuator::RunningStats>(name + "_members",
                                                                                 m_in.diagnostics_dir + "/" + name + "_members", mq);
        }
    }
}

void
Conductors::write_member_tables () const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    for (std::size_t t = 0; t < m_towers.size(); ++t) {
        const erf_towers::FrameTower* fr = frame_tower(t);
        if (!fr || !fr->has_member_checks()) { continue; }
        const std::string file = m_in.diagnostics_dir + "/tower_" + m_towers[t].name() + "_members.csv";
        std::ofstream out(file, std::ios::trunc);
        if (!out) { Abort("cannot write '" + file + "'"); }
        out << "member,role,joint_a,joint_b,temperature,length,r,L_r,KL_r,limit,w_t,yield,E,Fa,tension_capacity,compression_capacity,"
               "slender,thin\n" << std::setprecision(10);
        const auto checks = fr->member_checks(m_towers[t]);
        const auto& in = fr->frame().inputs();
        for (std::size_t m = 0; m < checks.size(); ++m) {
            const auto& c = checks[m];
            out << c.member << "," << erf_towers::role_name(c.role) << "," << in.members[m].joint_a << ","
                << in.members[m].joint_b << ","
                << fr->temperature()[m] << "," << c.length << "," << c.r << "," << c.slenderness << "," << c.effective << ","
                << c.limit << ","
                << c.w_t << "," << c.yield << "," << c.E << "," << c.compression_stress << "," << c.tension_capacity << ","
                << c.compression_capacity << "," << (c.slender ? 1 : 0) << "," << (c.thin ? 1 : 0) << "\n";
        }
        if (!out) { Abort("cannot write '" + file + "'"); }
    }
}

void
Conductors::write_tower_frames (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    for (std::size_t t = 0; t < m_towers.size(); ++t) {
        const erf_towers::FrameTower* fr = frame_tower(t);
        if (!fr) { continue; }
        const erf_towers::Frame& f = fr->frame();
        std::ofstream out;
        if (erf_actuator::open_log(out, m_in.diagnostics_dir + "/tower_" + m_towers[t].name() + "_frame.dat", first)) {
            // the header: the nodes' positions (frame axes, m), then the members checked
            out << "# nodes " << f.num_nodes() << ":";
            for (std::size_t n = 0; n < f.num_nodes(); ++n) {
                const auto& x = f.node_position(n);
                out << " " << x[0] << " " << x[1] << " " << x[2];
            }
            out << "\n# members " << (fr->has_member_checks() ? f.inputs().members.size() : 0) << ":";
            if (fr->has_member_checks()) { for (const auto& m : f.inputs().members) { out << " " << m.id; } }
            out << "\n# time, then dx dy dz per node (m), then the utilisation per member\n";
        }
        out << std::setprecision(10) << time;
        const auto u = fr->node_displacements();
        for (std::size_t n = 0; n < f.num_nodes(); ++n) { out << " " << u[6 * n] << " " << u[6 * n + 1] << " " << u[6 * n + 2]; }
        for (const auto& c : fr->member_checks(m_towers[t])) { out << " " << c.utilisation; }
        out << "\n";
    }
}

std::vector<Real>
Conductors::tower_wind (const MultiFab& U, const MultiFab& V, const MultiFab& W, const MultiFab* z_phys_nd, const Geometry& geom,
                        std::vector<Real>* nodes) const
{
    std::vector<Real> pos;
    // the members' current positions: a moving tower's nodes have left where they stood
    for (const auto& t : m_towers) { for (const auto& n : t.current_nodes()) { pos.insert(pos.end(), {n.pos[0], n.pos[1], n.pos[2]}); } }
    if (nodes != nullptr) { *nodes = pos; }
    std::vector<Real> uvw(pos.size(), 0.0);
    if (m_in.has_prescribed_velocity) {
        for (std::size_t p = 0; p < uvw.size() / 3; ++p) { for (int d = 0; d < 3; ++d) { uvw[3*p+d] = m_in.prescribed_velocity[d]; } }
        return uvw;
    }
    const Real reach = std::max(geom.CellSize(0), geom.CellSize(1));
    std::string outside;
    if (!erf_actuator::points_covered_by(U.boxArray(), geom, pos, reach, outside)) {
        Abort("erf.conductors: the tower node " + outside + " is not covered, with the cells around it, by the grids of the "
              "anchor level " + std::to_string(m_anchor) + "; refine around the whole tower or lower erf.conductors.anchor_level");
    }
    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, pos, uvw);
    const std::string err = erf_conductors::first_nonfinite(uvw, 3, "the wind (m/s) at tower node");
    if (!err.empty()) { Abort("erf.conductors: " + err + " (nodes of all towers in order); the flow on the anchor level holds NaN or Inf"); }
    return uvw;
}

void
Conductors::load_towers (const std::vector<Real>& wind)
{
    std::size_t off = 0;
    std::size_t total = 0;
    for (const auto& tw : m_towers) { total += 3 * tw.nodes().size(); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(wind.size() == total, "Conductors::load_towers: one wind vector is needed per tower node");
    m_tower_wind.resize(m_towers.size());
    for (std::size_t t = 0; t < m_towers.size(); ++t) {
        const std::size_t n = 3 * m_towers[t].nodes().size();
        m_tower_wind[t].assign(wind.begin() + static_cast<std::ptrdiff_t>(off), wind.begin() + static_cast<std::ptrdiff_t>(off + n));
        drag_on_tower(t);
        off += n;
    }
    load_towers_with_lines();
}

void
Conductors::drag_on_tower (std::size_t t)
{
    // the wind relative to the members: a tower that stands still has none of its own
    erf_towers::Tower& tw = m_towers[t];
    std::vector<Real> f;
    m_aero->loads(tw.current_nodes(), m_tower_wind[t], tw.node_velocities(), f);
    tw.set_loads(f);
}

erf_towers::FoundationLoad
Conductors::tower_foundation (std::size_t t) const
{
    erf_towers::FoundationLoad L;
    if (m_models[t] && m_models[t]->foundation(m_towers[t], L)) { return L; }
    return m_towers[t].foundation();
}

void
Conductors::move_tower (std::size_t t)
{
    const erf_towers::TowerModel& m = *m_models[t];
    const std::size_t n = m_towers[t].nodes().size();
    std::vector<Real> disp(3 * n), vel(3 * n), inertia(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto x = m.displacement(i), v = m.velocity(i), f = m.inertial_force(i);
        for (int d = 0; d < 3; ++d) { disp[3*i+d] = x[d]; vel[3*i+d] = v[d]; inertia[3*i+d] = f[d]; }
    }
    m_towers[t].set_motion(disp, vel, inertia);
}

void
Conductors::step_moving_group (const std::vector<std::size_t>& lines, double time, double dt)
{
    // the towers the group's lines hang from
    std::vector<std::size_t> towers;
    for (const std::size_t line : lines) {
        for (const auto& ta : m_line_towers[line]) {
            if (std::find(towers.begin(), towers.end(), ta.first) == towers.end()) { towers.push_back(ta.first); }
        }
    }
    // enough coupling steps for the fastest tower to take coupling_steps_per_period over its period
    double fmax = 0.0;
    for (const std::size_t t : towers) { fmax = std::max(fmax, static_cast<double>(m_models[t]->frequency())); }
    const int n = std::max(m_in.substeps, static_cast<int>(std::ceil(dt * fmax * coupling_steps_per_period - 1.0e-9)));
    const double h = dt / n;
    // the interface: every line's pull on every attachment of the group's towers, in one vector
    std::vector<std::size_t> first(towers.size() + 1, 0);
    for (std::size_t i = 0; i < towers.size(); ++i) { first[i+1] = first[i] + 3 * m_tower_lines[towers[i]].size(); }
    auto pulls = [&] () {
        std::vector<double> F(first.back());
        for (std::size_t i = 0; i < towers.size(); ++i) {
            const auto& tl = m_tower_lines[towers[i]];
            for (std::size_t a = 0; a < tl.size(); ++a) {
                const auto f = m_lines[tl[a].first]->tower_force(tl[a].second);
                for (int d = 0; d < 3; ++d) { F[first[i] + 3*a + static_cast<std::size_t>(d)] = static_cast<double>(f[d]); }
            }
        }
        return F;
    };
    for (int k = 0; k < n; ++k) {
        // the state at the start of the coupling step, to which each iteration returns
        std::vector<ConductorLine::State> line_state;
        for (const std::size_t line : lines) { line_state.push_back(m_lines[line]->state()); }
        std::vector<std::vector<double>> tower_state;
        std::vector<std::vector<std::array<Real,3>>> x0;
        for (const std::size_t t : towers) {
            tower_state.push_back(m_models[t]->state());
            std::vector<std::array<Real,3>> x(m_models[t]->num_attachments());
            for (std::size_t a = 0; a < x.size(); ++a) { x[a] = m_models[t]->attachment_displacement(a); }
            x0.push_back(x);
        }
        const std::vector<double> F0 = pulls();
        // the towers advance under the mean of the pulls at the start and the end of the coupling step,
        // and MoorDyn moves the lines' points to where the towers have taken them; the end pulls are
        // found by fixed-point iteration with Aitken's relaxation, so that a stiff span (a short, taut
        // one) cannot drive the exchange unstable
        std::vector<double> F = F0, r_prev;
        double omega = coupling_initial_relaxation;
        int it = 0;
        bool converged = false;
        for (; it < coupling_max_iterations; ++it) {
            for (std::size_t i = 0; i < towers.size(); ++i) {
                const std::size_t t = towers[i];
                if (it > 0) { m_models[t]->set_state(tower_state[i]); move_tower(t); }
                // the members' drag in the wind held for the step at the nodes' velocities at its start
                drag_on_tower(t);
                std::vector<std::array<Real,3>> mean(m_tower_lines[t].size());
                for (std::size_t a = 0; a < mean.size(); ++a) {
                    for (int d = 0; d < 3; ++d) {
                        const std::size_t c = first[i] + 3*a + static_cast<std::size_t>(d);
                        mean[a][d] = static_cast<Real>(0.5 * (F0[c] + F[c]));
                    }
                }
                m_models[t]->step(static_cast<Real>(h), m_towers[t].loads(), mean);
                move_tower(t);
            }
            for (std::size_t m = 0; m < lines.size(); ++m) {
                const std::size_t line = lines[m];
                if (it > 0) { m_lines[line]->set_state(line_state[m]); }
                const std::size_t nt = m_line_towers[line].size();
                std::vector<Real> start(3 * nt), velocity(3 * nt);
                for (std::size_t j = 0; j < nt; ++j) {
                    const auto [t, att] = m_line_towers[line][j];
                    const std::size_t i = static_cast<std::size_t>(std::find(towers.begin(), towers.end(), t) - towers.begin());
                    const auto x1 = m_models[t]->attachment_displacement(att);
                    for (int d = 0; d < 3; ++d) {
                        start[3*j+d] = x0[i][att][d];
                        velocity[3*j+d] = static_cast<Real>((x1[d] - x0[i][att][d]) / h);
                    }
                }
                m_lines[line]->step_coupled(time + k * h, h, start, velocity);
            }
            const std::vector<double> Fn = pulls();
            const std::string err = erf_conductors::coupling_converged(F0, F, Fn, coupling_tolerance, converged);
            if (!err.empty()) {
                Abort("erf.conductors: the coupling of " + m_lines[lines.front()]->name() + " and its towers at t = " +
                      std::to_string(time + k * h) + " s: " + err + " (MoorDyn or a tower model diverged); check "
                      "erf.conductors.moordyn_cfl and the tower types' frequency");
            }
            if (converged) { ++it; break; }
            std::vector<double> r(F.size());
            for (std::size_t c = 0; c < F.size(); ++c) { r[c] = Fn[c] - F[c]; }
            if (!r_prev.empty()) {
                double num = 0.0, den = 0.0;
                for (std::size_t c = 0; c < r.size(); ++c) { num += r_prev[c] * (r[c] - r_prev[c]); den += (r[c] - r_prev[c]) * (r[c] - r_prev[c]); }
                if (den > 0.0) { omega = std::clamp(-omega * num / den, 0.01, 1.0); }
            }
            for (std::size_t c = 0; c < F.size(); ++c) { F[c] += omega * r[c]; }
            r_prev = r;
        }
        m_coupling_iterations = std::max(m_coupling_iterations, it);
        if (!converged) {
            ++m_coupling_unconverged;
            if (!m_coupling_warned) {
                m_coupling_warned = true;
                Print() << "erf.conductors: the coupling of " << m_lines[lines.front()]->name() << " and its towers did not converge in "
                        << coupling_max_iterations << " iterations at t = " << time + k * h << " s; it goes on with the last iterate"
                        << " (each step's count is in coupling.dat)\n";
            }
        }
        for (const std::size_t t : towers) { load_tower_with_lines(t); }
    }
    for (const std::size_t line : lines) { m_lines[line]->check_clock(time + dt); }
}

void
Conductors::load_towers_with_lines ()
{
    for (std::size_t t = 0; t < m_towers.size(); ++t) { load_tower_with_lines(t); }
}

void
Conductors::load_tower_with_lines (std::size_t t)
{
    // each line where it hangs from the tower, which a moving tower has displaced
    const auto& lines = m_tower_lines[t];
    std::vector<std::array<Real,3>> force(lines.size()), at(lines.size());
    for (std::size_t a = 0; a < lines.size(); ++a) {
        const auto [line, j] = lines[a];
        force[a] = m_lines[line]->tower_force(j);
        at[a] = m_towers[t].attachments()[a];
        if (m_models[t]) {
            const auto x = m_models[t]->attachment_displacement(a);
            for (int d = 0; d < 3; ++d) { at[a][d] += x[d]; }
        }
    }
    m_towers[t].set_line_loads(force, at);
}

void
Conductors::write_towers (double time, bool first) const
{
    if (m_towers.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/towers.dat", first);
    if (header) {
        out << "time";
        for (std::size_t ti = 0; ti < m_towers.size(); ++ti) {
            const auto& t = m_towers[ti];
            const std::string& n = t.name();
            for (const char* c : {"_drag_Fx", "_drag_Fy", "_drag_Fz", "_line_Fx", "_line_Fy", "_line_Fz", "_shear", "_overturning",
                                  "_vertical", "_max_compression", "_max_uplift", "_over"}) { out << " " << n << c; }
            if (t.type().moves()) { out << " " << n << "_arm_dx " << n << "_arm_dy"; }
            if (m_member_stats[ti]) { out << " " << n << "_utilisation " << n << "_member"; }
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (std::size_t ti = 0; ti < m_towers.size(); ++ti) {
        const auto& t = m_towers[ti];
        const auto D = t.total_force();
        const auto& F = t.line_force();
        const auto L = tower_foundation(ti);
        out << " " << D[0] << " " << D[1] << " " << D[2] << " " << F[0] << " " << F[1] << " " << F[2] << " " << L.shear << " "
            << L.overturning << " " << L.vertical << " " << L.max_compression << " " << L.max_uplift << " " << (L.over_allowable ? 1 : 0);
        if (t.type().moves()) {
            const auto x = t.arm_displacement();
            out << " " << x[0] << " " << x[1];
        }
        if (m_member_stats[ti]) {
            const auto g = governing(member_checks(ti));
            out << " " << g.first << " " << g.second;
        }
    }
    out << "\n";
}

std::string
Conductors::place_transformers (const MultiFab* z_phys_nd, const Geometry& geom)
{
    if (m_in.transformers.empty()) { return std::string(); }
    std::vector<Real> pos;
    for (const auto& t : m_in.transformers) {
        for (int d = 0; d < 2; ++d) {
            if (t.position[d] < geom.ProbLo(d) || t.position[d] > geom.ProbHi(d)) {
                Abort("erf.conductors." + t.name + ".position (" + std::to_string(t.position[0]) + ", " +
                      std::to_string(t.position[1]) + ") lies outside the domain");
            }
        }
        pos.insert(pos.end(), {t.position[0], t.position[1], Real(0.0)});
    }
    std::vector<Real> h;
    ground_heights(z_phys_nd, geom, pos, h);
    m_transformers.clear();
    for (std::size_t t = 0; t < m_in.transformers.size(); ++t) { m_transformers.emplace_back(m_in.transformers[t], h[t]); }
    return erf_conductors::attach_line_ends(m_transformers, m_placed);
}

void
Conductors::add_transformer_stats ()
{
    m_tstats.clear();
    for (const Transformer& t : m_transformers) {
        const std::string name = "transformer_" + t.name();
        m_tstats.emplace_back(name, m_in.diagnostics_dir + "/" + name,
                              std::vector<std::string>{"horizontal_force", "overturning_moment", "over_allowable", "clearance", "clash"});
    }
}

void
Conductors::measure_transformers ()
{
    m_tload.resize(m_transformers.size());
    m_tclear.resize(m_transformers.size());
    std::vector<std::vector<Real>> paths;
    for (const auto& span : m_lines) { paths.push_back(span->conductor_path()); }
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const Transformer& tr = m_transformers[t];
        std::vector<std::array<Real,3>> at, force;
        for (const auto& e : tr.ends()) {
            const LineInputs& s = m_placed[e.line];
            at.push_back(e.end == 0 ? s.end_a : s.end_b);
            force.push_back(m_lines[e.line]->end_force(e.end));
        }
        m_tload[t] = tr.load(at, force);
        // every conductor, the ones ending on it as well: their ends clear the top by their standoff
        erf_conductors::Closest best;
        for (const auto& P : paths) {
            const erf_conductors::Closest c = erf_conductors::closest_polyline_box(P, tr.box_lo(), tr.box_hi());
            if (c.distance < best.distance) { best = c; }
        }
        m_tclear[t] = best;
    }
}

void
Conductors::write_transformers (double time, bool first) const
{
    if (m_transformers.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/transformers.dat", first);
    if (header) {
        out << "time";
        for (const Transformer& t : m_transformers) {
            const std::string& n = t.name();
            out << " " << n << "_Fx " << n << "_Fy " << n << "_Fz " << n << "_Fh " << n << "_Mx " << n << "_My " << n << "_Mh "
                << n << "_over " << n << "_clearance " << n << "_clash";
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (std::size_t t = 0; t < m_transformers.size(); ++t) {
        const auto& L = m_tload[t];
        out << " " << L.force[0] << " " << L.force[1] << " " << L.force[2] << " " << L.horizontal_force << " " << L.moment[0]
            << " " << L.moment[1] << " " << L.overturning_moment << " " << (L.over_allowable ? 1 : 0) << " " << m_tclear[t].distance
            << " " << (m_tclear[t].distance < m_in.flashover_distance ? 1 : 0);
    }
    out << "\n";
}

void
Conductors::add_stats (const LineInputs& s)
{
    std::vector<erf_actuator::RunningStats> st;
    for (int k = 0; k < s.num_spans(); ++k) {
        st.emplace_back(s.span_name(k), s.span_root(k),
                        std::vector<std::string>{"swing_deg", "mid_offset", "tension_a", "tension_b", "max_tension",
                                                 "min_clearance", "drag_y"});
    }
    if (s.has_insulators()) {
        std::vector<std::string> q;
        for (std::size_t j = 0; j < s.towers.size(); ++j) {
            const std::string t = "t" + std::to_string(j + 1);
            q.insert(q.end(), {t + "_swing_deg", t + "_across_deg", t + "_tension"});
        }
        st.emplace_back(s.name + "_insulators", s.output_root + "_insulators", q);
    }
    m_stats.push_back(std::move(st));
    if (m_in.gusts_on()) {
        std::vector<std::string> q;
        for (int k = 0; k < s.num_spans(); ++k) {
            const std::string sp = "span" + std::to_string(k + 1);
            q.insert(q.end(), {sp + "_wind", sp + "_normal_wind", sp + "_k"});
        }
        m_gust_stats.emplace_back(s.name + "_gusts", s.output_root + "_gusts", q);
    }
}

void
Conductors::add_pair_stats ()
{
    m_pairs.clear();
    m_pair_stats.clear();
    for (std::size_t i = 0; i < m_lines.size(); ++i) {
        for (std::size_t j = i + 1; j < m_lines.size(); ++j) {
            m_pairs.emplace_back(i, j);
            const std::string name = "separation_" + m_lines[i]->name() + "-" + m_lines[j]->name();
            m_pair_stats.emplace_back(name, m_in.diagnostics_dir + "/" + name, std::vector<std::string>{"distance", "clash"});
        }
    }
}

void
Conductors::measure_separation ()
{
    m_sep.resize(m_pairs.size());
    std::vector<std::vector<Real>> paths;
    for (const auto& span : m_lines) { paths.push_back(span->conductor_path()); }
    for (std::size_t p = 0; p < m_pairs.size(); ++p) {
        m_sep[p] = erf_conductors::closest_polylines(paths[m_pairs[p].first], paths[m_pairs[p].second]);
    }
}

void
Conductors::write_separation (double time, bool first) const
{
    if (m_pairs.empty() || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/separation.dat", first);
    if (header) {
        out << "time";
        for (const auto& pr : m_pairs) {
            const std::string n = m_lines[pr.first]->name() + "-" + m_lines[pr.second]->name();
            out << " " << n << "_distance " << n << "_x " << n << "_y " << n << "_z " << n << "_clash";
        }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (const auto& c : m_sep) {
        out << " " << c.distance;
        for (int d = 0; d < 3; ++d) { out << " " << Real(0.5) * (c.a[d] + c.b[d]); }
        out << " " << (c.distance < m_in.flashover_distance ? 1 : 0);
    }
    out << "\n";
}

std::vector<std::string>
Conductors::log_files () const
{
    std::vector<std::string> f;
    for (const LineInputs& s : m_placed) {
        for (int k = 0; k < s.num_spans(); ++k) { f.push_back(s.span_root(k) + ".dat"); }
        f.push_back(s.output_root + "_nodes.dat");
        if (s.has_insulators()) { f.push_back(s.output_root + "_insulators.dat"); }
    }
    f.push_back(m_in.diagnostics_dir + "/total_load.dat");
    f.push_back(m_in.diagnostics_dir + "/separation.dat");
    f.push_back(m_in.diagnostics_dir + "/transformers.dat");
    f.push_back(m_in.diagnostics_dir + "/coupling.dat");
    return f;
}

void
Conductors::restore (const std::string& dir)
{
    // the state file: "step = ", "time = ", "clock_offset = ", "surface_offset = " (absent in older
    // checkpoints) and one "line <name> <nodes>" record per line, in input order
    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(dir + "/state", chars);
    std::istringstream in(std::string(chars.dataPtr(), chars.size()));
    std::string line;
    bool have_step = false, have_time = false, have_t0 = false;
    double saved_offset = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::pair<std::string,unsigned>> saved;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key, eq;
        if (!(ls >> key)) { continue; }
        if (key == "step") {
            if (!(ls >> eq >> m_step) || eq != "=") { Abort("malformed step line in '" + dir + "/state'"); }
            have_step = true;
        } else if (key == "time") {
            if (!(ls >> eq >> m_time) || eq != "=") { Abort("malformed time line in '" + dir + "/state'"); }
            have_time = true;
        } else if (key == "clock_offset") {
            if (!(ls >> eq >> m_t0) || eq != "=") { Abort("malformed clock_offset line in '" + dir + "/state'"); }
            have_t0 = true;
        } else if (key == "surface_offset") {
            if (!(ls >> eq >> saved_offset) || eq != "=") { Abort("malformed surface_offset line in '" + dir + "/state'"); }
        } else if (key == "line" || key == "span") {   // "span" is read as the same record
            std::string name;
            unsigned nodes = 0;
            if (!(ls >> name >> nodes)) { Abort("malformed " + key + " line in '" + dir + "/state'"); }
            saved.emplace_back(name, nodes);
        }
    }
    if (!have_step || !have_time || !have_t0) {
        Abort("no step count, time or clock offset in the conductor checkpoint '" + dir + "/state'");
    }
    {
        const bool moving = std::any_of(m_models.begin(), m_models.end(), [] (const auto& m) { return m != nullptr; });
        const std::string err = erf_conductors::restart_mismatch(FileExists(dir + "/tower_motion"), moving, saved_offset,
                                                                 static_cast<double>(m_in.surface_offset));
        if (!err.empty()) { Abort("the conductor checkpoint '" + dir + "': " + err); }
    }
    // the moving towers' state first: their lines start with the cross-arms where the towers had taken them
    if (std::any_of(m_models.begin(), m_models.end(), [] (const auto& m) { return m != nullptr; })) {
        Vector<char> mchars;
        ParallelDescriptor::ReadAndBcastFile(dir + "/tower_motion", mchars);
        std::istringstream tm(std::string(mchars.dataPtr(), mchars.size()));
        for (std::size_t t = 0; t < m_towers.size(); ++t) {
            if (!m_models[t]) { continue; }
            std::string name;
            std::size_t n = 0;
            if (!(tm >> name >> n) || name != m_towers[t].name()) {
                Abort("the conductor checkpoint '" + dir + "/tower_motion' holds no state for tower " + m_towers[t].name() +
                      "; the lines' tower_type and erf.conductors.tower_types must match the run being restarted");
            }
            std::vector<double> st(n);
            for (auto& v : st) { if (!(tm >> v)) { Abort("the conductor checkpoint '" + dir + "/tower_motion' is truncated"); } }
            if (!m_models[t]->set_state(st)) { Abort("the conductor checkpoint '" + dir + "/tower_motion' does not fit tower " + m_towers[t].name()); }
            move_tower(t);
        }
    }
    if (saved.size() != m_placed.size()) {
        Abort("the conductor checkpoint '" + dir + "' holds " + std::to_string(saved.size()) + " line(s) but erf.conductors.lines lists " +
              std::to_string(m_placed.size()) + "; the erf.conductors block must match the run being restarted");
    }
    for (std::size_t i = 0; i < m_placed.size(); ++i) {
        const LineInputs& s = m_placed[i];
        if (saved[i].first != s.name) {
            Abort("the conductor checkpoint '" + dir + "' holds line " + saved[i].first + " where erf.conductors.lines lists " +
                  s.name + "; the erf.conductors block must match the run being restarted");
        }
        if (saved[i].second != static_cast<unsigned>(s.num_line_nodes())) {
            Abort("erf.conductors." + s.name + ": the checkpoint '" + dir + "' holds a line of " + std::to_string(saved[i].second) +
                  " nodes but the inputs give " + std::to_string(s.num_line_nodes()) +
                  " (spans, towers, segments and insulator strings must match the run being restarted)");
        }
        const std::string file = m_in.diagnostics_dir + "/" + s.name + ".moordyn.txt";
        std::vector<Real> moved;
        if (m_in.towers_move(s)) {
            for (std::size_t j = 0; j < s.towers.size(); ++j) {
                const auto [t, att] = m_line_towers[i][j];
                const auto x = m_models[t]->attachment_displacement(att);
                moved.insert(moved.end(), {x[0], x[1], x[2]});
            }
        }
        m_lines.push_back(std::make_unique<ConductorLine>(s, m_in, CONST_GRAV, file, dir + "/" + s.name + ".moordyn", moved));
        m_lines.back()->set_clock_offset(m_t0);
        add_stats(s);
        for (auto& st : m_stats.back()) {
            if (!st.read_state(dir)) {
                Abort("erf.conductors." + s.name + ": the checkpoint '" + dir + "' holds no statistics " + st.name());
            }
        }
        const ConductorLine& c = *m_lines.back();
        Print() << "erf.conductors." << s.name << ": continued from " << dir << " at t = " << m_time << " s (step " << m_step
                << "), mid-span offset " << c.mid_offset() << " m, sag " << c.mid_sag() << " m\n";
    }
    // the gusts are a diagnostic: a checkpoint written without them starts their statistics at the restart
    bool gusts_restored = !m_gust_stats.empty();
    for (auto& st : m_gust_stats) { gusts_restored = st.read_state(dir) && gusts_restored; }
    if (!m_gust_stats.empty() && !gusts_restored) {
        for (std::size_t i = 0; i < m_gust_stats.size(); ++i) {
            m_gust_stats[i] = erf_actuator::RunningStats(m_gust_stats[i].name(), m_placed[i].output_root + "_gusts",
                                                         m_gust_stats[i].quantities());
        }
        Print() << "erf.conductors: the checkpoint '" << dir << "' holds no gust statistics; they start at the restart\n";
    }
    if (!m_gust_z.empty()) {
        // the random gusts' processes; a checkpoint without them, or with other processes (another number, or a line
        // taking another's gusts differently), draws them afresh
        bool ok = false;
        if (FileExists(dir + "/gust_processes")) {
            Vector<char> gchars;
            ParallelDescriptor::ReadAndBcastFile(dir + "/gust_processes", gchars);
            std::istringstream gs(std::string(gchars.dataPtr(), gchars.size()));
            std::string key, setkey;
            std::size_t n = 0;
            int set = 0;
            if ((gs >> key >> n >> setkey >> set) && key == "processes" && setkey == "set" && n == m_gust_z.size()) {
                ok = true;
                for (std::size_t p = 0; p < m_gust_z.size() && ok; ++p) {
                    std::string name;
                    ok = static_cast<bool>(gs >> name >> m_gust_z[p]) && name == m_process_names[p];
                }
                m_gust_z_set = ok && set != 0;
            }
        }
        if (!ok) {
            std::fill(m_gust_z.begin(), m_gust_z.end(), 0.0);
            m_gust_z_set = false;
            Print() << "erf.conductors: the checkpoint '" << dir << "' holds no random gusts for these lines and towers; "
                    << "they are drawn afresh at the restart\n";
        }
    }
    add_pair_stats();
    for (auto& st : m_pair_stats) {
        if (!st.read_state(dir)) { Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name()); }
    }
    add_tower_stats();
    for (auto& st : m_tower_stats) {
        if (!st.read_state(dir)) {
            Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name() +
                  "; the lines' tower_type and erf.conductors.tower_types must match the run being restarted");
        }
    }
    for (auto& st : m_member_stats) {
        if (st && !st->read_state(dir)) {
            Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st->name() +
                  "; the towers' frames and their member design data must match the run being restarted");
        }
    }
    if (!m_towers.empty()) {
        // the members' last drag, so that the restored drag on the flow is the checkpointed step's
        Vector<char> chars;
        ParallelDescriptor::ReadAndBcastFile(dir + "/tower_loads", chars);
        std::istringstream tl(std::string(chars.dataPtr(), chars.size()));
        for (auto& t : m_towers) {
            std::vector<Real> f(3 * t.nodes().size());
            for (auto& v : f) {
                double x = 0.0;
                if (!(tl >> x)) { Abort("the conductor checkpoint '" + dir + "/tower_loads' holds too few tower node forces"); }
                v = static_cast<Real>(x);
            }
            t.set_loads(f);
        }
    }
    add_transformer_stats();
    for (auto& st : m_tstats) {
        if (!st.read_state(dir)) {
            Abort("the conductor checkpoint '" + dir + "' holds no statistics " + st.name() +
                  "; the erf.conductors.transformers must match the run being restarted");
        }
    }
    // the logs continue from the checkpoint: rows a run wrote after it are dropped
    if (ParallelDescriptor::IOProcessor()) {
        for (const std::string& f : log_files()) { erf_conductors::trim_log_after(f, m_time); }
        // the towers' and the gusts' rows carry the time their step starts at: the restarted run writes the row at m_time
        erf_conductors::trim_log_after(m_in.diagnostics_dir + "/towers.dat", m_time, true);
        erf_conductors::trim_log_after(m_in.diagnostics_dir + "/gust_series.dat", m_time, true);
        for (std::size_t t = 0; t < m_towers.size(); ++t) {
            if (frame_tower(t) && m_in.node_output_int > 0) {
                erf_conductors::trim_log_after(m_in.diagnostics_dir + "/tower_" + m_towers[t].name() + "_frame.dat", m_time, true);
            }
        }
    }
    write_member_tables();
    ParallelDescriptor::Barrier();
    m_restored = true;
}

void
Conductors::write_checkpoint (const std::string& chkdir) const
{
    if (!m_ground_set) { return; }
    const std::string dir = chkdir + "/conductors";
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(dir, 0755);
        std::ofstream out(dir + "/state", std::ios::trunc);
        if (!out) { Abort("cannot write the conductor checkpoint state '" + dir + "/state'"); }
        out << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "step = " << m_step << "\ntime = " << m_time << "\nclock_offset = " << m_t0
            << "\nsurface_offset = " << static_cast<double>(m_in.surface_offset) << "\n";
        for (const auto& line : m_lines) { out << "line " << line->name() << " " << line->num_nodes() << "\n"; }
        if (!out) { Abort("cannot write the conductor checkpoint state '" + dir + "/state'"); }
        // every rank holds the same line; one copy of each is saved
        for (const auto& span : m_lines) { span->save(dir + "/" + span->name() + ".moordyn"); }
        if (!m_gust_z.empty()) {
            // the random gusts' processes: their number and whether they hold values, then each process's name and value
            std::ofstream gz(dir + "/gust_processes", std::ios::trunc);
            gz << std::setprecision(std::numeric_limits<double>::max_digits10) << "processes " << m_gust_z.size() << " set "
               << (m_gust_z_set ? 1 : 0) << "\n";
            for (std::size_t p = 0; p < m_gust_z.size(); ++p) { gz << m_process_names[p] << " " << m_gust_z[p] << "\n"; }
            if (!gz) { Abort("cannot write the conductor checkpoint '" + dir + "/gust_processes'"); }
        }
    }
    ParallelDescriptor::Barrier();
    for (const auto& line : m_stats) { for (const auto& st : line) { st.write_state(dir); } }
    for (const auto& st : m_gust_stats) { st.write_state(dir); }
    for (const auto& st : m_pair_stats) { st.write_state(dir); }
    for (const auto& st : m_tstats) { st.write_state(dir); }
    for (const auto& st : m_tower_stats) { st.write_state(dir); }
    for (const auto& st : m_member_stats) { if (st) { st->write_state(dir); } }
    if (!m_towers.empty() && ParallelDescriptor::IOProcessor()) {
        std::ofstream out(dir + "/tower_loads", std::ios::trunc);
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto& t : m_towers) { for (const Real v : t.loads()) { out << static_cast<double>(v) << "\n"; } }
        if (!out) { Abort("cannot write the conductor checkpoint '" + dir + "/tower_loads'"); }
        if (std::any_of(m_models.begin(), m_models.end(), [] (const auto& m) { return m != nullptr; })) {
            // each moving tower's name, the size of its state and the state
            std::ofstream mo(dir + "/tower_motion", std::ios::trunc);
            mo << std::setprecision(std::numeric_limits<double>::max_digits10);
            for (std::size_t t = 0; t < m_towers.size(); ++t) {
                if (!m_models[t]) { continue; }
                const auto st = m_models[t]->state();
                mo << m_towers[t].name() << " " << st.size();
                for (const double v : st) { mo << " " << v; }
                mo << "\n";
            }
            if (!mo) { Abort("cannot write the conductor checkpoint '" + dir + "/tower_motion'"); }
        }
    }
    ParallelDescriptor::Barrier();
}

void
Conductors::restore_sources (int lev, const MultiFab& U, const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    if (lev != m_anchor || !m_restored || !m_in.drag_on_flow) { return; }
    spread_drag(U, z_phys_nd, detJ_cc, geom);
}

void
Conductors::check_nodes_in_domain (const ConductorLine& span, const std::vector<Real>& pos, const Geometry& geom) const
{
    for (std::size_t p = 0; p < pos.size() / 3; ++p) {
        for (int d = 0; d < 3; ++d) {
            const bool inside = geom.isPeriodic(d) ||
                                (pos[3*p+d] >= geom.ProbLo(d) && pos[3*p+d] <= geom.ProbHi(d));
            if (!inside) {
                Abort("erf.conductors." + span.name() + ": node " + std::to_string(p) + " at (" + std::to_string(pos[3*p]) + ", " +
                      std::to_string(pos[3*p+1]) + ", " + std::to_string(pos[3*p+2]) + ") m has left the domain");
            }
        }
    }
}

void
Conductors::set_ground_surface (const FArrayBox& surface, const Geometry& geom)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_ground_set, "Conductors::set_ground_surface: call before set_ground");
    m_surface = std::make_unique<FArrayBox>(surface.box(), 1, The_Pinned_Arena());
    m_surface->template copy<RunOn::Host>(surface, 0, 0, 1);
    m_surface_geom = geom;
}

void
Conductors::ground_heights (const MultiFab* z_phys_nd, const Geometry& geom, const std::vector<Real>& pos, std::vector<Real>& h) const
{
    if (!m_surface) {
        erf_actuator::terrain_heights(z_phys_nd, geom, pos, h);
        return;
    }
    // bilinear between the surface's nodes, the cell's four corners, as on a fitted mesh's bottom
    const Box& b = m_surface->box();
    const int k = b.smallEnd(2);
    const auto plo = m_surface_geom.ProbLoArray();
    const auto dxi = m_surface_geom.InvCellSizeArray();
    const auto s = m_surface->const_array();
    h.resize(pos.size() / 3);
    for (std::size_t p = 0; p < h.size(); ++p) {
        const Real xi = (pos[3*p] - plo[0]) * dxi[0], yi = (pos[3*p+1] - plo[1]) * dxi[1];
        const int i = std::clamp(static_cast<int>(std::floor(xi)), b.smallEnd(0), b.bigEnd(0) - 1);
        const int j = std::clamp(static_cast<int>(std::floor(yi)), b.smallEnd(1), b.bigEnd(1) - 1);
        const Real wx = xi - static_cast<Real>(i), wy = yi - static_cast<Real>(j);
        h[p] = (Real(1.0) - wy) * ((Real(1.0) - wx) * s(i, j, k) + wx * s(i+1, j, k)) + wy * ((Real(1.0) - wx) * s(i, j+1, k) + wx * s(i+1, j+1, k));
    }
}

void
Conductors::update_ground_under_nodes (const MultiFab* z_phys_nd, const Geometry& geom)
{
    for (auto& span : m_lines) {
        const std::vector<Real> pos = span->node_positions();
        check_nodes_in_domain(*span, pos, geom);
        std::vector<Real> h;
        ground_heights(z_phys_nd, geom, pos, h);
        span->set_ground_under_nodes(h);
    }
}

std::vector<Real>
Conductors::wind_at (const ConductorLine& span,
                     const MultiFab& U, const MultiFab& V, const MultiFab& W,
                     const MultiFab* z_phys_nd, const Geometry& geom, std::vector<Real>* nodes) const
{
    std::vector<Real> uvw(3 * static_cast<std::size_t>(span.num_kinematics_points()), 0.0);
    if (m_in.has_prescribed_velocity) {
        for (std::size_t p = 0; p < uvw.size() / 3; ++p) {
            for (int d = 0; d < 3; ++d) { uvw[3*p+d] = m_in.prescribed_velocity[d]; }
        }
        return uvw;
    }
    // at the line's current position: a blown-out span samples the wind metres away from where it
    // hung. MoorDyn lists the line nodes first, then its point entries (the attachment points, the
    // insulator strings' free lower ends) and one entry at its own origin, far outside ERF's domain;
    // only the line nodes carry a fluid load here, so the flow is sampled at the nodes and the entries
    // after them get zero wind
    const std::vector<Real> kin = span.kinematics_points();
    const std::vector<Real> pos(kin.begin(), kin.begin() + 3 * static_cast<std::ptrdiff_t>(span.num_nodes()));
    check_nodes_in_domain(span, pos, geom);
    // the sampler reads the cells around each point: on a refined anchor level they must be on its grids
    const Real reach = std::max(geom.CellSize(0), geom.CellSize(1));
    std::string outside;
    if (!erf_actuator::points_covered_by(U.boxArray(), geom, pos, reach, outside)) {
        Abort("erf.conductors." + span.name() + ": the point " + outside + " is not covered, with the cells around it, by the "
              "grids of the anchor level " + std::to_string(m_anchor) + "; refine around the whole line or lower erf.conductors.anchor_level");
    }
    std::vector<Real> at_nodes;
    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, pos, at_nodes);
    std::copy(at_nodes.begin(), at_nodes.end(), uvw.begin());
    if (nodes != nullptr) { *nodes = pos; }
    return uvw;
}

void
Conductors::advance (int lev, double time, double dt,
                     const MultiFab& U, const MultiFab& V, const MultiFab& W,
                     const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom,
                     const MultiFab* cons)
{
    if (lev != m_anchor) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt > 0.0 && std::isfinite(dt) && std::isfinite(time),
                                     "Conductors::advance: dt must be positive and finite");
    const bool gusts = m_in.gusts_on();
    const bool gusts_in_wind = m_in.gusts_in_wind();
    if (gusts && !m_closure_set) { Abort("Conductors::advance: gust_type = " + m_in.gust_type + " needs set_closure() first"); }
    if (gusts && cons == nullptr) {
        Abort("Conductors::advance: gust_type = " + m_in.gust_type + " needs the conserved state (rho, rho k)");
    }
    if (!m_ground_set) { set_ground(z_phys_nd, geom); }
    if (m_step == 0 && !m_restored) {
        // the lines' MoorDyn clocks start at zero at this step: at ERF's time zero in a fresh run, at the
        // restart time when a restart creates them afresh (a checkpoint without conductor state)
        m_t0 = time;
        m_time = time;
        for (auto& span : m_lines) { span->set_clock_offset(m_t0); }
    }
    ++m_step;
    const bool first = (m_step == 1);
    const bool write = first || (m_step % m_in.diagnostics_int == 0);
    if (first) { write_separation(time, true); write_transformers(time, true); }
    // the flow's wind at the start of the step, where the lines' nodes and the towers' members are, held over the
    // step; with gusts, also the nodes' positions and k there, every line's and tower's in one pass
    const bool sample_gusts_now = gusts && time + dt >= m_in.stats_start;
    const bool need_k = sample_gusts_now || gusts_in_wind;
    std::vector<std::vector<Real>> wind(m_lines.size()), pos(m_lines.size());
    for (std::size_t line = 0; line < m_lines.size(); ++line) {
        wind[line] = wind_at(*m_lines[line], U, V, W, z_phys_nd, geom, need_k ? &pos[line] : nullptr);
    }
    std::vector<Real> twind, tpos;
    if (!m_towers.empty()) { twind = tower_wind(U, V, W, z_phys_nd, geom, gusts_in_wind ? &tpos : nullptr); }
    std::vector<double> node_k, tower_k;
    if (need_k) {
        std::vector<Real> all;
        for (const auto& p : pos) { all.insert(all.end(), p.begin(), p.end()); }
        const std::size_t nline = all.size() / 3;
        all.insert(all.end(), tpos.begin(), tpos.end());
        node_k = sample_k(all, *cons, z_phys_nd, geom);
        tower_k.assign(node_k.begin() + static_cast<std::ptrdiff_t>(nline), node_k.end());
        node_k.resize(nline);
        std::size_t n0 = 0;
        for (const auto& line : m_lines) {
            for (unsigned m = 0; m < line->num_nodes(); ++m) {
                if (std::isnan(node_k[n0 + m])) {
                    Abort("erf.conductors." + line->name() + ": at node " + std::to_string(m) +
                          " the density is not positive or rho k is not finite");
                }
            }
            n0 += line->num_nodes();
        }
        // the towers' drag nodes are sampled only when the gusts go into the wind
        n0 = 0;
        for (std::size_t t = 0; t < m_towers.size() && !tower_k.empty(); ++t) {
            for (std::size_t m = 0; m < m_towers[t].nodes().size(); ++m) {
                if (std::isnan(tower_k[n0 + m])) {
                    Abort("erf.conductors: at drag node " + std::to_string(m) + " of tower " + m_towers[t].name() +
                          " the density is not positive or rho k is not finite");
                }
            }
            n0 += m_towers[t].nodes().size();
        }
    }
    // the gust statistics take the flow's wind, before any gust, with the other statistics' gate (the steps that end
    // at or after stats_start); a sample is the state at the step's start and is stamped with its time, as the towers' are
    if (sample_gusts_now) { sample_gusts(time, wind, node_k); }
    if (gusts_in_wind) {
        add_gusts(time, dt, pos, wind, node_k, tpos, twind, tower_k);
        if (first || m_step % m_in.diagnostics_int == 0) { write_gust_series(time, first); }
    }
    if (!m_towers.empty()) {
        // the members' drag from the wind at the start of the step, which the step holds
        load_towers(twind);
        if (first || m_step % m_in.diagnostics_int == 0) { write_towers(time, first); }
        if (m_in.node_output_int > 0 && (first || m_step % m_in.node_output_int == 0)) { write_tower_frames(time, first); }
        // the same gate as the other statistics: the steps that end at or after stats_start
        if (time + dt >= m_in.stats_start) {
            for (std::size_t t = 0; t < m_towers.size(); ++t) {
                const auto D = m_towers[t].total_force();
                const auto& F = m_towers[t].line_force();
                const auto L = tower_foundation(t);
                std::vector<Real> q{std::hypot(D[0], D[1]), std::hypot(F[0], F[1]), L.shear, L.overturning,
                                    L.max_compression, L.max_uplift, L.over_allowable ? Real(1.0) : Real(0.0)};
                if (m_models[t]) {
                    const auto x = m_towers[t].arm_displacement();
                    q.push_back(std::hypot(x[0], x[1]));
                }
                if (m_member_stats[t]) {
                    const auto checks = member_checks(t);
                    q.push_back(governing(checks).first);
                    std::vector<Real> mq;
                    mq.reserve(2 * checks.size());
                    for (const auto& c : checks) {
                        mq.push_back(static_cast<Real>(c.tension >= c.compression ? c.tension : -c.compression));
                        mq.push_back(static_cast<Real>(c.utilisation));
                    }
                    m_member_stats[t]->accumulate(time, mq);
                    if (write) { m_member_stats[t]->write(); }
                }
                m_tower_stats[t].accumulate(time, q);
                if (write) { m_tower_stats[t].write(); }
            }
        }
    }
    for (std::size_t line = 0; line < m_lines.size(); ++line) {
        ConductorLine& span = *m_lines[line];
        span.set_wind(wind[line], time + 0.5 * dt);
        if (first) {
            span.write_diagnostics(time, true);
            if (m_in.node_output_int > 0) { span.write_nodes(time, true); }
        }
    }
    // the lines on towers that move step with their towers, those sharing towers together
    m_coupling_iterations = 0;
    m_coupling_unconverged = 0;
    for (const auto& g : m_groups) {
        if (m_lines[g.front()]->towers_move()) {
            step_moving_group(g, time, dt);
        } else {
            for (const std::size_t line : g) { m_lines[line]->step(time, dt); }
        }
    }
    m_time = time + dt;
    if (std::any_of(m_models.begin(), m_models.end(), [] (const auto& m) { return m != nullptr; }) &&
        m_step % m_in.diagnostics_int == 0 && ParallelDescriptor::IOProcessor()) {
        // how hard the lines and their moving towers worked to agree over the step
        std::ofstream out;
        if (erf_actuator::open_log(out, m_in.diagnostics_dir + "/coupling.dat", m_step == m_in.diagnostics_int && !m_restored)) {
            out << "time max_iterations unconverged\n";
        }
        out << std::setprecision(10) << time + dt << " " << m_coupling_iterations << " " << m_coupling_unconverged << "\n";
    }
    // the lines' current positions: their clearance to the terrain, then the outputs and statistics
    update_ground_under_nodes(z_phys_nd, geom);
    // the air's drag on the lines and on the towers' members
    m_drag_total = {{0.0, 0.0, 0.0}};
    for (const auto& t : m_towers) {
        const auto F = t.total_force();
        for (int d = 0; d < 3; ++d) { m_drag_total[d] += F[d]; }
    }
    const bool sample = (time + dt >= m_in.stats_start);
    for (std::size_t i = 0; i < m_lines.size(); ++i) {
        const ConductorLine& span = *m_lines[i];
        const auto f = span.total_drag();
        for (int d = 0; d < 3; ++d) { m_drag_total[d] += f[d]; }
        if (m_step % m_in.diagnostics_int == 0) { span.write_diagnostics(time + dt, false); }
        if (m_in.node_output_int > 0 && m_step % m_in.node_output_int == 0) { span.write_nodes(time + dt, false); }
        if (!sample) { continue; }
        for (int k = 0; k < span.num_spans(); ++k) {
            unsigned low = 0;
            const Real cmin = span.min_clearance(low, k);
            m_stats[i][static_cast<std::size_t>(k)].accumulate(time + dt,
                {span.swing_angle(k) * rad2deg, span.mid_offset(k), span.tension_a(k), span.tension_b(k), span.max_tension(k),
                 cmin, span.span_drag(k)[1]});
        }
        if (span.num_insulators() > 0) {
            std::vector<Real> q;
            for (int j = 0; j < span.num_insulators(); ++j) {
                q.insert(q.end(), {span.insulator_swing(j) * rad2deg, span.insulator_swing_across(j) * rad2deg, span.insulator_tension(j)});
            }
            m_stats[i].back().accumulate(time + dt, q);
        }
        if (write) { for (const auto& st : m_stats[i]) { st.write(); } }
    }
    // how close the lines come to each other
    measure_separation();
    if (m_step % m_in.diagnostics_int == 0) { write_separation(time + dt, false); }
    if (sample) {
        for (std::size_t p = 0; p < m_pairs.size(); ++p) {
            m_pair_stats[p].accumulate(time + dt, {m_sep[p].distance, m_sep[p].distance < m_in.flashover_distance ? Real(1.0) : Real(0.0)});
            if (write) { m_pair_stats[p].write(); }
        }
    }
    // the lines' load on the transformers they end on, and how close any conductor comes to each
    measure_transformers();
    if (m_step % m_in.diagnostics_int == 0) { write_transformers(time + dt, false); }
    if (sample) {
        for (std::size_t t = 0; t < m_transformers.size(); ++t) {
            const auto& L = m_tload[t];
            const Real d = m_tclear[t].distance;
            m_tstats[t].accumulate(time + dt, {L.horizontal_force, L.overturning_moment, L.over_allowable ? Real(1.0) : Real(0.0),
                                               d, d < m_in.flashover_distance ? Real(1.0) : Real(0.0)});
            if (write) { m_tstats[t].write(); }
        }
    }
    // nothing before stats_start: means of no samples would read as calm air (every line is sampled together)
    if (write && !m_gust_stats.empty() && m_gust_stats.front().num_samples() > 0) {
        for (const auto& st : m_gust_stats) { st.write(); }
        write_gusts();
    }
    // the force the lines and the towers' members exert on the air (minus the air's drag on them),
    // spread into the momentum sources ERF adds over the next step
    if (m_in.drag_on_flow) { spread_drag(U, z_phys_nd, detJ_cc, geom); }
    if (write) { write_total_load(time + dt, first); }
}

std::vector<double>
Conductors::sample_k (const std::vector<Real>& pos, const MultiFab& cons, const MultiFab* z_phys_nd, const Geometry& geom) const
{
    std::vector<Real> rho, rhok;
    erf_actuator::sample_cell_scalar(cons, Rho_comp, z_phys_nd, geom, pos, rho);
    erf_actuator::sample_cell_scalar(cons, RhoKE_comp, z_phys_nd, geom, pos, rhok);
    std::vector<double> k(rho.size());
    for (std::size_t p = 0; p < k.size(); ++p) {
        const double r = static_cast<double>(rho[p]);
        const double rk = static_cast<double>(rhok[p]);
        k[p] = (r > 0.0 && std::isfinite(rk)) ? std::max(rk / r, 0.0) : std::numeric_limits<double>::quiet_NaN();
    }
    return k;
}

void
Conductors::sample_gusts (double t, const std::vector<std::vector<Real>>& uvw, const std::vector<double>& k)
{
    AMREX_ALWAYS_ASSERT(uvw.size() == m_lines.size());
    std::size_t first = 0;   // the line's first node in k
    for (std::size_t i = 0; i < m_lines.size(); ++i) {
        const ConductorLine& line = *m_lines[i];
        const LineInputs& s = m_placed[i];
        std::vector<Real> q;
        q.reserve(3 * static_cast<std::size_t>(s.num_spans()));
        for (int kk = 0; kk < s.num_spans(); ++kk) {
            const auto n = span_normal(i, kk);
            // root-mean-square over the span's nodes: the mean load goes with the mean of the square
            double wind2 = 0.0, normal2 = 0.0, kmean = 0.0;
            const unsigned n0 = line.span_first_node(kk), nn = line.span_num_nodes(kk);
            for (unsigned m = n0; m < n0 + nn; ++m) {
                const double u = static_cast<double>(uvw[i][3*m]), v = static_cast<double>(uvw[i][3*m+1]);
                const double un = u * n[0] + v * n[1];
                wind2 += u * u + v * v;
                normal2 += un * un;
                kmean += k[first + m];
            }
            q.insert(q.end(), {static_cast<Real>(std::sqrt(wind2 / nn)), static_cast<Real>(std::sqrt(normal2 / nn)),
                               static_cast<Real>(kmean / nn)});
        }
        m_gust_stats[i].accumulate(t, q);
        first += line.num_nodes();
    }
}

void
Conductors::setup_gusts (const Geometry& geom)
{
    using erf_conductors::GustType;
    const GustType g = m_in.gust();
    if (g == GustType::Event) {
        constexpr double pi = 3.14159265358979323846;
        const double a = static_cast<double>(m_in.gust_event_direction) * pi / 180.0;
        m_event.time = static_cast<double>(m_in.gust_event_time);
        m_event.speed = static_cast<double>(m_in.gust_event_speed);
        m_event.duration = static_cast<double>(m_in.gust_event_duration);
        m_event.ex = std::cos(a);
        m_event.ey = std::sin(a);
        m_event.x0 = m_in.has_gust_event_origin ? static_cast<double>(m_in.gust_event_origin[0]) : 0.5 * (geom.ProbLo(0) + geom.ProbHi(0));
        m_event.y0 = m_in.has_gust_event_origin ? static_cast<double>(m_in.gust_event_origin[1]) : 0.5 * (geom.ProbLo(1) + geom.ProbHi(1));
        Print() << "erf.conductors: a travelling 1 - cos gust of " << m_in.gust_peak_factor << " sigma_u crosses (" << m_event.x0 << ", "
                << m_event.y0 << ") at t = " << m_event.time << " s, moving at " << m_event.speed << " m/s towards "
                << m_in.gust_event_direction << " degrees from +x and lasting " << m_event.duration << " s at a point; written to "
                << m_in.diagnostics_dir << "/gust_series.dat\n";
    }
    if (g == GustType::Random) {
        // one process per span, numbered line by line; a line that shares another's towers or names it in
        // gust_with takes that line's spans' processes, so that the conductors of a circuit move together; then
        // one per tower
        m_span_process.assign(m_placed.size(), {});
        m_process_span.clear();
        std::vector<std::size_t> owner(m_placed.size());
        for (std::size_t i = 0; i < m_placed.size(); ++i) {
            const std::string& o = m_in.gust_owner(m_placed[i]).name;
            owner[i] = 0;
            while (owner[i] < m_placed.size() && m_placed[owner[i]].name != o) { ++owner[i]; }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owner[i] < m_placed.size(), "Conductors::setup_gusts: no line owns a line's gusts");
            if (owner[i] != i) { continue; }
            for (int k = 0; k < m_placed[i].num_spans(); ++k) {
                m_span_process[i].push_back(m_process_span.size());
                m_process_span.emplace_back(i, k);
            }
        }
        for (std::size_t i = 0; i < m_placed.size(); ++i) {
            if (owner[i] == i) { continue; }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owner[owner[i]] == owner[i] &&
                                             m_span_process[owner[i]].size() == static_cast<std::size_t>(m_placed[i].num_spans()),
                                             "Conductors::setup_gusts: a line taking another's gusts needs as many spans "
                                             "(validate_settings did not run)");
            m_span_process[i] = m_span_process[owner[i]];
        }
        m_gust_z.assign(m_process_span.size() + m_towers.size(), 0.0);
        m_gust_z_set = false;
        // each process's name, line_span<k> or the tower's, so that a restart takes a process's value only for the
        // same process
        m_process_names.clear();
        for (const auto& [i, k] : m_process_span) { m_process_names.push_back(m_placed[i].name + "_span" + std::to_string(k + 1)); }
        for (const auto& t : m_towers) { m_process_names.push_back(t.name()); }
        Print() << "erf.conductors: random gusts, " << m_process_span.size() << " span process(es) and " << m_towers.size()
                << " tower process(es), seed " << m_in.gust_seed << ", integral length ";
        if (m_in.has_gust_integral_length) { Print() << m_in.gust_integral_length << " m"; }
        else { Print() << "IEC 61400-1's 8.1 Lambda_1 at each height"; }
        Print() << "; written to " << m_in.diagnostics_dir << "/gust_series.dat\n";
    }
}

void
Conductors::add_gusts (double time, double dt, const std::vector<std::vector<Real>>& pos, std::vector<std::vector<Real>>& uvw,
                       const std::vector<double>& k, const std::vector<Real>& tpos, std::vector<Real>& tuvw, const std::vector<double>& tk)
{
    const bool event = (m_in.gust() == erf_conductors::GustType::Event);
    const double c = m_gust_sigma, g = static_cast<double>(m_in.gust_peak_factor);
    const double Ls = static_cast<double>(m_in.gust_span_length_scale);
    // the wind MoorDyn and the towers hold over the step is the wind at its middle
    const double t_mid = time + 0.5 * dt;
    // per line, per span: the mean k and the root-mean-square horizontal wind over its nodes
    std::vector<std::vector<double>> kspan(m_lines.size()), uspan(m_lines.size());
    std::size_t first = 0;
    for (std::size_t i = 0; i < m_lines.size(); ++i) {
        const ConductorLine& line = *m_lines[i];
        for (int s = 0; s < line.num_spans(); ++s) {
            const unsigned n0 = line.span_first_node(s), nn = line.span_num_nodes(s);
            double kk = 0.0, u2 = 0.0;
            for (unsigned m = n0; m < n0 + nn; ++m) {
                const double u = static_cast<double>(uvw[i][3*m]), v = static_cast<double>(uvw[i][3*m+1]);
                kk += k[first + m];
                u2 += u * u + v * v;
            }
            kspan[i].push_back(kk / nn);
            uspan[i].push_back(std::sqrt(u2 / nn));
        }
        first += line.num_nodes();
    }
    // per tower, over its drag nodes: the mean k, the root-mean-square horizontal wind, the mean height above the
    // base and the highest node
    const std::size_t nt = m_towers.size();
    std::vector<double> ktow(nt, 0.0), utow(nt, 0.0), zmean(nt, 0.0), ztop(nt, 0.0);
    std::vector<std::size_t> top(nt, 0);
    first = 0;
    for (std::size_t t = 0; t < nt; ++t) {
        const auto& nodes = m_towers[t].nodes();
        double u2 = 0.0;
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            const double u = static_cast<double>(tuvw[3*(first+j)]), v = static_cast<double>(tuvw[3*(first+j)+1]);
            const double z = static_cast<double>(nodes[j].pos[2] - m_towers[t].base()[2]);
            ktow[t] += tk[first + j];
            u2 += u * u + v * v;
            zmean[t] += z;
            if (j == 0 || z > ztop[t]) { ztop[t] = z; top[t] = j; }
        }
        const auto n = static_cast<double>(nodes.size());
        ktow[t] /= n;
        utow[t] = std::sqrt(u2 / n);
        zmean[t] /= n;
        first += nodes.size();
    }
    if (!event) {
        // step every process by dt, T = L_u / U; the first step that needs them draws them from the stationary law
        const auto seed = static_cast<std::uint64_t>(m_in.gust_seed);
        const auto step = static_cast<std::uint64_t>(m_step);
        auto advance_z = [&] (std::size_t p, double height, double U) {
            const double Lu = m_in.has_gust_integral_length ? static_cast<double>(m_in.gust_integral_length)
                                                            : erf_conductors::gust_integral_length(height);
            const double T = (U > 0.0) ? Lu / U : std::numeric_limits<double>::infinity();
            const double xi = erf_conductors::gust_normal(seed, static_cast<std::uint64_t>(p), step);
            m_gust_z[p] = m_gust_z_set ? erf_conductors::gust_ou_step(m_gust_z[p], dt, T, xi) : xi;
        };
        for (std::size_t p = 0; p < m_process_span.size(); ++p) {
            const auto [i, s] = m_process_span[p];
            advance_z(p, span_height(i, s), uspan[i][static_cast<std::size_t>(s)]);
        }
        for (std::size_t t = 0; t < nt; ++t) { advance_z(m_process_span.size() + t, zmean[t], utow[t]); }
        m_gust_z_set = true;
    }
    // the unit vector along the mean horizontal wind of the points [n0, n0 + nn) of w (3 components each): a span's,
    // string's or tower's gust acts along it at all its points; zero where that mean wind is zero
    auto direction = [] (const std::vector<Real>& w, std::size_t n0, std::size_t nn) {
        double u = 0.0, v = 0.0;
        for (std::size_t m = n0; m < n0 + nn; ++m) { u += static_cast<double>(w[3*m]); v += static_cast<double>(w[3*m+1]); }
        const double h = std::hypot(u, v);
        return (h > 0.0) ? std::array<double,2>{{u / h, v / h}} : std::array<double,2>{{0.0, 0.0}};
    };
    // add a gust (m/s) along the horizontal direction d to the wind w (3 components) at a point
    auto push = [] (Real* w, double gust, const std::array<double,2>& d) {
        w[0] = static_cast<Real>(static_cast<double>(w[0]) + gust * d[0]);
        w[1] = static_cast<Real>(static_cast<double>(w[1]) + gust * d[1]);
    };
    m_gust_series.clear();
    for (std::size_t i = 0; i < m_lines.size(); ++i) {
        const ConductorLine& line = *m_lines[i];
        // each span's amplitude g sigma_u (event) or gust sigma_u sqrt(B) z (random)
        std::vector<double> a(static_cast<std::size_t>(line.num_spans()));
        for (int s = 0; s < line.num_spans(); ++s) {
            const auto ss = static_cast<std::size_t>(s);
            const double sigma = c * std::sqrt(kspan[i][ss]);
            a[ss] = event ? g * sigma
                          : sigma * std::sqrt(erf_conductors::gust_background_factor(static_cast<double>(m_placed[i].chord(s)), Ls)) *
                            m_gust_z[m_span_process[i][ss]];
        }
        // the gust at node m of the line from the amplitude A: the event's shape there, at the step's middle
        auto gust_at = [&] (unsigned m, double A) {
            if (!event) { return A; }
            const double x = static_cast<double>(pos[i][3*m]), y = static_cast<double>(pos[i][3*m+1]);
            return A * erf_conductors::gust_event_shape(erf_conductors::gust_event_phase(m_event, x, y, t_mid));
        };
        for (int s = 0; s < line.num_spans(); ++s) {
            const unsigned n0 = line.span_first_node(s), nn = line.span_num_nodes(s);
            const double A = a[static_cast<std::size_t>(s)];
            const auto d = direction(uvw[i], n0, nn);
            for (unsigned m = n0; m < n0 + nn; ++m) { push(&uvw[i][3*m], gust_at(m, A), d); }
            m_gust_series.push_back(gust_at(n0 + (nn - 1) / 2, A));
        }
        for (int j = 0; j < line.num_insulators(); ++j) {
            // string j hangs at tower j + 1, between spans j and j + 1
            const auto jj = static_cast<std::size_t>(j);
            const double A = 0.5 * (a[jj] + a[jj + 1]);
            const unsigned n0 = line.string_first_node(j), nn = line.string_num_nodes(j);
            const auto d = direction(uvw[i], n0, nn);
            for (unsigned m = n0; m < n0 + nn; ++m) { push(&uvw[i][3*m], gust_at(m, A), d); }
        }
    }
    first = 0;
    for (std::size_t t = 0; t < nt; ++t) {
        const double sigma = c * std::sqrt(ktow[t]);
        const double A = event ? g * sigma
                               : sigma * std::sqrt(erf_conductors::tower_background_factor(ztop[t], Ls)) *
                                 m_gust_z[m_process_span.size() + t];
        const auto d = direction(tuvw, first, m_towers[t].nodes().size());
        for (std::size_t j = 0; j < m_towers[t].nodes().size(); ++j) {
            const std::size_t q = 3 * (first + j);
            const double gust = event ? A * erf_conductors::gust_event_shape(erf_conductors::gust_event_phase(
                                                m_event, static_cast<double>(tpos[q]), static_cast<double>(tpos[q+1]), t_mid))
                                      : A;
            push(&tuvw[q], gust, d);
            if (j == top[t]) { m_gust_series.push_back(gust); }
        }
        first += m_towers[t].nodes().size();
    }
}

void
Conductors::write_gust_series (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    if (erf_actuator::open_log(out, m_in.diagnostics_dir + "/gust_series.dat", first)) {
        out << "time";
        for (const LineInputs& s : m_placed) { for (int k = 0; k < s.num_spans(); ++k) { out << " " << s.span_name(k); } }
        for (const auto& t : m_towers) { out << " " << t.name(); }
        out << "\n";
    }
    out << std::setprecision(10) << time;
    for (const double gust : m_gust_series) { out << " " << gust; }
    out << "\n";
}

std::array<double,2>
Conductors::span_normal (std::size_t i, int k) const
{
    const LineInputs& s = m_placed[i];
    const auto& a = s.conductor_point(k);
    const auto& b = s.conductor_point(k + 1);
    const double dx = static_cast<double>(b[0] - a[0]), dy = static_cast<double>(b[1] - a[1]);
    const double h = std::hypot(dx, dy);
    // validate_settings refuses such a span with a gust_type
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(h > 1.0e-6, "Conductors::span_normal: the span has no horizontal extent");
    return {{-dy / h, dx / h}};
}

double
Conductors::span_height (std::size_t i, int k) const
{
    std::size_t ip = 0;
    for (std::size_t j = 0; j < i; ++j) { ip += static_cast<std::size_t>(m_placed[j].num_spans() + 1); }
    const LineInputs& s = m_placed[i];
    const double za = static_cast<double>(s.conductor_point(k)[2] - m_ground[ip + static_cast<std::size_t>(k)]);
    const double zb = static_cast<double>(s.conductor_point(k + 1)[2] - m_ground[ip + static_cast<std::size_t>(k) + 1]);
    return 0.5 * (za + zb);
}

void
Conductors::write_gusts () const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    const std::string file = m_in.diagnostics_dir + "/gusts.csv";
    std::ofstream out(file, std::ios::trunc);
    if (!out) { Abort("cannot write '" + file + "'"); }
    out << "line,span,height,chord,wind,normal_wind,k,sigma,normal_sigma,intensity,gust_response,gust_wind,mean_load,peak_load,valid\n"
        << std::setprecision(10);
    for (std::size_t i = 0; i < m_placed.size(); ++i) {
        const LineInputs& s = m_placed[i];
        const auto& st = m_gust_stats[i];
        for (int k = 0; k < s.num_spans(); ++k) {
            const std::size_t q = 3 * static_cast<std::size_t>(k);
            const double wind = static_cast<double>(st.mean(q));
            // the normal component's root-mean-square never exceeds the speed's; clip roundoff
            const double normal = std::min(static_cast<double>(st.mean(q + 1)), wind);
            const double kmean = static_cast<double>(st.mean(q + 2));
            const auto G = erf_conductors::span_gust(wind, normal, kmean, static_cast<double>(s.chord(k)),
                                                     static_cast<double>(s.diameter), static_cast<double>(s.drag_coefficient),
                                                     static_cast<double>(m_in.air_density), m_gust_sigma,
                                                     static_cast<double>(m_in.gust_peak_factor),
                                                     static_cast<double>(m_in.gust_span_length_scale));
            out << s.name << "," << k + 1 << "," << span_height(i, k) << "," << s.chord(k) << "," << wind << "," << normal << ","
                << kmean << "," << G.sigma << "," << G.normal_sigma << "," << G.intensity << "," << G.gust_response << ","
                << G.gust_wind << "," << G.mean_load << "," << G.peak_load << "," << (G.linear_valid ? 1 : 0) << "\n";
        }
    }
    if (!out) { Abort("cannot write '" + file + "'"); }
}

void
Conductors::spread_drag (const MultiFab& U, const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    const BoxArray ba = amrex::convert(U.boxArray(), IntVect(0,0,0));
    if (!m_src_defined || m_src_x.boxArray() != amrex::convert(ba, IntVect(1,0,0))) {
        const DistributionMapping& dm = U.DistributionMap();
        m_src_x.define(amrex::convert(ba, IntVect(1,0,0)), dm, 1, 0);
        m_src_y.define(amrex::convert(ba, IntVect(0,1,0)), dm, 1, 0);
        m_src_z.define(amrex::convert(ba, IntVect(0,0,1)), dm, 1, 0);
        m_src_defined = true;
    }
    // the force each node exerts on the air is minus the air's drag on it
    std::vector<Real> pos, force;
    for (const auto& span : m_lines) {
        const std::vector<Real> p = span->node_positions();
        pos.insert(pos.end(), p.begin(), p.end());
        for (unsigned n = 0; n < span->num_nodes(); ++n) {
            const auto f = span->node_drag(n);
            force.insert(force.end(), {-f[0], -f[1], -f[2]});
        }
    }
    for (const auto& t : m_towers) {
        const auto nodes = t.current_nodes();
        for (std::size_t n = 0; n < nodes.size(); ++n) {
            const auto& p = nodes[n].pos;
            pos.insert(pos.end(), {p[0], p[1], p[2]});
            force.insert(force.end(), {-t.loads()[3*n], -t.loads()[3*n+1], -t.loads()[3*n+2]});
        }
    }
    {
        std::string err = erf_conductors::first_nonfinite(force, 3, "the force (N) on the air of spread point");
        if (err.empty()) { err = erf_conductors::first_nonfinite(pos, 3, "the position (m) of spread point"); }
        if (!err.empty()) {
            Abort("erf.conductors.drag_on_flow: " + err + " (the line nodes, then the tower nodes); it is not spread into the flow");
        }
    }
    const Real eps = m_in.epsilon * geom.CellSize(0);
    erf_actuator::spread_forces(pos, force, eps, z_phys_nd, detJ_cc, geom, m_src_x, m_src_y, m_src_z);
    m_source_integral = {{erf_actuator::integrate_source(0, m_src_x, detJ_cc, geom),
                          erf_actuator::integrate_source(1, m_src_y, detJ_cc, geom),
                          erf_actuator::integrate_source(2, m_src_z, detJ_cc, geom)}};
}

void
Conductors::add_momentum_sources (int lev, MultiFab& xmom_src, MultiFab& ymom_src, MultiFab& zmom_src) const
{
    if (lev != m_anchor || !m_src_defined) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(xmom_src.boxArray() == m_src_x.boxArray(),
                                     "Conductors: the momentum sources were built on a different grid than ERF's");
    MultiFab::Add(xmom_src, m_src_x, 0, 0, 1, 0);
    MultiFab::Add(ymom_src, m_src_y, 0, 0, 1, 0);
    MultiFab::Add(zmom_src, m_src_z, 0, 0, 1, 0);
}

void
Conductors::cell_sources (int lev, MultiFab& dst, int comp) const
{
    dst.setVal(0.0, comp, 3, 0);
    if (lev != m_anchor || !m_src_defined) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::convert(m_src_x.boxArray(), IntVect(0,0,0)) == dst.boxArray() &&
                                     m_src_x.DistributionMap() == dst.DistributionMap(),
                                     "Conductors::cell_sources: the plot grid differs from the grid the sources were spread on");
    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto sx = m_src_x.const_array(mfi);
        const auto sy = m_src_y.const_array(mfi);
        const auto sz = m_src_z.const_array(mfi);
        const auto d = dst.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            d(i,j,k,comp)   = Real(0.5) * (sx(i,j,k) + sx(i+1,j,k));
            d(i,j,k,comp+1) = Real(0.5) * (sy(i,j,k) + sy(i,j+1,k));
            d(i,j,k,comp+2) = Real(0.5) * (sz(i,j,k) + sz(i,j,k+1));
        });
    }
}

void
Conductors::write_total_load (double time, bool first) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out;
    const bool header = erf_actuator::open_log(out, m_in.diagnostics_dir + "/total_load.dat", first);
    if (header) { out << "time drag_x drag_y drag_z force_on_air_x force_on_air_y force_on_air_z source_x source_y source_z\n"; }
    // the force the lines and the towers' members put into the air is minus the air's drag on them,
    // and only with drag_on_flow
    const Real on = m_in.drag_on_flow ? Real(-1.0) : Real(0.0);
    out << std::setprecision(10) << time;
    for (int d = 0; d < 3; ++d) { out << " " << m_drag_total[d]; }
    for (int d = 0; d < 3; ++d) { out << " " << on * m_drag_total[d]; }
    for (int d = 0; d < 3; ++d) { out << " " << m_source_integral[d]; }
    out << "\n";
}

namespace erf_conductors {

void trim_log_after (const std::string& fname, double t, bool at_too)
{
    std::ifstream in(fname);
    if (!in) { return; }
    std::vector<std::string> kept;
    std::string line;
    bool dropped = false;
    // the logs print the time to ten significant digits
    const double tol = 1.0e-9 * std::max(1.0, std::abs(t));
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        double row_t = 0.0;
        if ((ls >> row_t) && (row_t > t + tol || (at_too && row_t > t - tol))) { dropped = true; continue; }
        kept.push_back(line);
    }
    in.close();
    if (!dropped) { return; }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { Abort("cannot rewrite the conductor log '" + fname + "'"); }
    for (const auto& l : kept) { out << l << "\n"; }
}

std::string coupling_converged (const std::vector<double>& F0, const std::vector<double>& F, const std::vector<double>& Fn,
                                double tolerance, bool& converged)
{
    converged = false;
    for (const auto* v : {&F0, &F, &Fn}) {
        const std::string err = first_nonfinite(*v, 3, "the lines' pull (N) on tower attachment");
        if (!err.empty()) { return err; }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(F.size() == Fn.size(), "coupling_converged: the iterate and the pulls differ in size");
    double scale = 1.0;
    for (const double f : F0) { scale = std::max(scale, std::abs(f)); }
    double rmax = 0.0;
    for (std::size_t c = 0; c < F.size(); ++c) { rmax = std::max(rmax, std::abs(Fn[c] - F[c])); }
    converged = (rmax <= tolerance * scale);
    return std::string();
}

std::string restart_mismatch (bool saved_tower_motion, bool towers_move, double saved_offset, double surface_offset)
{
    if (towers_move && !saved_tower_motion) {
        return "it holds no tower sway, but a tower type of erf.conductors.tower_types has a frequency; the tower "
               "types must match the run being restarted";
    }
    if (saved_tower_motion && !towers_move) {
        return "it holds tower sway, but no tower type of erf.conductors.tower_types has a frequency; the tower types "
               "must match the run being restarted";
    }
    if (std::isfinite(saved_offset) && std::abs(saved_offset - surface_offset) > 1.0e-12 * std::max(1.0, std::abs(surface_offset))) {
        return "erf.conductors.surface_offset (" + std::to_string(surface_offset) + " m) differs from the checkpoint's (" +
               std::to_string(saved_offset) + " m), the frame MoorDyn's saved state is in";
    }
    return std::string();
}

} // namespace erf_conductors
