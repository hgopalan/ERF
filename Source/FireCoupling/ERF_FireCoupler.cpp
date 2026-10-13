#include <ERF_FireCoupler.H>
#include <ERF_FireAtmCoupling.H>
#include <ERF_FirePrerequisites.H>
#include <ERF_FirePrecip.H>
#include <ERF_FireSmokeEmission.H>
#include <ERF_FireUtils.H>
#include <ERF_IndexDefines.H>
#include <ERF.H>
#include <AMReX_Print.H>

using namespace amrex;

namespace {

// Where the fire grid sits, printed once at start-up, and the warning when a
// finer level covers part of it: the fire then reads that level's coarser wind,
// and its heat on this level is replaced by the finer level's average-down.
void report_fire_region(const ERF& erf, int lev, const FireGrid& fg)
{
    const Geometry& g = fg.geom;
    amrex::Print() << "[FIRE] Fire grid on level " << lev << " of 0-" << erf.finestLevel()
                   << ": x " << g.ProbLo(0) << " to " << g.ProbHi(0)
                   << " m, y " << g.ProbLo(1) << " to " << g.ProbHi(1) << " m, "
                   << g.Domain().length(0) << " x " << g.Domain().length(1) << " fire cells of "
                   << g.CellSize(0) << " x " << g.CellSize(1) << " m\n";

    if (lev >= erf.finestLevel()) { return; }

    // Share of the fire region that level lev + 1 covers, in columns of level lev.
    const Box region(IntVect(fg.atm_lo[0], fg.atm_lo[1], 0),
                     IntVect(fg.atm_lo[0] + g.Domain().length(0) / fg.C - 1,
                             fg.atm_lo[1] + g.Domain().length(1) / fg.C - 1, 0));
    BoxArray fine = erf.boxArray(lev + 1);
    fine.coarsen(erf.refRatio(lev));
    amrex::Long covered = 0;
    for (int n = 0; n < fine.size(); ++n) {
        Box b = fine[n];
        b.setSmall(2, 0);
        b.setBig(2, 0);
        b &= region;
        if (b.ok()) { covered += b.numPts(); }
    }
    const amrex::Real share = amrex::Real(covered) / amrex::Real(region.numPts());
    amrex::Print() << "[FIRE] WARNING: the fire grid is on level " << lev << " but the finest level is "
                   << erf.finestLevel() << " (erf.fire.anchor_level = " << lev << "), and level " << lev + 1
                   << " covers " << 100.0 * share << " % of the fire grid. The fire reads the wind of level "
                   << lev << " and its heat, moisture and smoke go into level " << lev << " only: with "
                   << "erf.coupling_type = TwoWay the average-down from the finer level replaces the heated cells "
                   << "under it, so that heat is lost, and with OneWay the finer level never sees the fire. "
                   << "Leave erf.fire.anchor_level unset to put the fire grid on the finest level.\n";
}

} // namespace

FireCoupler::FireCoupler (const FireCouplingParams& params, std::unique_ptr<FireEngine> engine)
    : m_params(params), m_engine(std::move(engine))
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_engine != nullptr, "[FIRE] FireCoupler needs a fire engine");
}

void
FireCoupler::initialize (const ERF& erf,
                         int lev,
                         const SurfaceLayer* surface_layer,
                         const MultiFab* z_phys_nd_atm)
{
    // The ERF configuration a fire grid needs, then the engine's own inputs
    verify_fire_prerequisites(erf, lev, surface_layer, m_params.grid_ratio);
    m_engine->verify_prerequisites(erf, lev);

    FireGrid fg = create_fire_grid(erf.boxArray(lev), erf.DistributionMap(lev),
                                   erf.Geom(lev), m_params.grid_ratio);
    fg.lev = lev;
    report_fire_region(erf, lev, fg);
    set_fire_grid(lev, fg, erf.boxArray(lev), erf.DistributionMap(lev),
                  erf.Geom(lev).Domain().length(2));

    m_engine->initialize(erf, lev, m_fg, surface_layer, z_phys_nd_atm);
}

void
FireCoupler::set_fire_grid (int lev, const FireGrid& fg,
                            const BoxArray& ba_atm,
                            const DistributionMapping& dm_atm,
                            int nz)
{
    m_lev = lev;
    m_fg  = fg;

    // The lagged fluxes live on the k = 0 slabs of the level's boxes, with the
    // level's owners. Ghost columns for the MRF fire thermal excess, which reads
    // the fluxes in the halo around every tile. ERF's state carries
    // ComputeGhostCells() + 1 columns, at most 5, and ComputeDiffusivityMRF
    // keeps its halo within the state's and checks these fluxes against it;
    // update_flux_buffer() fills them.
    BoxArray ba_atm_2d = ba_atm;
    ba_atm_2d.coarsen(IntVect(1, 1, nz));
    const IntVect ng_flux(5, 5, 0);
    m_Q_atm_prev     = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, ng_flux);
    m_Q_lat_atm_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, ng_flux);
    m_Q_atm_prev->setVal(0.0_rt);
    m_Q_lat_atm_prev->setVal(0.0_rt);
}

void
FireCoupler::advance (Real time, Real dt,
                      const MultiFab& S_old,
                      const MultiFab& xvel_old, const MultiFab& yvel_old,
                      const MultiFab& xvel_new, const MultiFab& yvel_new,
                      const MultiFab& z_phys_cc,
                      const BoxArray& ba2d,
                      const Geometry& geom,
                      const SurfacePrecipAccumulationSources& precip_sources)
{
    // Air temperature and RH at k=0 for the fuel moisture update, derived
    // from the pre-dycore state. This is used in both lagged and synchronous
    // modes: the moisture ODE uses time-averaged T, and the change in T from
    // fire heating within one atmospheric step is small relative to the
    // moisture time lag. The temperature is theta * Exner(p) (the moisture
    // curves, Balbi's T_a and the threshold ignition take an air
    // temperature); the plain k = 0 potential temperature used here before
    // was 14-16 K too warm at 840 hPa.
    MultiFab T_atm_k0(ba2d, S_old.DistributionMap(), 1, 0);
    compute_t_from_conservative(T_atm_k0, S_old);
    MultiFab RH_atm_k0(ba2d, S_old.DistributionMap(), 1, 0);
    compute_rh_from_conservative(RH_atm_k0, S_old, geom);

    // Surface precipitation accumulation of every column after this step's
    // microphysics, for an engine that takes the rain per column (the native
    // engine with erf.fire.precip_source = atmosphere); the engine differences
    // it against the accumulation it saw at its previous step.
    std::unique_ptr<MultiFab> precip_accum_k0;
    if (m_engine->wants_surface_precip()) {
        precip_accum_k0 = std::make_unique<MultiFab>(ba2d, S_old.DistributionMap(), 1, 0);
        fire_surface_precip_accum_k0(*precip_accum_k0, precip_sources, geom.Domain().smallEnd(2));
    }

    // In passive and lagged modes, fire uses pre-dycore wind (vars_old).
    // In synchronous mode, fire uses post-dycore wind (vars_new) so that
    // the fire spread rate at step n+1 reflects the atmospheric momentum
    // response to fire heating from step n.
    const bool post_dycore = m_params.is_synchronous();
    if (m_params.fire_debug) {
        const char* mode = m_params.is_synchronous() ? "SYNCHRONOUS coupling with POST-dycore"
                         : m_params.is_lagged()      ? "LAGGED coupling with PRE-dycore"
                                                     : "PASSIVE coupling with PRE-dycore";
        amrex::Print() << "[FIRE DEBUG] Fire advance using " << mode << " wind at t="
                       << time << ", dt=" << dt << std::endl;
    }
    const FireForcing forcing{time, dt,
                              post_dycore ? xvel_new : xvel_old,
                              post_dycore ? yvel_new : yvel_old,
                              z_phys_cc, T_atm_k0, RH_atm_k0, precip_accum_k0.get()};
    m_engine->advance(forcing);

    // Store the fire flux of this step for injection at the next step via the
    // Runge-Kutta source term (add_sources). This applies in both lagged and
    // synchronous modes; in passive mode the buffer keeps its zero unless the
    // smoke tracer needs it.
    update_flux_buffer(geom);
}

void
FireCoupler::update_flux_buffer (const Geometry& geom_atm)
{
    if (!m_params.injects_flux() && !m_params.smoke_enable) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping flux buffer update: coupling_type is passive (injects_flux=false) and smoke_enable=false" << std::endl;
        }
        return;
    }

    FireFluxes fluxes;
    m_engine->fill_fluxes(fluxes);
    if (!fluxes.sensible || !m_Q_atm_prev) { return; }

    coarsen_fire_flux_to_atm(*m_Q_atm_prev, *fluxes.sensible, geom_atm, m_fg);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Sensible flux to the atmosphere: max " << m_Q_atm_prev->max(0)
                       << " W/m2" << fluxes.debug_note << std::endl;
    }

    if (m_params.inject_latent && fluxes.latent) {
        coarsen_fire_flux_to_atm(*m_Q_lat_atm_prev, *fluxes.latent, geom_atm, m_fg);
    } else {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping latent heat injection (inject_latent=false or the engine reports no latent flux)" << std::endl;
        }
        m_Q_lat_atm_prev->setVal(0.0_rt);
    }

    // The MRF fire thermal excess reads these fluxes in the halo columns around
    // every tile; give them the neighbouring box's value (and the edge column's
    // outside a non-periodic face), so a column sees the same flux whichever box
    // holds it. On a finer level the columns outside the refined region have no
    // fire grid and keep the zero they were allocated with.
    fire_fill_boundary(*m_Q_atm_prev, geom_atm);
    fire_fill_boundary(*m_Q_lat_atm_prev, geom_atm);
}

void
FireCoupler::add_sources (MultiFab& cc_source,
                          const MultiFab& S_old,
                          const MultiFab& z_phys_cc,
                          const MultiFab* z_phys_nd,
                          const Geometry& geom,
                          bool has_moisture,
                          int step)
{
    // Heat and vapour, with the WRF-Fire profile (ERF_FireAtmCoupling.H)
    if (m_params.injects_flux() && m_Q_atm_prev && m_params.fire_atm_feedback > 0.0_rt) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Applying fire coupling to atmosphere (coupling_type="
                           << m_params.coupling_type << ", feedback=" << m_params.fire_atm_feedback
                           << ", max_Q_prev=" << m_Q_atm_prev->max(0) << " W/m2)" << std::endl;
        }

        const MultiFab* Q_lat_ptr = (m_params.inject_latent && has_moisture)
            ? m_Q_lat_atm_prev.get() : nullptr;

        // Pass fire_debug so apply_fire_tendency_to_cc_source prints tendency diagnostics.
        // With fire_debug=true you will see per-RK-stage output:
        //   [FIRE COUPLING] Q_atm_max=...  alfg=...
        //   [FIRE COUPLING] RhoTheta tendency sum=...  max=...  expected_surface_max=...
        apply_fire_tendency_to_cc_source(
            cc_source,
            *m_Q_atm_prev,
            Q_lat_ptr,
            z_phys_cc,
            S_old,
            geom,
            m_params.heat_flux_alfg,
            m_params.fire_atm_feedback,
            has_moisture,
            m_params.fire_debug,
            m_params.source_mode == "add",
            m_engine->heat_open_fraction_atm(),
            m_engine->roof_height_atm(),
            m_params.heat_open_fraction,
            m_params.heat_tendency_density,
            m_params.heat_tendency_exner);
    }

    // The smoke tracer, after the heat: proportional to the fuel burnt, which
    // the engine's heat per kilogram recovers from the lagged sensible flux
    if (m_params.smoke_enable && m_Q_atm_prev) {
        inject_smoke_from_fire(
            cc_source,
            *m_Q_atm_prev,
            z_phys_nd,
            geom,
            m_params.smoke_emission_factor,
            m_engine->smoke_heat_per_kg(),
            RhoSmoke_comp,
            m_params.fire_debug,
            step);
    }
}
