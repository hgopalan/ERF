#include <ERF_FireLayer.H>
#include <ERF_FireSuppression.H>
#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file ERF_FireCheckpoint.cpp
 * @brief The native fire engine's checkpoint: what FireLayer writes and reads
 *
 * Every field and scalar the native engine carries from one step to the next:
 * fields under Level_<lev> of the level the fire grid refines, the text files
 * FireState (and, with suppression, FireSuppression) at the top. The coupling's own
 * state (the lagged fluxes) is written by FireCoupler::write_checkpoint. The
 * file names and the FireState text file are those ERF wrote before the engine
 * owned its checkpoint, so earlier checkpoints restart unchanged.
 */

using namespace amrex;

void FireLayer::write_checkpoint (const std::string& checkpointname) const
{
    // The fire state lives on the level its grid refines (erf.fire.anchor_level).
    const int fire_lev = m_lev;
    if (const amrex::MultiFab* phi = get_levelset()) {
        amrex::Print() << "Writing fire level-set (phi) to checkpoint" << std::endl;
        VisMF::Write(*phi, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FirePhi"));
    }
    // Save fire arrival time (needed to restore burned interior on restart)
    if (const amrex::MultiFab* at = get_arrival_time()) {
        amrex::Print() << "Writing fire arrival time to checkpoint" << std::endl;
        VisMF::Write(*at, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireArrivalTime"));
    }
    // Balbi with heat_flux_coupling reads the previous step's flux before it is
    // recomputed; without it the first restarted step used a zero flux.
    if (const amrex::MultiFab* hf = get_heat_flux()) {
        VisMF::Write(*hf, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireHeatFlux"));
    }
    // The suppression hold test reads the previous step's flame length before
    // the step recomputes it; without it the first restarted step tested
    // every line against zero flames and held lines that should have failed.
    if (const amrex::MultiFab* fl = get_flame_length()) {
        VisMF::Write(*fl, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireFlameLength"));
    }
    if (const amrex::MultiFab* fuel = get_fuel_load()) {
        amrex::Print() << "Writing fire fuel load to checkpoint" << std::endl;
        VisMF::Write(*fuel, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireFuelLoad"));
    }
    // Fuel moisture evolves under erf.fire.moisture_dynamic, so without it a
    // restart silently reverts to the inputs-file moisture.
    if (const amrex::MultiFab* mc = get_fuel_mc()) {
        amrex::Print() << "Writing fire fuel moisture to checkpoint" << std::endl;
        VisMF::Write(*mc, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireFuelMC"));
    }
    // Shell moistures of the stick model (erf.fire.moisture_model = stick).
    if (const amrex::MultiFab* st = get_stick_mc()) {
        amrex::Print() << "Writing fire moisture stick shells to checkpoint" << std::endl;
        VisMF::Write(*st, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireStickMC"));
    }
    // Sub-cell displacement carried between FARSITE substeps.
    if (const amrex::MultiFab* disp = get_disp_accum()) {
        amrex::Print() << "Writing fire displacement accumulator to checkpoint" << std::endl;
        VisMF::Write(*disp, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireDispAccum"));
    }
    // Crown state; both are null unless erf.fire.crown.enable is set.
    if (const amrex::MultiFab* ca = get_crown_active()) {
        amrex::Print() << "Writing fire crown activation to checkpoint" << std::endl;
        VisMF::Write(*ca, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireCrownActive"));
    }
    if (const amrex::MultiFab* cl = get_crown_load()) {
        amrex::Print() << "Writing fire crown load to checkpoint" << std::endl;
        VisMF::Write(*cl, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireCrownLoad"));
    }
    // The surface precipitation accumulation the rain per column is measured from
    // (erf.fire.precip_source = atmosphere); without it the first restarted step
    // would read the whole accumulation since t = 0 as one step's rain.
    if (const amrex::MultiFab* pa = get_precip_accum_prev()) {
        amrex::Print() << "Writing fire precipitation snapshot to checkpoint" << std::endl;
        VisMF::Write(*pa, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FirePrecipAccumPrev"));
    }
    // Exposure accumulators; all null unless erf.fire.exposure.enable.
    if (const amrex::MultiFab* hl = get_heat_load()) {
        amrex::Print() << "Writing fire exposure accumulators to checkpoint" << std::endl;
        VisMF::Write(*hl, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireHeatLoad"));
    }
    if (const amrex::MultiFab* pk = get_peak_intensity()) {
        VisMF::Write(*pk, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FirePeakIntensity"));
    }
    if (const amrex::MultiFab* em = get_ember_landings()) {
        VisMF::Write(*em, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireEmberLandings"));
    }
    // Structure ignition state (state, ignition time, time above the
    // intensity threshold, cause); null unless erf.fire.structures.ignition.enable.
    if (const amrex::MultiFab* ss = get_structure_state()) {
        amrex::Print() << "Writing fire structure ignition state to checkpoint" << std::endl;
        VisMF::Write(*ss, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireStructureState"));
    }
    // State that carries across steps but was not written before: the spotting
    // diagnostics, which are recomputed only every spotting interval and held
    // in between; the temporal acceleration state; and the crown ROS carried
    // between steps. The crown fraction burned is a per-step diagnostic,
    // written only so the restart plotfile shows it.
    if (const amrex::MultiFab* ad = get_albini_data()) {
        VisMF::Write(*ad, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireAlbiniData"));
    }
    if (const amrex::MultiFab* as = get_accel_state()) {
        VisMF::Write(*as, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireAccelState"));
    }
    if (const amrex::MultiFab* cr = get_crown_ros_active()) {
        VisMF::Write(*cr, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireCrownRosActive"));
    }
    if (const amrex::MultiFab* cf = get_crown_fraction_burned()) {
        VisMF::Write(*cf, MultiFabFileFullPrefix(fire_lev, checkpointname, "Level_", "FireCrownFractionBurned"));
    }
    // The counters: the step (also implied by the atmosphere's) and the level-set
    // subcycle count that schedules the reinitialisation every N subcycles. Without
    // it a restart from a step that is not a multiple of N reinitialised on a
    // different schedule from the uninterrupted run and the fronts diverged.
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f(checkpointname + "/FireState");
        f << "step " << get_step() << "\n"
          << "levelset_subcycle_count " << get_levelset_subcycle_count() << "\n"
          << std::setprecision(17) << "f_dry_prev " << get_f_dry_prev() << "\n"
          // the boundary guard: when the fire first entered the band (-1 if never), so a
          // restart neither repeats the warning nor forgets that it fired
          << "edge_contact_time " << get_edge_contact_time() << "\n"
          << "edge_reach_checked " << (get_edge_reach_checked() ? 1 : 0) << "\n"
          // the lower end of the next ignition window (start of the last step), so
          // scheduled ignitions and the timed perimeter keep the (prev, current]
          // window across the restart instead of the dt-reconstructed one
          << "prev_window_end " << get_prev_window_end() << "\n"
          // where the fire grid is: its level and the lower corner of its region in
          // that level's index space, which a restart must reproduce
          << "anchor_level " << fire_lev << "\n"
          << "fire_region_lo_x " << get_fire_grid().atm_lo[0] << "\n"
          << "fire_region_lo_y " << get_fire_grid().atm_lo[1] << "\n";
        // the "already printed" flags of the arrival-time probes and of the
        // structure first-arrival lines, so a restart does not print them again
        const auto& pr = probe_reported();
        f << "probe_reported " << pr.size();
        for (bool b : pr) { f << " " << (b ? 1 : 0); }
        f << "\n";
        const auto& er = exposure_reported();
        f << "exposure_reported " << er.size();
        for (char c : er) { f << " " << (c ? 1 : 0); }
        f << "\n";
    }
    // Suppression fields, per-line built length, applied-id set and the
    // active actions (erf.fire.suppression.enable).
    if (const FireSuppression* sp = get_suppression()) {
        sp->write_checkpoint(checkpointname, fire_lev);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[FIRE] Fire state written to checkpoint " << checkpointname << "\n";
    }
}

void FireLayer::read_checkpoint (const std::string& restart_chkfile, int max_level, int step)
{
    // The fire state is on the level the fire grid refines. The checkpoint must
    // hold it on the same level and over the same region: reading it onto another
    // grid would put every cell in the wrong place.
    const int fire_lev = m_lev;
    int chk_fire_lev = -1;
    for (int l = 0; l <= max_level; ++l) {
        if (amrex::FileExists(restart_chkfile + "/Level_" + std::to_string(l) + "/FirePhi_H")) {
            chk_fire_lev = l;
            break;
        }
    }
    if (chk_fire_lev < 0) {
        amrex::Print() << "[FIRE] No fire state found in checkpoint; using ignition defaults.\n";
        return;
    }
    if (chk_fire_lev != fire_lev) {
        amrex::Abort("[FIRE] The checkpoint " + restart_chkfile + " holds the fire state on level "
                     + std::to_string(chk_fire_lev) + ", but this run puts the fire grid on level "
                     + std::to_string(fire_lev) + " (erf.fire.anchor_level, the finest level when unset). "
                     "Restart with erf.fire.anchor_level = " + std::to_string(chk_fire_lev)
                     + " and the refinement the checkpoint was written with.");
    }
    // FireState, a small text file of named scalars (and the two flag lists),
    // read once on the I/O rank and broadcast: every rank opening it was a
    // metadata storm on a parallel file system at scale. Keys are fixed, so
    // the parse happens on one rank and the values travel as numbers.
    int    chk_lo_x = 0, chk_lo_y = 0, chk_subcycles = 0, chk_edge_checked = 0;
    amrex::Real chk_f_dry = 1.0, chk_edge_time = -1.0, chk_window_end = 0.0;
    int    has_state = 0, has_subcycles = 0, has_f_dry = 0, has_edge_time = 0, has_edge_checked = 0, has_window = 0;
    std::vector<int> chk_probe_flags, chk_exposure_flags;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ifstream f(restart_chkfile + "/FireState");
        has_state = f ? 1 : 0;
        std::string key;
        while (f >> key) {
            if (key == "fire_region_lo_x")             { f >> chk_lo_x; }
            else if (key == "fire_region_lo_y")        { f >> chk_lo_y; }
            else if (key == "levelset_subcycle_count") { f >> chk_subcycles; has_subcycles = 1; }
            else if (key == "f_dry_prev")              { f >> chk_f_dry; has_f_dry = 1; }
            else if (key == "edge_contact_time")       { f >> chk_edge_time; has_edge_time = 1; }
            else if (key == "edge_reach_checked")      { f >> chk_edge_checked; has_edge_checked = 1; }
            else if (key == "prev_window_end")         { f >> chk_window_end; has_window = 1; }
            else if (key == "probe_reported" || key == "exposure_reported") {
                std::size_t n = 0; f >> n;
                std::vector<int>& v = (key == "probe_reported") ? chk_probe_flags : chk_exposure_flags;
                v.assign(n, 0);
                for (std::size_t i = 0; i < n; ++i) { f >> v[i]; }
            }
            else { std::string skip; f >> skip; }
        }
    }
    {
        int ints[10] = {chk_lo_x, chk_lo_y, chk_subcycles, chk_edge_checked, has_state, has_subcycles,
                        has_f_dry, has_edge_time, has_edge_checked, has_window};
        amrex::ParallelDescriptor::Bcast(ints, 10, amrex::ParallelDescriptor::IOProcessorNumber());
        chk_lo_x = ints[0]; chk_lo_y = ints[1]; chk_subcycles = ints[2]; chk_edge_checked = ints[3];
        has_state = ints[4]; has_subcycles = ints[5]; has_f_dry = ints[6]; has_edge_time = ints[7];
        has_edge_checked = ints[8]; has_window = ints[9];
        amrex::Real reals[3] = {chk_f_dry, chk_edge_time, chk_window_end};
        amrex::ParallelDescriptor::Bcast(reals, 3, amrex::ParallelDescriptor::IOProcessorNumber());
        chk_f_dry = reals[0]; chk_edge_time = reals[1]; chk_window_end = reals[2];
        int sizes[2] = {static_cast<int>(chk_probe_flags.size()), static_cast<int>(chk_exposure_flags.size())};
        amrex::ParallelDescriptor::Bcast(sizes, 2, amrex::ParallelDescriptor::IOProcessorNumber());
        chk_probe_flags.resize(sizes[0]);
        chk_exposure_flags.resize(sizes[1]);
        if (sizes[0] > 0) { amrex::ParallelDescriptor::Bcast(chk_probe_flags.data(), sizes[0], amrex::ParallelDescriptor::IOProcessorNumber()); }
        if (sizes[1] > 0) { amrex::ParallelDescriptor::Bcast(chk_exposure_flags.data(), sizes[1], amrex::ParallelDescriptor::IOProcessorNumber()); }
    }
    {
        // Region: its corner from FireState (checkpoints written before the fire
        // could sit on a finer level have none, and their fire grid is level 0's),
        // its size from the checkpointed level set's boxes.
        const amrex::VisMF chk_phi(amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", "FirePhi"));
        const amrex::Box chk_region = chk_phi.boxArray().minimalBox();
        const amrex::Box run_region = get_levelset()->boxArray().minimalBox();
        const amrex::IntVect& run_lo = get_fire_grid().atm_lo;
        if (chk_region != run_region || chk_lo_x != run_lo[0] || chk_lo_y != run_lo[1]) {
            std::ostringstream msg;
            msg << "[FIRE] The fire grid of checkpoint " << restart_chkfile << " covers " << chk_region
                << " starting at column (" << chk_lo_x << ", " << chk_lo_y << ") of level " << fire_lev
                << ", but this run's covers " << run_region << " starting at column (" << run_lo[0] << ", "
                << run_lo[1] << "). Restart with the refinement and erf.fire.grid_ratio the checkpoint "
                << "was written with.";
            amrex::Abort(msg.str());
        }
    }

    amrex::Print() << "[FIRE] Restoring fire state from checkpoint " << restart_chkfile << "\n";

    // The fire step counter continues from the step count of its level: the spotting
    // random seed is random_seed + step, so without this a restarted run drew
    // different brands from the straight run.
    set_step(step);

    VisMF::Read(*get_levelset_mut(),
        amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", "FirePhi"));
    fire_fill_boundary(*get_levelset_mut(), get_fire_geom());

    VisMF::Read(*get_arrival_time_mut(),
        amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", "FireArrivalTime"));

    VisMF::Read(*get_fuel_load_mut(),
        amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", "FireFuelLoad"));

    // The fields below were added after the original three. A checkpoint written
    // by an older build will not contain them, so each is restored only if it is
    // actually present; anything missing keeps the value initialize() gave it,
    // which is the behaviour every restart had before.
    auto restore_optional = [&] (amrex::MultiFab* mf, const char* name)
    {
        if (mf == nullptr) { return; }
        const std::string header = restart_chkfile + "/Level_" + std::to_string(fire_lev) + "/" + name + "_H";
        if (!amrex::FileExists(header)) {
            amrex::Print() << "[FIRE] Checkpoint has no " << name
                           << "; keeping the initialized values.\n";
            return;
        }
        // A width set by an input (FireStickMC = 3 x erf.fire.stick.n_shells)
        // that changed between the runs used to die inside VisMF::Read on an
        // assert naming neither width nor the input.
        const std::string prefix = amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", name);
        const int ncomp_disk = VisMF(prefix).nComp();
        // FireDispAccum grew from 2 to 4 components in 2026-10 (the FARSITE
        // distance clock and the clock at burn): an older checkpoint's two
        // are read and the clock starts from zero, which only means that for
        // one source stencil a cell burned before the restart under-counts
        // the distance a direct source may claim (the quadrant update covers it).
        if (std::string(name) == "FireDispAccum" && ncomp_disk == 2 && mf->nComp() == 4) {
            amrex::MultiFab two(mf->boxArray(), mf->DistributionMap(), 2, mf->nGrowVect());
            VisMF::Read(two, prefix);
            mf->setVal(0.0);
            mf->setVal(-1.0, 3, 1);   // the clock at burn: stamped by the next update from the restarted clock
            amrex::MultiFab::Copy(*mf, two, 0, 0, 2, 0);
            amrex::Print() << "[FIRE] Checkpoint FireDispAccum has 2 components (written before 2026-10);"
                           << " the FARSITE distance clock restarts from zero.\n";
            return;
        }
        if (ncomp_disk != mf->nComp()) {
            amrex::Abort("[FIRE] checkpoint field " + std::string(name) + " has " + std::to_string(ncomp_disk)
                         + " components but this run allocates " + std::to_string(mf->nComp())
                         + (std::string(name) == "FireStickMC"
                            ? " (3 x erf.fire.stick.n_shells: restart with the n_shells the checkpoint was written with)"
                            : ""));
        }
        VisMF::Read(*mf, prefix);
    };

    restore_optional(get_fuel_mc_mut(),      "FireFuelMC");
    // With erf.fire.moisture_live_model = "fixed" the live classes (components 3
    // and 4) are held at erf.fire.moisture_live, so they come from the inputs:
    // a restart may change the value, and a checkpoint written by a "legacy" run
    // or an older build holds live moisture pinned to [0.30, 0.40]. "legacy"
    // keeps the checkpointed values, which its update evolves.
    if (m_params.moisture_live_model == "fixed") {
        if (amrex::MultiFab* mc = get_fuel_mc_mut(); mc != nullptr && mc->nComp() >= 5) {
            mc->setVal(m_params.moisture_live, 3, 2);
        }
    }
    // The stick shells are allocated on the first advance; allocate them now
    // so the checkpoint's shells replace the fresh initialisation.
    if (m_params.moisture_model == "stick" && get_stick_mc() == nullptr) {
        allocate_stick_mc();
    }
    restore_optional(get_stick_mc_mut(),     "FireStickMC");
    restore_optional(get_disp_accum_mut(),   "FireDispAccum");
    restore_optional(get_crown_active_mut(), "FireCrownActive");
    restore_optional(get_crown_load_mut(),   "FireCrownLoad");
    restore_optional(get_heat_flux_mut(),      "FireHeatFlux");
    restore_optional(get_flame_length_mut(),   "FireFlameLength");
    // The precipitation snapshot (erf.fire.precip_source = atmosphere). A checkpoint
    // without it (an older build, or a run that used the uniform rate) cannot give
    // the first restarted step a rain rate: that step reports none and starts the
    // snapshot from its own accumulation, instead of reading the accumulation since
    // t = 0 as one step's rain.
    if (amrex::MultiFab* pa = get_precip_accum_prev_mut()) {
        const std::string header = restart_chkfile + "/Level_" + std::to_string(fire_lev) + "/FirePrecipAccumPrev_H";
        if (amrex::FileExists(header)) {
            VisMF::Read(*pa, amrex::MultiFabFileFullPrefix(fire_lev, restart_chkfile, "Level_", "FirePrecipAccumPrev"));
        } else {
            amrex::Print() << "[FIRE] Checkpoint has no FirePrecipAccumPrev; the first restarted step"
                           << " reports no rain and starts the precipitation snapshot from its own accumulation.\n";
            invalidate_precip_accum_prev();
        }
    }
    restore_optional(get_heat_load_mut(),      "FireHeatLoad");
    restore_optional(get_peak_intensity_mut(), "FirePeakIntensity");
    restore_optional(get_ember_landings_mut(), "FireEmberLandings");
    // The per-structure state that drives the ignition rule is rebuilt from
    // the field on every rank once it is read.
    if (get_structure_state_mut() != nullptr) {
        restore_optional(get_structure_state_mut(), "FireStructureState");
        restore_structure_ignition_state();
    }
    restore_optional(get_albini_data_mut(),           "FireAlbiniData");
    restore_optional(get_accel_state_mut(),           "FireAccelState");
    restore_optional(get_crown_ros_active_mut(),      "FireCrownRosActive");
    restore_optional(get_crown_fraction_burned_mut(), "FireCrownFractionBurned");

    // The level-set subcycle count, so the reinitialisation keeps the schedule of
    // the uninterrupted run; a checkpoint without FireState leaves it at zero,
    // which is right only when the checkpoint step is a multiple of the interval.
    if (has_state) {
        if (has_subcycles)    { set_levelset_subcycle_count(chk_subcycles); }
        if (has_f_dry)        { set_f_dry_prev(chk_f_dry); }   // the smoke emission divides the lagged flux by it
        if (has_edge_time)    { set_edge_contact_time(chk_edge_time); }
        if (has_edge_checked) { set_edge_reach_checked(chk_edge_checked != 0); }
        if (has_window)       { set_prev_window_end(chk_window_end); }
        // the probes and structures already reported before the checkpoint
        if (!chk_probe_flags.empty()) {
            std::vector<bool> pr(chk_probe_flags.size());
            for (std::size_t n = 0; n < pr.size(); ++n) { pr[n] = (chk_probe_flags[n] != 0); }
            set_probe_reported(pr);
        }
        if (!chk_exposure_flags.empty()) {
            std::vector<char> er(chk_exposure_flags.size());
            for (std::size_t n = 0; n < er.size(); ++n) { er[n] = static_cast<char>(chk_exposure_flags[n] != 0); }
            set_exposure_reported(er);
        }
    } else {
        amrex::Print() << "[FIRE] Checkpoint has no FireState; the reinitialisation"
                       << " schedule restarts from the checkpoint step.\n";
    }
    // Suppression: the checkpoint's actions (with geometry and state) replace the
    // file's copies of the same ids; the file only adds new ids, as at start-up.
    if (FireSuppression* sp = get_suppression_mut()) {
        sp->read_checkpoint(restart_chkfile, fire_lev);
    }
}
