#ifndef ERF_PHASE_TIMERS_H
#define ERF_PHASE_TIMERS_H

/** Wall-clock time of each phase of the moving bodies, for the summary erf.moving_bodies.timing_int asks for. */

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>

namespace erf_actuator {

/**
 * The seconds this rank spends in each phase of the moving bodies, summed from one timing summary to
 * the next. Off (erf.moving_bodies.timing_int = 0, the default) it records nothing. On, every timed phase
 * waits for the device when it starts and when it ends (a no-op on the CPU), so that it is charged for its
 * own kernels only, not for those queued before it nor only for their launches; the answers do not change.
 * A phase that holds a collective (an exchange, the gather of the logs) includes the wait for the slowest rank.
 *
 * The phases nest: advance holds checks, sample, openfast, diagnostics, spread, statistics and wake;
 * startup, add_sources, check_levels and checkpoint lie outside it.
 */
class PhaseTimers
{
public:
    enum Phase : int {
        Startup = 0,   //!< the constructor, set_ground() and read_checkpoint()
        Advance,       //!< the whole of advance() on the anchor level
        Checks,        //!< the coverage, density and turbine audits (first step, regrid) and a restart's wake-line set-up
        Sample,        //!< sampling the flow at every body's points, exchange included
        OpenFAST,      //!< OpenFAST's first solution and its step
        Diagnostics,   //!< the time-series logs, the farm totals and the source integrals, with their gather
        Spread,        //!< spreading the forces onto the momentum sources, exchange included
        Statistics,    //!< the running statistics and their files
        Wake,          //!< sampling and writing the wake lines (and setting them up on a fresh start)
        AddSources,    //!< adding the sources to ERF's right-hand sides, every RK stage
        CheckLevels,   //!< the finer-level check ERF calls every coarse step
        Checkpoint,    //!< writing the bodies' checkpoint
        NumPhases
    };

    static const char* name (int p)
    {
        static const std::array<const char*, NumPhases> names{{
            "startup", "advance", "checks", "sample", "openfast", "diagnostics", "spread", "statistics", "wake",
            "add_sources", "check_levels", "checkpoint"}};
        return names[static_cast<std::size_t>(p)];
    }

    void enable (bool on) { m_on = on; }
    bool enabled () const { return m_on; }
    void add (int p, double s) { m_seconds[static_cast<std::size_t>(p)] += s; }
    const std::array<double, NumPhases>& seconds () const { return m_seconds; }
    void reset () { m_seconds.fill(0.0); }

    /**
     * The summary text, from the per-phase minimum, sum and maximum over nranks ranks of the seconds
     * spent from step first_step to last_step. A phase no rank spent time in is left out.
     */
    static std::string summary (int first_step, int last_step, int nranks,
                                const double* mn, const double* sum, const double* mx)
    {
        std::ostringstream out;
        const int nsteps = last_step - first_step + 1;
        out << "Moving bodies timing, steps " << first_step << " to " << last_step << " (" << nsteps
            << (nsteps == 1 ? " step" : " steps") << "): wall-clock seconds, min / mean / max over " << nranks
            << (nranks == 1 ? " rank\n" : " ranks\n");
        out << std::scientific << std::setprecision(3);
        for (int p = 0; p < NumPhases; ++p) {
            if (!(mx[p] > 0.0)) { continue; }
            out << "  " << std::left << std::setw(13) << name(p) << std::right
                << std::setw(11) << mn[p] << std::setw(11) << sum[p] / nranks << std::setw(11) << mx[p] << "\n";
        }
        return out.str();
    }

    /**
     * The line on the OpenFAST step of each turbine, from its seconds over the same steps: the fastest,
     * the mean and the slowest turbine. Empty without turbines.
     */
    static std::string turbine_summary (const std::vector<std::string>& names, const std::vector<double>& sec)
    {
        if (sec.empty()) { return std::string(); }
        const auto lo = std::min_element(sec.begin(), sec.end());
        const auto hi = std::max_element(sec.begin(), sec.end());
        double sum = 0.0;
        for (const double s : sec) { sum += s; }
        std::ostringstream out;
        out << std::scientific << std::setprecision(3)
            << "  OpenFAST step per turbine (" << sec.size() << (sec.size() == 1 ? " turbine" : " turbines") << "): min " << *lo
            << " (" << names[static_cast<std::size_t>(lo - sec.begin())] << "), mean " << sum / static_cast<double>(sec.size())
            << ", max " << *hi << " (" << names[static_cast<std::size_t>(hi - sec.begin())] << ")\n";
        return out.str();
    }

private:
    bool m_on{false};
    std::array<double, NumPhases> m_seconds{};
};

/** Charges the time from its construction to its end to one phase of a PhaseTimers, when they are on. */
class ScopedPhase
{
public:
    ScopedPhase (PhaseTimers& t, int phase)
        : m_t(t.enabled() ? &t : nullptr), m_phase(phase)
    {
        if (m_t == nullptr) { return; }
        amrex::Gpu::synchronize();
        m_start = amrex::ParallelDescriptor::second();
    }
    ~ScopedPhase ()
    {
        // an abort that throws (amrex.throw_exception) passes through here: no device call then
        if (m_t == nullptr || std::uncaught_exceptions() > 0) { return; }
        amrex::Gpu::synchronize();
        m_t->add(m_phase, amrex::ParallelDescriptor::second() - m_start);
    }
    ScopedPhase (const ScopedPhase&) = delete;
    ScopedPhase& operator= (const ScopedPhase&) = delete;

private:
    PhaseTimers* m_t;
    int m_phase;
    double m_start{0.0};
};

} // namespace erf_actuator

#endif
