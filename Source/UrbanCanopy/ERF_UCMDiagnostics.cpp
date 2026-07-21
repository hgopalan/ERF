/**
 * @file ERF_UCMDiagnostics.cpp
 * @brief Implementation of diagnostics CSV writer for UCM statistics
 *
 * References:
 *  - Source/Dust/ERF_DustDiagnostics.cpp
 */

#include <UrbanCanopy/ERF_UCMDiagnostics.H>
#include <AMReex_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <iomanip>
#include <cmath>

UCMDiagnostics::UCMDiagnostics(const UCMParams& params, int lev)
    : m_params(params), m_lev(lev)
{
    // Build full path to diagnostics file
    m_diag_file = params.ucm_diag_file;
    if (m_diag_file.empty()) {
        m_diag_file = "ucm_diag.dat";
    }
    // TODO: In full ERF integration, prepend erf.plot_file_base if available

    // Initialize file on IO rank
    if (amrex::ParallelDescriptor::IOProcessor()) {
        initialize_file();
    }
}

UCMDiagnostics::~UCMDiagnostics() = default;

void UCMDiagnostics::initialize_file()
{
    // Check if file exists and is non-empty
    std::ifstream ifs(m_diag_file.c_str());
    bool file_exists = ifs.good();
    ifs.close();

    // Open file for append; if new, write header
    std::ofstream ofs(m_diag_file.c_str(), std::ios::app);

    if (!file_exists || ofs.tellp() == 0) {
        // Write header
        ofs << "step,time_s,T_skin_roof_max,T_skin_wall_max,T_skin_road_max,";
        ofs << "T_canyon_max,H_sensible_max,H_sensible_sum,LE_latent_max\n";
    }

    ofs.close();
}

void UCMDiagnostics::write_row(int nstep, amrex::Real time,
                               amrex::Real T_roof_max, amrex::Real T_wall_max, amrex::Real T_road_max,
                               amrex::Real T_canyon_max, amrex::Real H_max, amrex::Real H_sum,
                               amrex::Real LE_max)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream ofs(m_diag_file.c_str(), std::ios::app);
    ofs << std::setprecision(10);
    ofs << nstep << ","
        << time << ","
        << T_roof_max << ","
        << T_wall_max << ","
        << T_road_max << ","
        << T_canyon_max << ","
        << H_max << ","
        << H_sum << ","
        << LE_max << "\n";
    ofs.close();
}

void UCMDiagnostics::append(const UCMFields& fields, int nstep, amrex::Real time, int lev)
{
    // Duplicate-write guard
    if (nstep == m_last_write_step) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.4][UCMDiagnostics::append] (skipped, already written at step " << nstep << ")\n";
        }
        return;
    }
    m_last_write_step = nstep;

    // Check that required fields exist
    if (!fields.T_skin_roof || !fields.T_skin_wall || !fields.T_skin_road ||
        !fields.T_canyon_air || !fields.H_sensible || !fields.LE_latent ||
        !fields.is_urban) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.4][UCMDiagnostics::append] ERROR: One or more required fields is nullptr\n";
        }
        return;
    }

    // Compute statistics on all ranks, then reduce to IO rank
    amrex::Real T_roof_max_local = -std::numeric_limits<amrex::Real>::max();
    amrex::Real T_wall_max_local = -std::numeric_limits<amrex::Real>::max();
    amrex::Real T_road_max_local = -std::numeric_limits<amrex::Real>::max();
    amrex::Real T_canyon_max_local = -std::numeric_limits<amrex::Real>::max();
    amrex::Real H_max_local = -std::numeric_limits<amrex::Real>::max();
    amrex::Real H_sum_local = 0.0;
    amrex::Real LE_max_local = -std::numeric_limits<amrex::Real>::max();

    // Iterate over boxes to compute statistics
    for (amrex::MFIter mfi(*fields.T_skin_roof, false); mfi.isValid(); ++mfi) {
        auto roof_a    = fields.T_skin_roof->const_array(mfi);
        auto wall_a    = fields.T_skin_wall->const_array(mfi);
        auto road_a    = fields.T_skin_road->const_array(mfi);
        auto canyon_a  = fields.T_canyon_air->const_array(mfi);
        auto h_a       = fields.H_sensible->const_array(mfi);
        auto le_a      = fields.LE_latent->const_array(mfi);
        auto urban_a   = fields.is_urban->const_array(mfi);

        const amrex::Box& bx = mfi.validbox();

        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpSum,
                         amrex::ReduceOpMax> reduce_op;

        auto r = amrex::ParallelReduce(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real, amrex::Real, amrex::Real,
                                   amrex::Real>
            {
                if (urban_a(i, j, 0) == 0) {
                    return amrex::MakeTuple(
                        -std::numeric_limits<amrex::Real>::max(),
                        -std::numeric_limits<amrex::Real>::max(),
                        -std::numeric_limits<amrex::Real>::max(),
                        -std::numeric_limits<amrex::Real>::max(),
                        -std::numeric_limits<amrex::Real>::max(),
                        amrex::Real(0.0),
                        -std::numeric_limits<amrex::Real>::max()
                    );
                }
                return amrex::MakeTuple(
                    roof_a(i, j, 0, 0),
                    wall_a(i, j, 0, 0),
                    road_a(i, j, 0, 0),
                    canyon_a(i, j, 0, 0),
                    h_a(i, j, 0, 0),
                    h_a(i, j, 0, 0),  // sum component
                    le_a(i, j, 0, 0)
                );
            },
            reduce_op);

        T_roof_max_local   = std::max(T_roof_max_local,   amrex::get<0>(r));
        T_wall_max_local   = std::max(T_wall_max_local,   amrex::get<1>(r));
        T_road_max_local   = std::max(T_road_max_local,   amrex::get<2>(r));
        T_canyon_max_local = std::max(T_canyon_max_local, amrex::get<3>(r));
        H_max_local        = std::max(H_max_local,        amrex::get<4>(r));
        H_sum_local       += amrex::get<5>(r);
        LE_max_local       = std::max(LE_max_local,       amrex::get<6>(r));
    }

    // Reduce across MPI ranks
    amrex::ParallelDescriptor::ReduceRealMax(T_roof_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(T_wall_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(T_road_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(T_canyon_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(H_max_local);
    amrex::ParallelDescriptor::ReduceRealSum(H_sum_local);
    amrex::ParallelDescriptor::ReduceRealMax(LE_max_local);

    // Write to file on IO rank
    if (amrex::ParallelDescriptor::IOProcessor()) {
        write_row(nstep, time,
                 T_roof_max_local, T_wall_max_local, T_road_max_local,
                 T_canyon_max_local, H_max_local, H_sum_local, LE_max_local);

        // Debug trace
        amrex::Print() << "[UCM][1.4][UCMDiagnostics::append]\n";
        amrex::Print() << "  step=" << nstep << " time=" << time << "\n";
        amrex::Print() << "  T_skin_roof_max=" << T_roof_max_local
                       << " T_canyon_max=" << T_canyon_max_local << "\n";
        amrex::Print() << "  H_sensible_max=" << H_max_local
                       << " H_sensible_sum=" << H_sum_local << "\n";
    }
}
