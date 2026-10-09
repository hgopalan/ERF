#ifdef ERF_USE_DUST

#include <ERF_PhreeqcReader.H>
#include <ERF_DustGrid.H>
#include <ERF_DustParams.H>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

using namespace amrex;

// ---------------------------------------------------------------------------
// Internal helper: bilinear interpolation from a (nx_src x ny_src) flat
// row-major array onto a (nx_dst x ny_dst) flat row-major array.
// Coordinates are cell-centred: src cell i maps to physical x = (i+0.5)/nx_src.
// ---------------------------------------------------------------------------
static void bilinear_interpolate(const std::vector<Real>& src,
                                  int nx_src, int ny_src,
                                  std::vector<Real>&       dst,
                                  int nx_dst, int ny_dst)
{
    dst.resize(nx_dst * ny_dst);

    for (int jd = 0; jd < ny_dst; ++jd) {
        for (int id = 0; id < nx_dst; ++id) {
            // Map dst cell centre to src fractional index
            Real xs = (id + 0.5) * nx_src / nx_dst - 0.5;
            Real ys = (jd + 0.5) * ny_src / ny_dst - 0.5;

            int i0 = static_cast<int>(std::floor(xs));
            int j0 = static_cast<int>(std::floor(ys));
            int i1 = i0 + 1;
            int j1 = j0 + 1;

            // Clamp to valid range
            i0 = std::max(0, std::min(i0, nx_src - 1));
            i1 = std::max(0, std::min(i1, nx_src - 1));
            j0 = std::max(0, std::min(j0, ny_src - 1));
            j1 = std::max(0, std::min(j1, ny_src - 1));

            Real tx = xs - std::floor(xs);
            Real ty = ys - std::floor(ys);

            Real v00 = src[j0 * nx_src + i0];
            Real v10 = src[j0 * nx_src + i1];
            Real v01 = src[j1 * nx_src + i0];
            Real v11 = src[j1 * nx_src + i1];

            dst[jd * nx_dst + id] =
                (1.0 - tx) * (1.0 - ty) * v00 +
                       tx  * (1.0 - ty) * v10 +
                (1.0 - tx) *        ty  * v01 +
                       tx  *        ty  * v11;
        }
    }
}

bool read_phreeqc_csv(MultiFab& mf, const DustGrid& dg,
                      const std::string& filename,
                      const std::string& col_name,
                      Real nodata_fill)
{
    if (filename.empty()) return false;

    // Dust grid dimensions (target)
    Box dust_domain = dg.ba.minimalBox();
    int nx_dst = dust_domain.length(0);
    int ny_dst = dust_domain.length(1);

    // Variables filled by rank-0, then broadcast
    int nx_csv  = 0, ny_csv = 0;
    int col_idx = -1;
    int read_ok = 0;   // 1 = success
    std::vector<Real> csv_data;

    if (ParallelDescriptor::IOProcessor()) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            // abort, not warn: a run without the chemistry it was given is not
            // the run that was asked for (the road loader already aborted)
            amrex::Abort("[DUST] PHREEQC file cannot be opened: " + filename);
        } else {
            std::string line;

            // Read header row
            if (!std::getline(file, line)) {
                amrex::Print() << "[DUST] read_phreeqc_csv: could not read header from "
                               << filename << "\n";
            } else {
                // Find column index
                std::istringstream header_stream(line);
                std::string col;
                int idx = 0;
                while (std::getline(header_stream, col, ',')) {
                    col.erase(0, col.find_first_not_of(" \t\r\n"));
                    col.erase(col.find_last_not_of(" \t\r\n") + 1);
                    if (col == col_name) { col_idx = idx; break; }
                    ++idx;
                }

                if (col_idx < 0) {
                    amrex::Abort("[DUST] read_phreeqc_csv: column '" + col_name
                                 + "' not found in " + filename);
                } else {
                    // Read all data rows into a flat vector
                    std::vector<Real> rows_flat;
                    while (std::getline(file, line)) {
                        if (line.empty()) continue;
                        std::istringstream row_stream(line);
                        std::string cell;
                        int cell_idx = 0;
                        while (std::getline(row_stream, cell, ',')) {
                            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                            if (cell_idx == col_idx) {
                                try {
                                    Real val = std::stod(cell);
                                    if (std::abs(val - PhreeqcDustConst::NODATA_CSV) < 1.0e-6)
                                        val = nodata_fill;
                                    rows_flat.push_back(val);
                                } catch (...) {
                                    // a skipped cell shifted every later row by one cell
                                    amrex::Abort("[DUST] read_phreeqc_csv: " + filename + " column '" + col_name
                                                 + "': cannot parse '" + cell + "' at data row "
                                                 + std::to_string(rows_flat.size()));
                                }
                                break;
                            }
                            ++cell_idx;
                        }
                    }

                    int n_rows = static_cast<int>(rows_flat.size());
                    bool dim_ok = false;

                    if (n_rows == nx_dst * ny_dst) {
                        // Exact match — no interpolation needed
                        nx_csv  = nx_dst;
                        ny_csv  = ny_dst;
                        dim_ok  = true;
                    } else {
                        // Try factorisation consistent with dust grid aspect ratio
                        for (int nx_c = 1; nx_c <= n_rows; ++nx_c) {
                            if (n_rows % nx_c != 0) continue;
                            int ny_c = n_rows / nx_c;
                            if (nx_dst % nx_c == 0 && ny_dst % ny_c == 0) {
                                nx_csv = nx_c;
                                ny_csv = ny_c;
                                dim_ok = true;
                                break;
                            }
                        }

                        if (dim_ok) {
                            amrex::Print()
                                << "[DUST] WARNING: read_phreeqc_csv: CSV has "
                                << nx_csv << "x" << ny_csv << " (" << n_rows
                                << " rows) but dust grid is "
                                << nx_dst << "x" << ny_dst
                                << ". Bilinear interpolation will be applied.\n";
                        } else {
                            // Fallback: assume square
                            int sq = static_cast<int>(
                                std::round(std::sqrt(static_cast<double>(n_rows))));
                            if (sq * sq == n_rows) {
                                nx_csv = sq;
                                ny_csv = sq;
                                dim_ok = true;
                                amrex::Print()
                                    << "[DUST] WARNING: read_phreeqc_csv: CSV has "
                                    << n_rows << " rows (assumed " << sq << "x" << sq
                                    << "); dust grid is " << nx_dst << "x" << ny_dst
                                    << ". Bilinear interpolation will be applied.\n";
                            } else {
                                amrex::Abort("[DUST] read_phreeqc_csv: " + filename + " has "
                                             + std::to_string(n_rows) + " rows, which is not the dust grid ("
                                             + std::to_string(nx_dst) + "x" + std::to_string(ny_dst)
                                             + "), a coarsening of it, or a square");
                            }
                        }
                    }

                    if (dim_ok) {
                        csv_data = std::move(rows_flat);
                        read_ok  = 1;
                    }
                }
            }
            file.close();
        }
    } // end IOProcessor block

    // Broadcast metadata
    ParallelDescriptor::Bcast(&read_ok,  1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&col_idx,  1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&nx_csv,   1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&ny_csv,   1, ParallelDescriptor::IOProcessorNumber());

    if (!read_ok || col_idx < 0) return false;

    // Broadcast CSV data
    if (!ParallelDescriptor::IOProcessor()) {
        csv_data.resize(nx_csv * ny_csv);
    }
    ParallelDescriptor::Bcast(csv_data.data(), nx_csv * ny_csv,
                               ParallelDescriptor::IOProcessorNumber());

    // Interpolate onto dust grid if needed
    std::vector<Real> dst_data;
    if (nx_csv == nx_dst && ny_csv == ny_dst) {
        dst_data = std::move(csv_data);
    } else {
        bilinear_interpolate(csv_data, nx_csv, ny_csv, dst_data, nx_dst, ny_dst);
    }

    // Copy data to device
    Gpu::DeviceVector<Real> d_data(dst_data.size());
    Gpu::copy(Gpu::hostToDevice, dst_data.begin(), dst_data.end(), d_data.begin());

    // GPU ParallelFor over MultiFab tiles
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx     = mfi.tilebox();
        auto       mf_arr = mf.array(mfi);
        Real*      d_ptr  = d_data.data();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int ii = amrex::max(0, amrex::min(i, nx_dst - 1));
            int jj = amrex::max(0, amrex::min(j, ny_dst - 1));
            mf_arr(i, j, k) = d_ptr[jj * nx_dst + ii];
        });
    }

    // ParallelFor is asynchronous; d_data frees its device allocation when this
    // function returns. Let the kernels finish reading it.
    Gpu::streamSynchronize();

    return true;
}

bool read_phreeqc_netcdf(MultiFab& mf, const DustGrid& dg,
                         const std::string& filename,
                         const std::string& varname, Real nodata_fill)
{
    // Both branches below abort, so nothing here consumes the arguments.
    amrex::ignore_unused(mf, dg, filename, varname, nodata_fill);
#ifdef ERF_USE_NETCDF
    amrex::Abort("[DUST] read_phreeqc_netcdf: implementation not yet complete. "
                 "Use CSV format for PHREEQC output.");
    return false;
#else
    amrex::Abort("[DUST] read_phreeqc_netcdf requires ERF_ENABLE_NETCDF=ON");
    return false;
#endif
}

namespace {
/// Copy src into dst where site_id == site (1-based), valid cells only.
void copy_where_site(MultiFab& dst, const MultiFab& src, const MultiFab& site_id, int site)
{
    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto d = dst.array(mfi);
        auto s = src.const_array(mfi);
        auto id = site_id.const_array(mfi);
        const Real sid = Real(site);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (std::abs(id(i,j,k) - sid) < Real(0.5)) d(i,j,k) = s(i,j,k);
        });
    }
}
} // namespace

void update_dust_from_phreeqc(MultiFab&       dust_crust_index,
                              MultiFab&       dust_silt_frac,
                              MultiFab&       dust_efflor,
                              MultiFab&       dust_suppression,
                              const MultiFab* dust_site_id,
                              const DustGrid& dg,
                              const DustParams& params)
{
    if (params.phreeqc_output_file.empty()) return;

    // Select reader by file extension
    auto read_field = [&](MultiFab& mf, const std::string& filename, const std::string& varname) -> bool {
        if (filename.size() > 3) {
            std::string ext = filename.substr(filename.size() - 3);
            if (ext == ".nc")
                return read_phreeqc_netcdf(mf, dg, filename, varname, 0.0);
        }
        return read_phreeqc_csv(mf, dg, filename, varname, 0.0);
    };
    auto read_all = [&](const std::string& filename, MultiFab& crust, MultiFab& silt,
                        MultiFab& efflor, MultiFab& supp) {
        if (!params.phreeqc_crust_var.empty())  read_field(crust,  filename, params.phreeqc_crust_var);
        if (!params.phreeqc_efflor_var.empty()) read_field(efflor, filename, params.phreeqc_efflor_var);
        if (!params.phreeqc_silt_var.empty())   read_field(silt,   filename, params.phreeqc_silt_var);
        if (!params.phreeqc_supp_var.empty())   read_field(supp,   filename, params.phreeqc_supp_var);
    };

    // the global table everywhere
    read_all(params.phreeqc_output_file, dust_crust_index, dust_silt_frac, dust_efflor, dust_suppression);

    // then each site's own table over its cells (documented since the site
    // registry was added, but no table other than the global one was ever read)
    const int n_files = (int)params.site_phreeqc_files.size();
    if (n_files > 0 && dust_site_id) {
        MultiFab crust_s(dg.ba, dg.dm, 1, IntVect(1,1,0)), silt_s(dg.ba, dg.dm, 1, IntVect(1,1,0)),
                 efflor_s(dg.ba, dg.dm, 1, IntVect(1,1,0)), supp_s(dg.ba, dg.dm, 1, IntVect(1,1,0));
        for (int s = 0; s < n_files; ++s) {
            const std::string& f = params.site_phreeqc_files[s];
            if (f.empty()) continue;
            MultiFab::Copy(crust_s,  dust_crust_index, 0, 0, 1, IntVect(0));
            MultiFab::Copy(silt_s,   dust_silt_frac,   0, 0, 1, IntVect(0));
            MultiFab::Copy(efflor_s, dust_efflor,      0, 0, 1, IntVect(0));
            MultiFab::Copy(supp_s,   dust_suppression, 0, 0, 1, IntVect(0));
            read_all(f, crust_s, silt_s, efflor_s, supp_s);
            copy_where_site(dust_crust_index, crust_s,  *dust_site_id, s + 1);
            copy_where_site(dust_silt_frac,   silt_s,   *dust_site_id, s + 1);
            copy_where_site(dust_efflor,      efflor_s, *dust_site_id, s + 1);
            copy_where_site(dust_suppression, supp_s,   *dust_site_id, s + 1);
            amrex::Print() << "[DUST] PHREEQC site " << (s + 1) << " ("
                           << (s < (int)params.site_names.size() ? params.site_names[s] : std::string("?"))
                           << ") updated from " << f << "\n";
        }
    }
    // (the threshold is recomputed by DustLayer::advance from the new crust and
    // efflorescence with erf.dust.alpha_crust/alpha_efflor; the reader's own
    // update with hard-coded 0.5/0.3 and a 0.05 m/s floor, overwritten on the
    // same step, is gone, as is the metal-fraction column copied into the
    // emission flux and overwritten before it was read)

    amrex::Print() << "[DUST] PHREEQC update from file: "
                   << params.phreeqc_output_file << "\n";
}

#endif // ERF_USE_DUST