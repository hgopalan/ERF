#ifdef ERF_USE_DUST

#include <ERF_PhreeqcReader.H>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

// Forward declarations - these must be defined in DustLayer context
struct DustGrid {
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
};

struct DustParams {
    std::string phreeqc_output_file;
    std::string phreeqc_crust_var;
    std::string phreeqc_silt_var;
    std::string phreeqc_efflor_var;
    std::string phreeqc_supp_var;
    std::string phreeqc_metal_var;
    amrex::Real alpha_crust;
    amrex::Real alpha_efflor;
    amrex::Real phreeqc_update_interval_s;
};

using namespace amrex;

bool read_phreeqc_csv(MultiFab&          mf,
                      const DustGrid&    dg,
                      const std::string& filename,
                      const std::string& col_name,
                      Real               nodata_fill)
{
    // Return false if filename is empty
    if (filename.empty()) {
        return false;
    }

    int nx = 0, ny = 0;
    std::vector<Real> data;
    bool success = false;

    // Rank-0 reads the file
    if (ParallelDescriptor::IOProcessor()) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            amrex::Print() << "[DUST] read_phreeqc_csv: could not open file " 
                           << filename << "\n";
            return false;
        }

        std::string line;
        std::vector<std::string> header;
        int col_index = -1;

        // Read header row
        if (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                header.push_back(token);
            }

            // Find column index
            for (int i = 0; i < static_cast<int>(header.size()); ++i) {
                if (header[i] == col_name) {
                    col_index = i;
                    break;
                }
            }

            if (col_index < 0) {
                amrex::Print() << "[DUST] read_phreeqc_csv: column '" << col_name 
                               << "' not found in " << filename << "\n";
                infile.close();
                return false;
            }
        } else {
            amrex::Print() << "[DUST] read_phreeqc_csv: could not read header from " 
                           << filename << "\n";
            infile.close();
            return false;
        }

        // Get dust grid dimensions
        nx = dg.ba.getCellCenteredBox(0).length(0);
        ny = dg.ba.getCellCenteredBox(0).length(1);
        data.resize(nx * ny, nodata_fill);

        // Read data rows
        // CSV row order: row 0 = southernmost (j=0). No reversal needed, unlike ESRI ASCII.
        int row_count = 0;
        while (std::getline(infile, line)) {
            if (row_count >= nx * ny) break;

            std::istringstream iss(line);
            std::vector<std::string> row;
            std::string token;
            while (std::getline(iss, token, ',')) {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                row.push_back(token);
            }

            if (static_cast<int>(row.size()) > col_index) {
                try {
                    Real value = std::stod(row[col_index]);
                    if (value == PhreeqcDustConst::NODATA_CSV) {
                        data[row_count] = nodata_fill;
                    } else {
                        data[row_count] = value;
                    }
                } catch (...) {
                    data[row_count] = nodata_fill;
                }
            }
            ++row_count;
        }

        infile.close();
        success = true;
    }

    // Broadcast success flag
    ParallelDescriptor::Bcast(&success, 1, ParallelDescriptor::IOProcessorNumber());
    if (!success) return false;

    // Broadcast dimensions and data
    ParallelDescriptor::Bcast(&nx, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&ny, 1, ParallelDescriptor::IOProcessorNumber());
    
    if (!ParallelDescriptor::IOProcessor()) {
        data.resize(nx * ny);
    }
    ParallelDescriptor::Bcast(data.data(), data.size(), ParallelDescriptor::IOProcessorNumber());

    // Copy to device and fill MultiFab
    Real* d_data = (Real*) amrex::The_Arena()->alloc(data.size() * sizeof(Real));
    Gpu::copy(Gpu::hostToDevice, data.data(), data.data() + data.size(), d_data);

    // GPU fill
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto mf_arr = mf.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int ii = i;
            int jj = j;
            // Clamp to domain bounds
            if (ii < 0) ii = 0;
            if (ii >= nx) ii = nx - 1;
            if (jj < 0) jj = 0;
            if (jj >= ny) jj = ny - 1;
            mf_arr(i, j, k) = d_data[jj * nx + ii];
        });
    }

    amrex::The_Arena()->free(d_data);
    return true;
}

bool read_phreeqc_netcdf(MultiFab&          mf,
                         const DustGrid&    dg,
                         const std::string& filename,
                         const std::string& varname,
                         Real               nodata_fill)
{
#ifdef ERF_USE_NETCDF
    // NetCDF-C API: https://docs.unidata.ucar.edu/netcdf-c/current/
    // Full implementation deferred. Use CSV format for now.
    amrex::Abort("[DUST] read_phreeqc_netcdf: implementation not yet complete. "
                 "Use CSV format for PHREEQC output.");
    return false;
#else
    amrex::Abort("[DUST] read_phreeqc_netcdf requires ERF_ENABLE_NETCDF=ON");
    return false;
#endif
}

void update_ustar_t_from_chemistry(MultiFab&       ustar_t,
                                   const MultiFab& ustar_base,
                                   const MultiFab& crust,
                                   const MultiFab& efflor)
{
    // u*_t reduction from mineral crust and salt efflorescence.
    // Marticorena & Bergametti (1995), https://doi.org/10.1029/95JD00690
    const Real alpha_c    = PhreeqcDustConst::ALPHA_CRUST;
    const Real alpha_e    = PhreeqcDustConst::ALPHA_EFFLOR;
    const Real ustar_tmin = PhreeqcDustConst::USTAR_T_MIN;

    for (MFIter mfi(ustar_t, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx       = mfi.tilebox();
        auto ut_arr         = ustar_t.array(mfi);
        auto ut_base_arr    = ustar_base.const_array(mfi);
        auto crust_arr      = crust.const_array(mfi);
        auto efflor_arr     = efflor.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real ut = ut_base_arr(i,j,k)
                    * (Real(1.0) - alpha_c * crust_arr(i,j,k))
                    * (Real(1.0) - alpha_e * efflor_arr(i,j,k));
            ut_arr(i,j,k) = amrex::max(ut, ustar_tmin);
        });
    }
}

void update_dust_from_phreeqc(MultiFab&       dust_ustar_t,
                              const MultiFab& dust_ustar_base,
                              MultiFab&       dust_crust_index,
                              MultiFab&       dust_silt_frac,
                              MultiFab&       dust_efflor,
                              MultiFab&       dust_suppression,
                              MultiFab&       dust_emission,
                              const DustGrid& dg,
                              const DustParams& params)
{
    // Return immediately if no output file specified
    if (params.phreeqc_output_file.empty()) {
        return;
    }

    // Lambda to select reader by file extension
    auto read_field = [&](MultiFab& mf, const std::string& var_name, Real nodata) -> bool {
        const std::string& filename = params.phreeqc_output_file;
        if (filename.length() > 3) {
            std::string ext = filename.substr(filename.length() - 3);
            if (ext == ".nc" || ext == ".NC") {
                return read_phreeqc_netcdf(mf, dg, filename, var_name, nodata);
            }
        }
        return read_phreeqc_csv(mf, dg, filename, var_name, nodata);
    };

    // Read crust index
    if (!params.phreeqc_crust_var.empty()) {
        read_field(dust_crust_index, params.phreeqc_crust_var, 0.0);
    }

    // Read silt fraction
    if (!params.phreeqc_silt_var.empty()) {
        read_field(dust_silt_frac, params.phreeqc_silt_var, 0.1);
    }

    // Read efflorescence
    if (!params.phreeqc_efflor_var.empty()) {
        read_field(dust_efflor, params.phreeqc_efflor_var, 0.0);
    }

    // Read suppression modifier
    if (!params.phreeqc_supp_var.empty()) {
        read_field(dust_suppression, params.phreeqc_supp_var, 0.0);
    }

    // Read metal mass fraction (component 0)
    if (!params.phreeqc_metal_var.empty() && dust_emission.nComp() > 0) {
        // Create a temporary MultiFab for the metal field
        MultiFab metal_field(dust_emission.boxArray(), dust_emission.DistributionMap(), 1, 0);
        read_field(metal_field, params.phreeqc_metal_var, 0.0);
        // Copy to component 0 of dust_emission
        MultiFab::Copy(dust_emission, metal_field, 0, 0, 1, 0);
    }

    // Update u*_t from crust and efflorescence
    update_ustar_t_from_chemistry(dust_ustar_t, dust_ustar_base, 
                                  dust_crust_index, dust_efflor);

    amrex::Print() << "[DUST] PHREEQC update from file: " << params.phreeqc_output_file << "\n";
}

#endif  // ERF_USE_DUST
