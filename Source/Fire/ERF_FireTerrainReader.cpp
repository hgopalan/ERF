#include <ERF_FireTerrainReader.H>
#include <fstream>
#include <vector>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>

using namespace amrex;

namespace {

/**
 * Read an ERF terrain text file on the IO rank and broadcast it.
 * Returns false when the file cannot be opened; every rank agrees on the
 * result since the open status is broadcast with the sizes.
 */
bool read_erf_terrain_file(const std::string& fname,
                           int& nx_terrain, int& ny_terrain,
                           std::vector<Real>& x_coords,
                           std::vector<Real>& y_coords,
                           std::vector<Real>& z_values)
{
    nx_terrain = 0;
    ny_terrain = 0;
    int ok = 1;

    if (ParallelDescriptor::IOProcessor()) {
        std::ifstream file(fname);
        if (!file.is_open()) {
            amrex::Warning("Could not open terrain file: " + fname);
            ok = 0;
        } else {
            file >> nx_terrain >> ny_terrain;
            x_coords.resize(nx_terrain);
            y_coords.resize(ny_terrain);
            z_values.resize(static_cast<size_t>(nx_terrain) * ny_terrain);
            for (int i = 0; i < nx_terrain; ++i) { file >> x_coords[i]; }
            for (int j = 0; j < ny_terrain; ++j) { file >> y_coords[j]; }
            for (size_t n = 0; n < z_values.size(); ++n) { file >> z_values[n]; }
        }
    }

    ParallelDescriptor::Bcast(&ok, 1, ParallelDescriptor::IOProcessorNumber());
    if (!ok) { return false; }

    ParallelDescriptor::Bcast(&nx_terrain, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&ny_terrain, 1, ParallelDescriptor::IOProcessorNumber());
    if (!ParallelDescriptor::IOProcessor()) {
        x_coords.resize(nx_terrain);
        y_coords.resize(ny_terrain);
        z_values.resize(static_cast<size_t>(nx_terrain) * ny_terrain);
    }
    ParallelDescriptor::Bcast(x_coords.data(), nx_terrain, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(y_coords.data(), ny_terrain, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(z_values.data(), nx_terrain * ny_terrain, ParallelDescriptor::IOProcessorNumber());
    return true;
}

} // namespace

bool read_heightmap_nearest_onto_fire_cells(
    MultiFab&          h_fire_cc,
    const FireGrid&    fg,
    const std::string& fname)
{
    if (fname.empty()) { return false; }

    int nx_t = 0, ny_t = 0;
    std::vector<Real> x_coords, y_coords, z_values;
    if (!read_erf_terrain_file(fname, nx_t, ny_t, x_coords, y_coords, z_values)) {
        return false;
    }

    Gpu::DeviceVector<Real> x_device(nx_t);
    Gpu::DeviceVector<Real> y_device(ny_t);
    Gpu::DeviceVector<Real> z_device(static_cast<size_t>(nx_t) * ny_t);
    Gpu::copy(Gpu::hostToDevice, x_coords.begin(), x_coords.end(), x_device.begin());
    Gpu::copy(Gpu::hostToDevice, y_coords.begin(), y_coords.end(), y_device.begin());
    Gpu::copy(Gpu::hostToDevice, z_values.begin(), z_values.end(), z_device.begin());
    const Real* x_ptr = x_device.data();
    const Real* y_ptr = y_device.data();
    const Real* z_ptr = z_device.data();

    const auto prob_lo = fg.geom.ProbLoArray();
    const auto dx      = fg.geom.CellSizeArray();

    for (MFIter mfi(h_fire_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> h = h_fire_cc.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
            const Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];

            // Nearest source point along each axis. The coordinates are
            // sorted, so the first point at or beyond x and its predecessor
            // bracket it; pick the closer of the two.
            int ix = nx_t - 1;
            for (int n = 0; n < nx_t; ++n) {
                if (x_ptr[n] >= x) { ix = n; break; }
            }
            if (ix > 0 && (x - x_ptr[ix - 1]) < (x_ptr[ix] - x)) { ix -= 1; }

            int iy = ny_t - 1;
            for (int n = 0; n < ny_t; ++n) {
                if (y_ptr[n] >= y) { iy = n; break; }
            }
            if (iy > 0 && (y - y_ptr[iy - 1]) < (y_ptr[iy] - y)) { iy -= 1; }

            h(i, j, k) = z_ptr[ix * ny_t + iy];
        });
    }

    // The kernels read the device vectors asynchronously; let them finish
    // before the vectors are freed at scope exit.
    Gpu::streamSynchronize();
    return true;
}

int read_structure_ids_nearest_onto_fire_cells(
    MultiFab&          id_fire_cc,
    const FireGrid&    fg,
    const std::string& fname,
    Real               hmin)
{
    if (fname.empty()) { return -1; }

    int nx_t = 0, ny_t = 0;
    std::vector<Real> x_coords, y_coords, z_values;
    if (!read_erf_terrain_file(fname, nx_t, ny_t, x_coords, y_coords, z_values)) {
        return -1;
    }

    // Connected components of the points above hmin, 4-connected on the
    // file's grid, numbered in scan order. Every rank holds the whole file
    // after the broadcast, so every rank labels identically.
    const size_t npts = static_cast<size_t>(nx_t) * ny_t;
    std::vector<Real> labels(npts, 0.0);
    std::vector<int>  stack;
    int n_struct = 0;
    for (int ix = 0; ix < nx_t; ++ix) {
        for (int iy = 0; iy < ny_t; ++iy) {
            const size_t p = static_cast<size_t>(ix) * ny_t + iy;
            if (z_values[p] <= hmin || labels[p] > 0.0) { continue; }
            ++n_struct;
            labels[p] = static_cast<Real>(n_struct);
            stack.clear();
            stack.push_back(static_cast<int>(p));
            while (!stack.empty()) {
                const int q = stack.back(); stack.pop_back();
                const int qx = q / ny_t, qy = q % ny_t;
                const int nbx[4] = {qx - 1, qx + 1, qx, qx};
                const int nby[4] = {qy, qy, qy - 1, qy + 1};
                for (int n = 0; n < 4; ++n) {
                    if (nbx[n] < 0 || nbx[n] >= nx_t || nby[n] < 0 || nby[n] >= ny_t) { continue; }
                    const size_t r = static_cast<size_t>(nbx[n]) * ny_t + nby[n];
                    if (z_values[r] > hmin && labels[r] == 0.0) {
                        labels[r] = static_cast<Real>(n_struct);
                        stack.push_back(static_cast<int>(r));
                    }
                }
            }
        }
    }

    Gpu::DeviceVector<Real> x_device(nx_t);
    Gpu::DeviceVector<Real> y_device(ny_t);
    Gpu::DeviceVector<Real> l_device(npts);
    Gpu::copy(Gpu::hostToDevice, x_coords.begin(), x_coords.end(), x_device.begin());
    Gpu::copy(Gpu::hostToDevice, y_coords.begin(), y_coords.end(), y_device.begin());
    Gpu::copy(Gpu::hostToDevice, labels.begin(),   labels.end(),   l_device.begin());
    const Real* x_ptr = x_device.data();
    const Real* y_ptr = y_device.data();
    const Real* l_ptr = l_device.data();

    const auto prob_lo = fg.geom.ProbLoArray();
    const auto dx      = fg.geom.CellSizeArray();

    for (MFIter mfi(id_fire_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> id = id_fire_cc.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
            const Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
            // Same nearest-point rule as the height sampling.
            int ix = nx_t - 1;
            for (int n = 0; n < nx_t; ++n) {
                if (x_ptr[n] >= x) { ix = n; break; }
            }
            if (ix > 0 && (x - x_ptr[ix - 1]) < (x_ptr[ix] - x)) { ix -= 1; }
            int iy = ny_t - 1;
            for (int n = 0; n < ny_t; ++n) {
                if (y_ptr[n] >= y) { iy = n; break; }
            }
            if (iy > 0 && (y - y_ptr[iy - 1]) < (y_ptr[iy] - y)) { iy -= 1; }
            id(i, j, k) = l_ptr[ix * ny_t + iy];
        });
    }
    Gpu::streamSynchronize();
    return n_struct;
}

bool read_terrain_onto_fire_grid(
    MultiFab&   z_fire_nd,
    const FireGrid&    fg,
    const std::string& fname)
{
    // Return false if fname is empty
    if (fname.empty()) {
        return false;
    }

    int nx_terrain = 0;
    int ny_terrain = 0;
    std::vector<Real> x_coords;
    std::vector<Real> y_coords;
    std::vector<Real> z_values;
    if (!read_erf_terrain_file(fname, nx_terrain, ny_terrain, x_coords, y_coords, z_values)) {
        return false;
    }

    // Copy to GPU
    //Gpu::DeviceVector<Real> x_device(x_coords.begin(), x_coords.end());
    //Gpu::DeviceVector<Real> y_device(y_coords.begin(), y_coords.end());
    //Gpu::DeviceVector<Real> z_device(z_values.begin(), z_values.end());

    // Copy to GPU
    Gpu::DeviceVector<Real> x_device(nx_terrain);
    Gpu::DeviceVector<Real> y_device(ny_terrain);
    Gpu::DeviceVector<Real> z_device(nx_terrain * ny_terrain);

    Gpu::copy(Gpu::hostToDevice, x_coords.begin(), x_coords.end(), x_device.begin());
    Gpu::copy(Gpu::hostToDevice, y_coords.begin(), y_coords.end(), y_device.begin());
    Gpu::copy(Gpu::hostToDevice, z_values.begin(), z_values.end(), z_device.begin());

    Real* x_ptr = x_device.data();
    Real* y_ptr = y_device.data();
    Real* z_ptr = z_device.data();

    // Problem domain
    Real ProbLo_x = fg.geom.ProbLo(0);
    Real ProbLo_y = fg.geom.ProbLo(1);

    // Fire grid spacing
    auto dx_dy = fg.geom.CellSizeArray();
    Real dx_f = dx_dy[0];
    Real dy_f = dx_dy[1];

    // Get fire domain bounds (for clamping)
    const Box& domain_fire = fg.geom.Domain();
    int i_fire_lo = domain_fire.smallEnd(0);
    int i_fire_hi = domain_fire.bigEnd(0);
    int j_fire_lo = domain_fire.smallEnd(1);
    int j_fire_hi = domain_fire.bigEnd(1);

    // Get terrain domain bounds
    Real x_min = x_coords[0];
    Real x_max = x_coords[nx_terrain - 1];
    Real y_min = y_coords[0];
    Real y_max = y_coords[ny_terrain - 1];

    // Interpolate onto fire grid
    for (MFIter mfi(z_fire_nd, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> z_fire = z_fire_nd.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i_f = iv[0];
            int j_f = iv[1];

            // Clamp to domain for ghost cells
            int i_f_clamped = std::max(i_fire_lo, std::min(i_fire_hi, i_f));
            int j_f_clamped = std::max(j_fire_lo, std::min(j_fire_hi, j_f));

            // Compute physical position at fire-grid node
            Real x = ProbLo_x + i_f_clamped * dx_f;
            Real y = ProbLo_y + j_f_clamped * dy_f;

            // Clamp to terrain domain
            x = std::max(x_min, std::min(x_max, x));
            y = std::max(y_min, std::min(y_max, y));

            // Find surrounding terrain grid points via binary search or linear scan
            // For efficiency, use a linear scan (could optimize with binary search)
            int ix_lo = nx_terrain - 2;  // safe fallback for x == x_max
            int iy_lo = ny_terrain - 2;  // safe fallback for y == y_max

            // Find x index
            for (int ix = 0; ix < nx_terrain - 1; ++ix) {
                if (x_ptr[ix] <= x && x <= x_ptr[ix + 1]) {
                    ix_lo = ix;
                    break;
                }
            }

            // Find y index
            for (int iy = 0; iy < ny_terrain - 1; ++iy) {
                if (y_ptr[iy] <= y && y <= y_ptr[iy + 1]) {
                    iy_lo = iy;
                    break;
                }
            }

            int ix_hi = std::min(ix_lo + 1, nx_terrain - 1);
            int iy_hi = std::min(iy_lo + 1, ny_terrain - 1);

            // Get four corner values
            Real z_ll = z_ptr[ix_lo * ny_terrain + iy_lo];
            Real z_lr = z_ptr[ix_hi * ny_terrain + iy_lo];
            Real z_ul = z_ptr[ix_lo * ny_terrain + iy_hi];
            Real z_ur = z_ptr[ix_hi * ny_terrain + iy_hi];

            // Bilinear interpolation weights
            Real dx_x = (x_ptr[ix_hi] > x_ptr[ix_lo]) ? (x - x_ptr[ix_lo]) / (x_ptr[ix_hi] - x_ptr[ix_lo]) : 0.0;
            Real dy_y = (y_ptr[iy_hi] > y_ptr[iy_lo]) ? (y - y_ptr[iy_lo]) / (y_ptr[iy_hi] - y_ptr[iy_lo]) : 0.0;

            dx_x = std::max(0.0, std::min(1.0, dx_x));
            dy_y = std::max(0.0, std::min(1.0, dy_y));

            // Bilinear interpolation
            Real z_lo = z_ll * (1.0 - dx_x) + z_lr * dx_x;
            Real z_hi = z_ul * (1.0 - dx_x) + z_ur * dx_x;
            Real z = z_lo * (1.0 - dy_y) + z_hi * dy_y;

            z_fire(iv, 0) = z;
        });
    }

    // ParallelFor is asynchronous, and x_device/y_device/z_device free their
    // device allocations when this function returns. Let the kernels finish
    // reading them first.
    Gpu::streamSynchronize();

    return true;
}
