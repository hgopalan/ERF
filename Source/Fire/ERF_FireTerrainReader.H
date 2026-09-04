#ifndef ERF_FIRE_TERRAIN_READER_H
#define ERF_FIRE_TERRAIN_READER_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <string>

/**
 * @file ERF_FireTerrainReader.H
 * @brief Self-contained terrain reader for fire grid
 *
 * Reads terrain data from a standard ERF text file format and interpolates
 * onto the fire grid using bilinear interpolation.
 *
 * File format (line-based):
 *   Line 1: nx (integer)
 *   Line 2: ny (integer)
 *   Next nx lines: x-coordinates [m]
 *   Next ny lines: y-coordinates [m]
 *   Next nx*ny lines: z-values stored contiguous in y, i.e., z[ix*ny + iy]
 */

/**
 * @brief Read terrain data onto fire grid with bilinear interpolation
 *
 * Reads ERF standard terrain text file format:
 * 1. IO rank reads nx, ny from first two lines
 * 2. Reads nx x-coordinates, ny y-coordinates, nx*ny z-values
 * 3. Broadcasts nx, ny, and arrays to all MPI ranks
 * 4. Copies arrays to GPU with Gpu::DeviceVector
 * 5. Bilinearly interpolates terrain heights at fire-grid nodes
 * 6. Fills z_fire_nd MultiFab with interpolated values
 *
 * @param[out] z_fire_nd   Terrain height at fire-grid nodes (1 component, 1 ghost cell)
 *                         On fire grid, indexed by fire grid cell indices
 * @param[in]  fg          Fire grid structure
 * @param[in]  fname       File path (empty string means no file, returns false)
 *
 * @return true if file was successfully read and interpolation completed,
 *         false if fname is empty or file cannot be opened
 *
 * Ghost cells are handled by clamping indices to valid domain bounds before
 * computing physical positions.
 *
 * @note This is a host-side I/O operation followed by GPU-side interpolation.
 *       The function broadcasts to all MPI ranks so all can compute in parallel.
 */
bool read_terrain_onto_fire_grid(
    amrex::MultiFab&   z_fire_nd,
    const FireGrid&    fg,
    const std::string& fname);

/**
 * @brief Sample a heightmap onto fire-grid cell centres by nearest point
 *
 * Same file format and MPI handling as read_terrain_onto_fire_grid(), but the
 * value at each fire cell centre is the nearest heightmap point rather than a
 * bilinear blend. Bilinear interpolation turns a sharp wall into a ramp one
 * source cell wide, so building footprints are sampled this way to keep their
 * edges where the file puts them.
 *
 * @param[out] h_fire_cc   Height at fire-grid cell centres (1 component)
 * @param[in]  fg          Fire grid structure
 * @param[in]  fname       File path (empty string means no file, returns false)
 * @return true on success, false if fname is empty or cannot be opened
 */
bool read_heightmap_nearest_onto_fire_cells(
    amrex::MultiFab&   h_fire_cc,
    const FireGrid&    fg,
    const std::string& fname);

/**
 * @brief Label the connected structures of a heightmap and sample the labels
 *        onto fire cells
 *
 * Heightmap points above hmin are grouped by 4-connectivity on the file's
 * own grid and numbered 1..N in scan order; every other point is 0. Each fire
 * cell takes the label of its nearest point, the same sampling as
 * read_heightmap_nearest_onto_fire_cells(), so a cell carries a nonzero id
 * exactly when the structure mask built from the same file and threshold is
 * set there. Only the valid region is written.
 *
 * @param[out] id_fire_cc  Structure id at fire-grid cell centres (1 component)
 * @param[in]  fg          Fire grid structure
 * @param[in]  fname       Heightmap file
 * @param[in]  hmin        Height above which a point belongs to a structure [m]
 * @return Number of structures N, or -1 if the file cannot be read
 */
int read_structure_ids_nearest_onto_fire_cells(
    amrex::MultiFab&   id_fire_cc,
    const FireGrid&    fg,
    const std::string& fname,
    amrex::Real        hmin);

#endif
