#include <ERF_TerrainDrag.H>

using namespace amrex;

/*
  Constructor to get the terrain parameters:
  TreeType xc, yc, height
*/
TerrainDrag::TerrainDrag(std::string terrainfile)
{
  std::ifstream file(terrainfile, std::ios::in);
  if (!file.good()) {
    Abort("Cannot find terrain file: " + terrainfile);
  }
  // xc yc height
  Real value1, value2, value3;
  while (file >> value1 >> value2 >> value3) {
    m_x_terrain.push_back(value1);
    m_y_terrain.push_back(value2);
    m_height_terrain.push_back(value3);
  }
  file.close();
}

void
TerrainDrag::define_terrain_blank_field(
  const BoxArray& ba,
  const DistributionMapping& dm,
  Geometry& geom,
  MultiFab* z_phys_nd)
{
  // Geometry params
  const auto& dx = geom.CellSizeArray();
  const auto& prob_lo = geom.ProbLoArray();

  // Allocate the terrain blank MF
  // NOTE: 1 ghost cell for averaging to faces
  m_terrain_blank.reset();
  m_terrain_blank = std::make_unique<MultiFab>(ba, dm, 1, 1);
  m_terrain_blank->setVal(0.);

  // Set the terrain blank data
  for (MFIter mfi(*m_terrain_blank); mfi.isValid(); ++mfi) {
    Box gtbx = mfi.growntilebox();
    const Array4<Real>& levelBlank = m_terrain_blank->array(mfi);
    const Array4<const Real>& z_nd =
      (z_phys_nd) ? z_phys_nd->const_array(mfi) : Array4<const Real>{};
    ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Loop over terrain points
      for (unsigned ii = 0; ii < m_x_terrain.size(); ++ii) {
        Real ht = m_height_terrain[ii];
        Real xt = m_x_terrain[ii];
        Real yt = m_y_terrain[ii];
        // Physical positions of cell-centers
        const Real x = prob_lo[0] + (i + 0.5) * dx[0];
        const Real y = prob_lo[1] + (j + 0.5) * dx[1];
        Real z = prob_lo[2] + (k + 0.5) * dx[2];
        if (z_nd) {
          z = 0.125 * (z_nd(i, j, k) + z_nd(i + 1, j, k) + z_nd(i, j + 1, k) +
                       z_nd(i + 1, j + 1, k) + z_nd(i, j, k + 1) +
                       z_nd(i + 1, j, k + 1) + z_nd(i, j + 1, k + 1) +
                       z_nd(i + 1, j + 1, k + 1));
        }
        z = std::max(z, 0.0);
        const Real cell_radius = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1]);
        // Proximity to the terrain
        const Real radius =
          std::sqrt((x - xt) * (x - xt) + (y - yt) * (y - yt));
        const Real terrain_point =
          (radius <= cell_radius && z <= ht) ? 1.0 : 0.0;
        levelBlank(i, j, k) = terrain_point;
        if(terrain_point==1){
            break;
        }
      }
    });
  } // mfi

// Fillboundary for periodic ghost cell copy
m_terrain_blank->FillBoundary(geom.periodicity());

} // init_terrain_blank_field
