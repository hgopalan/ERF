/**
 * @file ERF_LNGAtmReturn.H
 * @brief Return fields from the ERF 3D conserved state to the 2D LNG grid.
 *
 * Implements the ERF-Atm -> ERF-LNG return path (Phase 4). One field
 * is extracted from the 3D solver and mapped to the 2D LNG grid:
 *
 *   lng_conc_sfc:  RhoLNG at k=klo [kg/m^3] from S_new conserved state.
 *                  Feeds future loading feedback (Phase 5+).
 *
 * Coarsening pattern (same as fill_lng_scalar_from_atm in ERF_LNGWindExtract.H):
 *   lng_field(i,j,0) = atm_field(i/C, j/C, klo)
 *
 * Pattern references:
 *   Source/Dust/ERF_DustAtmReturn.H (Phase 13 analog)
 *   Source/Fire/ERF_FireWindExtract.H (algorithm source)
 *
 * References:
 *   Hong, S.-Y., & Pan, H.-L. (1996). Mon. Wea. Rev., 124, 2322.
 *     https://doi.org/10.1175/1520-0493(1996)124<2322:NBLVDI>2.0.CO;2
 */

#ifndef ERF_LNG_ATM_RETURN_H
#define ERF_LNG_ATM_RETURN_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>

/**
 * @brief Map RhoLNG at k=klo from the 3D conserved state to the 2D LNG grid.
 *
 * lng_conc_sfc(i,j,0) = max(S_cons(i/C, j/C, klo, lng_comp), 0)
 *
 * @param[out] lng_conc_sfc  2D MultiFab on LNG grid [kg/m^3].
 * @param[in]  S_cons        3D conserved state (S_new) on atm grid.
 * @param[in]  lng_comp      Component index for LNG scalar.
 * @param[in]  geom_atm      Atmospheric geometry.
 * @param[in]  C             Grid ratio (LNG grid refinement factor).
 */
inline void fill_lng_conc_from_atm(
    amrex::MultiFab&       lng_conc_sfc,
    const amrex::MultiFab& S_cons,
    int                    lng_comp,
    const amrex::Geometry& geom_atm,
    int                    C)
{
    const int klo = geom_atm.Domain().smallEnd(2);
    for (amrex::MFIter mfi(lng_conc_sfc, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto conc = lng_conc_sfc.array(mfi);
        auto cons = S_cons.const_array(mfi);
        const int comp = lng_comp;
        const int kl   = klo;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            conc(i,j,k) = amrex::max(cons(i/C, j/C, kl, comp), 0.0);
        });
    }
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_ATM_RETURN_H */
