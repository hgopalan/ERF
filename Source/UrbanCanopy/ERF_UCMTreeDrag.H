/**
 * @file ERF_UCMTreeDrag.H
 * @brief Phase 6.1 explicit tree-canopy momentum drag on ATM source terms.
 */

#ifndef ERF_UCM_TREE_DRAG_H
#define ERF_UCM_TREE_DRAG_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <UrbanCanopy/ERF_UCMParams.H>

void apply_ucm_tree_drag_to_source(
    amrex::MultiFab&       xmom_src,
    amrex::MultiFab&       ymom_src,
    const amrex::MultiFab& S_cons,
    const amrex::MultiFab& S_xmom,
    const amrex::MultiFab& S_ymom,
    const amrex::iMultiFab& is_tree_atm,
    const amrex::MultiFab& H_tree_atm,
    const amrex::MultiFab& H_crown_base_atm,
    const amrex::MultiFab& LAD_atm,
    const amrex::MultiFab& crown_area_frac_atm,
    const amrex::MultiFab& Cd_leaf_atm,
    const amrex::MultiFab* z_phys_nd,
    const amrex::Geometry& geom_atm,
    TreeDragMode           drag_mode,
    int                    lev,
    bool                   ucm_debug = false);

#endif
