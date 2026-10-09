#include <ERF_Rothermel.H>
#include <ERF_FuelModels.H>


using namespace amrex;


RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           Real moisture_1hr,
                                           Real moisture_10hr,
                                           Real moisture_100hr,
                                           bool use_wind_limit,
                                           int wind_limit_mode)
{
    return rothermel_coefficients(fp, moisture_1hr, moisture_10hr, moisture_100hr, use_wind_limit, wind_limit_mode);
}


void compute_ros_field(
    MultiFab& fire_ros,
    const MultiFab& fire_wind,
    const MultiFab& fire_slopes,
    const RothermelComputed& rc)
{
    for (MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> ros = fire_ros.array(mfi);
        Array4<const Real> wind = fire_wind.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            Real ux = wind(i, j, k, 0);
            Real uy = wind(i, j, k, 1);
            Real sx = slopes(i, j, k, 0);
            Real sy = slopes(i, j, k, 1);

            ros(i, j, k) = rothermel_ros_cell(ux, uy, sx, sy, rc);
        });
    }
}

std::vector<RothermelComputed> build_fuel_rothermel_table(
    Real moisture_1hr,
    Real moisture_10hr,
    Real moisture_100hr,
    int fuel_set,
    Real moisture_live,
    bool use_wind_limit,
    const FuelModelParams* fp_tbl,
    int wind_limit_mode)
{
    std::vector<RothermelComputed> table(ROTHERMEL_TABLE_SIZE);

    // Code 0 is non-burnable: every coefficient zero, so R0 = 0 and the
    // kernel returns zero spread whatever the wind and slope.
    table[0] = RothermelComputed{};

    // Slots 1-13 hold the Anderson models at their own codes, 14-53 the
    // Scott-Burgan models and 54-69 the deck-defined ones, which only the
    // caller's slot table knows.
    for (int slot = 1; slot < ROTHERMEL_TABLE_SIZE; ++slot) {
        if (fp_tbl == nullptr && slot >= FUEL_SLOT_CUSTOM_BASE) {
            table[slot] = RothermelComputed{};   // no slot table, so no deck-defined fuel: no spread
            continue;
        }
        const FuelModelParams fp = (fp_tbl != nullptr)
            ? fp_tbl[slot]
            : get_fuel_params(fuel_code_from_slot(slot),
                              (slot >= FUEL_SLOT_SB40_BASE) ? FUEL_SET_SCOTT_BURGAN40 : fuel_set,
                              moisture_live);
        // A slot with no fuel gets the zeroed entry slot 0 carries rather than
        // the coefficients of a zero load, which divide by the packing ratio and
        // come out non-finite. Custom slots the deck leaves undefined are the
        // case that reaches this; a zero-load entry can never spread anyway.
        if (!(fuel_total_load_kg_m2(fp) > 0.0)) {
            table[slot] = RothermelComputed{};
            continue;
        }
        table[slot] = compute_rothermel_params(fp, moisture_1hr, moisture_10hr, moisture_100hr, use_wind_limit, wind_limit_mode);
    }
    return table;
}

void compute_ros_field(
    MultiFab& fire_ros,
    const MultiFab& fire_wind,
    const MultiFab& fire_slopes,
    const RothermelComputed& rc_default,
    const MultiFab* fuel_model,
    const RothermelComputed* table,
    int table_size,
    int fuel_set,
    const RothermelCellInputs* cell_mc)
{
    const bool per_cell_mc = (cell_mc != nullptr) && cell_mc->active();
    const bool per_fuel    = (fuel_model != nullptr && table != nullptr && table_size > 0);
    if (!per_fuel && !per_cell_mc) {
        compute_ros_field(fire_ros, fire_wind, fire_slopes, rc_default);
        return;
    }

    // per-cell coefficients (erf.fire.rothermel_cell_moisture): the packed
    // field built once per step by build_cell_rothermel_coefficients
    const bool has_codes = (fuel_model != nullptr);

    for (MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> ros = fire_ros.array(mfi);
        Array4<const Real> wind = fire_wind.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);
        Array4<const Real> fuel;
        if (has_codes) { fuel = fuel_model->const_array(mfi); }
        Array4<const Real> rcc;
        if (per_cell_mc) { rcc = cell_mc->rc_cell->const_array(mfi); }

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            Real ux = wind(i, j, k, 0);
            Real uy = wind(i, j, k, 1);
            Real sx = slopes(i, j, k, 0);
            Real sy = slopes(i, j, k, 1);

            const int code = has_codes ? static_cast<int>(fuel(i, j, k) + Real(0.5)) : -1;   // nearest integer, as every reader
            RothermelComputed rc = rc_default;
            if (per_cell_mc) {
                rc = unpack_rothermel(rcc, i, j);
            } else if (per_fuel) {
                const int idx = fuel_table_index(code, fuel_set);
                if (idx >= 0 && idx < table_size) { rc = table[idx]; }
            }
            ros(i, j, k) = rothermel_ros_cell(ux, uy, sx, sy, rc);
        });
    }
}

void build_cell_rothermel_coefficients(MultiFab& rc_cell,
                                       const MultiFab& fuel_mc,
                                       const MultiFab* fuel_model,
                                       const FuelModelParams* fp_tbl,
                                       int fp_tbl_size,
                                       int fuel_set,
                                       Real M_live,
                                       const FuelModelParams& fp_uniform,
                                       bool use_wind_limit,
                                       int wind_limit_mode)
{
    AMREX_ALWAYS_ASSERT(rc_cell.nComp() >= ROTHERMEL_RC_NCOMP);
    const bool has_codes = (fuel_model != nullptr);
    for (MFIter mfi(rc_cell, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> rcc = rc_cell.array(mfi);
        Array4<const Real> mc = fuel_mc.const_array(mfi);
        Array4<const Real> fuel;
        if (has_codes) { fuel = fuel_model->const_array(mfi); }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // a code outside the table keeps the uniform fuel, as the per-fuel
            // table path keeps the uniform entry for it
            const int code = has_codes ? static_cast<int>(fuel(i, j, k) + Real(0.5)) : -1;
            const FuelModelParams fp_cell = (has_codes && fuel_table_index(code, fuel_set) >= 0)
                ? fuel_params_at_code(code, fuel_set, M_live, fp_tbl, fp_tbl_size)
                : fp_uniform;
            // the dead moistures as the domain-mean rebuild clamps them
            const Real m1   = amrex::max(Real(0.01), amrex::min(mc(i, j, k, 0), Real(0.40)));
            const Real m10  = amrex::max(Real(0.01), amrex::min(mc(i, j, k, 1), Real(0.40)));
            const Real m100 = amrex::max(Real(0.01), amrex::min(mc(i, j, k, 2), Real(0.40)));
            // a fuel with no load (a non-burnable code) cannot spread
            const RothermelComputed rc = (fuel_total_load_kg_m2(fp_cell) > 0.0)
                ? rothermel_coefficients(fp_cell, m1, m10, m100, use_wind_limit, wind_limit_mode)
                : RothermelComputed{};
            pack_rothermel(rc, rcc, i, j);
        });
    }
}
