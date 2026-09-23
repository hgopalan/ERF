#include <ERF_CustomFuel.H>

#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

using amrex::Real;

namespace {

/// SI to the US units FuelModelParams carries.
constexpr Real KG_M2_TO_LB_FT2  = 1.0 / 4.88243;
constexpr Real INV_M_TO_INV_FT  = 1.0 / 3.28084;
constexpr Real M_TO_FT          = 1.0 / 0.3048;
constexpr Real J_KG_TO_BTU_LB   = 1.0 / 2326.0;
constexpr Real KG_M3_TO_LB_FT3  = 1.0 / 16.0185;

}  // namespace

void
CustomFuelTable::init_from_inputs ()
{
    amrex::ParmParse pp("erf.fire");

    std::vector<int> codes;
    if (!pp.queryarr("custom_fuel.codes", codes) || codes.empty()) { return; }

    auto fail = [] (const std::string& msg) { amrex::Abort("ERF-Fire custom fuel: " + msg); };

    for (int code : codes) {
        if (!custom_fuel_code(code)) {
            fail("erf.fire.custom_fuel.codes holds " + std::to_string(code) + ", outside the custom range "
                 + std::to_string(CUSTOM_FUEL_CODE_BASE) + "-"
                 + std::to_string(CUSTOM_FUEL_CODE_BASE + CUSTOM_FUEL_MAX - 1));
        }
        const int slot = code - CUSTOM_FUEL_CODE_BASE;
        if (m_defined[slot]) {
            fail("erf.fire.custom_fuel.codes lists " + std::to_string(code) + " twice");
        }

        const std::string pre = "custom_fuel." + std::to_string(code) + ".";
        const std::string key = "erf.fire." + pre;

        // Every property is required but the four with a standard value, so a
        // deck cannot define a fuel model by accident with a partial block.
        auto need = [&] (const char* name, Real& v) {
            if (!pp.query((pre + name).c_str(), v)) { fail(key + name + " is required"); }
        };
        auto band = [&] (const char* name, Real v, Real lo, Real hi) {
            if (!(v >= lo && v <= hi)) {
                fail(key + name + " must be in [" + std::to_string(lo) + ", " + std::to_string(hi)
                     + "], got " + std::to_string(v));
            }
        };

        Real w_1h = 0.0, w_10h = 0.0, w_100h = 0.0, w_lh = 0.0, w_lw = 0.0;
        need("w_1h_kg_m2", w_1h);
        pp.query((pre + "w_10h_kg_m2").c_str(),  w_10h);
        pp.query((pre + "w_100h_kg_m2").c_str(), w_100h);
        pp.query((pre + "w_live_herb_kg_m2").c_str(),  w_lh);
        pp.query((pre + "w_live_woody_kg_m2").c_str(), w_lw);

        Real sav_1h = 0.0, depth = 0.0, Mx = 0.0, heat = 0.0;
        need("sav_1h_1_m",        sav_1h);
        need("depth_m",           depth);
        need("moisture_ext",      Mx);
        need("heat_content_J_kg", heat);

        Real sav_lh = sav_1h, sav_lw = sav_1h, rho_p = 512.0, burn_s = 0.0;
        pp.query((pre + "sav_live_herb_1_m").c_str(),  sav_lh);
        pp.query((pre + "sav_live_woody_1_m").c_str(), sav_lw);
        pp.query((pre + "density_kg_m3").c_str(), rho_p);
        // The flaming burn time only matters under erf.fire.burnout_model = sfire;
        // 7 s is the published grass value and a poor default for a coarse fuel,
        // so a deck that runs that model on a custom fuel has to say.
        pp.query((pre + "burnout_time_s").c_str(), burn_s);

        // Bands wide enough for anything from cured grass to a structural fuel,
        // and tight enough to catch a unit mistake (a load given in lb/ft2, a
        // heat content in BTU/lb, a SAV in 1/ft).
        band("w_1h_kg_m2",            w_1h,   0.0,     500.0);
        band("w_10h_kg_m2",           w_10h,  0.0,     500.0);
        band("w_100h_kg_m2",          w_100h, 0.0,     500.0);
        band("w_live_herb_kg_m2",     w_lh,   0.0,     500.0);
        band("w_live_woody_kg_m2",    w_lw,   0.0,     500.0);
        band("sav_1h_1_m",            sav_1h, 10.0,    12000.0);
        band("sav_live_herb_1_m",     sav_lh, 10.0,    12000.0);
        band("sav_live_woody_1_m",    sav_lw, 10.0,    12000.0);
        band("moisture_ext",          Mx,     0.01,    2.0);
        band("heat_content_J_kg",     heat,   5.0e6,   5.0e7);
        band("density_kg_m3", rho_p, 50.0,    2000.0);
        band("burnout_time_s",        burn_s, 0.0,     1.0e5);

        // Balbi returns zero spread below a 0.01 m bed depth, silently, so a
        // depth under that would make the fuel model unburnable with no message.
        if (!(depth > 0.01 && depth <= 50.0)) {
            fail(key + "depth_m must be in (0.01, 50], got " + std::to_string(depth)
                 + "; the Balbi models return zero spread at or below 0.01 m");
        }
        if (!(w_1h + w_10h + w_100h + w_lh + w_lw > 0.0)) {
            fail(key + "* has no fuel load: at least one of w_1h_kg_m2, w_10h_kg_m2, "
                 "w_100h_kg_m2, w_live_herb_kg_m2, w_live_woody_kg_m2 must be > 0");
        }

        CustomFuelEntry e;
        e.fp.w_d1   = w_1h   * KG_M2_TO_LB_FT2;
        e.fp.w_d10  = w_10h  * KG_M2_TO_LB_FT2;
        e.fp.w_d100 = w_100h * KG_M2_TO_LB_FT2;
        e.fp.w_lh   = w_lh   * KG_M2_TO_LB_FT2;
        e.fp.w_lw   = w_lw   * KG_M2_TO_LB_FT2;
        e.fp.sigma_d1 = sav_1h * INV_M_TO_INV_FT;
        e.fp.sigma_lh = (w_lh > 0.0) ? sav_lh * INV_M_TO_INV_FT : Real(0.0);
        e.fp.sigma_lw = (w_lw > 0.0) ? sav_lw * INV_M_TO_INV_FT : Real(0.0);
        e.fp.delta        = depth * M_TO_FT;
        e.fp.Mx           = Mx;
        e.fp.heat_content = heat  * J_KG_TO_BTU_LB;
        e.fp.rho_p        = rho_p * KG_M3_TO_LB_FT3;
        e.burnout_time_s  = burn_s;
        pp.query((pre + "name").c_str(), e.name);
        if (e.name.empty()) { e.name = "custom_" + std::to_string(code); }

        m_entry[slot]   = e;
        m_defined[slot] = true;
        ++m_count;
    }
}

const FuelModelParams&
CustomFuelTable::params (int code) const
{
    if (!has_code(code)) {
        amrex::Abort("ERF-Fire custom fuel: fuel code " + std::to_string(code)
                     + " has no erf.fire.custom_fuel." + std::to_string(code) + ".* block");
    }
    return m_entry[code - CUSTOM_FUEL_CODE_BASE].fp;
}

Real
CustomFuelTable::burnout_time_s (int code) const
{
    if (!has_code(code)) {
        amrex::Abort("ERF-Fire custom fuel: fuel code " + std::to_string(code)
                     + " has no erf.fire.custom_fuel." + std::to_string(code) + ".* block");
    }
    return m_entry[code - CUSTOM_FUEL_CODE_BASE].burnout_time_s;
}

void
CustomFuelTable::print_summary () const
{
    if (!active()) { return; }
    amrex::Print() << "[FIRE DEBUG] Custom fuel models: " << m_count << "\n";
    for (int i = 0; i < CUSTOM_FUEL_MAX; ++i) {
        if (!m_defined[i]) { continue; }
        const CustomFuelEntry& e = m_entry[i];
        amrex::Print() << "[FIRE DEBUG]   code " << (CUSTOM_FUEL_CODE_BASE + i)
                       << " slot " << (FUEL_SLOT_CUSTOM_BASE + i)
                       << " \"" << e.name << "\""
                       << "  load=" << fuel_total_load_kg_m2(e.fp) << " kg/m2"
                       << "  sav_1h=" << e.fp.sigma_d1 * 3.28084 << " 1/m"
                       << "  depth=" << e.fp.delta * 0.3048 << " m"
                       << "  Mx=" << e.fp.Mx
                       << "  h=" << e.fp.heat_content * 2326.0 << " J/kg"
                       << "  rho_p=" << e.fp.rho_p * 16.0185 << " kg/m3"
                       << "  burnout=" << e.burnout_time_s << " s\n";
    }
}

void
CustomFuelTable::validate_map_codes (const amrex::MultiFab& fuel_model,
                                    const std::vector<int>& nonburnable_codes) const
{
    // A raster cell holding an undefined custom code would otherwise fall
    // through to the fuel set's unknown-code default and burn as grass.
    // has_code() is host-side, so the accepted set travels into the kernel as a
    // bitmask captured by value, one bit per custom slot.
    //
    // A code the deck marks non-burnable is accepted without a block: it never
    // spreads, so it needs no properties. build_nonburnable_mask() reads the
    // same list later, and the slot table already gives an undefined custom
    // slot a zero load, so the two agree.
    int defined_mask = 0;
    for (int i = 0; i < CUSTOM_FUEL_MAX; ++i) {
        if (m_defined[i]) { defined_mask |= (1 << i); }
    }
    for (int code : nonburnable_codes) {
        if (custom_fuel_code(code)) { defined_mask |= (1 << (code - CUSTOM_FUEL_CODE_BASE)); }
    }

    amrex::Gpu::DeviceScalar<int> d_bad(0);
    int* p_bad = d_bad.dataPtr();

    for (amrex::MFIter mfi(fuel_model); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& code = fuel_model.const_array(mfi);

        // The largest offending code is enough for the message, and taking the
        // maximum makes it the same on every rank count and box decomposition.
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int c = static_cast<int>(code(i, j, k) + amrex::Real(0.5));
            if (custom_fuel_code(c) && ((defined_mask >> (c - CUSTOM_FUEL_CODE_BASE)) & 1) == 0) {
                amrex::Gpu::Atomic::Max(p_bad, c);
            }
        });
    }
    amrex::Gpu::streamSynchronize();

    int bad_code = d_bad.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(bad_code);

    if (bad_code > 0) {
        amrex::Abort("ERF-Fire custom fuel: the fuel map holds code " + std::to_string(bad_code)
                     + ", which is in the custom range but has no erf.fire.custom_fuel."
                     + std::to_string(bad_code) + ".* block; add one or list the code in "
                     "erf.fire.fuel_map.nonburnable_codes");
    }
}
