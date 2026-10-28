/**
 * @file ERF_NativeFireEngine.cpp
 * @brief Registers the native fire engine (FireLayer) with ERF
 *
 * erf.fire.engine = native selects it. The coupling keys (erf.fire.grid_ratio,
 * coupling_type, ...) are read once by ERF (FireCouplingParams) and handed
 * here; FireParams reads the engine's own keys and takes the coupling keys
 * from ERF, so the two never disagree.
 */
#include <ERF_FireEngineRegistry.H>
#include <ERF_FireLayer.H>

namespace {

FireParams
native_fire_params (const FireCouplingParams& c)
{
    FireParams p;
    p.engine                = c.engine;
    p.grid_ratio            = c.grid_ratio;
    p.anchor_level          = c.anchor_level;
    p.coupling_type         = c.coupling_type;
    p.heat_flux_alfg        = c.heat_flux_alfg;
    p.fire_atm_feedback     = c.fire_atm_feedback;
    p.inject_latent         = c.inject_latent;
    p.source_mode           = c.source_mode;
    p.source_mode_given     = c.source_mode_given;
    p.heat_open_fraction    = c.heat_open_fraction;
    p.heat_tendency_density = c.heat_tendency_density;
    p.heat_tendency_exner   = c.heat_tendency_exner;
    p.smoke_enable          = c.smoke_enable;
    p.smoke_emission_factor = c.smoke_emission_factor;
    p.fire_debug            = c.fire_debug;
    return p;
}

class NativeFireEngine : public FireLayer
{
public:
    explicit NativeFireEngine (const FireCouplingParams& c) : FireLayer(native_fire_params(c)) {}
};

} // namespace

ERF_REGISTER_FIRE_ENGINE(NativeFireEngine, "native")
