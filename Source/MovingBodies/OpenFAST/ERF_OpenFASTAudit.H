#ifndef ERF_OPENFAST_AUDIT_H
#define ERF_OPENFAST_AUDIT_H

#include <array>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_MovingBodiesInputs.H"
#include "ERF_OpenFASTDriver.H"

namespace erf_openfast {

/**
 * The value field of an OpenFAST input line "value  Name  - description" for the named key,
 * or an empty string when the file has no such line. Quotes around the value are removed.
 */
std::string openfast_value (const std::string& fname, const std::string& key);

/** The path of the module file the primary file names under key (AeroFile, EDFile, ...), resolved against the primary file's directory. */
std::string openfast_module_path (const std::string& fst_file, const std::string& key);

/** Whether the file is an OpenFAST primary input file (it names CompAero or AirDens); the bundled stub's deck is not. */
bool is_openfast_model (const std::string& fst_file);

/** The model's air density: AirDens of the primary file, or of the AeroDyn file when the primary says "default". False when neither gives a number. */
bool openfast_air_density (const std::string& fst_file, amrex::Real& rho);

/** The model's gravity (Gravity of the primary file). False when it is not there. */
bool openfast_gravity (const std::string& fst_file, amrex::Real& g);

/**
 * One finding of the start-up audit: fatal ones abort the run once every finding has been
 * printed, so the user sees the whole list at once.
 */
struct AuditFinding
{
    bool fatal;
    std::string message;
};

/**
 * The model files against the body's inputs: the air density the model uses must equal the
 * body's air_density (fatal), aerodynamics must be on when the loads go into the flow
 * (fatal), the model's gravity should be ERF's (warning), and a forced tower needs AeroDyn's
 * tower aerodynamics (warning). A deck that is not an OpenFAST primary file (the stub) gets
 * one non-fatal note and no file checks.
 */
std::vector<AuditFinding> audit_model (const MovingBodyInputs& b, amrex::Real erf_gravity);

/**
 * The rotor against the domain and the mesh: the swept disc must clear the ground, the top
 * and every non-periodic side (fatal) and the base must lie in the domain (fatal); the mesh
 * should resolve the rotor (at least 8 cells across the diameter), the kernel should be at
 * least a cell wide, an actuator line's points should be no farther apart than the kernel,
 * and the kernel should not be cut by a non-periodic boundary (warnings).
 * @param prob_lo,prob_hi the domain, dx the cell sizes, periodic 1 for each periodic direction
 */
std::vector<AuditFinding> audit_geometry (const TurbineState& t, const MovingBodyInputs& b, amrex::Real epsilon,
                                          const std::array<amrex::Real,3>& prob_lo, const std::array<amrex::Real,3>& prob_hi,
                                          const std::array<amrex::Real,3>& dx, const std::array<int,3>& periodic);

/** Two rotors whose swept discs intersect (fatal). */
std::vector<AuditFinding> audit_overlap (const std::vector<TurbineState>& turbs);

/**
 * ERF's density at the hub against the model's: they must agree within the relative tolerance
 * (fatal), since the loads OpenFAST computes with its density are put into a flow of ERF's.
 */
std::vector<AuditFinding> audit_density (const std::string& name, amrex::Real rho_model, amrex::Real rho_hub, amrex::Real tolerance);

/** The rotor against the wind at its hub: facing away from it, or yawed more than 30 degrees (warnings). */
std::vector<AuditFinding> audit_facing (const TurbineState& t, const std::array<amrex::Real,3>& u_hub);

/** Print every finding under a heading for the body; sets any_fatal when one is fatal. */
void report_audit (const std::string& name, const std::vector<AuditFinding>& findings, bool& any_fatal);

} // namespace erf_openfast

#endif
