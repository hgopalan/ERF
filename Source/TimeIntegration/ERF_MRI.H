#ifndef ERF_MRI_H
#define ERF_MRI_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntegratorBase.H>

#include <ERF_TI_slow_headers.H>
#include <ERF_TI_fast_headers.H>

#include <functional>

template<class T>
class MRISplitIntegrator
{
private:
   /**
    * \brief rhs is the right-hand-side function the integrator will use.
    */
    std::function<void(T&, const T&,     const amrex::Real, const amrex::Real     )> rhs;
    std::function<void(T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int)> slow_rhs_pre;
    std::function<void(T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int)> slow_rhs_inc;
    std::function<void(T&, T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int )> slow_rhs_post;
    std::function<void(int, int, int, T&, const T&, T&, T&, T&, const amrex::Real, const amrex::Real,
                                                                const amrex::Real, const amrex::Real)> fast_rhs;

   /**
    * \brief Integrator timestep size (Real)
    */
    amrex::Real timestep;

   /**
    * \brief The ratio of slow timestep size / fast timestep size (int)
    */
    int slow_fast_timestep_ratio = 0;

   /**
    * \brief Should we not do acoustic substepping
    */
    int no_substepping;

   /**
    * \brief Should we use the anelastic integrator
    */
    int anelastic;

   /**
    * \brief How many components in the cell-centered MultiFab
    */
    int ncomp_cons;

   /**
    * \brief Do we follow the recommendation to only perform a single substep in the first RK stage
    */
    int force_stage1_single_substep;

   /**
    * \brief The no_substep function is called when we have no acoustic substepping
    */
    std::function<void (T&, T&, T&, amrex::Real, amrex::Real, int)> no_substep;


    amrex::Vector<std::unique_ptr<T> > T_store;
    T* S_sum;
    T* S_scratch;
    T* F_slow;

    void initialize_data (const T& S_data)
    {
        // TODO: We can optimize memory by making the cell-centered part of S_sum, S_scratch
        //       have only 2 components, not ncomp_cons components
        const bool include_ghost = true;
        amrex::IntegratorOps<T>::CreateLike(T_store, S_data, include_ghost);
        S_sum = T_store[0].get();
        amrex::IntegratorOps<T>::CreateLike(T_store, S_data, include_ghost);
        S_scratch = T_store[1].get();
        amrex::IntegratorOps<T>::CreateLike(T_store, S_data, include_ghost);
        F_slow = T_store[2].get();
    }

public:
    MRISplitIntegrator () = default;

    MRISplitIntegrator (const T& S_data)
    {
        initialize_data(S_data);
    }

    void initialize (const T& S_data)
    {
        initialize_data(S_data);
    }

    ~MRISplitIntegrator () = default;

    // Declare a default move constructor so we ensure the destructor is
    // not called when we return an object of this class by value
    MRISplitIntegrator(MRISplitIntegrator&&)  noexcept = default;

    // Declare a default move assignment operator
    MRISplitIntegrator& operator=(MRISplitIntegrator&& other)  noexcept = default;

    // Delete the copy constructor and copy assignment operators because
    // the integrator allocates internal memory that is best initialized
    // from scratch when needed instead of making a copy.

    // Delete the copy constructor
    MRISplitIntegrator(const MRISplitIntegrator& other) = delete;
    //
    // Delete the copy assignment operator
    MRISplitIntegrator& operator=(const MRISplitIntegrator& other) = delete;

    void setNcompCons(int _ncomp_cons)
    {
        ncomp_cons = _ncomp_cons;
    }

    void setAnelastic(int _anelastic)
    {
        anelastic = _anelastic;
    }

    void setNoSubstepping(int _no_substepping)
    {
        no_substepping = _no_substepping;
    }

    void setForceFirstStageSingleSubstep(int _force_stage1_single_substep)
    {
        force_stage1_single_substep = _force_stage1_single_substep;
    }

    void set_slow_rhs_pre (std::function<void(T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int)> F)
    {
        slow_rhs_pre = F;
    }
    void set_slow_rhs_inc (std::function<void(T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int)> F)
    {
        slow_rhs_inc = F;
    }
    void set_slow_rhs_post (std::function<void(T&, T&, T&, T&, T&, const amrex::Real, const amrex::Real, const amrex::Real, const int)> F)
    {
        slow_rhs_post = F;
    }

    void set_fast_rhs (std::function<void(int, int, int, T&, const T&, T&, T&, T&,
                                          const amrex::Real, const amrex::Real,
                                          const amrex::Real, const amrex::Real)> F)
    {
        fast_rhs = F;
    }

    void set_slow_fast_timestep_ratio (const int timestep_ratio = 1)
    {
        slow_fast_timestep_ratio = timestep_ratio;
    }

    int get_slow_fast_timestep_ratio ()
    {
        return slow_fast_timestep_ratio;
    }

    void set_no_substep (std::function<void (T&, T&, T&, amrex::Real, amrex::Real, int)> F)
    {
        no_substep = F;
    }

    std::function<void(T&, const T&, const amrex::Real, int)> get_rhs ()
    {
        return rhs;
    }

    amrex::Real advance (T& S_old, T& S_new, amrex::Real time, const amrex::Real time_step)
    {
    BL_PROFILE_REGION("MRI_advance");
        using namespace amrex;

        // *******************************************************************************
        // !no_substepping: we only update the fast variables every fast timestep, then update
        //                  the slow variables after the acoustic sub-stepping.  This has
        //                  two calls to slow_rhs so that we can update the slow variables
        //                  with the velocity field after the acoustic substepping using
        //                  the time-averaged velocity from the substepping
        // no_substepping:  we don't do any acoustic subcyling so we only make one call per RK
        //                  stage to slow_rhs
        // *******************************************************************************
        timestep = time_step;

        const int substep_ratio = get_slow_fast_timestep_ratio();

        if (!no_substepping) {
            AMREX_ALWAYS_ASSERT(substep_ratio > 1 && substep_ratio % 2 == 0);
        }

        // Assume before advance() that S_old is valid data at the current time ("time" argument)
        // And that if data is a MultiFab, both S_old and S_new contain ghost cells for evaluating a stencil based RHS
        // We need this from S_old. This is convenient for S_new to have so we can use it
        // as scratch space for stage values without creating a new scratch MultiFab with ghost cells.

        // NOTE: In the following, we use S_new to hold S*, S**, and finally, S^(n+1) at the new time
        // DEFINITIONS:
        // S_old  = S^n
        // S_sum  = S(t)
        // F_slow = F(S_stage)

        int n_data = IntVars::NumTypes;

        /**********************************************/
        /* RK3 Integration with Acoustic Sub-stepping */
        /**********************************************/
        Vector<int> num_vars = {ncomp_cons, 1, 1, 1};
        for (int i(0); i<n_data; ++i)
        {
            // Copy old -> new
            MultiFab::Copy(S_new[i],S_old[i],0,0,num_vars[i],S_old[i].nGrowVect());
        }

        // Timestep taken by the fast integrator
        amrex::Real dtau;

        // How many timesteps taken by the fast integrator
        int nsubsteps;

        // This is the final time of the full timestep (also the 3rd RK stage)
        // Real new_time = time + timestep;

        amrex::Real time_stage = time;
        amrex::Real old_time_stage;

        const amrex::Real sub_timestep = timestep / substep_ratio;

        if (!anelastic) {
          // RK3 for compressible integrator
          for (int nrk = 0; nrk < 3; nrk++)
          {
            // Capture the time we got to in the previous RK step
            old_time_stage = time_stage;

            if (nrk == 0) {
                if (force_stage1_single_substep || no_substepping) {
                    nsubsteps = 1;               dtau = timestep / 3.0;
                } else {
                    nsubsteps = substep_ratio/3; dtau = sub_timestep  ;
                }
                time_stage = time + timestep / 3.0;
            }
            if (nrk == 1) {
                if (no_substepping) {
                    nsubsteps =               1; dtau = 0.5 * timestep;
                } else {
                    nsubsteps = substep_ratio/2; dtau =   sub_timestep;
                }
                time_stage = time + timestep / 2.0;
            }
            if (nrk == 2) {
                if (no_substepping) {
                    nsubsteps =             1;   dtau =     timestep;
                } else {
                    nsubsteps = substep_ratio;   dtau = sub_timestep;

                    // STRT HACK -- this hack can be used to approximate the no-substepping algorithm
                    // nsubsteps = 1;   dtau = timestep;
                    // END HACK
                }
                time_stage = time + timestep;
            }

            // step 1 starts with S_stage = S^n  and we always start substepping at the old time
            // step 2 starts with S_stage = S^*  and we always start substepping at the old time
            // step 3 starts with S_stage = S^** and we always start substepping at the old time

            // S_scratch also holds the average momenta over the fast iterations --
            //    to be used to update the slow variables -- we will initialize with
            //    the momenta used in the first call to the slow_rhs, then update
            //    inside fast_rhs, then use these values in the later call to slow_rhs

            slow_rhs_pre(*F_slow, S_old, S_new, *S_scratch, time, old_time_stage, time_stage, nrk);

            amrex::Real inv_fac = 1.0 / static_cast<amrex::Real>(nsubsteps);

            // ****************************************************
            // Acoustic substepping
            // ****************************************************
            if (!no_substepping)
            {
                // *******************************************************************************
                // Update the fast variables
                // *******************************************************************************
                for (int ks = 0; ks < nsubsteps; ++ks)
                {
                    fast_rhs(ks, nsubsteps, nrk, *F_slow, S_old, S_new, *S_sum, *S_scratch, dtau, inv_fac,
                             time + ks*dtau, time + (ks+1) * dtau);

                } // ks

            } else {
                no_substep(*S_sum, S_old, *F_slow, time + nsubsteps*dtau, nsubsteps*dtau, nrk);
            }

            // ****************************************************
            // Evaluate F_slow(S_stage) only for the slow variables
            // Note that we are using the current stage versions (in S_new) of the slow variables
            //      (because we didn't update the slow variables in the substepping)
            //       but we are using the "new" versions (in S_sum) of the velocities
            //      (because we did    update the fast variables in the substepping)
            // ****************************************************
            slow_rhs_post(*F_slow, S_old, S_new, *S_sum, *S_scratch, time, old_time_stage, time_stage, nrk);
          } // nrk

        } else {
          // RK2 for anelastic integrator
          for (int nrk = 0; nrk < 2; nrk++)
          {
            // Capture the time we got to in the previous RK step
            old_time_stage = time_stage;

            if (nrk == 0) { nsubsteps = 1; dtau = timestep; time_stage = time + timestep; }
            if (nrk == 1) { nsubsteps = 1; dtau = timestep; time_stage = time + timestep; }

            // S_scratch also holds the average momenta over the fast iterations --
            //    to be used to update the slow variables -- we will initialize with
            //    the momenta used in the first call to the slow_rhs, then update
            //    inside fast_rhs, then use these values in the later call to slow_rhs

            slow_rhs_inc(*F_slow, S_old, S_new, *S_scratch, time, old_time_stage, time_stage, nrk);

            no_substep(*S_sum, S_old, *F_slow, time + nsubsteps*dtau, nsubsteps*dtau, nrk);

            // ****************************************************
            // Evaluate F_slow(S_stage) only for the slow variables
            // Note that we are using the current stage versions (in S_new) of the slow variables
            //      (because we didn't update the slow variables in the substepping)
            //       but we are using the "new" versions (in S_sum) of the velocities
            //      (because we did    update the fast variables in the substepping)
            // ****************************************************
            slow_rhs_post(*F_slow, S_old, S_new, *S_sum, *S_scratch, time, old_time_stage, time_stage, nrk);
          } // nrk
        }

        // Return timestep
        return timestep;
    }

    void map_data (std::function<void(T&)> Map)
    {
        for (auto& F : T_store) {
            Map(*F);
        }
    }
};

#endif
