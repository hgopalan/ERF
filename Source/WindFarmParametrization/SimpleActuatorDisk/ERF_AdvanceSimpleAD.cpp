#include <ERF_SimpleAD.H>
#include <ERF_IndexDefines.H>
#include <ERF_Interpolation_1D.H>

using namespace amrex;

void
SimpleAD::advance (const Geometry& geom,
                   const Real& dt_advance,
                   MultiFab& cons_in,
                   MultiFab& mf_vars_simpleAD,
                   MultiFab& U_old,
                   MultiFab& V_old,
                   MultiFab& W_old,
                   const MultiFab& mf_Nturb,
                   const MultiFab& mf_SMark,
                   const Real& time)
{
    AMREX_ALWAYS_ASSERT(W_old.nComp() > 0);
    AMREX_ALWAYS_ASSERT(mf_Nturb.nComp() > 0);
    compute_freestream_velocity(cons_in, U_old, V_old, mf_SMark);
    source_terms_cellcentered(geom, cons_in, mf_SMark, mf_vars_simpleAD);
    update(dt_advance, cons_in, U_old, V_old, mf_vars_simpleAD);
    compute_power_output(time);
}

void
SimpleAD::compute_power_output (const Real& time)
{
     get_turb_loc(xloc, yloc);
     get_turb_spec(rotor_rad, hub_height, thrust_coeff_standing,
                  wind_speed, thrust_coeff, power);

     const int n_spec_table = wind_speed.size();
  // Compute power based on the look-up table

    if (ParallelDescriptor::IOProcessor()){
        static std::ofstream file("power_output_SimpleAD.txt", std::ios::app);
        // Check if the file opened successfully
        if (!file.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            Abort("Could not open file to write power output in ERF_AdvanceSimpleAD.cpp");
        }
        Real total_power = 0.0;
        for(int it=0; it<xloc.size(); it++){
            Real avg_vel = freestream_velocity[it]/(disk_cell_count[it] + 1e-10);
            Real turb_power = interpolate_1d(wind_speed.data(), power.data(), avg_vel, n_spec_table);
            total_power = total_power + turb_power;
            //printf("avg vel and power is %d %0.15g, %0.15g\n", it, avg_vel, turb_power);
        }
        file << time << " " << total_power << "\n";
        file.flush();
    }
}

void
SimpleAD::update (const Real& dt_advance,
                  MultiFab& cons_in,
                  MultiFab& U_old, MultiFab& V_old,
                  const MultiFab& mf_vars_simpleAD)
{

    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        Box tbx = mfi.nodaltilebox(0);
        Box tby = mfi.nodaltilebox(1);

        auto simpleAD_array = mf_vars_simpleAD.array(mfi);
        auto u_vel       = U_old.array(mfi);
        auto v_vel       = V_old.array(mfi);

        ParallelFor(tbx, tby,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            u_vel(i,j,k) = u_vel(i,j,k) + (simpleAD_array(i-1,j,k,0) + simpleAD_array(i,j,k,0))/2.0*dt_advance;
        },
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            v_vel(i,j,k) = v_vel(i,j,k) + (simpleAD_array(i,j-1,k,1) + simpleAD_array(i,j,k,1))/2.0*dt_advance;
        });
    }
}

void SimpleAD::compute_freestream_velocity (const MultiFab& cons_in,
                                            const MultiFab& U_old,
                                            const MultiFab& V_old,
                                            const MultiFab& mf_SMark)
{
     get_turb_loc(xloc, yloc);
     freestream_velocity.clear();
     freestream_phi.clear();
     disk_cell_count.clear();
     freestream_velocity.resize(xloc.size(),0.0);
     freestream_phi.resize(xloc.size(),0.0);
     disk_cell_count.resize(xloc.size(),0.0);

     Gpu::DeviceVector<Real> d_freestream_velocity(xloc.size());
     Gpu::DeviceVector<Real> d_freestream_phi(yloc.size());
     Gpu::DeviceVector<Real> d_disk_cell_count(yloc.size());
     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());

     Real* d_freestream_velocity_ptr = d_freestream_velocity.data();
     Real* d_freestream_phi_ptr = d_freestream_phi.data();
     Real* d_disk_cell_count_ptr     = d_disk_cell_count.data();


     for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        auto SMark_array    = mf_SMark.array(mfi);
        auto u_vel          = U_old.array(mfi);
        auto v_vel          = V_old.array(mfi);
        Box tbx = mfi.nodaltilebox(0);

        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

            if(SMark_array(i,j,k,0) != -1.0) {
                int turb_index = static_cast<int>(SMark_array(i,j,k,0));
                Real phi = std::atan2(v_vel(i,j,k),u_vel(i,j,k)); // Wind direction w.r.t the x-dreiction
                Gpu::Atomic::Add(&d_freestream_velocity_ptr[turb_index],std::pow(u_vel(i,j,k)*u_vel(i,j,k) + v_vel(i,j,k)*v_vel(i,j,k),0.5));
                Gpu::Atomic::Add(&d_disk_cell_count_ptr[turb_index],1.0);
                Gpu::Atomic::Add(&d_freestream_phi_ptr[turb_index],phi);
            }
        });
    }

    // Copy back to host
    Gpu::copy(Gpu::deviceToHost, d_freestream_velocity.begin(), d_freestream_velocity.end(), freestream_velocity.begin());
    Gpu::copy(Gpu::deviceToHost, d_freestream_phi.begin(), d_freestream_phi.end(), freestream_phi.begin());
    Gpu::copy(Gpu::deviceToHost, d_disk_cell_count.begin(), d_disk_cell_count.end(), disk_cell_count.begin());

    // Reduce the data on every processor
    amrex::ParallelAllReduce::Sum(freestream_velocity.data(),
                                  freestream_velocity.size(),
                                  amrex::ParallelContext::CommunicatorAll());

    amrex::ParallelAllReduce::Sum(freestream_phi.data(),
                                  freestream_phi.size(),
                                  amrex::ParallelContext::CommunicatorAll());


   amrex::ParallelAllReduce::Sum(disk_cell_count.data(),
                                 disk_cell_count.size(),
                                 amrex::ParallelContext::CommunicatorAll());

    get_turb_loc(xloc, yloc);
    /*if (ParallelDescriptor::IOProcessor()){
        for(int it=0; it<xloc.size(); it++){

            std::cout << "turbine index, freestream velocity is " << it << " " << freestream_velocity[it] << " " <<
                                                               disk_cell_count[it]  <<  " " <<
                                                                freestream_velocity[it]/(disk_cell_count[it] + 1e-10) << " " <<
                                                                freestream_phi[it]/(disk_cell_count[it] + 1e-10) << "\n";
        }
    }*/
}

void
SimpleAD::source_terms_cellcentered (const Geometry& geom,
                                     const MultiFab& cons_in,
                                     const MultiFab& mf_SMark,
                                     MultiFab& mf_vars_simpleAD)
{

    get_turb_loc(xloc, yloc);
    get_turb_spec(rotor_rad, hub_height, thrust_coeff_standing,
                  wind_speed, thrust_coeff, power);

    Gpu::DeviceVector<Real> d_xloc(xloc.size());
    Gpu::DeviceVector<Real> d_yloc(yloc.size());
    Gpu::copy(Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    Gpu::copy(Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());

      auto dx = geom.CellSizeArray();

  // Domain valid box
      const amrex::Box& domain = geom.Domain();
      int domlo_x = domain.smallEnd(0);
      int domhi_x = domain.bigEnd(0) + 1;
      int domlo_y = domain.smallEnd(1);
      int domhi_y = domain.bigEnd(1) + 1;
      int domlo_z = domain.smallEnd(2);
      int domhi_z = domain.bigEnd(2) + 1;

      // The order of variables are - Vabs dVabsdt, dudt, dvdt, dTKEdt
      mf_vars_simpleAD.setVal(0.0);

      long unsigned int nturbs = xloc.size();

     Gpu::DeviceVector<Real> d_freestream_velocity(nturbs);
     Gpu::DeviceVector<Real> d_freestream_phi(nturbs);
     Gpu::DeviceVector<Real> d_disk_cell_count(nturbs);
     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());

     Real* d_freestream_velocity_ptr = d_freestream_velocity.data();
     Real* d_freestream_phi_ptr = d_freestream_phi.data();
     Real* d_disk_cell_count_ptr     = d_disk_cell_count.data();

    get_turb_disk_angle(turb_disk_angle);
    Real nx = -std::cos(turb_disk_angle);
    Real ny = -std::sin(turb_disk_angle);
    Real d_turb_disk_angle = turb_disk_angle;

    Gpu::DeviceVector<Real> d_wind_speed(wind_speed.size());
    Gpu::DeviceVector<Real> d_thrust_coeff(thrust_coeff.size());

    // Copy data from host vectors to device vectors
    Gpu::copy(Gpu::hostToDevice, wind_speed.begin(), wind_speed.end(), d_wind_speed.begin());
    Gpu::copy(Gpu::hostToDevice, thrust_coeff.begin(), thrust_coeff.end(), d_thrust_coeff.begin());

    const Real* wind_speed_d     = d_wind_speed.dataPtr();
    const Real* thrust_coeff_d   = d_thrust_coeff.dataPtr();
    const int n_spec_table = d_wind_speed.size();

    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& gbx      = mfi.growntilebox(1);
        auto SMark_array    = mf_SMark.array(mfi);
        auto simpleAD_array = mf_vars_simpleAD.array(mfi);

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int ii = amrex::min(amrex::max(i, domlo_x), domhi_x);
            int jj = amrex::min(amrex::max(j, domlo_y), domhi_y);
            int kk = amrex::min(amrex::max(k, domlo_z), domhi_z);


            Real source_x = 0.0;
            Real source_y = 0.0;

            int it = static_cast<int>(SMark_array(ii,jj,kk,1));

              if(it != -1) {
                Real avg_vel  = d_freestream_velocity_ptr[it]/(d_disk_cell_count_ptr[it] + 1e-10);
                Real phi      = d_freestream_phi_ptr[it]/(d_disk_cell_count_ptr[it] + 1e-10);

                Real C_T = interpolate_1d(wind_speed_d, thrust_coeff_d, avg_vel, n_spec_table);
                Real a;
                if(C_T <= 1) {
                    a = 0.5 - 0.5*std::pow(1.0-C_T,0.5);
                }
                Real Uinfty_dot_nhat = avg_vel*(std::cos(phi)*nx + std::sin(phi)*ny);
                    if(C_T <= 1) {
                        source_x = -2.0*std::pow(Uinfty_dot_nhat, 2.0)*a*(1.0-a)*dx[1]*dx[2]*std::cos(d_turb_disk_angle)/(dx[0]*dx[1]*dx[2])*std::cos(phi);
                        source_y = -2.0*std::pow(Uinfty_dot_nhat, 2.0)*a*(1.0-a)*dx[1]*dx[2]*std::cos(d_turb_disk_angle)/(dx[0]*dx[1]*dx[2])*std::sin(phi);
                    }
                    else {
                        source_x = -0.5*C_T*std::pow(Uinfty_dot_nhat, 2.0)*dx[1]*dx[2]*std::cos(d_turb_disk_angle)/(dx[0]*dx[1]*dx[2])*std::cos(phi);
                        source_y = -0.5*C_T*std::pow(Uinfty_dot_nhat, 2.0)*dx[1]*dx[2]*std::cos(d_turb_disk_angle)/(dx[0]*dx[1]*dx[2])*std::sin(phi);
                    }
             }

            simpleAD_array(i,j,k,0) = source_x;
            simpleAD_array(i,j,k,1) = source_y;
         });
    }
}
