/**
 * @file ERF_UCMCloudCSVReader.cpp
 * @brief Implementation of cloud fraction CSV reader (Phase 4.2)
 *
 * Loads hourly cloud fraction from CSV and provides interpolation.
 */

#include <ERF_UCMCloudCSVReader.H>
#include <fstream>
#include <sstream>
#include <cmath>
#include <AMReX_Print.H>

UCMCloudCSVReader::UCMCloudCSVReader(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        amrex::Error("UCMCloudCSVReader: Cannot open file: " + csv_path);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Skip header row (any line whose first character isn't a digit, '-', or '+')
        char c0 = line[0];
        if (!(std::isdigit(static_cast<unsigned char>(c0)) || c0 == '-' || c0 == '+')) {
            continue;
        }

        // Parse comma-separated values
        std::stringstream ss(line);
        std::string time_str, cf_str;
        
        if (!std::getline(ss, time_str, ',') || !std::getline(ss, cf_str, ',')) {
            amrex::Warning("UCMCloudCSVReader: Line " + std::to_string(line_num) + 
                          " format error, skipping");
            continue;
        }

        try {
            amrex::Real t = std::stod(time_str);
            amrex::Real cf = std::stod(cf_str);
            
            // Guard against unphysical cloud fractions
            cf = amrex::max(0.0, amrex::min(1.0, cf));
            
            time_s_.push_back(t);
            cloud_fraction_.push_back(cf);
        } catch (...) {
            amrex::Warning("UCMCloudCSVReader: Line " + std::to_string(line_num) + 
                          " conversion error, skipping");
            continue;
        }
    }

    file.close();

    if (time_s_.empty()) {
        amrex::Error("UCMCloudCSVReader: No valid data rows in " + csv_path);
    }

    amrex::Print() << "[UCM][4.2][cloud-csv] Loaded " << time_s_.size() 
                   << " cloud fraction samples from " << csv_path << "\n";
}

amrex::Real UCMCloudCSVReader::get_cloud_fraction_at(amrex::Real sim_time_s, int ucm_debug) const
{
    // Wrap to 24-hour cycle (86400 seconds)
    constexpr amrex::Real cycle_s = 86400.0;
    amrex::Real t_eff = std::fmod(sim_time_s, cycle_s);
    if (t_eff < 0.0) t_eff += cycle_s;  // Handle negative time (shouldn't happen but be safe)

    // Find bracketing indices
    int i_lower = 0;
    int i_upper = time_s_.size() - 1;

    // Handle wrap-around: if t_eff is between last sample and end of cycle
    if (t_eff >= time_s_.back()) {
        // Between last sample and 24h; interpolate to start of next cycle
        i_lower = time_s_.size() - 1;
        i_upper = 0;  // Next cycle's first sample
        
        if (ucm_debug) {
            amrex::Print() << "[UCM][4.2][cloud-csv] sim_time_s=" << sim_time_s 
                           << " t_eff=" << t_eff 
                           << " wrap: bracket [" << i_lower << "," << i_upper << "]"
                           << " t=[" << time_s_[i_lower] << "," << cycle_s << "->0"
                           << "] cf=[" << cloud_fraction_[i_lower] << "," 
                           << cloud_fraction_[i_upper] << "]\n";
        }
        
        // Interpolate from last sample to 86400, then 0 to first
        amrex::Real t0 = time_s_[i_lower];
        amrex::Real t1 = cycle_s + time_s_[i_upper];  // Conceptually next cycle's 0
        amrex::Real cf0 = cloud_fraction_[i_lower];
        amrex::Real cf1 = cloud_fraction_[i_upper];
        
        amrex::Real frac = (t_eff - t0) / (t1 - t0);
        return cf0 + frac * (cf1 - cf0);
    }

    // Normal case: find bracket in this cycle
    for (int i = 0; i < static_cast<int>(time_s_.size()) - 1; ++i) {
        if (t_eff >= time_s_[i] && t_eff <= time_s_[i+1]) {
            i_lower = i;
            i_upper = i + 1;
            break;
        }
    }

    // If t_eff < first sample, wrap to previous cycle
    if (t_eff < time_s_[0]) {
        i_lower = time_s_.size() - 1;
        i_upper = 0;
        
        if (ucm_debug) {
            amrex::Print() << "[UCM][4.2][cloud-csv] sim_time_s=" << sim_time_s 
                           << " t_eff=" << t_eff 
                           << " wrap: bracket [" << i_lower << "," << i_upper << "]"
                           << " t=[" << (time_s_[i_lower] - cycle_s) << "," 
                           << time_s_[i_upper] << "] cf=[" 
                           << cloud_fraction_[i_lower] << "," 
                           << cloud_fraction_[i_upper] << "]\n";
        }
        
        amrex::Real t0 = time_s_[i_lower] - cycle_s;
        amrex::Real t1 = time_s_[i_upper];
        amrex::Real cf0 = cloud_fraction_[i_lower];
        amrex::Real cf1 = cloud_fraction_[i_upper];
        
        amrex::Real frac = (t_eff - t0) / (t1 - t0);
        return cf0 + frac * (cf1 - cf0);
    }

    if (ucm_debug) {
        amrex::Print() << "[UCM][4.2][cloud-csv] sim_time_s=" << sim_time_s 
                       << " t_eff=" << t_eff 
                       << " bracket [" << i_lower << "," << i_upper << "]"
                       << " t=[" << time_s_[i_lower] << "," << time_s_[i_upper] 
                       << "] cf=[" << cloud_fraction_[i_lower] << "," 
                       << cloud_fraction_[i_upper] << "]\n";
    }

    // Linear interpolation
    amrex::Real t0 = time_s_[i_lower];
    amrex::Real t1 = time_s_[i_upper];
    amrex::Real cf0 = cloud_fraction_[i_lower];
    amrex::Real cf1 = cloud_fraction_[i_upper];
    
    if (t1 == t0) {
        // Avoid division by zero (shouldn't happen with well-formed CSV)
        return cf0;
    }

    amrex::Real frac = (t_eff - t0) / (t1 - t0);
    return cf0 + frac * (cf1 - cf0);
}
