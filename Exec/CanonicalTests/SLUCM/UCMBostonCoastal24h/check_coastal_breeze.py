#!/usr/bin/env python3
"""
Phase 5.7 verifier: land–sea breeze reversal on coastal Boston canonical.

Loads hourly plotfiles, extracts near-surface u-velocity at a coastal ATM cell
sampling line, and asserts the four criteria in the goals section:

1. Sea-breeze onset: u >= 1 m/s onshore between local hours 11-14
2. Sea-breeze peak: u >= 2 m/s sustained for >= 2 hours around 15-17
3. Land-breeze reversal: u <= -0.5 m/s between local hours 01-05
4. Nighttime UHI: T_urban - T_rural >= 2 K averaged over 02-05

Plotfile indexing (start_hour_local = 0, midnight):
  - Hour 0 → plt_coastal_00000 (00:00 local)
  - Hour 6 → plt_coastal_06 (06:00 local) [if using yt's native naming]
  - ...
  - Hour 23 → plt_coastal_23 (23:00 local)
"""

import os
import glob
import sys
import numpy as np

# Try to import yt; if not available, provide helpful error message
try:
    import yt
except ImportError:
    print("[FAIL] check_coastal_breeze.py requires yt (yt-project.org)")
    print("  Install: pip install yt")
    sys.exit(1)


def get_plotfile_list(plt_dir=""):
    """Collect sorted plotfiles from directory (defaults to pwd)."""
    pattern = os.path.join(plt_dir, "plt_coastal_*")
    pltfiles = sorted(glob.glob(pattern))
    if not pltfiles:
        print(f"[FAIL] No plotfiles matching {pattern}")
        return None
    return pltfiles


def extract_coastal_diagnostics(pltfiles):
    """
    Load hourly plotfiles and extract:
      - u-velocity at coastal strip (x near 5000 m, y center)
      - T_skin at urban and rural cells for UHI
    
    Returns: (u_coast, T_urban, T_rural) arrays [24 hours]
    """
    
    n_files = len(pltfiles)
    u_coast = np.zeros(n_files)
    T_urban = np.zeros(n_files)
    T_rural = np.zeros(n_files)
    
    # Coastal sampling line geometry
    # Coast is at x ≈ 5000-6000 m (coast transition band in coastal layout)
    # Sample near y_center = 10000 m (middle of domain in y)
    X_LO_UCM = 4500.0  # coastal strip left
    X_HI_UCM = 5500.0  # coastal strip right
    Y_C_UCM = 10000.0  # center in y
    K_SFC = 0          # lowest ATM cell
    
    # Urban cell: center of urban region (x ≈ 10000 m, y ≈ 10000 m)
    X_URBAN = 10000.0
    Y_URBAN = 10000.0
    
    # Rural cell: inland rural region (x ≈ 17500 m, y ≈ 10000 m)
    X_RURAL = 17500.0
    Y_RURAL = 10000.0
    
    for i, pf in enumerate(pltfiles):
        try:
            ds = yt.load(pf)
            # Get covering grid at lowest refinement level
            cg = ds.covering_grid(0, [0, 0, 0], ds.domain_dimensions)
            
            # Extract u-velocity at coastal strip
            # Grid indexing: index = x / dx, convert physical coords to grid coords
            # dx_atm = 20000 / 20 = 1000 m, so grid cell i covers [i*1000, (i+1)*1000)
            i_lo = int(X_LO_UCM / 1000.0)
            i_hi = int(X_HI_UCM / 1000.0) + 1
            j_center = int(Y_C_UCM / 1000.0)
            
            u_coast_loc = cg['x_velocity'][i_lo:i_hi, j_center:j_center+1, K_SFC].mean()
            u_coast[i] = u_coast_loc
            
            # Extract T at urban and rural cells (at k=0, lowest ATM level)
            i_urban = int(X_URBAN / 1000.0)
            j_urban = int(Y_URBAN / 1000.0)
            i_rural = int(X_RURAL / 1000.0)
            j_rural = int(Y_RURAL / 1000.0)
            
            T_urban[i] = cg['theta'][i_urban, j_urban, K_SFC]
            T_rural[i] = cg['theta'][i_rural, j_rural, K_SFC]
            
        except Exception as e:
            print(f"[WARN] Error loading {pf}: {e}")
            return None
    
    return u_coast, T_urban, T_rural


def verify_coastal_breeze(u_coast, T_urban, T_rural):
    """
    Verify four coastal sea-breeze criteria.
    
    Returns: (all_pass, details_dict)
    """
    details = {}
    all_pass = True
    
    # Criterion 1: Sea-breeze onset (u >= 1 m/s onshore between 11-14 local time)
    u_onset_window = u_coast[11:15]
    onset_peak = u_onset_window.max()
    criterion_1 = onset_peak >= 1.0
    details['onset_peak'] = onset_peak
    details['onset_pass'] = criterion_1
    all_pass = all_pass and criterion_1
    
    # Criterion 2: Sea-breeze peak (u >= 2 m/s for >= 2 hours in 15-18 local)
    u_peak_window = u_coast[15:18]
    peak_hours = (u_peak_window >= 2.0).sum()
    criterion_2 = peak_hours >= 2
    details['peak_max'] = u_peak_window.max()
    details['peak_hours'] = peak_hours
    details['peak_pass'] = criterion_2
    all_pass = all_pass and criterion_2
    
    # Criterion 3: Land-breeze reversal (u <= -0.5 m/s between 01-05 local)
    u_reversal_window = u_coast[1:6]
    reversal_min = u_reversal_window.min()
    criterion_3 = reversal_min <= -0.5
    details['reversal_min'] = reversal_min
    details['reversal_pass'] = criterion_3
    all_pass = all_pass and criterion_3
    
    # Criterion 4: Nighttime UHI (T_urban - T_rural >= 2 K, averaged 02-05)
    uhi_night = (T_urban[2:6] - T_rural[2:6]).mean()
    criterion_4 = uhi_night >= 2.0
    details['uhi_night'] = uhi_night
    details['uhi_pass'] = criterion_4
    all_pass = all_pass and criterion_4
    
    return all_pass, details


def main():
    """Main verifier entry point."""
    
    # Parse command-line args
    plt_dir = ""
    if "--plt_dir" in sys.argv:
        idx = sys.argv.index("--plt_dir")
        if idx + 1 < len(sys.argv):
            plt_dir = sys.argv[idx + 1]
    
    # Load plotfiles
    pltfiles = get_plotfile_list(plt_dir)
    if pltfiles is None:
        return 1
    
    print(f"[INFO] Loaded {len(pltfiles)} plotfiles from {plt_dir or '.'}")
    
    # Extract diagnostics
    result = extract_coastal_diagnostics(pltfiles)
    if result is None:
        return 1
    
    u_coast, T_urban, T_rural = result
    
    # Verify criteria
    all_pass, details = verify_coastal_breeze(u_coast, T_urban, T_rural)
    
    # Print results
    print("\n" + "="*70)
    if all_pass:
        print("[PASS] Phase 5.7 coastal canonical")
    else:
        print("[FAIL] Phase 5.7 coastal canonical")
    print("="*70)
    print(f"  Sea-breeze onset (hour 11-14):  {details['onset_peak']:6.2f} m/s {'✓' if details['onset_pass'] else '✗'}")
    print(f"  Sea-breeze peak (hour 15-18):   {details['peak_max']:6.2f} m/s ({details['peak_hours']} hours >= 2 m/s) {'✓' if details['peak_pass'] else '✗'}")
    print(f"  Land-breeze min (hour 01-05):   {details['reversal_min']:6.2f} m/s {'✓' if details['reversal_pass'] else '✗'}")
    print(f"  Nighttime UHI (hour 02-05):     {details['uhi_night']:6.2f} K {'✓' if details['uhi_pass'] else '✗'}")
    print("="*70 + "\n")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
