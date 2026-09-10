# MRF_Single_Pass_Comparison

Old vs new MRF after `ComputeDiffusivityMRF` stopped running Passes 2-5 twice.

## What changed

Before this change, ERF-Hazard's MRF ran Passes 2-5 (w\*/VPERT, corrector,
w\* recompute, zero-Ri extent) twice in every tile. The first set matches
upstream ERF: it has the Deardorff convective w\* blend, the QNSE option, the
VH96 shear term, the `theta_v_klo` and `Rib - Rib0` guards, and the PBLH
smoothing, and it copies its corrector to SurfaceLayer. The second set came in
with the fire coupling and overwrote the first. It has none of those guards
and options, and it sets w\* = max(Deardorff scale over h, u\*/phi_m) in
every column, whether or not a fire is burning. So:

- the K-profile (`Lturb`, `Kmv`) used the second corrector and w\*;
- SurfaceLayer (the 2D `pblh`, the Beljaars w\* term, dust) stored the first
  corrector.

Now there is a single set of passes. Pass 2 stores the fire buoyancy flux. Pass
4 raises w\* to max(w\*_fire, w\*) only in columns where
`erf.pbl_mrf_fire_thermal_excess = true` and the fire heat flux exceeds
`erf.mrf_fire_q_threshold`. Without fire the passes are those of upstream ERF.
See the "Pass Order and Fire Coupling" section of `Docs/sphinx_doc/theory/PBLschemes.rst`.

## Cases

| case | deck | overrides |
|---|---|---|
| `mrf_unstable` | `CanonicalTests/ABL/mrf_unstable` | 9 h, `erf.fixed_dt = 1.5` |
| `mrf_unstable_enhanced` | same | + `enable_mrf_countergradient`, `enable_vh96_shear_correction`, `enable_pblh_smoothing` |
| `fire_mrf_unstable` | `CanonicalTests/Fire/Atmospheric_Boundary_Layer/ABL_with_MRF/inputs_fire_abl_mrf_unstable` | 512 s, `erf.fixed_dt = 0.32` |
| `fire_mrf_unstable_boost` | same | + `erf.pbl_mrf_fire_thermal_excess = true` |

Every case also sets `erf.most.pblh_calc = MRF`. That only makes the 2D
plotfile report the PBLH that MRF writes to SurfaceLayer, which is otherwise
−999; nothing else in these decks reads it. The time step is fixed so that both
binaries write at the same times. The decks are read from `Exec/CanonicalTests`
and are not copied.

## Running

```bash
MPIRUN="mpirun -np 1" ./run_comparison.sh /path/to/old/erf_exec /path/to/new/erf_exec
```

`OLD` is a build of ERF-Hazard at `5764e850b` or earlier; `NEW` has the
single set of passes. Runs go to `./mrf_single_pass_runs/<case>/{old,new}`.
`SKIP_RUN=1` reprints the tables, and `LABELS=old` (or `new`) runs one binary
only. The tables come from `compare_mrf.py`, which needs `yt`:

- `Lturb`: horizontal mean of the PBLH the K-profile used.
- `pblh`: horizontal mean of the PBLH stored in SurfaceLayer.
- `Kmv_max`: domain maximum of rho K_m [kg/(m s)].
- `Kmv@z`, `th@z`: horizontal means at the level nearest z.
- `th_max`: domain maximum of theta.
- `d%`: the new value relative to the old.

The last block of each table gives the largest difference in the mean theta and
Kmv profiles.

## Results

Run on 2026-09-10, Release with MPI, 1 rank each. OLD is a build of 8de28ef36,
whose tree is identical to 7fb20776f. It lacks #393, but every deck here is a
single box with `mfiter_tile_size = 1024`, so the valid box is the tile. NEW
has the single set of passes.

Values at the last output time (theta and Kmv are horizontal means):

| case | Lturb old → new | pblh old → new | Kmv@100 m | Kmv@500 m | theta@100 m [K] | max \|Δ⟨theta⟩\| |
|---|---|---|---|---|---|---|
| `mrf_unstable` (9 h) | 1162.4 → 1167.4 (+0.4%) | 1162.4 → 1167.4 | +11.6% | +12.3% | 302.316 → 302.399 | 0.34 K (0.65 K at 2 h) |
| `mrf_unstable_enhanced` (9 h) | 1259.2 → 1349.1 (+7.1%) | 1267.3 → 1349.1 (+6.5%) | +14.5% | +24.2% | 302.597 → 302.963 | 3.30 K at 1235 m |
| `fire_mrf_unstable` (512 s) | 1028.9 → 1029.3 (+0.04%) | same | +15.3% | +15.4% | 300.242 → 300.249 | 0.08 K |
| `fire_mrf_unstable_boost` (512 s) | 1028.8 → 1029.2 (+0.04%) | same | +14.9% | +15.0% | 300.239 → 300.246 | 0.08 K |

- **Kmv rises 12-15% everywhere.** The removed Pass 4 used w\* = max(Deardorff
  scale over h, u\*/phi_m); Pass 4 now uses the cube-sum blend of the two.
  PBLH follows only slowly (under 1% by 9 h) because countergradient
  transport is off in these decks, so VPERT is zero.
- **The enhanced case shows the inconsistency.** In the old run the K-profile
  PBLH (`Lturb`) and SurfaceLayer's (`pblh`) differ, 1259 vs 1267 m at 9 h,
  because only the first corrector had VH96 and smoothing; the new run has one
  value. VH96, VPERT from the blended w\* and the larger K deepen the PBL by 7%
  and move the entrainment zone, hence the 3.3 K theta difference near 1.2 km.
- **The fire boost itself is unchanged.** In the boost case, both runs report
  the same `[MRF FIRE] wstar_max`: a peak of 7.81 m/s, and 4.596 vs 4.597 m/s
  at 512 s. `Kmv_max` over the fire is 345 in both, the Kmax = 300 m²/s cap.
  The change is in the columns without fire, as in the case without the boost.
- `th_max` is the top of the sounding (312.58 K) in every run, not a plume.

Full output of `run_comparison.sh`:

```text
== mrf_unstable  (z levels used: 100 m -> 102.7 m, 500 m -> 509.1 m)
   step      t[s]  run     Lturb     pblh  Kmv_max  Kmv@100  Kmv@500   th@100   th@500   th_max
      0       0.0  old       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
      0       0.0  new       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
                   d%        n/a      n/a      n/a      n/a      n/a   +0.00%   +0.00%   +0.00%
   1200    1800.0  old    1048.6   1048.6   145.38    81.69   120.68  300.581  300.304  312.582
   1200    1800.0  new    1051.0   1051.0   165.97    93.10   138.10  300.593  300.341  312.582
                   d%     +0.24%   +0.24%  +14.17%  +13.96%  +14.43%   +0.00%   +0.01%   -0.00%
   2400    3600.0  old    1082.7   1082.7   151.37    83.02   129.62  301.037  300.759  312.582
   2400    3600.0  new    1085.2   1085.2   171.22    93.74   146.94  301.072  300.820  312.582
                   d%     +0.24%   +0.24%  +13.12%  +12.92%  +13.36%   +0.01%   +0.02%   -0.00%
   3600    5400.0  old    1100.0   1100.0   154.23    83.60   134.01  301.487  301.196  312.582
   3600    5400.0  new    1103.2   1103.2   173.56    93.89   151.20  301.537  301.265  312.582
                   d%     +0.29%   +0.29%  +12.53%  +12.30%  +12.82%   +0.02%   +0.02%   -0.00%
   4800    7200.0  old    1130.4   1130.4   159.63    84.72   141.76  301.902  301.603  312.582
   4800    7200.0  new    1141.4   1141.4   180.48    95.07   161.42  301.961  301.690  312.582
                   d%     +0.98%   +0.98%  +13.06%  +12.21%  +13.87%   +0.02%   +0.03%   -0.00%
   6000    9000.0  old    1162.4   1162.4   165.33    85.88   149.81  302.316  302.035  312.582
   6000    9000.0  new    1167.4   1167.4   185.08    95.83   168.19  302.399  302.139  312.582
                   d%     +0.42%   +0.42%  +11.94%  +11.59%  +12.27%   +0.03%   +0.03%   -0.00%
   step  max|d<theta>| [K] at z[m]   max|d<Kmv>| at z[m]
      0          0.0000      0.0         0.000      0.0
   1200          0.1791   1026.3        20.595    339.2
   2400          0.2780   1026.3        19.853    339.2
   3600          0.3221   1091.8        19.363    363.6
   4800          0.6450   1091.8        20.928    389.4
   6000          0.3356   1161.3        19.777    389.4

== mrf_unstable_enhanced  (z levels used: 100 m -> 102.7 m, 500 m -> 509.1 m)
   step      t[s]  run     Lturb     pblh  Kmv_max  Kmv@100  Kmv@500   th@100   th@500   th_max
      0       0.0  old       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
      0       0.0  new       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
                   d%        n/a      n/a      n/a      n/a      n/a   +0.00%   +0.00%   +0.00%
   1200    1800.0  old    1086.5   1100.7   152.28    83.31   130.82  300.602  300.369  312.582
   1200    1800.0  new    1116.4   1116.4   178.66    95.80   157.14  300.681  300.518  312.582
                   d%     +2.76%   +1.43%  +17.32%  +14.99%  +20.12%   +0.03%   +0.05%   -0.00%
   2400    3600.0  old    1115.6   1128.0   157.25    84.35   138.25  301.125  300.878  312.582
   2400    3600.0  new    1175.6   1175.6   188.94    97.35   172.51  301.300  301.121  312.582
                   d%     +5.38%   +4.22%  +20.15%  +15.40%  +24.78%   +0.06%   +0.08%   -0.00%
   3600    5400.0  old    1167.1   1176.8   166.57    86.27   151.34  301.617  301.374  312.582
   3600    5400.0  new    1237.5   1237.5   200.28    99.11   188.29  301.885  301.688  312.582
                   d%     +6.03%   +5.16%  +20.24%  +14.88%  +24.42%   +0.09%   +0.10%   -0.00%
   4800    7200.0  old    1211.9   1225.9   174.64    87.85   162.39  302.114  301.867  312.582
   4800    7200.0  new    1294.8   1294.8   210.78   100.68   202.32  302.439  302.234  312.582
                   d%     +6.84%   +5.62%  +20.70%  +14.62%  +24.59%   +0.11%   +0.12%   -0.00%
   6000    9000.0  old    1259.2   1267.3   183.22    89.46   173.78  302.597  302.360  312.582
   6000    9000.0  new    1349.1   1349.1   221.61   102.47   215.85  302.963  302.758  312.582
                   d%     +7.14%   +6.46%  +20.95%  +14.54%  +24.21%   +0.12%   +0.13%   -0.00%
   step  max|d<theta>| [K] at z[m]   max|d<Kmv>| at z[m]
      0          0.0000      0.0         0.000      0.0
   1200          0.9846   1091.8        27.123    416.7
   2400          2.8295   1091.8        34.260    509.1
   3600          2.9692   1161.3        37.182    543.6
   4800          2.4399   1161.3        40.853    619.0
   6000          3.2951   1235.0        43.724    619.0

== fire_mrf_unstable  (z levels used: 100 m -> 102.7 m, 500 m -> 509.1 m)
   step      t[s]  run     Lturb     pblh  Kmv_max  Kmv@100  Kmv@500   th@100   th@500   th_max
      0       0.0  old       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
      0       0.0  new       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
                   d%        n/a      n/a      n/a      n/a      n/a   +0.00%   +0.00%   +0.00%
    400     128.0  old    1018.8   1018.8   140.46    80.55   112.77  300.071  300.000  312.581
    400     128.0  new    1019.0   1019.0   163.76    93.96   131.58  300.077  300.000  312.581
                   d%     +0.01%   +0.01%  +16.59%  +16.65%  +16.68%   +0.00%   +0.00%   -0.00%
    800     256.0  old    1023.3   1023.3   141.64    80.72   113.98  300.148  300.006  312.581
    800     256.0  new    1023.5   1023.5   164.29    93.60   132.22  300.155  300.008  312.581
                   d%     +0.02%   +0.02%  +15.99%  +15.95%  +16.00%   +0.00%   +0.00%   -0.00%
   1200     384.0  old    1026.3   1026.3   145.04    80.84   114.79  300.200  300.025  312.581
   1200     384.0  new    1026.6   1026.6   167.43    93.42   132.74  300.207  300.030  312.581
                   d%     +0.03%   +0.03%  +15.44%  +15.57%  +15.64%   +0.00%   +0.00%   -0.00%
   1600     512.0  old    1028.9   1028.9   154.75    80.94   115.48  300.242  300.041  312.581
   1600     512.0  new    1029.3   1029.3   177.11    93.32   133.24  300.249  300.050  312.581
                   d%     +0.04%   +0.04%  +14.45%  +15.30%  +15.38%   +0.00%   +0.00%   +0.00%
   step  max|d<theta>| [K] at z[m]   max|d<Kmv>| at z[m]
      0          0.0000      0.0         0.000      0.0
    400          0.0526      0.0        23.362    316.2
    800          0.0608      0.0        22.519    316.2
   1200          0.0706    964.4        22.067    316.2
   1600          0.0809    964.4        21.759    316.2

== fire_mrf_unstable_boost  (z levels used: 100 m -> 102.7 m, 500 m -> 509.1 m)
   step      t[s]  run     Lturb     pblh  Kmv_max  Kmv@100  Kmv@500   th@100   th@500   th_max
      0       0.0  old       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
      0       0.0  new       0.0      n/a     0.00     0.00     0.00  300.000  300.000  312.582
                   d%        n/a      n/a      n/a      n/a      n/a   +0.00%   +0.00%   +0.00%
    400     128.0  old    1018.4   1018.4   345.27    82.04   114.67  300.071  300.001  312.581
    400     128.0  new    1018.6   1018.6   345.27    95.32   133.30  300.078  300.001  312.581
                   d%     +0.02%   +0.02%   -0.00%  +16.19%  +16.25%   +0.00%   +0.00%   -0.00%
    800     256.0  old    1023.0   1023.0   343.80    82.08   115.84  300.145  300.009  312.581
    800     256.0  new    1023.3   1023.3   343.80    94.81   133.87  300.152  300.011  312.581
                   d%     +0.03%   +0.03%   -0.00%  +15.50%  +15.57%   +0.00%   +0.00%   -0.00%
   1200     384.0  old    1026.1   1026.1   343.73    81.89   116.24  300.196  300.027  312.581
   1200     384.0  new    1026.5   1026.5   343.73    94.32   133.96  300.204  300.031  312.581
                   d%     +0.03%   +0.03%   -0.00%  +15.17%  +15.25%   +0.00%   +0.00%   -0.00%
   1600     512.0  old    1028.8   1028.8   325.25    81.83   116.73  300.239  300.043  312.582
   1600     512.0  new    1029.2   1029.2   325.40    94.04   134.24  300.246  300.051  312.582
                   d%     +0.04%   +0.04%   +0.05%  +14.92%  +15.00%   +0.00%   +0.00%   +0.00%
   step  max|d<theta>| [K] at z[m]   max|d<Kmv>| at z[m]
      0          0.0000      0.0         0.000      0.0
    400          0.0482      0.0        23.143    316.2
    800          0.0565      0.0        22.256    316.2
   1200          0.0698    964.4        21.783    316.2
   1600          0.0808    964.4        21.451    316.2
```
