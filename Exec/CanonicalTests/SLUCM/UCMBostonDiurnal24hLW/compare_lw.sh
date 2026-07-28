python3.9 << 'EOF'
import re

def extract_last_hour(logfile, facet):
    """Extract [min, max] T_skin_<facet> from all [3.5A-hotfix3][entry] banners.
    Returns list of (min, max) tuples."""
    pattern = rf'T_skin_{facet}=\[([\d.]+),([\d.]+)\] K'
    with open(logfile) as f:
        matches = re.findall(pattern, f.read())
    return [(float(a), float(b)) for a, b in matches]

for facet in ['wall', 'road', 'roof']:
    s = extract_last_hour('run_single.log', facet)
    m = extract_last_hour('run_multi_lagged.log', facet)

    if not s or not m:
        print(f'{facet}: no matches')
        continue

    # Number of steps in last hour ≈ 2500 (dt~1.44s, 3600s/hr)
    # Sample last 2500 for "last hour" nighttime (steps 57500-60000 ≈ 23:00-24:00)
    N = min(2500, len(s), len(m))
    s_last = s[-N:]
    m_last = m[-N:]

    # Peak (max of max) and trough (min of min) across the window
    s_peak = max(x[1] for x in s_last)
    m_peak = max(x[1] for x in m_last)
    s_mean = sum(x[1] for x in s_last) / N
    m_mean = sum(x[1] for x in m_last) / N

    print(f'{facet.upper():5s}  (last {N} steps = last hour ≈ 23:00-24:00 local)')
    print(f'   single      T_skin_{facet}_max: peak={s_peak:.2f} K   mean={s_mean:.2f} K')
    print(f'   multi-lag   T_skin_{facet}_max: peak={m_peak:.2f} K   mean={m_mean:.2f} K')
    print(f'   DELTA peak={m_peak-s_peak:+.2f} K   mean={m_mean-s_mean:+.2f} K')
    print(f'   {"✓ multi warmer (LW trapping visible)" if m_mean > s_mean else "✗ no visible LW enhancement"}')
    print()
EOF
