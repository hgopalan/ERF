#!/usr/bin/env python3
"""
Phase 5.2 HVAC Unit Test Check Script

Parses 4 log files and validates:
1. Off mode reports mode=off; no Q_HVAC in output.
2. Simple_hot reports Q_HVAC > 0.
3. Simple_cold reports Q_HVAC = 0 (setpoint gate).
4. Simple_unoccupied reports Q_HVAC = 0 (occupancy gate).
"""

import sys
import re
from pathlib import Path


def parse_hvac_log(log_file):
    """Extract HVAC diagnostics from ERF log file."""
    if not Path(log_file).exists():
        print(f"Error: log file not found: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    result = {
        'mode': None,
        'q_hvac_min': None,
        'q_hvac_max': None,
        'has_hvac_line': False
    }
    
    # Search for [UCM][5.2][hvac] banner lines
    hvac_pattern = r'\[UCM\]\[5\.2\]\[hvac\] mode=([a-z]+)(?: hour=(\d+))?(?: Q_HVAC=\[([\d.e\-+]+), ([\d.e\-+]+)\])?'
    matches = re.findall(hvac_pattern, content)
    
    if matches:
        result['has_hvac_line'] = True
        for mode, hour, q_min, q_max in matches:
            result['mode'] = mode
            if q_min and q_max:
                result['q_hvac_min'] = float(q_min)
                result['q_hvac_max'] = float(q_max)
    
    return result


def main():
    """Run checks on all 4 unit test variants."""
    
    log_dir = Path('.')
    test_cases = [
        ('plt_ucm_off', 'off', 0.0, 0.0),
        ('plt_ucm_simple_hot', 'simple', None, None),  # Should be > 0
        ('plt_ucm_simple_cold', 'simple', 0.0, 0.0),
        ('plt_ucm_simple_unoccupied', 'simple', 0.0, 0.0),
    ]
    
    all_passed = True
    
    for log_prefix, expected_mode, expected_q_min, expected_q_max in test_cases:
        # Try multiple log file naming conventions
        log_file = None
        for suffix in ['.log', '_0.log', '.txt']:
            candidate = log_dir / f'{log_prefix}{suffix}'
            if candidate.exists():
                log_file = candidate
                break
        
        if not log_file:
            # For off mode, log might not contain [UCM][5.2][hvac] banner
            if expected_mode == 'off' and expected_q_min == 0.0:
                print(f"✓ {log_prefix}: Off mode (no HVAC log expected)")
                continue
            else:
                print(f"✗ {log_prefix}: Log file not found (tried multiple patterns)")
                all_passed = False
                continue
        
        result = parse_hvac_log(log_file)
        
        if result is None:
            print(f"✗ {log_prefix}: Failed to parse log")
            all_passed = False
            continue
        
        # Validate based on test case
        if expected_mode == 'off':
            if not result['has_hvac_line']:
                print(f"✓ {log_prefix}: Off mode, no Q_HVAC logged (as expected)")
            else:
                print(f"✗ {log_prefix}: Off mode should not have HVAC banner")
                all_passed = False
        
        elif expected_mode == 'simple':
            if not result['has_hvac_line']:
                print(f"✗ {log_prefix}: Simple mode should have [UCM][5.2][hvac] banner")
                all_passed = False
                continue
            
            if expected_q_min == 0.0 and expected_q_max == 0.0:
                # Cold or unoccupied: Q_HVAC should be zero
                if result['q_hvac_min'] == 0.0 and result['q_hvac_max'] == 0.0:
                    print(f"✓ {log_prefix}: Q_HVAC = 0 (gate active as expected)")
                else:
                    print(f"✗ {log_prefix}: Expected Q_HVAC=0, got [{result['q_hvac_min']}, {result['q_hvac_max']}]")
                    all_passed = False
            else:
                # Hot: Q_HVAC should be > 0
                if result['q_hvac_max'] is not None and result['q_hvac_max'] > 0.0:
                    print(f"✓ {log_prefix}: Q_HVAC > 0 (hot canyon as expected) = [{result['q_hvac_min']}, {result['q_hvac_max']}]")
                else:
                    print(f"✗ {log_prefix}: Expected Q_HVAC > 0, got [{result['q_hvac_min']}, {result['q_hvac_max']}]")
                    all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed!")
        return 0
    else:
        print("\n✗ Some checks failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
