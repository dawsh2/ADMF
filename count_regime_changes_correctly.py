#!/usr/bin/env python3
"""
Count regime changes correctly by looking at actual transitions
"""

import re

def count_regime_changes(log_file, start_line=0):
    """Count actual regime changes by tracking transitions"""
    
    current_regime = None
    regime_changes = 0
    
    with open(log_file, 'r') as f:
        # Skip to start line
        for _ in range(start_line):
            f.readline()
            
        for line in f:
            # Only look at test period
            if not any(date in line for date in ['2025-01-27', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02']):
                continue
                
            # Look for CLASSIFICATION events
            if "Publishing event: Event(type=CLASSIFICATION" in line:
                curr_match = re.search(r"'classification': '(\w+)'", line)
                if curr_match:
                    new_regime = curr_match.group(1)
                    if current_regime and new_regime != current_regime:
                        regime_changes += 1
                    current_regime = new_regime
                    
    return regime_changes

# Count regime changes in both logs
print("Counting actual regime transitions in test period...")

opt_changes = count_regime_changes('logs/admf_20250523_175845.log', 2602000)
print(f"\nOptimization: {opt_changes} regime changes")

prod_changes = count_regime_changes('logs/admf_20250523_183637.log')
print(f"Production: {prod_changes} regime changes")

if opt_changes == prod_changes:
    print("\n✓ Both have the same number of regime changes!")
else:
    print(f"\n✗ Different regime change counts! Difference: {abs(opt_changes - prod_changes)}")