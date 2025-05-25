#!/usr/bin/env python3
"""
Analyze regime distribution in adaptive test section
"""

import re
from collections import Counter

# Read the optimization log starting from adaptive test
start_line = 2602000
regime_counter = Counter()
signal_counter = 0

with open('logs/admf_20250523_175845.log', 'r') as f:
    # Skip to adaptive test section
    for _ in range(start_line):
        f.readline()
    
    # Process remaining lines
    for line in f:
        # Look for signals with regime information
        if "Publishing event: Event(type=SIGNAL" in line:
            signal_counter += 1
            # Extract regime from reason field
            regime_match = re.search(r"Regime:\s*(\w+)", line)
            if regime_match:
                regime = regime_match.group(1)
                regime_counter[regime] += 1

print(f"Signals in adaptive test section: {signal_counter}")
print(f"\nRegime distribution:")
for regime, count in regime_counter.most_common():
    percentage = (count / signal_counter * 100) if signal_counter > 0 else 0
    print(f"  {regime}: {count} signals ({percentage:.1f}%)")

# Check if regimes actually changed
unique_regimes = len(regime_counter)
print(f"\nUnique regimes found: {unique_regimes}")
if unique_regimes > 1:
    print("✓ The optimization IS switching regimes during adaptive test!")
else:
    print("✗ The optimization is stuck in one regime")