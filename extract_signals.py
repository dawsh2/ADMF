#!/usr/bin/env python3
"""Extract and compare signals from optimization and production logs"""

import re
import sys

def extract_signals(log_file, start_marker=None, end_marker=None):
    """Extract signals with their regimes and strengths"""
    signals = []
    in_section = start_marker is None  # If no marker, process entire file
    
    with open(log_file, 'r') as f:
        for line in f:
            if start_marker and start_marker in line:
                in_section = True
            if end_marker and end_marker in line:
                break
                
            if in_section and 'Publishing event: Event(type=SIGNAL' in line:
                # Extract timestamp, signal_type, regime, and strength
                match = re.search(r"Timestamp\('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^']*'\).*?signal_type':\s*([+-]?\d+).*?Regime:\s*(\w+).*?signal_strength':\s*([+-]?\d+\.?\d*)", line)
                if match:
                    timestamp = match.group(1)
                    signal_type = int(match.group(2))
                    regime = match.group(3)
                    strength = float(match.group(4))
                    signals.append((timestamp, signal_type, regime, strength))
    
    return signals

# Extract signals from both logs
print("Extracting signals from logs...")

# Production log - entire file
prod_signals = extract_signals('/Users/daws/ADMF/logs/admf_20250522_225839.log')
print(f"Production: Found {len(prod_signals)} signals")

# For optimization, we need to find the adaptive test section
# Let's extract all signals for now
opt_signals = extract_signals('/Users/daws/ADMF/logs/admf_20250522_225752.log')
print(f"Optimization: Found {len(opt_signals)} signals")

# Find signals starting from test data (April 24, 2024)
prod_test_signals = [s for s in prod_signals if s[0].startswith('2024-04-24')]
opt_test_signals = [s for s in opt_signals if s[0].startswith('2024-04-24')]

print(f"\nTest data signals starting April 24, 2024:")
print(f"Production: {len(prod_test_signals)} signals")
print(f"Optimization: {len(opt_test_signals)} signals")

# Compare first 10 signals
print("\nFirst 10 signals comparison:")
print("-" * 80)
print(f"{'Time':<20} {'Prod Signal':<12} {'Prod Regime':<15} {'Opt Signal':<12} {'Opt Regime':<15} {'Match'}")
print("-" * 80)

for i in range(min(10, len(prod_test_signals), len(opt_test_signals))):
    p_time, p_sig, p_reg, p_str = prod_test_signals[i]
    o_time, o_sig, o_reg, o_str = opt_test_signals[i]
    
    # Extract just the time part
    p_time_short = p_time.split()[1] if ' ' in p_time else p_time
    
    match = "✓" if (p_sig == o_sig and p_reg == o_reg) else "✗"
    print(f"{p_time_short:<20} {p_sig:<12} {p_reg:<15} {o_sig:<12} {o_reg:<15} {match}")

# Find first divergence
print("\nSearching for first divergence...")
for i in range(min(len(prod_test_signals), len(opt_test_signals))):
    p_time, p_sig, p_reg, p_str = prod_test_signals[i]
    o_time, o_sig, o_reg, o_str = opt_test_signals[i]
    
    if p_sig != o_sig or p_reg != o_reg:
        print(f"\nFIRST DIVERGENCE at signal #{i+1}:")
        print(f"  Production: {p_time} signal={p_sig} regime={p_reg} strength={p_str}")
        print(f"  Optimization: {o_time} signal={o_sig} regime={o_reg} strength={o_str}")
        break
else:
    print("\nNo divergence found in first", min(len(prod_test_signals), len(opt_test_signals)), "signals")