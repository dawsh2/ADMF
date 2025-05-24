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
                # Extract timestamp, signal_type, and try to get regime
                timestamp_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                signal_match = re.search(r"'signal_type': ([+-]?\d+)", line)
                regime_match = re.search(r"Regime:\s*(\w+)", line)
                strength_match = re.search(r"'signal_strength': ([+-]?\d+\.?\d*)", line)
                
                if timestamp_match and signal_match:
                    timestamp = timestamp_match.group(1)
                    signal_type = int(signal_match.group(1))
                    regime = regime_match.group(1) if regime_match else 'unknown'
                    strength = float(strength_match.group(1)) if strength_match else 1.0
                    signals.append((timestamp, signal_type, regime, strength))
    
    return signals

# Extract signals from both logs
print("Extracting signals from logs...")

# Get log files from command line or use defaults
prod_log = sys.argv[1] if len(sys.argv) > 1 else '/Users/daws/ADMF/logs/admf_20250523_175758.log'
opt_log = sys.argv[2] if len(sys.argv) > 2 else '/Users/daws/ADMF/logs/admf_20250523_175845.log'

# Production log - entire file
prod_signals = extract_signals(prod_log)
print(f"Production: Found {len(prod_signals)} signals")

# For optimization, we need to find the adaptive test section
# Look for signals after "ENABLING ADAPTIVE MODE"
opt_signals = extract_signals(opt_log, start_marker="ENABLING ADAPTIVE MODE")
print(f"Optimization: Found {len(opt_signals)} signals after ADAPTIVE MODE")

# Find signals starting from test data (2025-01-27)
test_date = '2025-01-27'
prod_test_signals = [s for s in prod_signals if s[0] >= test_date]
opt_test_signals = [s for s in opt_signals if s[0] >= test_date]

print(f"\nTest data signals starting {test_date}:")
print(f"Production: {len(prod_test_signals)} signals")
print(f"Optimization: {len(opt_test_signals)} signals")

# Check for duplicate timestamps
print("\nChecking for duplicate signals...")
prod_timestamps = [s[0] for s in prod_test_signals]
opt_timestamps = [s[0] for s in opt_test_signals]

prod_unique = len(set(prod_timestamps))
opt_unique = len(set(opt_timestamps))

print(f"Production: {prod_unique} unique timestamps out of {len(prod_timestamps)} signals")
print(f"Optimization: {opt_unique} unique timestamps out of {len(opt_timestamps)} signals")

if len(opt_timestamps) > opt_unique:
    print(f"\nWARNING: Optimization has {len(opt_timestamps) - opt_unique} duplicate signals!")
    # Find most common duplicates
    from collections import Counter
    counts = Counter(opt_timestamps)
    duplicates = [(ts, count) for ts, count in counts.items() if count > 1]
    print(f"Found {len(duplicates)} timestamps with duplicates")
    if duplicates:
        print("First 5 duplicate timestamps:")
        for ts, count in sorted(duplicates)[:5]:
            print(f"  {ts}: appears {count} times")

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