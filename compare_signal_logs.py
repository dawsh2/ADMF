#!/usr/bin/env python3
"""Extract and compare signal generation between runs"""

import re
import sys

def extract_signals_from_log(log_file, label):
    """Extract all signal-related events from a log file"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {label}: {log_file}")
    print('='*60)
    
    signals = []
    regime_changes = []
    parameter_changes = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for signal events
                if 'Publishing SIGNAL event' in line and 'direction' in line:
                    match = re.search(r'timestamp.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*direction["\']:\s*([-\d.]+)', line)
                    if match:
                        timestamp, direction = match.groups()
                        signals.append((timestamp, float(direction)))
                
                # Look for regime changes
                if 'Regime changed from' in line or 'CLASSIFICATION' in line:
                    regime_changes.append(line.strip())
                
                # Look for parameter updates
                if 'Applying regime-specific parameters' in line or 'Final weights after parameter update' in line:
                    parameter_changes.append(line.strip())
    
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\nFound {len(signals)} signals")
    print(f"Found {len(regime_changes)} regime changes")
    print(f"Found {len(parameter_changes)} parameter updates")
    
    # Show first 10 signals
    if signals:
        print("\nFirst 10 signals:")
        for i, (ts, direction) in enumerate(signals[:10]):
            print(f"  {i+1}. {ts}: direction = {direction}")
    
    # Show first 5 regime changes
    if regime_changes:
        print("\nFirst 5 regime changes:")
        for i, change in enumerate(regime_changes[:5]):
            print(f"  {i+1}. {change[:100]}...")
    
    # Show first 5 parameter changes
    if parameter_changes:
        print("\nFirst 5 parameter updates:")
        for i, change in enumerate(parameter_changes[:5]):
            print(f"  {i+1}. {change[:100]}...")
    
    return signals, regime_changes, parameter_changes


# Compare the two logs
print("SIGNAL LOG COMPARISON")
print("="*80)

# Optimization log
opt_log = "/Users/daws/ADMF/logs/admf_20250522_225752.log"
opt_signals, opt_regimes, opt_params = extract_signals_from_log(opt_log, "OPTIMIZATION (with adaptive test)")

# Production log
prod_log = "/Users/daws/ADMF/logs/admf_20250522_225839.log"
prod_signals, prod_regimes, prod_params = extract_signals_from_log(prod_log, "PRODUCTION")

# Compare results
if opt_signals and prod_signals:
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    print(f"Optimization signals: {len(opt_signals)}")
    print(f"Production signals: {len(prod_signals)}")
    
    # Find first divergence
    print("\nChecking for first divergence in signals...")
    for i, ((opt_ts, opt_dir), (prod_ts, prod_dir)) in enumerate(zip(opt_signals, prod_signals)):
        if opt_dir != prod_dir:
            print(f"DIVERGENCE at signal #{i+1}:")
            print(f"  Optimization: {opt_ts} direction={opt_dir}")
            print(f"  Production: {prod_ts} direction={prod_dir}")
            break
    else:
        if len(opt_signals) == len(prod_signals):
            print("All signals match!")
        else:
            print(f"Signal counts differ after {min(len(opt_signals), len(prod_signals))} signals")