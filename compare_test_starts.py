#!/usr/bin/env python3
"""
Compare the start times and regime detection between optimization and production logs
"""

import sys
import re
from datetime import datetime

def find_test_start(log_file, start_line=0):
    """Find when test data starts being processed"""
    
    first_test_bar = None
    first_test_signal = None
    regime_at_start = None
    test_date = '2025-01-27'
    
    with open(log_file, 'r') as f:
        # Skip to start line if specified
        for _ in range(start_line):
            f.readline()
            
        for line in f:
            # Look for first BAR event with test date
            if not first_test_bar and "Publishing event: Event(type=BAR" in line:
                if test_date in line:
                    timestamp_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                    if timestamp_match:
                        first_test_bar = timestamp_match.group(1)
                        
            # Look for first SIGNAL event with test date
            if not first_test_signal and "Publishing event: Event(type=SIGNAL" in line:
                if test_date in line:
                    timestamp_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                    regime_match = re.search(r"Regime:\s*(\w+)", line)
                    if timestamp_match:
                        first_test_signal = timestamp_match.group(1)
                        regime_at_start = regime_match.group(1) if regime_match else 'unknown'
                        
            # Stop after finding both
            if first_test_bar and first_test_signal:
                break
                
    return first_test_bar, first_test_signal, regime_at_start

def analyze_regime_detection(log_file, start_line=0):
    """Analyze regime detection patterns"""
    
    regimes_detected = set()
    regime_changes = 0
    test_date = '2025-01-27'
    
    with open(log_file, 'r') as f:
        # Skip to start line if specified
        for _ in range(start_line):
            f.readline()
            
        prev_regime = None
        for line in f:
            # Only look at test period
            if test_date not in line:
                continue
                
            # Look for classification events
            if "Publishing event: Event(type=CLASSIFICATION" in line:
                regime_match = re.search(r"'classification': '(\w+)'", line)
                if regime_match:
                    regime = regime_match.group(1)
                    regimes_detected.add(regime)
                    if prev_regime and prev_regime != regime:
                        regime_changes += 1
                    prev_regime = regime
                    
    return regimes_detected, regime_changes

def main():
    print("="*80)
    print("Comparing Test Start Times and Regime Detection")
    print("="*80)
    
    # Analyze optimization log (adaptive test section)
    print("\nOptimization Adaptive Test (from line 2602000):")
    opt_bar, opt_signal, opt_regime = find_test_start('logs/admf_20250523_175845.log', 2602000)
    opt_regimes, opt_changes = analyze_regime_detection('logs/admf_20250523_175845.log', 2602000)
    
    print(f"  First test BAR: {opt_bar}")
    print(f"  First test SIGNAL: {opt_signal}")
    print(f"  Regime at start: {opt_regime}")
    print(f"  Regimes detected: {sorted(opt_regimes)}")
    print(f"  Regime changes: {opt_changes}")
    
    # Analyze production logs
    prod_logs = [
        ('Production (RegimeAdaptive)', 'logs/admf_20250523_175758.log'),
        ('Production (Ensemble)', 'logs/admf_20250523_183637.log'),
        ('Production (Fixed)', 'logs/admf_20250523_185455.log')
    ]
    
    for name, log_file in prod_logs:
        print(f"\n{name}:")
        try:
            prod_bar, prod_signal, prod_regime = find_test_start(log_file)
            prod_regimes, prod_changes = analyze_regime_detection(log_file)
            
            print(f"  First test BAR: {prod_bar}")
            print(f"  First test SIGNAL: {prod_signal}")
            print(f"  Regime at start: {prod_regime}")
            print(f"  Regimes detected: {sorted(prod_regimes)}")
            print(f"  Regime changes: {prod_changes}")
            
            # Compare with optimization
            if opt_bar and prod_bar:
                if opt_bar == prod_bar:
                    print(f"  ✓ Same start time as optimization")
                else:
                    print(f"  ✗ Different start time! Opt: {opt_bar}, Prod: {prod_bar}")
                    
            if opt_regime and prod_regime:
                if opt_regime == prod_regime:
                    print(f"  ✓ Same starting regime as optimization")
                else:
                    print(f"  ✗ Different starting regime! Opt: {opt_regime}, Prod: {prod_regime}")
                    
        except FileNotFoundError:
            print(f"  Log file not found: {log_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    
    if opt_changes > 0:
        print("✓ Optimization IS detecting regime changes during test")
    else:
        print("✗ Optimization is NOT detecting regime changes")
        
    print(f"\nOptimization detected {len(opt_regimes)} unique regimes with {opt_changes} changes")
    print("Production runs should show similar regime diversity for accurate comparison")

if __name__ == "__main__":
    main()