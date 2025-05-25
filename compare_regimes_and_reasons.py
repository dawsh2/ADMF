#!/usr/bin/env python3
"""
Compare regime classifications and signal reasons between optimization and production
"""

import re
from datetime import datetime
from collections import Counter

def extract_regime_info_from_signals(log_file, is_optimization=False):
    """Extract regime information from signal reasons"""
    regimes = []
    signals_by_regime = Counter()
    
    # For optimization, we need to identify the test phase
    in_test_phase = False
    test_start_time = datetime.strptime("2025-01-27 18:06:00", "%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check if we're in test phase for optimization logs
            if is_optimization:
                if "ADAPTIVE TEST" in line or "!!! ADAPTIVE TEST !!!" in line:
                    in_test_phase = True
                elif "Optimization complete" in line or "Enhanced Grid Search with Train/Test Ended" in line:
                    in_test_phase = False
            
            # Look for SIGNAL events with regime information
            if "Publishing event: Event(type=SIGNAL" in line:
                # Extract timestamp
                ts_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                if ts_match:
                    signal_time = ts_match.group(1).split('+')[0]
                    signal_dt = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase signals
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include signals from test period
                    if signal_dt < test_start_time:
                        continue
                    
                    # Extract regime from reason field
                    reason_match = re.search(r"'reason': '([^']+)'", line)
                    if reason_match:
                        reason = reason_match.group(1)
                        # Extract regime from reason like "Ensemble_Voting(MediumSignal(strength=0.500), Regime: trending_down)"
                        regime_match = re.search(r"Regime: (\w+)", reason)
                        if regime_match:
                            regime = regime_match.group(1)
                            regimes.append({
                                'timestamp': signal_time,
                                'regime': regime,
                                'reason': reason
                            })
                            signals_by_regime[regime] += 1
    
    return regimes, signals_by_regime

def extract_weight_info(log_file):
    """Extract MA and RSI weight information"""
    weights = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for weight log entries
            if "EnsembleSignalStrategy weights:" in line:
                ma_match = re.search(r"MA=([\d.]+)", line)
                rsi_match = re.search(r"RSI=([\d.]+)", line)
                if ma_match and rsi_match:
                    weights.append({
                        'ma': float(ma_match.group(1)),
                        'rsi': float(rsi_match.group(1))
                    })
    
    return weights

def main():
    print("Comparing regime classifications and weights...\n")
    
    # Extract regime information
    opt_regimes, opt_regime_counts = extract_regime_info_from_signals('logs/admf_20250523_202211.log', is_optimization=True)
    prod_regimes, prod_regime_counts = extract_regime_info_from_signals('logs/admf_20250523_202911.log', is_optimization=False)
    
    print("=== REGIME DISTRIBUTION IN SIGNALS ===")
    print("\nOptimization regime counts:")
    for regime, count in sorted(opt_regime_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {count} signals")
    
    print("\nProduction regime counts:")
    for regime, count in sorted(prod_regime_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {regime}: {count} signals")
    
    # Extract weight information
    print("\n=== WEIGHT ANALYSIS ===")
    opt_weights = extract_weight_info('logs/admf_20250523_202211.log')
    prod_weights = extract_weight_info('logs/admf_20250523_202911.log')
    
    if opt_weights:
        print(f"\nOptimization weights found: {len(opt_weights)}")
        # Show unique weight combinations
        unique_opt = set((w['ma'], w['rsi']) for w in opt_weights)
        for ma, rsi in unique_opt:
            print(f"  MA={ma}, RSI={rsi}")
    
    if prod_weights:
        print(f"\nProduction weights found: {len(prod_weights)}")
        # Show unique weight combinations
        unique_prod = set((w['ma'], w['rsi']) for w in prod_weights)
        for ma, rsi in unique_prod:
            print(f"  MA={ma}, RSI={rsi}")
    
    # Show first few signal reasons to understand format
    print("\n=== SAMPLE SIGNAL REASONS ===")
    print("\nOptimization (first 3):")
    for i, regime_info in enumerate(opt_regimes[:3]):
        print(f"  {regime_info['timestamp']}: {regime_info['reason']}")
    
    print("\nProduction (first 3):")
    for i, regime_info in enumerate(prod_regimes[:3]):
        print(f"  {regime_info['timestamp']}: {regime_info['reason']}")

if __name__ == "__main__":
    main()