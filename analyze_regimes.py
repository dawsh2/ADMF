#!/usr/bin/env python3
"""
Analyze regime patterns and parameter switching in logs
"""

import sys
import re
from collections import Counter, defaultdict

def analyze_regime_patterns(log_file):
    """Extract and analyze regime changes and parameter applications"""
    
    regime_changes = []
    parameter_applications = []
    signals_by_regime = defaultdict(list)
    current_regime = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Track regime changes
            if "Market regime changed from" in line:
                match = re.search(r"regime changed from '(\w+)' to '(\w+)' at ([^']+)", line)
                if match:
                    from_regime, to_regime, timestamp = match.groups()
                    regime_changes.append((timestamp, from_regime, to_regime))
                    current_regime = to_regime
            
            # Track parameter applications
            if "Applying parameters for" in line or "ADAPTIVE TEST.*Applying parameters" in line:
                match = re.search(r"regime '(\w+)': ({[^}]+})", line)
                if match:
                    regime, params = match.groups()
                    parameter_applications.append((regime, params))
                    
            # Track signals with their regimes
            if "Publishing event: Event(type=SIGNAL" in line:
                timestamp_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                signal_match = re.search(r"'signal_type': ([+-]?\d+)", line)
                strength_match = re.search(r"'signal_strength': ([+-]?\d+\.?\d*)", line)
                reason_match = re.search(r"Regime:\s*(\w+)", line)
                
                if timestamp_match and signal_match:
                    timestamp = timestamp_match.group(1)
                    signal_type = int(signal_match.group(1))
                    strength = float(strength_match.group(1)) if strength_match else 1.0
                    regime = reason_match.group(1) if reason_match else current_regime or 'unknown'
                    
                    signals_by_regime[regime].append({
                        'timestamp': timestamp,
                        'signal': signal_type,
                        'strength': strength
                    })
    
    # Analyze results
    print(f"\n{'='*60}")
    print(f"REGIME ANALYSIS FOR: {log_file}")
    print(f"{'='*60}")
    
    # Regime changes
    print(f"\nTotal regime changes: {len(regime_changes)}")
    if regime_changes:
        print("\nFirst 10 regime changes:")
        for i, (ts, from_r, to_r) in enumerate(regime_changes[:10]):
            print(f"  {i+1}. {ts}: {from_r} → {to_r}")
            
        # Count regime transitions
        transitions = Counter([(from_r, to_r) for _, from_r, to_r in regime_changes])
        print(f"\nMost common regime transitions:")
        for (from_r, to_r), count in transitions.most_common(10):
            print(f"  {from_r} → {to_r}: {count} times")
    
    # Parameter applications
    print(f"\nParameter applications: {len(parameter_applications)}")
    if parameter_applications:
        print("\nUnique parameter sets by regime:")
        regime_params = defaultdict(set)
        for regime, params in parameter_applications:
            regime_params[regime].add(params)
        
        for regime, param_sets in regime_params.items():
            print(f"\n  {regime}: {len(param_sets)} unique parameter set(s)")
            for params in list(param_sets)[:1]:  # Show first param set
                print(f"    {params[:100]}...")
    
    # Signals by regime
    print(f"\nSignals by regime:")
    total_signals = 0
    for regime, signals in sorted(signals_by_regime.items()):
        total_signals += len(signals)
        # Calculate average strength
        avg_strength = sum(s['strength'] for s in signals) / len(signals) if signals else 0
        print(f"  {regime}: {len(signals)} signals (avg strength: {avg_strength:.3f})")
    
    print(f"\nTotal signals: {total_signals}")
    
    # Check for regime detection during test period
    test_date = '2025-01-27'
    test_regime_changes = [(ts, from_r, to_r) for ts, from_r, to_r in regime_changes if ts >= test_date]
    print(f"\nRegime changes during test period (>= {test_date}): {len(test_regime_changes)}")
    
    if test_regime_changes:
        # Count unique regimes in test period
        test_regimes = set()
        for _, from_r, to_r in test_regime_changes:
            test_regimes.add(from_r)
            test_regimes.add(to_r)
        print(f"Unique regimes in test period: {sorted(test_regimes)}")
    
    return regime_changes, signals_by_regime

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_regimes.py <log_file>")
        sys.exit(1)
    
    analyze_regime_patterns(sys.argv[1])