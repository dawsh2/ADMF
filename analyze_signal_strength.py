#!/usr/bin/env python3
"""
Analyze signal strength distribution between logs
"""

import sys
import re
from collections import Counter

def extract_signals_with_strength(log_file, start_marker=None):
    """Extract signals with their strengths and regimes"""
    signals = []
    in_section = start_marker is None
    
    with open(log_file, 'r') as f:
        for line in f:
            if start_marker and start_marker in line:
                in_section = True
                
            if in_section and 'Publishing event: Event(type=SIGNAL' in line:
                # Extract all relevant fields
                timestamp_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                signal_match = re.search(r"'signal_type': ([+-]?\d+)", line)
                regime_match = re.search(r"Regime:\s*(\w+)", line)
                strength_match = re.search(r"'signal_strength': ([+-]?\d+\.?\d*)", line)
                reason_match = re.search(r"'reason': '([^']+)'", line)
                
                if timestamp_match and signal_match:
                    timestamp = timestamp_match.group(1)
                    signal_type = int(signal_match.group(1))
                    regime = regime_match.group(1) if regime_match else 'unknown'
                    strength = float(strength_match.group(1)) if strength_match else 1.0
                    reason = reason_match.group(1) if reason_match else ''
                    
                    signals.append({
                        'timestamp': timestamp,
                        'signal_type': signal_type,
                        'regime': regime,
                        'strength': strength,
                        'reason': reason
                    })
    
    return signals

def analyze_strengths(signals, name):
    """Analyze signal strength distribution"""
    print(f"\n{'='*60}")
    print(f"Signal Strength Analysis: {name}")
    print(f"{'='*60}")
    
    # Count unique strengths
    strengths = [s['strength'] for s in signals]
    strength_counts = Counter(strengths)
    
    print(f"Total signals: {len(signals)}")
    print(f"Unique strength values: {len(strength_counts)}")
    
    # Show distribution
    print(f"\nStrength distribution:")
    for strength, count in sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(signals)) * 100
        print(f"  {strength:6.2f}: {count:4d} signals ({percentage:5.1f}%)")
    
    # Check for position sizing patterns
    fractional_strengths = [s for s in strengths if s != 1.0 and s != 0.0]
    print(f"\nFractional strengths: {len(fractional_strengths)} ({len(fractional_strengths)/len(signals)*100:.1f}%)")
    
    if fractional_strengths:
        print(f"Range: {min(fractional_strengths):.3f} to {max(fractional_strengths):.3f}")
        
    # Sample some fractional strength signals
    frac_signals = [s for s in signals if s['strength'] not in [0.0, 1.0]][:5]
    if frac_signals:
        print(f"\nSample fractional strength signals:")
        for sig in frac_signals:
            print(f"  {sig['timestamp']}: strength={sig['strength']:.3f}, reason='{sig['reason'][:50]}...'")
    
    return strength_counts

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_signal_strength.py <production_log> <optimization_log>")
        sys.exit(1)
        
    prod_log = sys.argv[1]
    opt_log = sys.argv[2]
    
    print("Extracting signals...")
    prod_signals = extract_signals_with_strength(prod_log)
    opt_signals = extract_signals_with_strength(opt_log, start_marker="ENABLING ADAPTIVE MODE")
    
    # Filter for test data
    test_date = '2025-01-27'
    prod_test = [s for s in prod_signals if s['timestamp'] >= test_date]
    opt_test = [s for s in opt_signals if s['timestamp'] >= test_date]
    
    print(f"\nTest data signals (>= {test_date}):")
    print(f"Production: {len(prod_test)}")
    print(f"Optimization: {len(opt_test)}")
    
    # Analyze strengths
    prod_strengths = analyze_strengths(prod_test, "Production")
    opt_strengths = analyze_strengths(opt_test, "Optimization")
    
    # Compare signal generation
    print(f"\n{'='*60}")
    print("Signal Generation Comparison")
    print(f"{'='*60}")
    
    # Calculate expected trades based on strengths
    prod_expected_trades = sum(count * strength for strength, count in prod_strengths.items())
    opt_expected_trades = sum(count * strength for strength, count in opt_strengths.items())
    
    print(f"\nExpected 'trade units' based on signal strengths:")
    print(f"Production: {prod_expected_trades:.1f}")
    print(f"Optimization: {opt_expected_trades:.1f}")
    print(f"Ratio: {opt_expected_trades/prod_expected_trades:.2f}x")
    
    print(f"\nThis could explain the performance difference!")
    print(f"If optimization uses position sizing based on signal strength,")
    print(f"it would generate {opt_expected_trades/prod_expected_trades:.1f}x more trading volume.")

if __name__ == "__main__":
    main()