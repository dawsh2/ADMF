#!/usr/bin/env python3
"""
Compare signals between optimization test phase and production runs
"""

import re
from datetime import datetime
from collections import defaultdict

def extract_signals(log_file, is_optimization=False):
    """Extract signal events from log file"""
    signals = []
    
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
            
            # Look for SIGNAL events
            if "Publishing event: Event(type=SIGNAL" in line:
                # Extract timestamp from the signal
                ts_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                if ts_match:
                    signal_time = ts_match.group(1).split('+')[0]  # Remove timezone
                    signal_dt = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase signals
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include signals from test period (after 2025-01-27 18:06:00)
                    if signal_dt < test_start_time:
                        continue
                    
                    # Extract signal details
                    signal_type_match = re.search(r"'signal_type': (-?[0-9]+)", line)
                    strength_match = re.search(r"'signal_strength': ([0-9.]+)", line)
                    price_match = re.search(r"'price_at_signal': ([0-9.]+)", line)
                    reason_match = re.search(r"'reason': '([^']+)'", line)
                    
                    if signal_type_match:
                        signal_type = int(signal_type_match.group(1))
                        action = "BUY" if signal_type == 1 else "SELL" if signal_type == -1 else "NEUTRAL"
                        
                        signal = {
                            'timestamp': signal_time,
                            'action': action,
                            'signal_type': signal_type,
                            'strength': float(strength_match.group(1)) if strength_match else None,
                            'price': float(price_match.group(1)) if price_match else None,
                            'reason': reason_match.group(1) if reason_match else None,
                            'line': line.strip()
                        }
                        signals.append(signal)
    
    return signals

def compare_signals(opt_signals, prod_signals):
    """Compare signals between optimization and production"""
    print(f"\n=== SIGNAL COMPARISON ===")
    print(f"Optimization signals in test period: {len(opt_signals)}")
    print(f"Production signals in test period: {len(prod_signals)}")
    
    # Group signals by timestamp for easier comparison
    opt_by_time = {s['timestamp']: s for s in opt_signals}
    prod_by_time = {s['timestamp']: s for s in prod_signals}
    
    # Find common timestamps
    common_times = set(opt_by_time.keys()) & set(prod_by_time.keys())
    opt_only_times = set(opt_by_time.keys()) - set(prod_by_time.keys())
    prod_only_times = set(prod_by_time.keys()) - set(opt_by_time.keys())
    
    print(f"\nCommon signal times: {len(common_times)}")
    print(f"Optimization-only signals: {len(opt_only_times)}")
    print(f"Production-only signals: {len(prod_only_times)}")
    
    # Check for differences in common signals
    differences = []
    for time in sorted(common_times)[:10]:  # First 10 for brevity
        opt_sig = opt_by_time[time]
        prod_sig = prod_by_time[time]
        
        if (opt_sig['action'] != prod_sig['action'] or 
            opt_sig['strength'] != prod_sig['strength']):
            differences.append({
                'time': time,
                'opt': opt_sig,
                'prod': prod_sig
            })
    
    if differences:
        print(f"\n=== SIGNAL DIFFERENCES (first {len(differences)}) ===")
        for diff in differences:
            print(f"\nTime: {diff['time']}")
            print(f"  Optimization: {diff['opt']['action']} @ strength {diff['opt']['strength']}")
            print(f"  Production:   {diff['prod']['action']} @ strength {diff['prod']['strength']}")
    
    # Show first few unique signals
    if opt_only_times:
        print(f"\n=== FIRST 5 OPTIMIZATION-ONLY SIGNALS ===")
        for time in sorted(opt_only_times)[:5]:
            sig = opt_by_time[time]
            print(f"{time}: {sig['action']} @ strength {sig['strength']}")
    
    if prod_only_times:
        print(f"\n=== FIRST 5 PRODUCTION-ONLY SIGNALS ===")
        for time in sorted(prod_only_times)[:5]:
            sig = prod_by_time[time]
            print(f"{time}: {sig['action']} @ strength {sig['strength']}")
    
    # Analyze signal strength distribution
    print(f"\n=== SIGNAL STRENGTH ANALYSIS ===")
    opt_strengths = [s['strength'] for s in opt_signals if s['strength']]
    prod_strengths = [s['strength'] for s in prod_signals if s['strength']]
    
    if opt_strengths:
        print(f"Optimization strengths: min={min(opt_strengths):.3f}, max={max(opt_strengths):.3f}, unique={set(opt_strengths)}")
    if prod_strengths:
        print(f"Production strengths: min={min(prod_strengths):.3f}, max={max(prod_strengths):.3f}, unique={set(prod_strengths)}")
    
    return differences

def main():
    print("Comparing signals between optimization and production runs...")
    
    # Extract signals from optimization log
    print("\nExtracting signals from optimization log...")
    opt_signals = extract_signals('logs/admf_20250523_202211.log', is_optimization=True)
    
    # Extract signals from production log
    print("Extracting signals from production log...")
    # Use the most recent production log
    prod_signals = extract_signals('logs/admf_20250523_202911.log', is_optimization=False)
    
    # Compare signals
    differences = compare_signals(opt_signals, prod_signals)
    
    # If signals are identical, check regime events
    if len(differences) == 0 and len(opt_signals) == len(prod_signals):
        print("\n\nSignals appear identical! Checking regime/classification events...")
        # TODO: Add regime comparison

if __name__ == "__main__":
    main()