#!/usr/bin/env python3
"""
Final comparison of signals between production and optimizer with matching configs.
"""

import re
from datetime import datetime

def extract_signals(log_file):
    """Extract signals from log file."""
    signals = []
    current_timestamp = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract timestamp from BAR lines
        bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', line)
        if bar_match:
            timestamp_str = bar_match.group(1).split('+')[0]
            try:
                current_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Extract signals
        signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=(-?\d+)', line)
        if signal_match and current_timestamp and current_timestamp.year == 2024:
            signals.append({
                'timestamp': current_timestamp,
                'type': int(signal_match.group(2)),
                'direction': 'BUY' if int(signal_match.group(2)) == 1 else 'SELL'
            })
    
    return signals

def main():
    print("FINAL SIGNAL COMPARISON")
    print("="*70)
    print("Production: Using data/1000_1min.csv with 0.80 split and RSI disabled")
    print("Optimizer:  Using data/1000_1min.csv with 0.80 split")
    print("="*70)
    
    # Extract signals
    prod_signals = extract_signals('logs/admf_20250524_214722.log')
    opt_signals = extract_signals('logs/admf_20250523_230532.log')
    
    print(f"\nSignal counts:")
    print(f"  Production: {len(prod_signals)} signals")
    print(f"  Optimizer:  {len(opt_signals)} signals")
    
    # Create timestamp lookups
    prod_by_time = {s['timestamp']: s for s in prod_signals}
    opt_by_time = {s['timestamp']: s for s in opt_signals}
    
    # Find matches and mismatches
    all_times = sorted(set(prod_by_time.keys()) | set(opt_by_time.keys()))
    
    print("\nDetailed comparison:")
    print("-"*70)
    print("Timestamp            | Production | Optimizer  | Match")
    print("-"*70)
    
    matches = 0
    for t in all_times:
        prod_sig = prod_by_time.get(t)
        opt_sig = opt_by_time.get(t)
        
        if prod_sig and opt_sig:
            match = "✓" if prod_sig['type'] == opt_sig['type'] else "✗"
            if prod_sig['type'] == opt_sig['type']:
                matches += 1
            print(f"{t} | {prod_sig['direction']:^10s} | {opt_sig['direction']:^10s} | {match:^5s}")
        elif prod_sig:
            print(f"{t} | {prod_sig['direction']:^10s} | {'---':^10s} | Prod only")
        else:
            print(f"{t} | {'---':^10s} | {opt_sig['direction']:^10s} | Opt only")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("-"*70)
    
    common_times = set(prod_by_time.keys()) & set(opt_by_time.keys())
    prod_only = set(prod_by_time.keys()) - set(opt_by_time.keys())
    opt_only = set(opt_by_time.keys()) - set(prod_by_time.keys())
    
    print(f"Common timestamps: {len(common_times)}")
    print(f"  Matching signals: {matches}")
    print(f"  Different signals: {len(common_times) - matches}")
    if common_times:
        print(f"  Match rate: {matches/len(common_times)*100:.1f}%")
    
    print(f"\nProduction-only signals: {len(prod_only)}")
    for t in sorted(prod_only):
        print(f"  {t}: {prod_by_time[t]['direction']}")
    
    print(f"\nOptimizer-only signals: {len(opt_only)}")
    for t in sorted(opt_only):
        print(f"  {t}: {opt_by_time[t]['direction']}")
    
    print("\n" + "="*70)
    if len(prod_signals) == len(opt_signals) and matches == len(common_times) and len(prod_only) == 0 and len(opt_only) == 0:
        print("✓ PERFECT MATCH! Production and optimizer generate identical signals.")
    else:
        print("✗ Still have discrepancies:")
        print(f"  - Signal count difference: {abs(len(prod_signals) - len(opt_signals))}")
        print(f"  - Unmatched timestamps: {len(prod_only) + len(opt_only)}")

if __name__ == "__main__":
    main()