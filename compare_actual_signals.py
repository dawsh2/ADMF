#!/usr/bin/env python3
"""
Compare actual signals between production and optimizer at matching timestamps
"""
import re
from datetime import datetime

def extract_signals(log_file):
    """Extract all signals with their details"""
    signals = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            if '🚨 SIGNAL GENERATED' in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2}).*Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+)', line)
                if match:
                    timestamp = match.group(1)
                    signal_type = int(match.group(2))
                    price = float(match.group(3))
                    regime = match.group(4)
                    
                    signals[timestamp] = {
                        'type': signal_type,
                        'price': price,
                        'regime': regime,
                        'full_line': line.strip()
                    }
    
    return signals

def compare_signals(prod_signals, opt_signals):
    """Compare signals at matching timestamps"""
    
    # Get all unique timestamps
    all_timestamps = sorted(set(prod_signals.keys()) | set(opt_signals.keys()))
    
    print("SIGNAL COMPARISON BY TIMESTAMP")
    print("=" * 140)
    print(f"{'Timestamp':^25} | {'Production':^40} | {'Optimizer':^40} | {'Match':^10}")
    print("-" * 140)
    
    matches = 0
    prod_only = 0
    opt_only = 0
    mismatches = 0
    
    for ts in all_timestamps:
        prod = prod_signals.get(ts)
        opt = opt_signals.get(ts)
        
        if prod and opt:
            # Both have signals at this time
            if prod['type'] == opt['type'] and prod['regime'] == opt['regime']:
                matches += 1
                match_str = "✓ MATCH"
                prod_str = f"Type={prod['type']:2d}, P={prod['price']:.2f}, R={prod['regime']}"
                opt_str = f"Type={opt['type']:2d}, P={opt['price']:.2f}, R={opt['regime']}"
            else:
                mismatches += 1
                match_str = "✗ MISMATCH"
                prod_str = f"Type={prod['type']:2d}, P={prod['price']:.2f}, R={prod['regime']}"
                opt_str = f"Type={opt['type']:2d}, P={opt['price']:.2f}, R={opt['regime']}"
                
            print(f"{ts:^25} | {prod_str:^40} | {opt_str:^40} | {match_str:^10}")
            
        elif prod and not opt:
            prod_only += 1
            prod_str = f"Type={prod['type']:2d}, P={prod['price']:.2f}, R={prod['regime']}"
            print(f"{ts:^25} | {prod_str:^40} | {'NO SIGNAL':^40} | {'PROD ONLY':^10}")
            
        elif opt and not prod:
            opt_only += 1
            opt_str = f"Type={opt['type']:2d}, P={opt['price']:.2f}, R={opt['regime']}"
            print(f"{ts:^25} | {'NO SIGNAL':^40} | {opt_str:^40} | {'OPT ONLY':^10}")
    
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(f"Total Production Signals: {len(prod_signals)}")
    print(f"Total Optimizer Signals:  {len(opt_signals)}")
    print(f"\nMatching signals:    {matches}")
    print(f"Mismatched signals:  {mismatches}")
    print(f"Production only:     {prod_only}")
    print(f"Optimizer only:      {opt_only}")
    
    if matches > 0:
        match_rate = matches / (matches + mismatches) * 100 if (matches + mismatches) > 0 else 0
        print(f"\nMatch rate (when both have signals): {match_rate:.1f}%")

def main():
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_002523.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print(f"Comparing signals from:")
    print(f"  Production: {prod_file}")
    print(f"  Optimizer:  {opt_file}")
    print()
    
    prod_signals = extract_signals(prod_file)
    opt_signals = extract_signals(opt_file)
    
    compare_signals(prod_signals, opt_signals)

if __name__ == "__main__":
    main()