#!/usr/bin/env python3
"""
Compare actual signals between production and optimizer at matching timestamps
"""
import re
from collections import defaultdict

def extract_signals_with_context(log_file):
    """Extract all signals with their bar timestamps"""
    signals = {}
    current_bar_timestamp = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # Track current bar timestamp
        if '📊 BAR_' in line:
            bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', line)
            if bar_match:
                current_bar_timestamp = bar_match.group(1)
        
        # Extract signals
        if '🚨 SIGNAL GENERATED' in line:
            match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+)', line)
            if match and current_bar_timestamp:
                signal_num = match.group(1)
                signal_type = int(match.group(2))
                price = float(match.group(3))
                regime = match.group(4)
                
                # Extract additional details
                ma_match = re.search(r'MA_signal=([+-]?\d+)', line)
                rsi_match = re.search(r'RSI_signal=([+-]?\d+)', line)
                strength_match = re.search(r'Combined_strength=([+-]?[0-9.]+)', line)
                
                signals[current_bar_timestamp] = {
                    'num': signal_num,
                    'type': signal_type,
                    'price': price,
                    'regime': regime,
                    'ma_signal': int(ma_match.group(1)) if ma_match else 0,
                    'rsi_signal': int(rsi_match.group(1)) if rsi_match else 0,
                    'strength': float(strength_match.group(1)) if strength_match else 0.0
                }
    
    return signals

def compare_signals(prod_signals, opt_signals):
    """Compare signals at matching timestamps"""
    
    # Get all unique timestamps
    all_timestamps = sorted(set(prod_signals.keys()) | set(opt_signals.keys()))
    
    print("SIGNAL COMPARISON BY BAR TIMESTAMP")
    print("=" * 140)
    print(f"{'Bar Timestamp':^25} | {'Production':^50} | {'Optimizer':^50} | {'Match':^10}")
    print("-" * 140)
    
    matches = 0
    type_matches = 0
    regime_matches = 0
    prod_only = 0
    opt_only = 0
    mismatches = 0
    
    # Track signals by hour for pattern analysis
    hourly_matches = defaultdict(int)
    hourly_mismatches = defaultdict(int)
    
    for ts in all_timestamps:
        prod = prod_signals.get(ts)
        opt = opt_signals.get(ts)
        
        # Extract hour from timestamp for pattern analysis
        hour = ts.split(' ')[1].split(':')[0] if ' ' in ts else 'unknown'
        
        if prod and opt:
            # Both have signals at this time
            type_match = prod['type'] == opt['type']
            regime_match = prod['regime'] == opt['regime']
            
            if type_match and regime_match:
                matches += 1
                match_str = "✓ MATCH"
                hourly_matches[hour] += 1
            else:
                mismatches += 1
                if type_match:
                    match_str = "✗ REGIME"
                    type_matches += 1
                elif regime_match:
                    match_str = "✗ TYPE"
                    regime_matches += 1
                else:
                    match_str = "✗ BOTH"
                hourly_mismatches[hour] += 1
                
            prod_str = f"#{prod['num']:>2} T={prod['type']:2d} P={prod['price']:.2f} R={prod['regime']:>15} MA={prod['ma_signal']:2d} RSI={prod['rsi_signal']:2d}"
            opt_str = f"#{opt['num']:>2} T={opt['type']:2d} P={opt['price']:.2f} R={opt['regime']:>15} MA={opt['ma_signal']:2d} RSI={opt['rsi_signal']:2d}"
            
            print(f"{ts[:19]:^25} | {prod_str:^50} | {opt_str:^50} | {match_str:^10}")
            
        elif prod and not opt:
            prod_only += 1
            prod_str = f"#{prod['num']:>2} T={prod['type']:2d} P={prod['price']:.2f} R={prod['regime']:>15}"
            print(f"{ts[:19]:^25} | {prod_str:^50} | {'NO SIGNAL':^50} | {'PROD ONLY':^10}")
            
        elif opt and not prod:
            opt_only += 1
            opt_str = f"#{opt['num']:>2} T={opt['type']:2d} P={opt['price']:.2f} R={opt['regime']:>15}"
            print(f"{ts[:19]:^25} | {'NO SIGNAL':^50} | {opt_str:^50} | {'OPT ONLY':^10}")
    
    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)
    print(f"Total Production Signals: {len(prod_signals)}")
    print(f"Total Optimizer Signals:  {len(opt_signals)}")
    print(f"\nMatching signals:       {matches}")
    print(f"Type matches only:      {type_matches}")
    print(f"Regime matches only:    {regime_matches}")
    print(f"Complete mismatches:    {mismatches - type_matches - regime_matches}")
    print(f"Production only:        {prod_only}")
    print(f"Optimizer only:         {opt_only}")
    
    if matches + mismatches > 0:
        match_rate = matches / (matches + mismatches) * 100
        print(f"\nMatch rate (when both have signals): {match_rate:.1f}%")
    
    # Show hourly pattern
    print(f"\nHOURLY PATTERN:")
    for hour in sorted(set(hourly_matches.keys()) | set(hourly_mismatches.keys())):
        m = hourly_matches[hour]
        mm = hourly_mismatches[hour]
        total = m + mm
        if total > 0:
            rate = m / total * 100
            print(f"  Hour {hour}:00 - {m} matches, {mm} mismatches ({rate:.0f}% match rate)")

def main():
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_203019.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print(f"Comparing signals from:")
    print(f"  Production: {prod_file}")
    print(f"  Optimizer:  {opt_file}")
    print()
    
    prod_signals = extract_signals_with_context(prod_file)
    opt_signals = extract_signals_with_context(opt_file)
    
    compare_signals(prod_signals, opt_signals)

if __name__ == "__main__":
    main()