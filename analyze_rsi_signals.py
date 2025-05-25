#!/usr/bin/env python3
"""
Analyze RSI values for all signals in both production and optimizer runs
"""
import re
from collections import defaultdict

def extract_signals_with_rsi(log_file):
    """Extract all signals with their RSI values and context"""
    signals = []
    current_bar_info = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Track current bar information
        if '📊 BAR_' in line and 'INDICATORS:' in line:
            bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=(\w+)', line)
            if bar_match:
                current_bar_info = {
                    'bar_num': bar_match.group(1),
                    'timestamp': bar_match.group(2),
                    'price': float(bar_match.group(3)),
                    'ma_short': bar_match.group(4),
                    'ma_long': bar_match.group(5),
                    'rsi': bar_match.group(6),
                    'rsi_thresholds': bar_match.group(7),
                    'regime': bar_match.group(8)
                }
        
        # Extract RSI calculation details
        if 'RSI calculation:' in line and current_bar_info:
            rsi_calc_match = re.search(r'RSI calculation: period=(\d+), values_count=(\d+), value=([0-9.]+|N/A)', line)
            if rsi_calc_match:
                current_bar_info['rsi_period'] = int(rsi_calc_match.group(1))
                current_bar_info['rsi_values_count'] = int(rsi_calc_match.group(2))
                current_bar_info['rsi_calc_value'] = rsi_calc_match.group(3)
        
        # Extract signals
        if '🚨 SIGNAL GENERATED' in line and current_bar_info:
            signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\), Combined_strength=([+-]?[0-9.]+), Final_multiplier=([0-9.]+)', line)
            if signal_match:
                signal_data = {
                    'signal_num': int(signal_match.group(1)),
                    'type': int(signal_match.group(2)),
                    'price': float(signal_match.group(3)),
                    'regime': signal_match.group(4),
                    'ma_signal': int(signal_match.group(5)),
                    'ma_weight': float(signal_match.group(6)),
                    'rsi_signal': int(signal_match.group(7)),
                    'rsi_weight': float(signal_match.group(8)),
                    'combined_strength': float(signal_match.group(9)),
                    'final_multiplier': float(signal_match.group(10)),
                    'bar_timestamp': current_bar_info.get('timestamp', 'Unknown'),
                    'rsi_value': current_bar_info.get('rsi', 'N/A'),
                    'rsi_period': current_bar_info.get('rsi_period', 'Unknown'),
                    'rsi_values_count': current_bar_info.get('rsi_values_count', 'Unknown'),
                    'rsi_thresholds': current_bar_info.get('rsi_thresholds', 'Unknown')
                }
                signals.append(signal_data)
    
    return signals

def analyze_rsi_patterns(prod_signals, opt_signals):
    """Analyze RSI patterns in signals"""
    print("\nRSI ANALYSIS FOR ALL SIGNALS")
    print("=" * 120)
    
    # Group signals by timestamp for comparison
    prod_by_time = {s['bar_timestamp']: s for s in prod_signals}
    opt_by_time = {s['bar_timestamp']: s for s in opt_signals}
    
    # All unique timestamps
    all_timestamps = sorted(set(prod_by_time.keys()) | set(opt_by_time.keys()))
    
    print(f"{'Timestamp':^20} | {'Production RSI':^30} | {'Optimizer RSI':^30} | {'Signal Match':^15} | {'RSI Match':^10}")
    print("-" * 120)
    
    rsi_stats = {
        'both_na': 0,
        'both_valid': 0,
        'prod_na_opt_valid': 0,
        'prod_valid_opt_na': 0,
        'matching_signals': 0,
        'mismatched_signals': 0
    }
    
    for ts in all_timestamps:
        prod = prod_by_time.get(ts)
        opt = opt_by_time.get(ts)
        
        if prod and opt:
            # Both have signals
            prod_rsi = prod['rsi_value']
            opt_rsi = opt['rsi_value']
            
            # Check RSI states
            if prod_rsi == 'N/A' and opt_rsi == 'N/A':
                rsi_stats['both_na'] += 1
                rsi_match = "Both N/A"
            elif prod_rsi != 'N/A' and opt_rsi != 'N/A':
                rsi_stats['both_valid'] += 1
                try:
                    rsi_match = "✓" if abs(float(prod_rsi) - float(opt_rsi)) < 0.01 else f"✗ ({float(prod_rsi):.2f} vs {float(opt_rsi):.2f})"
                except:
                    rsi_match = "Parse Error"
            elif prod_rsi == 'N/A' and opt_rsi != 'N/A':
                rsi_stats['prod_na_opt_valid'] += 1
                rsi_match = "P:N/A O:Valid"
            else:
                rsi_stats['prod_valid_opt_na'] += 1
                rsi_match = "P:Valid O:N/A"
            
            # Check signal match
            signal_match = "✓" if prod['type'] == opt['type'] and prod['rsi_signal'] == opt['rsi_signal'] else "✗"
            if signal_match == "✓":
                rsi_stats['matching_signals'] += 1
            else:
                rsi_stats['mismatched_signals'] += 1
            
            prod_info = f"RSI={prod_rsi} (Per={prod.get('rsi_period', '?')}) Sig={prod['rsi_signal']}"
            opt_info = f"RSI={opt_rsi} (Per={opt.get('rsi_period', '?')}) Sig={opt['rsi_signal']}"
            
            print(f"{ts[:19]:^20} | {prod_info:^30} | {opt_info:^30} | {signal_match:^15} | {rsi_match:^10}")
        
        elif prod:
            # Production only
            prod_info = f"RSI={prod['rsi_value']} Sig={prod['rsi_signal']}"
            print(f"{ts[:19]:^20} | {prod_info:^30} | {'NO SIGNAL':^30} | {'Prod Only':^15} | {'N/A':^10}")
        
        elif opt:
            # Optimizer only
            opt_info = f"RSI={opt['rsi_value']} Sig={opt['rsi_signal']}"
            print(f"{ts[:19]:^20} | {'NO SIGNAL':^30} | {opt_info:^30} | {'Opt Only':^15} | {'N/A':^10}")
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("RSI STATISTICS")
    print("=" * 120)
    print(f"Total Production Signals: {len(prod_signals)}")
    print(f"Total Optimizer Signals: {len(opt_signals)}")
    print(f"\nWhen both have signals:")
    print(f"  Both RSI N/A: {rsi_stats['both_na']}")
    print(f"  Both RSI Valid: {rsi_stats['both_valid']}")
    print(f"  Prod N/A, Opt Valid: {rsi_stats['prod_na_opt_valid']}")
    print(f"  Prod Valid, Opt N/A: {rsi_stats['prod_valid_opt_na']}")
    print(f"  Matching signals: {rsi_stats['matching_signals']}")
    print(f"  Mismatched signals: {rsi_stats['mismatched_signals']}")
    
    # Analyze RSI signal patterns
    print(f"\nRSI SIGNAL PATTERNS:")
    print("-" * 60)
    
    # Count RSI signal types
    prod_rsi_signals = defaultdict(int)
    opt_rsi_signals = defaultdict(int)
    
    for sig in prod_signals:
        key = f"RSI_signal={sig['rsi_signal']} (RSI={'Valid' if sig['rsi_value'] != 'N/A' else 'N/A'})"
        prod_rsi_signals[key] += 1
    
    for sig in opt_signals:
        key = f"RSI_signal={sig['rsi_signal']} (RSI={'Valid' if sig['rsi_value'] != 'N/A' else 'N/A'})"
        opt_rsi_signals[key] += 1
    
    print("Production RSI signals:")
    for key, count in sorted(prod_rsi_signals.items()):
        print(f"  {key}: {count}")
    
    print("\nOptimizer RSI signals:")
    for key, count in sorted(opt_rsi_signals.items()):
        print(f"  {key}: {count}")

def main():
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_203019.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print("RSI SIGNAL ANALYSIS")
    print("=" * 120)
    print(f"Production: {prod_file}")
    print(f"Optimizer:  {opt_file}")
    
    # Extract signals with RSI context
    prod_signals = extract_signals_with_rsi(prod_file)
    opt_signals = extract_signals_with_rsi(opt_file)
    
    # Analyze RSI patterns
    analyze_rsi_patterns(prod_signals, opt_signals)

if __name__ == "__main__":
    main()