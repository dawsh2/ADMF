#!/usr/bin/env python3
"""
Analyze the first two optimizer signals that production is missing.
Compare bar metadata (indicator values, regimes) at those exact timestamps.
"""

import re
from datetime import datetime

def extract_bar_data(log_file, target_timestamps):
    """Extract bar data for specific timestamps."""
    bar_data = {}
    signals = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Extract BAR indicator data
        bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([\d.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\([^)]+\), Regime=(\w+), Weights=\(MA:([\d.]+),RSI:([\d.]+)\)', line)
        if bar_match:
            timestamp_str = bar_match.group(2).split('+')[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                if timestamp in target_timestamps:
                    bar_data[timestamp] = {
                        'bar_num': int(bar_match.group(1)),
                        'price': float(bar_match.group(3)),
                        'ma_short': bar_match.group(4),
                        'ma_long': bar_match.group(5),
                        'rsi': bar_match.group(6),
                        'regime': bar_match.group(7),
                        'ma_weight': float(bar_match.group(8)),
                        'rsi_weight': float(bar_match.group(9))
                    }
            except:
                pass
        
        # Check for signals at these timestamps
        signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=(-?\d+), Price=([\d.]+)', line)
        if signal_match and i > 0:
            # Look back for the timestamp
            for j in range(i-1, max(0, i-5), -1):
                ts_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[j])
                if ts_match:
                    timestamp_str = ts_match.group(1).split('+')[0]
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        if timestamp in target_timestamps:
                            signals[timestamp] = {
                                'type': int(signal_match.group(2)),
                                'direction': 'BUY' if int(signal_match.group(2)) == 1 else 'SELL',
                                'price': float(signal_match.group(3))
                            }
                        break
                    except:
                        pass
    
    return bar_data, signals

def main():
    print("ANALYZING MISSING OPTIMIZER SIGNALS")
    print("="*70)
    
    # The two missing optimizer signals
    missing_signals = [
        datetime(2024, 3, 28, 13, 46, 0),  # First signal
        datetime(2024, 3, 28, 14, 0, 0),   # Second signal
    ]
    
    print("Missing optimizer signals:")
    for ts in missing_signals:
        print(f"  {ts}")
    
    print("\n" + "="*70)
    print("EXTRACTING BAR DATA")
    print("="*70)
    
    # Extract data from both logs
    print("Extracting from optimizer log...")
    opt_bars, opt_signals = extract_bar_data('logs/admf_20250523_230532.log', missing_signals)
    
    print("Extracting from production log (0.72 split)...")
    prod_bars, prod_signals = extract_bar_data('logs/perfect_match_072.log', missing_signals)
    
    # Analyze each missing signal
    for ts in missing_signals:
        print(f"\n{'='*70}")
        print(f"ANALYZING SIGNAL AT {ts}")
        print(f"{'='*70}")
        
        # Optimizer data
        if ts in opt_bars:
            opt_bar = opt_bars[ts]
            print(f"\nOPTIMIZER BAR DATA:")
            print(f"  Bar Number: {opt_bar['bar_num']}")
            print(f"  Price: {opt_bar['price']}")
            print(f"  MA_short: {opt_bar['ma_short']}")
            print(f"  MA_long: {opt_bar['ma_long']}")
            print(f"  RSI: {opt_bar['rsi']}")
            print(f"  Regime: {opt_bar['regime']}")
            print(f"  Weights: MA={opt_bar['ma_weight']}, RSI={opt_bar['rsi_weight']}")
            
            # Calculate MA signal
            if opt_bar['ma_short'] != 'N/A' and opt_bar['ma_long'] != 'N/A':
                ma_short_val = float(opt_bar['ma_short'])
                ma_long_val = float(opt_bar['ma_long'])
                ma_signal = "BUY" if ma_short_val > ma_long_val else "SELL"
                print(f"  MA Signal: {ma_signal} (short={ma_short_val:.4f} vs long={ma_long_val:.4f})")
            else:
                print(f"  MA Signal: N/A (indicators not ready)")
        else:
            print(f"\n❌ OPTIMIZER: No bar data found for {ts}")
        
        if ts in opt_signals:
            opt_sig = opt_signals[ts]
            print(f"\n  🚨 OPTIMIZER SIGNAL: {opt_sig['direction']} @ {opt_sig['price']}")
        else:
            print(f"\n  ❌ OPTIMIZER: No signal found for {ts}")
        
        # Production data
        if ts in prod_bars:
            prod_bar = prod_bars[ts]
            print(f"\nPRODUCTION BAR DATA:")
            print(f"  Bar Number: {prod_bar['bar_num']}")
            print(f"  Price: {prod_bar['price']}")
            print(f"  MA_short: {prod_bar['ma_short']}")
            print(f"  MA_long: {prod_bar['ma_long']}")
            print(f"  RSI: {prod_bar['rsi']}")
            print(f"  Regime: {prod_bar['regime']}")
            print(f"  Weights: MA={prod_bar['ma_weight']}, RSI={prod_bar['rsi_weight']}")
            
            # Calculate MA signal
            if prod_bar['ma_short'] != 'N/A' and prod_bar['ma_long'] != 'N/A':
                ma_short_val = float(prod_bar['ma_short'])
                ma_long_val = float(prod_bar['ma_long'])
                ma_signal = "BUY" if ma_short_val > ma_long_val else "SELL"
                print(f"  MA Signal: {ma_signal} (short={ma_short_val:.4f} vs long={ma_long_val:.4f})")
            else:
                print(f"  MA Signal: N/A (indicators not ready)")
        else:
            print(f"\n❌ PRODUCTION: No bar data found for {ts}")
        
        if ts in prod_signals:
            prod_sig = prod_signals[ts]
            print(f"\n  🚨 PRODUCTION SIGNAL: {prod_sig['direction']} @ {prod_sig['price']}")
        else:
            print(f"\n  ❌ PRODUCTION: No signal generated for {ts}")
        
        # Compare and diagnose
        print(f"\nDIAGNOSIS:")
        if ts in opt_bars and ts in prod_bars:
            opt_bar = opt_bars[ts]
            prod_bar = prod_bars[ts]
            
            # Compare indicator states
            if opt_bar['ma_short'] != 'N/A' and prod_bar['ma_short'] == 'N/A':
                print(f"  🔍 Optimizer has warmed MA indicators, production doesn't")
            elif opt_bar['ma_short'] == 'N/A' and prod_bar['ma_short'] != 'N/A':
                print(f"  🔍 Production has warmed MA indicators, optimizer doesn't")
            elif opt_bar['ma_short'] != 'N/A' and prod_bar['ma_short'] != 'N/A':
                print(f"  🔍 Both have warmed MA indicators")
                opt_short = float(opt_bar['ma_short'])
                prod_short = float(prod_bar['ma_short'])
                opt_long = float(opt_bar['ma_long'])
                prod_long = float(prod_bar['ma_long'])
                
                print(f"     Optimizer: short={opt_short:.4f}, long={opt_long:.4f}")
                print(f"     Production: short={prod_short:.4f}, long={prod_long:.4f}")
                
                if abs(opt_short - prod_short) > 0.01 or abs(opt_long - prod_long) > 0.01:
                    print(f"  ⚠️  MA values differ significantly!")
            
            # Compare regimes
            if opt_bar['regime'] != prod_bar['regime']:
                print(f"  ⚠️  Regime mismatch: opt={opt_bar['regime']}, prod={prod_bar['regime']}")
            
            # Compare weights
            if opt_bar['ma_weight'] != prod_bar['ma_weight']:
                print(f"  ⚠️  MA weight mismatch: opt={opt_bar['ma_weight']}, prod={prod_bar['ma_weight']}")
        
        elif ts not in prod_bars:
            print(f"  🔍 Production data doesn't include this timestamp")
            print(f"     This suggests the 0.72 split doesn't go back far enough")

if __name__ == "__main__":
    main()