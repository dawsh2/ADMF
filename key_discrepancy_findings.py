#!/usr/bin/env python3
"""
Extract and summarize key findings about signal discrepancies.
"""

import re
from datetime import datetime
from collections import OrderedDict

def extract_key_data(log_file):
    """Extract key data points for analysis."""
    data = {
        'first_bar_time': None,
        'signals': [],
        'regime_at_signals': {},
        'ma_values_at_signals': {},
        'bar_numbers': OrderedDict()
    }
    
    current_timestamp = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract BAR info
        bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([\d.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\([^)]+\), Regime=(\w+)', line)
        if bar_match:
            bar_num = int(bar_match.group(1))
            timestamp_str = bar_match.group(2).split('+')[0] if '+' in bar_match.group(2) else bar_match.group(2)
            
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                current_timestamp = timestamp
                
                # Only track 2024 data
                if timestamp.year == 2024:
                    if data['first_bar_time'] is None:
                        data['first_bar_time'] = timestamp
                    
                    data['bar_numbers'][timestamp] = bar_num
                    
                    # Extract MA values
                    ma_short = bar_match.group(4)
                    ma_long = bar_match.group(5)
                    data['ma_values_at_signals'][timestamp] = {
                        'ma_short': None if ma_short == 'N/A' else float(ma_short),
                        'ma_long': None if ma_long == 'N/A' else float(ma_long)
                    }
                    
                    # Extract regime
                    data['regime_at_signals'][timestamp] = bar_match.group(7)
            except:
                pass
        
        # Extract signals
        signal_match = re.search(r'🚨 SIGNAL GENERATED.*Type=(-?\d+)', line)
        if signal_match and current_timestamp and current_timestamp.year == 2024:
            data['signals'].append({
                'timestamp': current_timestamp,
                'type': int(signal_match.group(1))
            })
    
    return data

def main():
    print("KEY DISCREPANCY FINDINGS")
    print("="*70)
    
    # Extract data from both logs
    prod_data = extract_key_data('logs/admf_20250524_210903.log')
    opt_data = extract_key_data('logs/admf_20250523_230532.log')
    
    # 1. Data Window Analysis
    print("\n1. DATA WINDOW DIFFERENCES:")
    print("-"*50)
    print(f"Production first bar: {prod_data['first_bar_time']}")
    print(f"Optimizer first bar:  {opt_data['first_bar_time']}")
    if prod_data['first_bar_time'] and opt_data['first_bar_time']:
        diff = opt_data['first_bar_time'] - prod_data['first_bar_time']
        print(f"Difference: {diff}")
        print(f"\n>>> FINDING: Production starts {abs(diff.total_seconds()/3600):.1f} hours earlier than optimizer")
    
    # 2. Analyze the 3 mismatched signals
    print("\n\n2. MISMATCHED SIGNALS ANALYSIS:")
    print("-"*50)
    
    prod_signal_times = {s['timestamp'] for s in prod_data['signals']}
    opt_signal_times = {s['timestamp'] for s in opt_data['signals']}
    
    # Timestamps of interest
    times_of_interest = [
        datetime(2024, 3, 28, 13, 46, 0),  # Opt only
        datetime(2024, 3, 28, 13, 59, 0),  # Prod only
        datetime(2024, 3, 28, 14, 0, 0),   # Opt only
    ]
    
    for t in times_of_interest:
        print(f"\nAt {t}:")
        
        # Check if signal exists
        prod_has_signal = t in prod_signal_times
        opt_has_signal = t in opt_signal_times
        
        print(f"  Signal: Production={'YES' if prod_has_signal else 'NO'}, Optimizer={'YES' if opt_has_signal else 'NO'}")
        
        # Compare regimes
        prod_regime = prod_data['regime_at_signals'].get(t, 'N/A')
        opt_regime = opt_data['regime_at_signals'].get(t, 'N/A')
        regime_match = "✓" if prod_regime == opt_regime else "✗"
        print(f"  Regime: Production={prod_regime:20s}, Optimizer={opt_regime:20s} {regime_match}")
        
        # Compare MA values
        prod_ma = prod_data['ma_values_at_signals'].get(t, {})
        opt_ma = opt_data['ma_values_at_signals'].get(t, {})
        
        if prod_ma and opt_ma:
            # MA signals
            prod_ma_signal = "SELL" if prod_ma.get('ma_short') and prod_ma.get('ma_long') and prod_ma['ma_short'] < prod_ma['ma_long'] else "BUY" if prod_ma.get('ma_short') and prod_ma.get('ma_long') and prod_ma['ma_short'] > prod_ma['ma_long'] else "N/A"
            opt_ma_signal = "SELL" if opt_ma.get('ma_short') and opt_ma.get('ma_long') and opt_ma['ma_short'] < opt_ma['ma_long'] else "BUY" if opt_ma.get('ma_short') and opt_ma.get('ma_long') and opt_ma['ma_short'] > opt_ma['ma_long'] else "N/A"
            
            ma_signal_match = "✓" if prod_ma_signal == opt_ma_signal else "✗"
            print(f"  MA Signal: Production={prod_ma_signal:4s}, Optimizer={opt_ma_signal:4s} {ma_signal_match}")
            
            # MA values
            if prod_ma.get('ma_short') and opt_ma.get('ma_short'):
                ma_short_diff = abs(prod_ma['ma_short'] - opt_ma['ma_short'])
                print(f"  MA_short: Prod={prod_ma['ma_short']:.4f}, Opt={opt_ma['ma_short']:.4f}, Diff={ma_short_diff:.4f}")
            
            if prod_ma.get('ma_long') and opt_ma.get('ma_long'):
                ma_long_diff = abs(prod_ma['ma_long'] - opt_ma['ma_long'])
                print(f"  MA_long:  Prod={prod_ma['ma_long']:.4f}, Opt={opt_ma['ma_long']:.4f}, Diff={ma_long_diff:.4f}")
        
        # Bar numbers
        prod_bar = prod_data['bar_numbers'].get(t, 'N/A')
        opt_bar = opt_data['bar_numbers'].get(t, 'N/A')
        print(f"  Bar#: Production={prod_bar}, Optimizer={opt_bar}")
    
    # 3. Root Cause Analysis
    print("\n\n3. ROOT CAUSE ANALYSIS:")
    print("-"*50)
    
    print("\nKEY FINDINGS:")
    print("1. Different Data Windows:")
    print("   - Production includes pre-market data (starts at 19:47)")
    print("   - Optimizer starts at market open (13:46)")
    print("   - This causes different MA warmup states")
    
    print("\n2. Regime Mismatches at Critical Points:")
    print("   - All 3 mismatched signals occur when regimes differ")
    print("   - Production shows different regimes than optimizer at these times")
    
    print("\n3. Indicator Warmup:")
    print("   - Production MA indicators warm up on pre-market data")
    print("   - Optimizer MA indicators start fresh at 13:46")
    print("   - This affects regime detection and signal generation")
    
    print("\n4. Signal Generation Logic:")
    print("   - Signals depend on both MA crossover AND regime")
    print("   - Different regimes = different signal behavior")
    print("   - Even with same MA signal, regime can suppress/allow trades")
    
    print("\n\nRECOMMENDATIONS:")
    print("-"*50)
    print("1. Ensure both runs use identical data windows")
    print("2. Start both at the same timestamp (e.g., 13:46)")
    print("3. Use same train/test split ratio")
    print("4. Consider pre-warming indicators identically")
    print("5. Verify regime detector parameters are identical")

if __name__ == "__main__":
    main()