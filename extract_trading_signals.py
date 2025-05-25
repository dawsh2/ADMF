#!/usr/bin/env python3
"""Extract actual trading signals (not initialization) from logs."""

import re
from datetime import datetime

def extract_trading_signals(log_file):
    """Extract trading signals with timestamps during the actual backtest."""
    signals = []
    current_timestamp = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        # First capture BAR timestamps
        bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', line)
        if bar_match:
            timestamp_str = bar_match.group(1)
            try:
                if '+' in timestamp_str:
                    timestamp_str = timestamp_str.split('+')[0]
                current_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                pass
        
        # Look for signal generation lines
        signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=(-?\d+), Price=([\d.]+), Regime=(\w+)', line)
        if signal_match and current_timestamp:
            bar_num = int(signal_match.group(1))
            signal_type = int(signal_match.group(2))
            price = float(signal_match.group(3))
            regime = signal_match.group(4)
            
            # Filter out initialization signals (those from today)
            if current_timestamp.year == 2024:  # Only include backtest data from 2024
                signals.append({
                    'timestamp': current_timestamp,
                    'bar_num': bar_num,
                    'signal_type': signal_type,
                    'price': price,
                    'regime': regime
                })
    
    return signals

def main():
    # Extract signals from both logs
    prod_signals = extract_trading_signals('logs/admf_20250524_210903.log')
    opt_signals = extract_trading_signals('logs/admf_20250523_230532.log')
    
    print("TRADING SIGNALS COMPARISON")
    print("=" * 60)
    print(f"Production (RSI disabled): {len(prod_signals)} signals")
    print(f"Optimizer: {len(opt_signals)} signals")
    print()
    
    # Show production signals
    print("Production signals (RSI weight=0):")
    print("-" * 60)
    for i, sig in enumerate(prod_signals[:20]):  # Show first 20
        direction = "BUY" if sig['signal_type'] == 1 else "SELL"
        print(f"{i+1:2d}. {sig['timestamp']} | {direction:4s} @ {sig['price']:.2f} | Regime: {sig['regime']}")
    if len(prod_signals) > 20:
        print(f"... and {len(prod_signals) - 20} more signals")
    
    print()
    print("Optimizer signals:")
    print("-" * 60)
    for i, sig in enumerate(opt_signals[:20]):  # Show first 20
        direction = "BUY" if sig['signal_type'] == 1 else "SELL"
        print(f"{i+1:2d}. {sig['timestamp']} | {direction:4s} @ {sig['price']:.2f} | Regime: {sig['regime']}")
    if len(opt_signals) > 20:
        print(f"... and {len(opt_signals) - 20} more signals")
    
    # Compare signals at matching timestamps
    print()
    print("Signal matching analysis:")
    print("-" * 60)
    
    # Create timestamp lookup
    prod_by_time = {sig['timestamp']: sig for sig in prod_signals}
    opt_by_time = {sig['timestamp']: sig for sig in opt_signals}
    
    # Find matches
    matches = 0
    mismatches = 0
    for ts, prod_sig in prod_by_time.items():
        if ts in opt_by_time:
            opt_sig = opt_by_time[ts]
            if prod_sig['signal_type'] == opt_sig['signal_type']:
                matches += 1
            else:
                mismatches += 1
                print(f"MISMATCH at {ts}: Prod={prod_sig['signal_type']}, Opt={opt_sig['signal_type']}")
    
    print(f"\nMatching timestamps: {matches + mismatches}")
    print(f"  Same signal: {matches}")
    print(f"  Different signal: {mismatches}")
    print(f"  Match rate: {matches / (matches + mismatches) * 100:.1f}%" if (matches + mismatches) > 0 else "N/A")
    
    # Count prod-only and opt-only signals
    prod_only = len([ts for ts in prod_by_time if ts not in opt_by_time])
    opt_only = len([ts for ts in opt_by_time if ts not in prod_by_time])
    print(f"\nProduction-only signals: {prod_only}")
    print(f"Optimizer-only signals: {opt_only}")

if __name__ == "__main__":
    main()