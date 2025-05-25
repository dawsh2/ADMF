#!/usr/bin/env python3
"""Compare signals from warmup test with optimizer."""

import re
from datetime import datetime

def extract_signals_from_log(log_file):
    """Extract signals with timestamps."""
    signals = []
    current_timestamp = None
    
    with open(log_file, 'r') as f:
        for line in f:
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
    # Get signals from both runs
    prod_signals = extract_signals_from_log('logs/warmup_test_output.log')
    opt_signals = extract_signals_from_log('logs/admf_20250523_230532.log')
    
    print("WARMUP TEST SIGNAL COMPARISON")
    print("="*60)
    print(f"Production (0.75 split): {len(prod_signals)} signals")
    print(f"Optimizer: {len(opt_signals)} signals")
    print()
    
    # Find differences
    prod_times = {s['timestamp'] for s in prod_signals}
    opt_times = {s['timestamp'] for s in opt_signals}
    
    missing_in_prod = opt_times - prod_times
    extra_in_prod = prod_times - opt_times
    
    if missing_in_prod:
        print("Missing in production:")
        for t in sorted(missing_in_prod):
            opt_sig = next(s for s in opt_signals if s['timestamp'] == t)
            print(f"  {t}: {opt_sig['direction']}")
    
    if extra_in_prod:
        print("\nExtra in production:")
        for t in sorted(extra_in_prod):
            prod_sig = next(s for s in prod_signals if s['timestamp'] == t)
            print(f"  {t}: {prod_sig['direction']}")
    
    # Check first few signals
    print("\nFirst 5 signals comparison:")
    print("-"*60)
    print("Production" + " "*30 + "Optimizer")
    print("-"*60)
    
    for i in range(min(5, len(prod_signals), len(opt_signals))):
        p = prod_signals[i] if i < len(prod_signals) else None
        o = opt_signals[i] if i < len(opt_signals) else None
        
        if p and o:
            print(f"{p['timestamp']} {p['direction']:4s} | {o['timestamp']} {o['direction']:4s}")
        elif p:
            print(f"{p['timestamp']} {p['direction']:4s} | ---")
        else:
            print(f"--- | {o['timestamp']} {o['direction']:4s}")

if __name__ == "__main__":
    main()