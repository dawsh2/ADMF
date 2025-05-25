#!/usr/bin/env python3
"""
Verify that the 0.720 split not only gives 16 signals but they're at the same timestamps.
"""

import re
from datetime import datetime

def extract_signals_from_log(log_file):
    """Extract signals with timestamps."""
    signals = []
    current_timestamp = None
    
    try:
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
    except FileNotFoundError:
        print(f"File not found: {log_file}")
        return []
    
    return signals

def main():
    print("VERIFYING PERFECT MATCH - 0.720 SPLIT")
    print("="*70)
    
    # Extract signals from both logs
    prod_signals = extract_signals_from_log('logs/perfect_match_072.log')
    opt_signals = extract_signals_from_log('logs/admf_20250523_230532.log')
    
    print(f"Production (0.720 split): {len(prod_signals)} signals")
    print(f"Optimizer: {len(opt_signals)} signals")
    
    if len(prod_signals) != len(opt_signals):
        print(f"❌ Signal count mismatch: {len(prod_signals)} vs {len(opt_signals)}")
        return
    
    print("\nDetailed signal comparison:")
    print("-"*70)
    print("Timestamp            | Production | Optimizer  | Match")
    print("-"*70)
    
    perfect_match = True
    for i in range(len(prod_signals)):
        p_sig = prod_signals[i]
        o_sig = opt_signals[i]
        
        timestamp_match = p_sig['timestamp'] == o_sig['timestamp']
        direction_match = p_sig['direction'] == o_sig['direction']
        overall_match = timestamp_match and direction_match
        
        if not overall_match:
            perfect_match = False
        
        match_symbol = "✓" if overall_match else "✗"
        
        print(f"{p_sig['timestamp']} | {p_sig['direction']:^10s} | {o_sig['direction']:^10s} | {match_symbol:^5s}")
        
        if not timestamp_match:
            print(f"   ⚠️  Timestamp mismatch: prod={p_sig['timestamp']}, opt={o_sig['timestamp']}")
        if not direction_match:
            print(f"   ⚠️  Direction mismatch: prod={p_sig['direction']}, opt={o_sig['direction']}")
    
    print("-"*70)
    
    if perfect_match:
        print("🎉 PERFECT MATCH! All 16 signals occur at identical timestamps with identical directions!")
        print("\n✅ SOLUTION CONFIRMED:")
        print(f"   Use train_test_split_ratio: 0.720 in production config")
        print(f"   This gives production the exact same warmup as optimizer")
        print(f"   Result: Identical signal generation")
    else:
        print("❌ Signals don't match perfectly. Investigating differences...")
        
        # Show mismatches
        prod_times = {s['timestamp'] for s in prod_signals}
        opt_times = {s['timestamp'] for s in opt_signals}
        
        missing_in_prod = opt_times - prod_times
        extra_in_prod = prod_times - opt_times
        
        if missing_in_prod:
            print(f"\nMissing in production ({len(missing_in_prod)}):")
            for t in sorted(missing_in_prod):
                print(f"  {t}")
        
        if extra_in_prod:
            print(f"\nExtra in production ({len(extra_in_prod)}):")
            for t in sorted(extra_in_prod):
                print(f"  {t}")

if __name__ == "__main__":
    main()