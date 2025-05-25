#!/usr/bin/env python3
"""
Debug validation signals to understand timing.
"""

import re
import subprocess

def main():
    # Run validation and capture output
    print("Running validation...")
    result = subprocess.run(['python', 'run_exact_validation.py'], 
                          capture_output=True, text=True)
    
    lines = result.stdout.split('\n') + result.stderr.split('\n')
    
    # Track bars and signals together
    current_bar = 0
    current_timestamp = None
    signals_by_bar = {}
    bar_798_found = False
    
    for line in lines:
        # Track current bar
        bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\]', line)
        if bar_match:
            current_bar = int(bar_match.group(1))
            current_timestamp = bar_match.group(2)
            if current_bar == 798:
                bar_798_found = True
                print(f"\nFound BAR_798 at timestamp: {current_timestamp}")
            
        # Track signals
        if "SIGNAL GENERATED" in line and current_bar > 0:
            if current_bar not in signals_by_bar:
                signals_by_bar[current_bar] = []
            
            signal_match = re.search(r'Type=([+-]?\d+), Price=([\d.]+)', line)
            if signal_match:
                signals_by_bar[current_bar].append({
                    'type': int(signal_match.group(1)),
                    'price': float(signal_match.group(2)),
                    'timestamp': current_timestamp
                })
                
                # Print signals around bar 798
                if 795 <= current_bar <= 805:
                    print(f"  Signal at bar {current_bar}: Type {signal_match.group(1)}, Price {signal_match.group(2)}")
    
    # Summary
    print("\n=== SIGNAL DISTRIBUTION ===")
    warmup_signals = sum(len(sigs) for bar, sigs in signals_by_bar.items() if bar < 798)
    test_signals = sum(len(sigs) for bar, sigs in signals_by_bar.items() if bar >= 798)
    
    print(f"Warmup period signals (bar < 798): {warmup_signals}")
    print(f"Test period signals (bar >= 798): {test_signals}")
    
    # Show last warmup signals
    warmup_bars = sorted([b for b in signals_by_bar.keys() if b < 798])
    if warmup_bars:
        print(f"\nLast warmup signal at bar: {warmup_bars[-1]}")
        
    # Show first test signals
    test_bars = sorted([b for b in signals_by_bar.keys() if b >= 798])
    if test_bars:
        print(f"First test signal at bar: {test_bars[0]}")
    else:
        print("\nNo test signals found!")
        
    # Debug: Check bars around 798
    print("\n=== BARS AROUND TRANSITION ===")
    for bar in range(795, 805):
        if bar in signals_by_bar:
            print(f"Bar {bar}: {len(signals_by_bar[bar])} signals")
        
    return 0

if __name__ == "__main__":
    main()