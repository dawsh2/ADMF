#!/usr/bin/env python3
"""
Analyze validation signals to count test period signals.
"""

import re
import subprocess
from datetime import datetime

def main():
    # Run validation and capture output
    print("Running validation...")
    result = subprocess.run(['python', 'run_exact_validation.py'], 
                          capture_output=True, text=True)
    
    lines = result.stdout.split('\n') + result.stderr.split('\n')
    
    # Find test start timestamp
    test_start_timestamp = None
    warmup_complete_bar = None
    
    for line in lines:
        if "WARMUP COMPLETE at bar" in line:
            match = re.search(r'bar (\d+)', line)
            if match:
                warmup_complete_bar = int(match.group(1))
        elif "Starting test phase at" in line:
            match = re.search(r'at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                test_start_timestamp = match.group(1)
                
    print(f"\nWarmup complete at bar: {warmup_complete_bar}")
    print(f"Test start timestamp: {test_start_timestamp}")
    
    # Extract all signals with their bar numbers
    signals = []
    current_bar = 0
    
    for line in lines:
        # Track current bar
        bar_match = re.search(r'BAR_(\d+) \[([^\]]+)\]', line)
        if bar_match:
            current_bar = int(bar_match.group(1))
            
        # Extract signals
        signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([\d.]+)', line)
        if signal_match:
            signals.append({
                'number': int(signal_match.group(1)),
                'bar': current_bar,
                'type': int(signal_match.group(2)),
                'price': float(signal_match.group(3))
            })
    
    # Count total and test period signals
    total_signals = len(signals)
    test_signals = [s for s in signals if s['bar'] >= 798]
    
    print(f"\nTotal signals generated: {total_signals}")
    print(f"Test period signals (bar >= 798): {len(test_signals)}")
    
    # Show first and last few test signals
    if test_signals:
        print(f"\nFirst test signal: Bar {test_signals[0]['bar']}, Type {test_signals[0]['type']}, Price ${test_signals[0]['price']:.2f}")
        print(f"Last test signal: Bar {test_signals[-1]['bar']}, Type {test_signals[-1]['type']}, Price ${test_signals[-1]['price']:.2f}")
        
        print(f"\nFirst 5 test signals:")
        for i, sig in enumerate(test_signals[:5]):
            print(f"  {i+1}. Bar {sig['bar']}, Type {sig['type']}, Price ${sig['price']:.2f}")
            
    # Compare with expected optimizer results
    print("\n=== COMPARISON WITH OPTIMIZER ===")
    print("Expected optimizer test signals: 17")
    print(f"Actual validation test signals: {len(test_signals)}")
    print(f"Match: {'YES' if len(test_signals) == 17 else 'NO'}")
    
    return 0

if __name__ == "__main__":
    main()