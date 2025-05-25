#!/usr/bin/env python3
"""
Count test signals properly by looking at timestamps.
"""
import subprocess
import re
from datetime import datetime

# Run validation
print("Running validation with regime detection...")
result = subprocess.run(['python', 'run_exact_validation_v2.py'], 
                      capture_output=True, text=True)

output = result.stdout + result.stderr

# Parse all signals with timestamps
test_start = datetime.strptime('2024-03-28 13:46:00', '%Y-%m-%d %H:%M:%S')
all_signals = []
test_signals = []

lines = output.split('\n')
for i, line in enumerate(lines):
    if 'SIGNAL GENERATED' in line:
        # Extract signal info
        match = re.search(r'Type=([+-]?\d+), Price=([\d.]+), Regime=(\w+)', line)
        if match:
            signal_type = int(match.group(1))
            price = float(match.group(2))
            regime = match.group(3)
            
            # Look for timestamp in previous lines
            timestamp = None
            for j in range(max(0, i-10), i):
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[j])
                if ts_match:
                    timestamp = ts_match.group(1)
                    
            signal_info = {
                'type': signal_type,
                'price': price,
                'regime': regime,
                'timestamp': timestamp
            }
            all_signals.append(signal_info)
            
            # Check if test signal
            if timestamp:
                try:
                    signal_dt = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')
                    if signal_dt >= test_start:
                        test_signals.append(signal_info)
                except:
                    pass

print(f"\nTotal signals: {len(all_signals)}")
print(f"Test signals (>= 2024-03-28 13:46:00): {len(test_signals)}")

# Group test signals by regime
test_by_regime = {}
for sig in test_signals:
    regime = sig['regime']
    test_by_regime[regime] = test_by_regime.get(regime, 0) + 1

print("\nTest signals by regime:")
for regime, count in sorted(test_by_regime.items()):
    print(f"  {regime}: {count}")

# Show first few test signals
print("\nFirst 5 test signals:")
for i, sig in enumerate(test_signals[:5]):
    print(f"  {i+1}. {sig['timestamp']}, {sig['regime']}, "
          f"{'BUY' if sig['type'] == 1 else 'SELL'} at ${sig['price']:.2f}")
    
print(f"\nComparison:")
print(f"  Optimizer test signals: 16")
print(f"  Validation test signals: {len(test_signals)}")
print(f"  Match: {'YES!' if len(test_signals) == 16 else 'NO'}")