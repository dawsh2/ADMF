#!/usr/bin/env python3
"""Debug validation signals."""
import subprocess
import re
from datetime import datetime

# Run validation
result = subprocess.run(['python', 'run_exact_validation_v2.py'], 
                      capture_output=True, text=True)

output = result.stdout + result.stderr
lines = output.split('\n')

# Find all signals
signals = []
for line in lines:
    if "SIGNAL GENERATED" in line:
        # Extract signal details
        match = re.search(r'#(\d+):.*Type=([+-]?\d+), Price=([\d.]+)', line)
        if match:
            bar_num = int(match.group(1))
            signal_type = int(match.group(2))
            price = float(match.group(3))
            
            # Get timestamp from previous lines
            timestamp = None
            for prev_line in lines[lines.index(line)-5:lines.index(line)]:
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', prev_line)
                if ts_match:
                    timestamp = ts_match.group(1)
                    break
                    
            signals.append({
                'bar': bar_num,
                'timestamp': timestamp,
                'type': signal_type,
                'price': price
            })

# Analyze
test_start = datetime.strptime('2024-03-28 13:46:00', '%Y-%m-%d %H:%M:%S')
test_signals = [s for s in signals if s['timestamp'] and 
                datetime.strptime(s['timestamp'], '%Y-%m-%d %H:%M:%S') >= test_start]

print(f"Total signals: {len(signals)}")
print(f"Test signals (>= 2024-03-28 13:46:00): {len(test_signals)}")

if test_signals:
    print("\nTest signals:")
    for i, sig in enumerate(test_signals):
        print(f"{i+1}. Bar {sig['bar']}, {sig['timestamp']}, "
              f"Type {'BUY' if sig['type'] == 1 else 'SELL'}, ${sig['price']:.2f}")