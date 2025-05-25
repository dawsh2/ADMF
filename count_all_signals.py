#!/usr/bin/env python3
"""Count all signals from validation run."""
import subprocess
import re

# Run validation
print("Running validation...")
result = subprocess.run(['python', 'run_exact_validation_v2.py'], 
                      capture_output=True, text=True)

output = result.stdout + result.stderr

# Count all SIGNAL GENERATED lines
all_signals = []
for line in output.split('\n'):
    if 'SIGNAL GENERATED' in line:
        # Extract bar number
        match = re.search(r'#(\d+):', line)
        if match:
            bar = int(match.group(1))
            all_signals.append(bar)

print(f"\nTotal signals found: {len(all_signals)}")
print(f"Signal bar numbers: {all_signals}")

# Count test signals (bar >= 798)
test_signals = [b for b in all_signals if b >= 798]
print(f"\nTest signals (bar >= 798): {len(test_signals)}")
print(f"Test signal bars: {test_signals}")

# Also check by searching for timestamps
test_date_signals = []
for line in output.split('\n'):
    if 'SIGNAL GENERATED' in line and '2024-03-28' in line:
        test_date_signals.append(line)
        
print(f"\nSignals on test date (2024-03-28): {len(test_date_signals)}")