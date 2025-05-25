#!/usr/bin/env python3
"""
Verify MA-only signal generation by running a simple test.
"""

import subprocess
import re

print("Running MA-only optimization test...")
print("This will show actual signal counts from the optimizer")

# Run optimizer with MA-only mode
cmd = [
    'python', 'main.py',
    '--optimize',
    '--optimize-ma',    # MA-only mode
    '--iterations', '1',
    '--bars', '1000',
    '--log-level', 'INFO'
]

print(f"\nCommand: {' '.join(cmd)}")
print("Running... (this may take a minute)")

result = subprocess.run(cmd, capture_output=True, text=True)
output = result.stdout + result.stderr

# Save full output
with open('ma_only_test.log', 'w') as f:
    f.write(output)

print("\nFull output saved to ma_only_test.log")

# Extract key metrics
lines = output.split('\n')

# Look for test results
print("\n=== SEARCHING FOR TEST RESULTS ===")
for line in lines:
    if 'test' in line.lower() and any(word in line.lower() for word in ['trade', 'signal', 'value', 'metric']):
        if any(char.isdigit() for char in line):  # Has numbers
            print(line.strip())

# Count signals
signal_count = 0
test_signals = []
for line in lines:
    if 'SIGNAL GENERATED' in line:
        signal_count += 1
        if '2024-03-28' in line:  # Test date
            test_signals.append(line)

print(f"\n=== SIGNAL COUNTS ===")
print(f"Total signals found: {signal_count}")
print(f"Test date signals: {len(test_signals)}")

# Look for specific numbers
for target in [13, 17, 52]:
    for line in lines:
        if str(target) in line and any(word in line.lower() for word in ['signal', 'trade']):
            print(f"\nFound {target}: {line.strip()}")