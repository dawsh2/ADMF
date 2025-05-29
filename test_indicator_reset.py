#!/usr/bin/env python3
"""
Test that indicators properly reset to cold start.
"""

# Run optimization
print("Running optimization...")
import subprocess
opt_result = subprocess.run(
    ["python", "main_ultimate.py", "--config", "config/test_ensemble_optimization.yaml", 
     "--bars", "1000", "--optimize"],
    capture_output=True, text=True
)

# Check for indicator reset messages
opt_log = opt_result.stdout + opt_result.stderr
print("\nChecking optimization indicator resets...")
for line in opt_log.split('\n'):
    if 'RESET - clearing' in line and 'Indicator' in line:
        print(f"  {line.strip()}")

print("\n" + "="*60)

# Run standalone test
print("\nRunning standalone test...")
test_result = subprocess.run(
    ["python", "main_ultimate.py", "--config", "config/test_ensemble_optimization.yaml", 
     "--bars", "1000", "--dataset", "test"],
    capture_output=True, text=True
)

# Check for indicator reset messages
test_log = test_result.stdout + test_result.stderr
print("\nChecking test run indicator resets...")
for line in test_log.split('\n'):
    if 'RESET - clearing' in line and 'Indicator' in line:
        print(f"  {line.strip()}")

print("\n" + "="*60)
print("\nSummary:")

# Count bars cleared
opt_bars_cleared = []
test_bars_cleared = []

import re
for line in opt_log.split('\n'):
    match = re.search(r'RESET - clearing (\d+) bars', line)
    if match:
        opt_bars_cleared.append(int(match.group(1)))

for line in test_log.split('\n'):
    match = re.search(r'RESET - clearing (\d+) bars', line)
    if match:
        test_bars_cleared.append(int(match.group(1)))

print(f"Optimization: {len(opt_bars_cleared)} resets, bars cleared: {opt_bars_cleared}")
print(f"Test run: {len(test_bars_cleared)} resets, bars cleared: {test_bars_cleared}")

if all(b == 0 for b in opt_bars_cleared[-10:]) and all(b == 0 for b in test_bars_cleared):
    print("\n✅ SUCCESS: Both runs now start with true cold indicators (0 bars)!")
else:
    print("\n❌ ISSUE: Indicators still have buffered data")