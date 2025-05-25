#\!/usr/bin/env python3
"""Test optimizer signal generation."""
import subprocess
import re

print("Running optimizer with MA-only mode...")
cmd = ['python', 'main.py', '--optimize', '--iterations', '1', '--optimize-ma', '--bars', '1000']
result = subprocess.run(cmd, capture_output=True, text=True)

output = result.stdout + result.stderr
print("Searching for test results...")

# Look for test metrics
for line in output.split('\n'):
    if 'test' in line.lower() and ('trade' in line or 'signal' in line):
        print(line)

# Save output
with open('optimizer_test_output.log', 'w') as f:
    f.write(output)
print("\nFull output saved to optimizer_test_output.log")
EOF < /dev/null