#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/Users/daws/ADMF')

print("Running quick standalone test to check indicator logging...")
cmd = [sys.executable, 'main.py', '--config', 'config/config_adaptive_production.yaml', '--bars', '1000']

result = subprocess.run(cmd, capture_output=True, text=True)

print("Return code:", result.returncode)
print("\nFirst 100 lines of output:")
print("=" * 50)
lines = result.stdout.split('\n')
for i, line in enumerate(lines[:100]):
    if '📊 BAR_' in line or '🚨 SIGNAL' in line or 'Final Portfolio' in line:
        print(f"{i:3d}: {line}")

if result.stderr:
    print("\nErrors:")
    print(result.stderr[:2000])  # First 2000 chars of errors