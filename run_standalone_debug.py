#!/usr/bin/env python3
import sys
sys.path.append('.')
import subprocess

print("Running standalone test with detailed logging...")
result = subprocess.run([sys.executable, 'main.py', '--config', 'config/config_adaptive_production.yaml', '--bars', '100000'], 
                       capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)