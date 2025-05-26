#!/usr/bin/env python3
"""
Simple test to verify we can run a backtest with new components.
"""
import subprocess
import sys

# Run a short backtest using the existing main.py
print("Running backtest with new component architecture...")
print("="*50)

# Use existing config but limit the run
result = subprocess.run([
    sys.executable, 
    "main.py", 
    "--config", "config/config_production.yaml",
    "--mode", "backtest",
    "--max-bars", "50"  # Limit to 50 bars for testing
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)