#!/usr/bin/env python3
"""
Diagnose why regime changes aren't matching between optimizer and validation.
"""

import subprocess
import re

# Run validation and capture output
print("Running validation with adaptive mode...")
result = subprocess.run(['python', 'run_exact_validation_v2.py'], 
                      capture_output=True, text=True)

output = result.stdout + result.stderr

# Extract regime information
regime_changes = []
parameter_changes = []

for line in output.split('\n'):
    # Look for regime changes
    if 'CLASSIFICATION' in line and 'New=' in line:
        regime_changes.append(line)
    
    # Look for parameter updates
    if 'Applying parameters for' in line or 'parameters updated' in line:
        parameter_changes.append(line)
        
    # Look for window changes
    if 'short_window' in line and ('5' in line or 'changed' in line):
        parameter_changes.append(line)

print("\n=== REGIME CHANGES DETECTED ===")
if regime_changes:
    for change in regime_changes[:10]:
        print(change)
else:
    print("No regime changes detected!")
    
print("\n=== PARAMETER CHANGES ===")
if parameter_changes:
    for change in parameter_changes[:10]:
        print(change)
else:
    print("No parameter changes detected!")

# Check indicators at key times
print("\n=== INDICATORS AT KEY TIMES ===")
for line in output.split('\n'):
    if '14:11:00' in line and 'INDICATORS' in line:
        print("At 14:11:00:", line)
        
# Count signals by regime
print("\n=== SIGNALS BY REGIME ===")
regimes = {}
for line in output.split('\n'):
    if 'SIGNAL GENERATED' in line:
        match = re.search(r'Regime=(\w+)', line)
        if match:
            regime = match.group(1)
            regimes[regime] = regimes.get(regime, 0) + 1

for regime, count in regimes.items():
    print(f"{regime}: {count} signals")
    
print("\n=== SUMMARY ===")
print("The validation is likely using fixed parameters throughout.")
print("Need to ensure regime detector triggers the same changes as optimizer.")