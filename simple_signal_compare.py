#!/usr/bin/env python3
"""Simple signal comparison"""

import subprocess

# Extract first 20 signals from production
print("PRODUCTION SIGNALS (first 20):")
print("-" * 60)
prod_cmd = "grep 'Publishing event: Event(type=SIGNAL' /Users/daws/ADMF/logs/admf_20250522_225839.log | head -20 | grep -o \"Timestamp('[^']*').*signal_type': [^,]*.*Regime: [^']*'\" | cut -d'(' -f2"
prod_result = subprocess.run(prod_cmd, shell=True, capture_output=True, text=True)
print(prod_result.stdout)

# Extract first 20 signals from optimization  
print("\nOPTIMIZATION SIGNALS (first 20 from April 24):")
print("-" * 60)
opt_cmd = "grep 'Publishing event: Event(type=SIGNAL' /Users/daws/ADMF/logs/admf_20250522_225752.log | grep '2024-04-24' | head -20 | grep -o \"Timestamp('[^']*').*signal_type': [^,]*.*Regime: [^']*'\" | cut -d'(' -f2"
opt_result = subprocess.run(opt_cmd, shell=True, capture_output=True, text=True)
print(opt_result.stdout)

# Just count total signals
print("\nSIGNAL COUNTS:")
prod_count_cmd = "grep -c 'Publishing event: Event(type=SIGNAL' /Users/daws/ADMF/logs/admf_20250522_225839.log"
prod_count = subprocess.run(prod_count_cmd, shell=True, capture_output=True, text=True)
print(f"Production total signals: {prod_count.stdout.strip()}")

opt_count_cmd = "grep 'Publishing event: Event(type=SIGNAL' /Users/daws/ADMF/logs/admf_20250522_225752.log | grep -c '2024-04-24'"
opt_count = subprocess.run(opt_count_cmd, shell=True, capture_output=True, text=True)
print(f"Optimization signals on April 24: {opt_count.stdout.strip()}")