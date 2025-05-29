#!/usr/bin/env python3
"""
Quick test to verify determinism after comprehensive reset.
"""

import subprocess
import re
import time

def run_test(dataset_type):
    """Run a test and extract first signals."""
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--bars", "100",  # Smaller for faster test
        "--dataset", dataset_type,
        "--log-level", "INFO"
    ]
    
    print(f"Running {dataset_type} dataset...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Extract signals
    signals = []
    for match in re.finditer(r"Strategy 'strategy' generated signal: ([-\d]+)", output):
        signals.append(int(match.group(1)))
        if len(signals) >= 10:  # Get first 10
            break
    
    # Extract comprehensive reset
    reset_found = "COMPREHENSIVE RESET COMPLETE" in output
    
    # Extract final return
    return_match = re.search(r"Total Return: ([-\d.]+)%", output)
    total_return = float(return_match.group(1)) if return_match else None
    
    return {
        'signals': signals,
        'reset_found': reset_found,
        'total_return': total_return
    }

print("="*60)
print("DETERMINISM TEST WITH COMPREHENSIVE RESET")
print("="*60)

# Run test dataset twice
test1 = run_test("test")
print(f"\nTest Run 1:")
print(f"  Reset found: {test1['reset_found']}")
print(f"  First 10 signals: {test1['signals'][:10]}")
print(f"  Total return: {test1['total_return']}%")

time.sleep(1)  # Brief pause

test2 = run_test("test")
print(f"\nTest Run 2:")
print(f"  Reset found: {test2['reset_found']}")
print(f"  First 10 signals: {test2['signals'][:10]}")
print(f"  Total return: {test2['total_return']}%")

# Compare
print("\n" + "="*60)
print("COMPARISON:")
print("="*60)

if test1['signals'] == test2['signals']:
    print("✅ Signals are IDENTICAL between runs!")
else:
    print("❌ Signals DIFFER between runs")
    # Find first difference
    for i, (s1, s2) in enumerate(zip(test1['signals'], test2['signals'])):
        if s1 != s2:
            print(f"  First difference at signal {i+1}: {s1} vs {s2}")
            break

if test1['total_return'] == test2['total_return']:
    print("✅ Returns are IDENTICAL between runs!")
else:
    print(f"❌ Returns DIFFER: {test1['total_return']}% vs {test2['total_return']}%")