#!/usr/bin/env python3
"""
Test script to verify the cold start fix works correctly.
Runs both optimizer OOS test and production backtest to compare results.
"""

import subprocess
import sys
import json
import re

def extract_final_value(output, pattern):
    """Extract final portfolio value from output."""
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None

def run_optimizer_oos_test():
    """Run the optimizer and extract OOS test results."""
    print("="*60)
    print("Running Optimizer with OOS Test...")
    print("="*60)
    
    cmd = [sys.executable, "main.py", "--config", "config/config_debug_comparison.yaml", "--optimize-joint"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Look for OOS test results in output
    oos_pattern = r"Final portfolio value on test set: \$?([\d,]+\.?\d*)"
    oos_value = extract_final_value(result.stdout, oos_pattern)
    
    if not oos_value:
        # Try alternative pattern
        oos_pattern = r"test.*final.*value.*\$?([\d,]+\.?\d*)"
        oos_value = extract_final_value(result.stdout, oos_pattern)
    
    print(f"OOS Test Final Value: ${oos_value:.2f}" if oos_value else "Could not extract OOS value")
    return oos_value

def run_production_backtest():
    """Run production backtest on test data."""
    print("\n" + "="*60)
    print("Running Production Backtest on Test Data...")
    print("="*60)
    
    cmd = [sys.executable, "run_production_backtest_v2.py", "-c", "config/config_debug_comparison.yaml"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Look for final value in output
    prod_pattern = r"Final Portfolio Value: \$?([\d,]+\.?\d*)"
    prod_value = extract_final_value(result.stdout, prod_pattern)
    
    if not prod_value:
        # Try alternative pattern
        prod_pattern = r"final.*value.*\$?([\d,]+\.?\d*)"
        prod_value = extract_final_value(result.stdout, prod_pattern)
    
    print(f"Production Final Value: ${prod_value:.2f}" if prod_value else "Could not extract production value")
    return prod_value

def compare_results(oos_value, prod_value):
    """Compare OOS and production results."""
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    if oos_value and prod_value:
        difference = abs(oos_value - prod_value)
        pct_diff = (difference / oos_value) * 100
        
        print(f"OOS Test Result:    ${oos_value:.2f}")
        print(f"Production Result:  ${prod_value:.2f}")
        print(f"Difference:         ${difference:.2f} ({pct_diff:.3f}%)")
        
        if pct_diff < 0.01:  # Less than 0.01% difference
            print("\n✅ SUCCESS: Results match within tolerance!")
        else:
            print(f"\n❌ MISMATCH: Results differ by {pct_diff:.3f}%")
            print("\nPossible causes:")
            print("- Random number generation differences")
            print("- Floating point precision")
            print("- Component initialization order")
    else:
        print("❌ ERROR: Could not extract values for comparison")

def main():
    """Run the test."""
    print("Testing Cold Start Fix for OOS vs Production Matching")
    print("="*60)
    
    # Run both tests
    oos_value = run_optimizer_oos_test()
    prod_value = run_production_backtest()
    
    # Compare results
    compare_results(oos_value, prod_value)

if __name__ == "__main__":
    main()