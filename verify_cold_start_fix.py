#!/usr/bin/env python3
"""
Verify the cold start fix by running both optimizer and production with the same configuration.
"""

import subprocess
import sys
import re

def extract_results(output, test_type):
    """Extract final value and regimes from output."""
    # Look for final portfolio value
    value_patterns = [
        r"Final portfolio value: ([\d.]+)",
        r"Final Portfolio Value: ([\d.]+)",
        r"final.*value.*?(\d+\.?\d*)",
        r"Test metric: ([\d.]+)"
    ]
    
    final_value = None
    for pattern in value_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            final_value = float(match.group(1))
            break
    
    # Look for regimes detected
    regime_pattern = r"Regimes detected: (.+?)(?:\n|$)"
    regime_match = re.search(regime_pattern, output)
    regimes = regime_match.group(1).split(", ") if regime_match else []
    
    # Also check for regime performance section
    if "PERFORMANCE BY REGIME:" in output:
        regime_perf_pattern = r"^\s*(\w+):\s*\d+\s*trades"
        regimes_from_perf = re.findall(regime_perf_pattern, output, re.MULTILINE)
        if regimes_from_perf and not regimes:
            regimes = regimes_from_perf
    
    return final_value, regimes

def run_test():
    """Run both optimizer and production to compare results."""
    print("="*80)
    print("VERIFYING COLD START FIX")
    print("="*80)
    
    # Test 1: Run optimizer with V2 and debug config
    print("\n1. Running Optimizer with EnhancedOptimizerV2...")
    print("-"*80)
    
    cmd1 = [
        sys.executable, 
        "main.py", 
        "--config", "config/config_debug_comparison.yaml",
        "--optimize-joint",
        "--log-level", "ERROR"
    ]
    
    print(f"Command: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    # Extract OOS test results
    oos_value, oos_regimes = extract_results(result1.stdout, "OOS")
    print(f"\nOOS Test Results:")
    print(f"  Final Value: ${oos_value:.2f}" if oos_value else "  Could not extract value")
    print(f"  Regimes: {', '.join(oos_regimes)}" if oos_regimes else "  No regimes found")
    
    # Test 2: Run production backtest with same config
    print("\n\n2. Running Production Backtest...")
    print("-"*80)
    
    cmd2 = [
        sys.executable,
        "run_production_backtest_v2.py",
        "--config", "config/config_debug_comparison.yaml",
        "--dataset", "test",
        "--log-level", "ERROR"
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    # Extract production results
    prod_value, prod_regimes = extract_results(result2.stdout, "Production")
    print(f"\nProduction Results:")
    print(f"  Final Value: ${prod_value:.2f}" if prod_value else "  Could not extract value")
    print(f"  Regimes: {', '.join(prod_regimes)}" if prod_regimes else "  No regimes found")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    if oos_value and prod_value:
        difference = abs(oos_value - prod_value)
        pct_diff = (difference / oos_value) * 100
        
        print(f"OOS Test:    ${oos_value:.2f}")
        print(f"Production:  ${prod_value:.2f}")
        print(f"Difference:  ${difference:.2f} ({pct_diff:.3f}%)")
        
        if pct_diff < 0.01:
            print("\n✅ SUCCESS: Results match within tolerance!")
        else:
            print(f"\n❌ Results still differ by {pct_diff:.3f}%")
            
        # Compare regimes
        if set(oos_regimes) != set(prod_regimes):
            print(f"\n⚠️  Regime mismatch:")
            print(f"   OOS regimes: {oos_regimes}")
            print(f"   Prod regimes: {prod_regimes}")
    else:
        print("❌ Could not extract values for comparison")
        if result1.stderr:
            print(f"\nOptimizer stderr:\n{result1.stderr}")
        if result2.stderr:
            print(f"\nProduction stderr:\n{result2.stderr}")

if __name__ == "__main__":
    run_test()