#!/usr/bin/env python3
"""
Test the V3 optimizer to verify it produces consistent results.
"""

import subprocess
import sys

def test_v3_optimizer():
    """Test that V3 optimizer produces consistent results with production."""
    
    print("\n" + "="*80)
    print("TESTING ENHANCED OPTIMIZER V3 - CLEAN STATE ISOLATION")
    print("="*80)
    print("")
    print("This test verifies that:")
    print("1. Each optimization run starts with clean state")
    print("2. The adaptive test matches production results")
    print("3. No state leaks between runs")
    print("")
    
    # Run optimizer with V3
    print("Running optimizer with V3 (clean state for every backtest)...")
    print("-"*80)
    
    cmd = [
        sys.executable, "main.py",
        "--config", "config/config.yaml",
        "--optimize-joint",
        "--log-level", "ERROR"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract results - look for adaptive test result
    optimizer_value = None
    lines = result.stdout.split('\n')
    for line in lines:
        # Look for the adaptive test result line
        if "Adaptive GA Ensemble Strategy Test final_portfolio_value:" in line:
            try:
                value_str = line.split(':')[1].strip()
                optimizer_value = float(value_str)
                break
            except:
                pass
    
    print(result.stdout)
    
    # Run production test
    print("\n" + "-"*80)
    print("Running production test for comparison...")
    print("-"*80)
    
    prod_cmd = [
        sys.executable, "run_production_backtest_v2.py",
        "--config", "config/config.yaml",
        "--strategy", "regime_adaptive",
        "--dataset", "test",
        "--adaptive-params", "regime_optimized_parameters.json",
        "--log-level", "ERROR"
    ]
    
    prod_result = subprocess.run(prod_cmd, capture_output=True, text=True)
    
    # Extract production value
    prod_value = None
    if "Final Portfolio Value:" in prod_result.stdout:
        lines = prod_result.stdout.split('\n')
        for line in lines:
            if "Final Portfolio Value:" in line:
                try:
                    value_str = line.split(':')[1].strip()
                    prod_value = float(value_str)
                except:
                    pass
    
    print(prod_result.stdout)
    
    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    if optimizer_value and prod_value:
        print(f"Optimizer V3 adaptive test: ${optimizer_value:.2f}")
        print(f"Production test: ${prod_value:.2f}")
        
        diff = abs(optimizer_value - prod_value)
        pct_diff = (diff / optimizer_value) * 100 if optimizer_value else 0
        
        print(f"Difference: ${diff:.2f} ({pct_diff:.3f}%)")
        
        if pct_diff < 0.01:
            print("\n✅ SUCCESS! Results match within tolerance.")
            print("V3 successfully provides clean state isolation.")
        else:
            print(f"\n⚠️  Results differ by {pct_diff:.3f}%")
            print("There may still be some state leakage.")
    else:
        print("❌ Could not extract values for comparison")
        
    print("="*80)

if __name__ == "__main__":
    test_v3_optimizer()