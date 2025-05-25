#!/usr/bin/env python3
"""
Direct comparison of optimizer adaptive test vs production run.
"""

import subprocess
import re
import os
from datetime import datetime

def run_optimizer_v3():
    """Run EnhancedOptimizerV3 and extract adaptive test results."""
    print("Running EnhancedOptimizerV3...")
    
    # Run the optimizer
    cmd = [
        "/usr/bin/python3", "main.py",
        "-c", "config/config_optimization_exact.yaml",
        "-m", "optimization"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract final equity from output
    final_equity_match = re.search(r'Final Portfolio Value: \$([0-9,]+\.\d+)', result.stdout)
    if final_equity_match:
        equity = final_equity_match.group(1).replace(',', '')
        print(f"Optimizer Final Equity: ${equity}")
        return float(equity)
    
    return None

def run_production():
    """Run production with adaptive parameters."""
    print("\nRunning Production...")
    
    # Run production
    cmd = [
        "/usr/bin/python3", "run_production_adaptive.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract final equity
    final_equity_match = re.search(r'Final Portfolio Value: \$([0-9,]+\.\d+)', result.stdout)
    if final_equity_match:
        equity = final_equity_match.group(1).replace(',', '')
        print(f"Production Final Equity: ${equity}")
        return float(equity)
    
    return None

def main():
    print("="*80)
    print("DIRECT COMPARISON: OPTIMIZER V3 vs PRODUCTION")
    print("="*80)
    
    # Run both
    opt_equity = run_optimizer_v3()
    prod_equity = run_production()
    
    if opt_equity and prod_equity:
        diff = abs(opt_equity - prod_equity)
        pct_diff = (diff / opt_equity) * 100
        
        print("\n" + "="*80)
        print("RESULTS:")
        print("="*80)
        print(f"Optimizer:  ${opt_equity:,.2f}")
        print(f"Production: ${prod_equity:,.2f}")
        print(f"Difference: ${diff:,.2f} ({pct_diff:.3f}%)")
        
        if pct_diff < 0.01:
            print("\n✅ EXCELLENT! Difference is less than 0.01%")
        elif pct_diff < 0.05:
            print("\n⚠️  ACCEPTABLE: Difference is less than 0.05%")
        else:
            print("\n❌ UNACCEPTABLE: Difference exceeds 0.05%")

if __name__ == "__main__":
    main()