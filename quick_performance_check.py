#!/usr/bin/env python3
"""
Quick performance check to see current results.
"""

import subprocess
import re

def check_performance():
    """Run a quick backtest and check performance."""
    
    print("Running quick performance check...")
    
    # Run a simple backtest
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--bars", "2000",
        "--log-level", "WARNING"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Extract key metrics
    total_return = re.search(r"Total Return: ([-\d.]+)%", output)
    num_trades = re.search(r"Number of Trades: (\d+)", output)
    sharpe = re.search(r"Sharpe Ratio: ([-\d.]+)", output)
    
    print("\n" + "="*60)
    print("PERFORMANCE CHECK RESULTS")
    print("="*60)
    
    if total_return:
        print(f"Total Return: {total_return.group(1)}%")
    else:
        print("Total Return: NOT FOUND")
        
    if num_trades:
        print(f"Number of Trades: {num_trades.group(1)}")
    else:
        print("Number of Trades: NOT FOUND")
        
    if sharpe:
        print(f"Sharpe Ratio: {sharpe.group(1)}")
    else:
        print("Sharpe Ratio: NOT FOUND")
    
    # Check for errors
    if "ERROR" in output or "error" in output:
        print("\n⚠️  ERRORS DETECTED IN OUTPUT")
        errors = re.findall(r"ERROR.*", output)
        for err in errors[:5]:
            print(f"  {err}")
    
    # Save full output
    with open("quick_performance_check.log", "w") as f:
        f.write(output)
    
    print(f"\nFull output saved to: quick_performance_check.log")

if __name__ == "__main__":
    check_performance()