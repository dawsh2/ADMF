#!/usr/bin/env python3
"""
Run the optimizer with debug logging enabled to trace OOS test behavior.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_optimizer_with_debug():
    """Run the optimizer with debug configuration."""
    print(f"\n{'='*60}")
    print(f"Running Optimizer with Debug Logging")
    print(f"Time: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Run the optimizer with debug config
    cmd = [
        "/usr/bin/python3", 
        "main.py", 
        "--config", "config/config_debug_comparison.yaml",
        "--optimize-joint"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
            
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
        # Extract key information from logs
        print(f"\n{'='*60}")
        print("Extracting Debug Information from Logs...")
        print(f"{'='*60}\n")
        
        # Find the latest log file
        log_dir = "logs"
        if os.path.exists(log_dir):
            log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("debug_comparison_")])
            if log_files:
                latest_log = os.path.join(log_dir, log_files[-1])
                print(f"Analyzing log file: {latest_log}\n")
                
                # Extract regime debug lines
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    
                regime_debug_lines = [line for line in lines if "[REGIME_DEBUG]" in line]
                backtest_debug_lines = [line for line in lines if "[BACKTEST_DEBUG]" in line]
                
                if regime_debug_lines:
                    print("Regime Debug Info (first 20 bars):")
                    for line in regime_debug_lines[:80]:  # Show first 20 bars (4 lines per bar)
                        print(line.strip())
                        
                if backtest_debug_lines:
                    print("\nBacktest Debug Info:")
                    for line in backtest_debug_lines[:20]:
                        print(line.strip())
                        
        return result.returncode
        
    except Exception as e:
        print(f"Error running optimizer: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_optimizer_with_debug()
    sys.exit(exit_code)