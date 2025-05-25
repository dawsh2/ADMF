#!/usr/bin/env python3
"""
Controlled production run that matches optimizer behavior.

Instead of modifying core files, we'll create a custom data handler
that feeds bars in the correct sequence to match optimizer behavior.
"""

import pandas as pd
import numpy as np
import subprocess
import os
import json
from datetime import datetime

def analyze_optimizer_vs_production():
    """Analyze the exact differences."""
    print("ANALYZING OPTIMIZER VS PRODUCTION")
    print("=" * 60)
    
    # Load data to understand the split
    df = pd.read_csv('data/1000_1min.csv')
    split_idx = int(len(df) * 0.8)
    
    print(f"Data: {len(df)} total bars")
    print(f"Training: bars 0-{split_idx-1}")
    print(f"Test: bars {split_idx}-{len(df)-1}")
    print(f"Test starts at: {df.iloc[split_idx]['timestamp']}")
    
    # Key insight
    print("\nKEY INSIGHT:")
    print("- Optimizer: Processes training data (warms indicators), then test data")
    print("- Production: Only processes test data (cold indicators)")
    print("- Result: Production misses signals at 13:46 and 14:00")
    
    return split_idx

def create_offset_config(offset_bars=30):
    """
    Create a config that starts earlier to give production warmup time.
    
    The idea: If production needs ~30 bars to warm up, start it 30 bars
    earlier so indicators are ready when we reach the critical timestamps.
    """
    
    # Calculate adjusted split ratio
    # We want test to start 30 bars earlier
    total_bars = 998
    original_split = 798
    new_split = original_split - offset_bars
    new_ratio = new_split / total_bars
    
    config_content = f"""# Config with adjusted split for warmup
system:
  name: "ADMF-Production-Offset"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 1.0
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.0

  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    timestamp_column: "timestamp"
    train_test_split_ratio: {new_ratio:.3f}  # Start {offset_bars} bars earlier
    open_column: "Open"
    high_column: "High"
    low_column: "Low"
    close_column: "Close"
    volume_column: "Volume"

  basic_portfolio:
    initial_cash: 100000.00

  basic_risk_manager:
    target_trade_quantity: 100

  simulated_execution_handler:
    default_quantity: 100
    commission_per_trade: 0.005
    commission_type: "per_share"
    passthrough: false
    fill_price_logic: "signal_price"
"""
    
    config_path = f"config/config_offset_{offset_bars}.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nCreated config with {offset_bars}-bar offset")
    print(f"Original split: 798/998 = 0.800")
    print(f"New split: {new_split}/{total_bars} = {new_ratio:.3f}")
    print(f"This gives production {offset_bars} extra bars to warm up")
    
    return config_path

def find_optimal_offset():
    """
    Find the offset that makes production generate a signal at 13:46:00.
    """
    print("\nFINDING OPTIMAL OFFSET")
    print("=" * 60)
    
    target_timestamp = "2024-03-28 13:46:00"
    
    for offset in [20, 25, 30, 35, 40]:
        print(f"\nTesting offset: {offset} bars")
        
        # Create config
        config_path = create_offset_config(offset)
        
        # Run production
        cmd = ["python", "main.py", "--config", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error with offset {offset}")
            continue
        
        # Check if we get a signal at target timestamp
        log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
        if log_files:
            latest_log = f"logs/{log_files[0]}"
            
            # Look for signal at target time
            check_cmd = f"grep -A1 -B1 '{target_timestamp}' {latest_log} | grep -c 'SIGNAL GENERATED' || true"
            count = subprocess.check_output(check_cmd, shell=True, text=True).strip()
            
            if count != "0":
                print(f"SUCCESS! Got signal at {target_timestamp} with offset {offset}")
                
                # Count total signals
                total_cmd = f"grep -c '🚨 SIGNAL GENERATED' {latest_log}"
                total = subprocess.check_output(total_cmd, shell=True, text=True).strip()
                print(f"Total signals: {total}")
                
                return offset, config_path
            else:
                print(f"No signal at {target_timestamp}")
    
    return None, None

def main():
    """Main execution."""
    # First, understand the problem
    split_idx = analyze_optimizer_vs_production()
    
    # Try to find optimal offset
    optimal_offset, optimal_config = find_optimal_offset()
    
    if optimal_offset:
        print(f"\n" + "="*60)
        print(f"SOLUTION FOUND!")
        print(f"="*60)
        print(f"Use offset of {optimal_offset} bars to match optimizer signals")
        print(f"Config: {optimal_config}")
    else:
        print("\nNo simple offset solution found.")
        print("The problem requires actual warmup on training data.")

if __name__ == "__main__":
    main()