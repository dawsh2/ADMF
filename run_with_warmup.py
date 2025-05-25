#!/usr/bin/env python3
"""
Run production with warmup-aware strategy to match optimizer signals exactly.
"""

import sys
import subprocess
import os

# First, patch the ensemble strategy
from warmup_ensemble_strategy import patch_ensemble_strategy

def create_full_data_config():
    """Create config that processes ALL data (no split in data handler)."""
    config_content = """# Production config for full data processing with warmup
system:
  name: "ADMF-Production-Full-Data"
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
    # Set split to 1.0 to process ALL data
    train_test_split_ratio: 1.0
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
    
    config_path = "config/config_full_data.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_with_warmup():
    """Run production with warmup handling."""
    print("PRODUCTION RUN WITH WARMUP")
    print("=" * 60)
    
    # Apply the patch
    patch_ensemble_strategy()
    
    # Create config
    config_path = create_full_data_config()
    print(f"Created config: {config_path}")
    
    # Run production (it will now handle warmup internally)
    print("\nRunning production with warmup-aware strategy...")
    print("First 798 bars: warmup phase (no signals)")
    print("Last 200 bars: evaluation phase (signals generated)")
    print("-" * 40)
    
    # Import and run main directly to ensure our patch is active
    from main import main
    import argparse
    
    # Create args
    args = argparse.Namespace(
        config=config_path,
        verbose=False,
        debug=True,
        bars=None
    )
    
    # Run main
    result = main(args)
    
    if result != 0:
        print("Error in production run")
        return
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)
    
    # Find latest log
    log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
    if log_files:
        latest_log = f"logs/{log_files[0]}"
        
        # Count signals
        signal_count_cmd = f"grep -c '🚨 SIGNAL GENERATED' {latest_log}"
        result = subprocess.run(signal_count_cmd, shell=True, capture_output=True, text=True)
        print(f"Total signals generated: {result.stdout.strip()}")
        
        # Show warmup completion
        print("\nWarmup phase:")
        warmup_cmd = f"grep 'WARMUP' {latest_log} | tail -3"
        subprocess.run(warmup_cmd, shell=True)
        
        # Show first few signals
        print("\nFirst 3 signals after warmup:")
        signals_cmd = f"grep '🚨 SIGNAL GENERATED' {latest_log} | head -3"
        subprocess.run(signals_cmd, shell=True)
        
        # Extract exact timestamps
        print("\nSignal timestamps:")
        ts_cmd = f"""grep -B1 "🚨 SIGNAL GENERATED" {latest_log} | grep "📊 BAR" | grep -o "\\[20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]" | head -5"""
        subprocess.run(ts_cmd, shell=True)

if __name__ == "__main__":
    run_with_warmup()