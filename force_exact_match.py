#!/usr/bin/env python3
"""
Force production to exactly match optimizer behavior.
Key: Disable adaptive mode and ensure identical parameters.
"""

import os
import shutil
import subprocess
import json

def disable_adaptive_mode():
    """Temporarily disable adaptive mode by renaming the parameters file."""
    params_file = "regime_optimized_parameters.json"
    backup_file = params_file + ".backup"
    
    if os.path.exists(params_file):
        shutil.move(params_file, backup_file)
        print(f"Moved {params_file} to {backup_file} to disable adaptive mode")
        return backup_file
    return None

def restore_adaptive_params(backup_file):
    """Restore the adaptive parameters file."""
    if backup_file and os.path.exists(backup_file):
        shutil.move(backup_file, "regime_optimized_parameters.json")
        print("Restored adaptive parameters file")

def create_exact_match_config():
    """Create config that exactly matches optimizer settings."""
    config = """system:
  name: "ADMF-Exact-Match"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    # Initial weights - optimizer will adjust these
    ma_rule.weight: 0.6
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.4

  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    timestamp_column: "timestamp"
    train_test_split_ratio: 0.8
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

  # Include regime detector but it should stay in default mode
  MyPrimaryRegimeDetector:
    min_regime_duration: 2
    verbose_logging: false
    summary_interval: 50
    indicators:
      rsi_14:
        type: "rsi"
        parameters: {"period": 14}
      atr_20:
        type: "atr"
        parameters: {"period": 10}
      trend_10_30:
        type: "simple_ma_trend"
        parameters: {"short_period": 5, "long_period": 20}
    regime_thresholds:
      trending_up_volatile:
        trend_10_30: {"min": 0.02}
        atr_20: {"min": 0.15}
      trending_up_low_vol:
        trend_10_30: {"min": 0.02}
        atr_20: {"max": 0.15}
      ranging_low_vol:
        trend_10_30: {"min": -0.01, "max": 0.01}
        atr_20: {"max": 0.12}
      trending_down:
        trend_10_30: {"max": -0.01}
      oversold_in_uptrend:
        rsi_14: {"max": 40}
        trend_10_30: {"min": 0.01}
"""
    
    config_path = "config/config_exact_match.yaml"
    with open(config_path, 'w') as f:
        f.write(config)
    
    return config_path

def check_weight_adjustment():
    """Check if production adjusts weights like optimizer."""
    print("\nCHECKING WEIGHT ADJUSTMENT BEHAVIOR")
    print("=" * 60)
    
    print("Optimizer behavior:")
    print("1. Starts with MA=0.6, RSI=0.4")
    print("2. Detects MA optimization mode")
    print("3. Adjusts to MA=0.8, RSI=0.2") 
    print("4. For some reason, uses MA=1.0, RSI=0.0 in actual signals")
    
    print("\nProduction should do the same IF:")
    print("- It's not in adaptive mode")
    print("- It detects optimization context")

def run_exact_match_test():
    """Run production with exact optimizer settings."""
    print("EXACT MATCH TEST")
    print("=" * 60)
    
    # Disable adaptive mode
    backup = disable_adaptive_mode()
    
    try:
        # Create config
        config_path = create_exact_match_config()
        print(f"Created config: {config_path}")
        
        # Run production
        print("\nRunning production with adaptive mode disabled...")
        result = subprocess.run(
            ["python", "main.py", "--config", config_path, "--bars", "1000"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return
        
        # Count signals
        import glob
        logs = sorted(glob.glob("logs/admf*.log"), key=os.path.getmtime, reverse=True)
        if logs:
            latest_log = logs[0]
            
            # Count signals
            signal_count_cmd = f"grep -c 'SIGNAL GENERATED' {latest_log}"
            count_result = subprocess.run(signal_count_cmd, shell=True, capture_output=True, text=True)
            signal_count = int(count_result.stdout.strip() or 0)
            
            print(f"\nProduction signals: {signal_count}")
            
            # Check weights
            print("\nWeight check:")
            weight_cmd = f"grep 'EnsembleSignalStrategy weights' {latest_log}"
            subprocess.run(weight_cmd, shell=True)
            
            # Check adaptive mode
            print("\nAdaptive mode check:")
            adaptive_cmd = f"grep -i 'adaptive.*mode' {latest_log} | head -3"
            subprocess.run(adaptive_cmd, shell=True)
            
            # Check first signal
            print("\nFirst signal:")
            signal_cmd = f"grep -m1 -B2 'SIGNAL GENERATED' {latest_log}"
            subprocess.run(signal_cmd, shell=True)
            
            # Check if weights were adjusted
            print("\nWeight adjustment check:")
            adjust_cmd = f"grep 'adjusting weights\\|Detected.*optimization' {latest_log}"
            subprocess.run(adjust_cmd, shell=True)
            
    finally:
        # Restore adaptive params
        restore_adaptive_params(backup)

def analyze_remaining_differences():
    """Analyze why there are still differences."""
    print("\n\nREMAINING DIFFERENCES ANALYSIS")
    print("=" * 60)
    
    print("Even with identical configs, differences remain because:")
    print("1. WARMUP: Optimizer processes training data first (bars 0-797)")
    print("2. Production starts directly on test data (bars 798-997)")
    print("3. This means indicators have different states at test start")
    print("4. First signals will differ due to MA indicator warmup")
    
    print("\nTo achieve EXACT match, production must:")
    print("- Process the same training bars as optimizer")
    print("- Only count signals from test period")
    print("- Have identical indicator states at bar 798")

def main():
    """Run the exact match test."""
    # Check weight adjustment
    check_weight_adjustment()
    
    # Run test
    run_exact_match_test()
    
    # Analyze differences
    analyze_remaining_differences()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The fundamental issue is the warmup difference.")
    print("Optimizer: Processes bars 0-797 (train), then 798-997 (test)")
    print("Production: Only processes bars 798-997 (test)")
    print("\nThis causes different indicator states and different signals.")

if __name__ == "__main__":
    main()