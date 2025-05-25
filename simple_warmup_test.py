#!/usr/bin/env python3
"""
Simple test to run production with warmup bars included.
Uses a clever trick: adjust the train/test split to include warmup bars in "test" data.
"""

import subprocess
import os

def create_warmup_config():
    """Create config that includes warmup bars in test data."""
    config = """# Production config with warmup trick
system:
  name: "ADMF-Production-Warmup-Test"
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
    # TRICK: Use 0.75 split to include ~20 extra bars for warmup
    # Original: 0.80 split = 800 train, 200 test
    # New: 0.75 split = 750 train, 250 test (50 extra bars for warmup)
    train_test_split_ratio: 0.75
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
    
    config_path = 'config/config_warmup_test.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    return config_path

def main():
    print("TESTING PRODUCTION WITH WARMUP BARS")
    print("="*70)
    print("Strategy: Use 0.75 split instead of 0.80 to include warmup bars")
    print("This gives us ~50 extra bars at the start of 'test' data")
    print("="*70)
    
    # Create config
    config_path = create_warmup_config()
    print(f"\nCreated config: {config_path}")
    
    # Run production
    print("\nRunning production with extended test period...")
    cmd = ["python", "main.py", "--config", config_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Count signals
        signal_count = result.stdout.count('🚨 SIGNAL GENERATED')
        print(f"\n✓ Run completed successfully")
        print(f"Total signals generated: {signal_count}")
        
        # Compare with optimizer (16 signals)
        print(f"\nComparison:")
        print(f"  Optimizer: 16 signals")
        print(f"  Production: {signal_count} signals")
        print(f"  Difference: {abs(16 - signal_count)} signals")
        
        if signal_count == 16:
            print("\n🎉 PERFECT MATCH! Production now generates same number of signals as optimizer!")
        else:
            print("\nStill have a difference. May need to adjust warmup period or check other factors.")
            
        # Save output for analysis
        with open('logs/warmup_test_output.log', 'w') as f:
            f.write(result.stdout)
        print("\nFull output saved to: logs/warmup_test_output.log")
        
    else:
        print("\n✗ Run failed")
        print("Error:", result.stderr)

if __name__ == "__main__":
    main()