#!/usr/bin/env python3
"""
Find the exact number of warmup bars needed to match optimizer's first signal.
"""

import pandas as pd
import subprocess

def analyze_data_structure():
    """Analyze the data structure to understand bar positioning."""
    df = pd.read_csv('data/1000_1min.csv', index_col='timestamp', parse_dates=True)
    
    print("DATA STRUCTURE ANALYSIS")
    print("="*50)
    print(f"Total bars: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Find the optimizer's test start
    split_80 = int(len(df) * 0.8)
    opt_test_start = df.iloc[split_80].name
    print(f"\nOptimizer test start (0.8 split): {opt_test_start}")
    
    # Find optimizer's first signal timestamp
    first_signal_time = pd.Timestamp('2024-03-28 13:46:00')
    try:
        signal_idx = df.index.get_loc(first_signal_time)
        print(f"First signal time {first_signal_time} is at index: {signal_idx}")
        print(f"Optimizer test start is at index: {split_80}")
        
        if signal_idx == split_80:
            print("✓ First signal is exactly at test start - optimizer has pre-warmed indicators")
        else:
            print(f"✗ Mismatch: signal at {signal_idx}, test at {split_80}")
    except KeyError:
        print(f"❌ First signal timestamp {first_signal_time} not found in data")
        return None
    
    # The key insight: we need to start our test data early enough to warm up
    # but not so early that we get different regime behavior
    
    # MA needs 20 bars to warm up (long_window=20)
    warmup_needed = 20
    ideal_test_start_idx = signal_idx - warmup_needed
    
    if ideal_test_start_idx < 0:
        print(f"❌ Not enough data for {warmup_needed} bar warmup")
        return None
    
    ideal_split_ratio = ideal_test_start_idx / len(df)
    ideal_test_start_time = df.iloc[ideal_test_start_idx].name
    
    print(f"\nPROPOSED SOLUTION:")
    print(f"Start test data at index: {ideal_test_start_idx}")
    print(f"Start test data at time: {ideal_test_start_time}")
    print(f"Required split ratio: {ideal_split_ratio:.4f}")
    print(f"This gives {warmup_needed} bars of warmup before first signal")
    
    return ideal_split_ratio

def test_exact_split(split_ratio):
    """Test the exact split ratio."""
    config = f"""# Exact warmup test
system:
  name: "ADMF-Exact-Warmup"
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
    train_test_split_ratio: {split_ratio:.6f}
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
        parameters: {{"period": 14}}
      atr_20:
        type: "atr"
        parameters: {{"period": 10}}
      trend_10_30:
        type: "simple_ma_trend"
        parameters: {{"short_period": 5, "long_period": 20}}
    regime_thresholds:
      trending_up_volatile:
        trend_10_30: {{"min": 0.02}}
        atr_20: {{"min": 0.15}}
      trending_up_low_vol:
        trend_10_30: {{"min": 0.02}}
        atr_20: {{"max": 0.15}}
      ranging_low_vol:
        trend_10_30: {{"min": -0.01, "max": 0.01}}
        atr_20: {{"max": 0.12}}
      trending_down:
        trend_10_30: {{"max": -0.01}}
      oversold_in_uptrend:
        rsi_14: {{"max": 40}}
        trend_10_30: {{"min": 0.01}}
"""
    
    config_path = 'config/config_exact_warmup.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"\nTesting split ratio {split_ratio:.6f}...")
    cmd = ["python", "main.py", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        signal_count = result.stdout.count('🚨 SIGNAL GENERATED')
        print(f"✓ Run successful: {signal_count} signals")
        
        # Save output for analysis
        with open('logs/exact_warmup_test.log', 'w') as f:
            f.write(result.stdout)
        
        return signal_count
    else:
        print(f"✗ Run failed: {result.stderr}")
        return 0

def main():
    print("FINDING EXACT WARMUP REQUIREMENTS")
    print("="*70)
    
    # Analyze what we need
    ideal_ratio = analyze_data_structure()
    
    if ideal_ratio is None:
        print("❌ Cannot determine ideal split ratio")
        return
    
    # Test the calculated ratio
    signal_count = test_exact_split(ideal_ratio)
    
    print(f"\nRESULT:")
    print(f"Split ratio {ideal_ratio:.6f} produced {signal_count} signals")
    print(f"Target: 16 signals")
    
    if signal_count == 16:
        print("🎉 SUCCESS! This should be the exact match.")
        print("\nNext step: Verify signal timestamps match exactly")
    else:
        print(f"❌ Still off by {16 - signal_count} signals")
        print("May need to adjust warmup period or check regime alignment")

if __name__ == "__main__":
    main()