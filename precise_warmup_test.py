#!/usr/bin/env python3
"""
Precise warmup test to exactly match optimizer's indicator state.
Find the exact split ratio that gives us the same warmup as optimizer.
"""

import subprocess
import pandas as pd
import os

def test_split_ratio(split_ratio, test_name):
    """Test a specific split ratio and return signal count."""
    
    config = f"""# Precise warmup test
system:
  name: "ADMF-Warmup-Test-{test_name}"
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
    train_test_split_ratio: {split_ratio}
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
    
    config_path = f'config/config_test_{test_name}.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    # Run test
    cmd = ["python", "main.py", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        signal_count = result.stdout.count('🚨 SIGNAL GENERATED')
        
        # Extract first signal timestamp to check warmup timing
        first_signal = None
        if '🚨 SIGNAL GENERATED' in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if '🚨 SIGNAL GENERATED' in line:
                    # Look for timestamp in surrounding lines
                    break
        
        return signal_count, result.stdout
    else:
        print(f"Test {test_name} failed: {result.stderr}")
        return 0, ""

def analyze_optimizer_warmup():
    """Analyze exactly how many warmup bars the optimizer uses."""
    print("ANALYZING OPTIMIZER WARMUP STATE")
    print("="*50)
    
    # Load the dataset
    df = pd.read_csv('data/1000_1min.csv', index_col='timestamp', parse_dates=True)
    total_bars = len(df)
    
    # Optimizer uses 0.8 split
    split_idx = int(total_bars * 0.8)
    test_start_time = df.iloc[split_idx].name
    
    print(f"Total bars: {total_bars}")
    print(f"Optimizer split (0.8): {split_idx} train, {total_bars - split_idx} test")
    print(f"Test starts at: {test_start_time}")
    
    # The optimizer's first signal is at 13:46
    # Check what bar number this would be if we started earlier
    first_signal_time = pd.Timestamp('2024-03-28 13:46:00')
    
    # Find this timestamp in the data
    signal_idx = df.index.get_loc(first_signal_time)
    print(f"First signal timestamp {first_signal_time} is at index {signal_idx}")
    
    # Calculate how many extra bars we need
    extra_bars_needed = split_idx - signal_idx
    print(f"Need {extra_bars_needed} extra bars before test to include first signal")
    
    # Calculate new split ratio
    new_split_idx = signal_idx
    new_split_ratio = new_split_idx / total_bars
    
    print(f"New split ratio needed: {new_split_ratio:.4f}")
    print(f"This gives us {total_bars - new_split_idx} bars in test (vs {total_bars - split_idx} originally)")
    
    return new_split_ratio

def main():
    print("PRECISE WARMUP MATCHING")
    print("="*70)
    
    # First, analyze what split ratio we need
    optimal_ratio = analyze_optimizer_warmup()
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT SPLIT RATIOS")
    print("="*70)
    
    # Test different ratios around the calculated optimal
    test_ratios = [
        (0.70, "070"),
        (0.72, "072"), 
        (optimal_ratio, "optimal"),
        (0.75, "075"),
        (0.77, "077"),
    ]
    
    results = []
    
    for ratio, name in test_ratios:
        print(f"\nTesting split ratio {ratio:.3f} ({name})...")
        signal_count, output = test_split_ratio(ratio, name)
        results.append((ratio, name, signal_count))
        print(f"  Result: {signal_count} signals")
        
        # Check if we got the target 16 signals
        if signal_count == 16:
            print(f"  🎉 PERFECT MATCH! Split ratio {ratio:.3f} gives 16 signals!")
            
            # Save the successful output for detailed analysis
            with open(f'logs/perfect_match_{name}.log', 'w') as f:
                f.write(output)
            print(f"  Saved detailed output to logs/perfect_match_{name}.log")
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("Split Ratio | Test Name | Signals | Target")
    print("-"*50)
    
    for ratio, name, signals in results:
        status = "✓ MATCH" if signals == 16 else f"✗ {16-signals:+d}"
        print(f"{ratio:>10.3f} | {name:>8s} | {signals:>7d} | {status}")
    
    # Find best match
    best_match = min(results, key=lambda x: abs(x[2] - 16))
    print(f"\nBest match: Split ratio {best_match[0]:.3f} with {best_match[1]} signals")
    
    if best_match[2] == 16:
        print("\n🎉 SOLUTION FOUND!")
        print(f"Use split ratio {best_match[0]:.3f} to get exact match with optimizer")
    else:
        print(f"\nClosest we got: {best_match[2]} signals (off by {16 - best_match[2]})")
        print("May need to try more precise ratios or check other factors")

if __name__ == "__main__":
    main()