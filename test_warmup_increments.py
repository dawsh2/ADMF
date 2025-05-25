#!/usr/bin/env python3
"""
Test different warmup bar counts to find optimal match
"""
import subprocess
import time
import re

def test_split_ratio(split_ratio, warmup_bars):
    """Test a specific train/test split ratio"""
    print(f"\n{'='*60}")
    print(f"Testing split ratio {split_ratio} (~{warmup_bars} warmup bars)")
    print(f"{'='*60}")
    
    # Create config with this split ratio
    config_content = f"""# Adaptive Production configuration with {split_ratio} split
system:
  name: "ADMF-Adaptive-Production-Test"
  version: "0.1.0"

logging:
  level: "INFO"

components:
  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 0.2
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.8

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
    
    # Write config file
    config_path = f"config/config_test_split_{split_ratio}.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Run the test
    cmd = ["python", "main.py", "--config", config_path]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    duration = time.time() - start_time
    print(f"Completed in {duration:.1f} seconds")
    
    # Find the log file
    log_files = subprocess.run(["ls", "-t", "logs/admf_*.log"], 
                              capture_output=True, text=True, shell=True)
    latest_log = log_files.stdout.strip().split('\n')[0] if log_files.stdout else None
    
    if not latest_log:
        print("ERROR: No log file found")
        return None
    
    # Count signals
    signal_count = subprocess.run(["grep", "-c", "SIGNAL GENERATED", latest_log], 
                                 capture_output=True, text=True)
    signals = int(signal_count.stdout.strip()) if signal_count.stdout else 0
    
    # Check first bar timestamp
    first_bar_cmd = subprocess.run(["grep", "-m1", "📊 BAR_", latest_log], 
                                  capture_output=True, text=True)
    first_bar_match = re.search(r'\[([^\]]+)\]', first_bar_cmd.stdout)
    first_bar_time = first_bar_match.group(1) if first_bar_match else "Unknown"
    
    print(f"Results:")
    print(f"  Signal count: {signals}")
    print(f"  First bar: {first_bar_time}")
    print(f"  Log file: {latest_log}")
    
    return {
        'split_ratio': split_ratio,
        'warmup_bars': warmup_bars,
        'signals': signals,
        'first_bar': first_bar_time,
        'log_file': latest_log
    }

def main():
    print("Testing different train/test split ratios to find optimal warmup")
    print("Target: 16 signals (matching optimizer)")
    
    # Test different split ratios
    # 0.80 = 200 test bars (no warmup)
    # 0.79 = 210 test bars (~10 warmup)
    # 0.78 = 220 test bars (~20 warmup)
    # 0.77 = 230 test bars (~30 warmup)
    
    test_configs = [
        (0.80, 0),    # Original - no warmup
        (0.795, 5),   # 5 warmup bars
        (0.79, 10),   # 10 warmup bars
        (0.785, 15),  # 15 warmup bars
        (0.78, 20),   # 20 warmup bars
        (0.775, 25),  # 25 warmup bars
        (0.77, 30),   # 30 warmup bars (current)
    ]
    
    results = []
    
    for split_ratio, warmup_bars in test_configs:
        result = test_split_ratio(split_ratio, warmup_bars)
        if result:
            results.append(result)
        time.sleep(2)  # Brief pause between runs
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Signal Count vs Warmup Bars")
    print("="*60)
    print(f"{'Split':>6} | {'Warmup':>7} | {'Signals':>7} | {'First Bar'}")
    print("-"*60)
    
    for r in results:
        print(f"{r['split_ratio']:>6.3f} | {r['warmup_bars']:>7} | {r['signals']:>7} | {r['first_bar'][:19]}")
    
    print(f"\nOptimizer baseline: 16 signals starting at 2024-03-28 13:46:00")

if __name__ == "__main__":
    main()