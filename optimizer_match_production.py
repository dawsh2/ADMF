#!/usr/bin/env python3
"""
Option 2: Make optimizer's adaptive test match production's data loading.
Creates a new data handler with ONLY test data for the adaptive test.
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path

sys.path.append('.')

def create_test_only_data():
    """
    Extract only the test portion of data and save to a new CSV.
    This ensures the optimizer sees data indexed 0-199 like production.
    """
    import pandas as pd
    
    print("Creating test-only dataset...")
    
    # Load full data
    df = pd.read_csv('data/1000_1min.csv')
    
    # Calculate split
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:].copy()
    
    # Reset index so it starts from 0
    test_df.reset_index(drop=True, inplace=True)
    
    # Save to temporary file
    test_file = 'data/test_only_200.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"Created test-only data: {test_file}")
    print(f"Bars: {len(test_df)} (indexed 0-{len(test_df)-1})")
    
    return test_file

def create_test_only_config(test_data_file):
    """
    Create a config that uses only test data.
    """
    config_content = f"""# Configuration for test-only data
system:
  name: "ADMF-Test-Only"
  version: "0.1.0"

logging:
  level: "ERROR"

components:
  # Data handler pointing to test-only file
  data_handler_csv:
    csv_file_path: "{test_data_file}"
    symbol: "SPY"
    train_test_split_ratio: 0.99  # Almost all data, avoiding the 1.0 validation error
    
  # Portfolio
  basic_portfolio:
    initial_cash: 100000.0
    
  # Risk Manager
  basic_risk_manager:
    risk_per_trade: 0.02
    max_portfolio_risk: 0.06
    max_position_count: 3
    
  # Execution Handler
  simulated_execution_handler:
    slippage_percent: 0.0
    commission_per_trade: 0.0
    
  # Regime Detector (same as original)
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
      trending_down:
        trend_10_30: {{"max": -0.02}}
      ranging_high_vol:
        trend_10_30: {{"min": -0.02, "max": 0.02}}
        atr_20: {{"min": 0.15}}
      ranging_low_vol:
        trend_10_30: {{"min": -0.02, "max": 0.02}}
        atr_20: {{"max": 0.15}}
        
  # Strategy
  ensemble_strategy:
    symbol: "SPY"
    parameters:
      ma_short_period: 20
      ma_long_period: 50
      rsi_period: 14
      bb_period: 20
      bb_std: 2.0
      rsi_oversold: 30
      rsi_overbought: 70
      volume_factor: 1.5
      ma_weight: 0.3
      rsi_weight: 0.3
      bb_weight: 0.2
      volume_weight: 0.2
      
  # Optimizer
  optimizer:
    parameter_ranges:
      ma_short_period: [10, 30]
      ma_long_period: [40, 60]
      rsi_period: [10, 20]
      bb_period: [15, 25]
      bb_std: [1.5, 2.5]
      rsi_oversold: [20, 40]
      rsi_overbought: [60, 80]
      volume_factor: [1.0, 2.0]
      ma_weight: [0.1, 0.4]
      rsi_weight: [0.1, 0.4]
      bb_weight: [0.1, 0.3]
      volume_weight: [0.1, 0.3]
    train_ratio: 1.0  # Use all test data
    metric: "sharpe_ratio"
    n_initial_samples: 5
    max_iterations: 2
    cv_splits: 2
    top_n_performers: 3
"""
    
    config_file = 'config/config_test_only.yaml'
    with open(config_file, 'w') as f:
        f.write(config_content)
        
    print(f"Created test-only config: {config_file}")
    return config_file

def run_modified_optimizer_test():
    """
    Run a special test where optimizer uses only test data.
    """
    print("\n" + "="*80)
    print("OPTION 2: OPTIMIZER MATCHING PRODUCTION'S DATA LOADING")
    print("="*80)
    print("Creating test-only dataset (200 bars indexed 0-199)")
    print("")
    
    # Create test-only data
    test_data_file = create_test_only_data()
    
    # Create test-only config
    config_file = create_test_only_config(test_data_file)
    
    # Run a simple backtest with regime adaptive strategy
    print("\nRunning backtest with test-only data...")
    
    cmd = [
        "python", "run_production_backtest_v2.py",
        "--config", config_file,
        "--strategy", "regime_adaptive",
        "--dataset", "full",  # Use "full" since our data is already test-only
        "--adaptive-params", "regime_optimized_parameters.json",
        "--log-level", "ERROR"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    print("="*80)
    print("This should match production's result of $99,870.04")
    print("="*80)
    
    # Cleanup
    print("\nCleaning up temporary files...")
    Path(test_data_file).unlink(missing_ok=True)
    Path(config_file).unlink(missing_ok=True)

if __name__ == "__main__":
    run_modified_optimizer_test()