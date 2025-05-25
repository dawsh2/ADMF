#!/usr/bin/env python3
"""
Run the system in adaptive mode like the optimization test phase
"""

import json
import subprocess
import sys

def main():
    # First, we need to run with the ensemble strategy in adaptive mode
    # The key is to use the exact same setup as the optimization test phase
    
    print("=" * 80)
    print("Running in ADAPTIVE MODE - like optimization test phase")
    print("=" * 80)
    
    # Create a special config that forces EnsembleSignalStrategy
    config_content = """system:
  name: "ADMF-Trader-MVP"
  version: "0.1.0"

logging:
  level: "DEBUG"

components:
  dummy_service:
    some_setting: "Dummy setting"

  # Force EnsembleSignalStrategy (not RegimeAdaptiveStrategy)
  strategy:
    class_path: "src.strategy.implementations.ensemble_strategy.EnsembleSignalStrategy"
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 0.6
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.4

  ensemble_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 0.6
    rsi_indicator:
      period: 14
    rsi_rule:
      oversold_threshold: 30.0
      overbought_threshold: 70.0
      weight: 0.4

  regime_adaptive_strategy:
    symbol: "SPY"
    short_window_default: 10
    long_window_default: 20
    ma_rule.weight: 0.6
    regime_detector_service_name: "MyPrimaryRegimeDetector"
    regime_params_file_path: "regime_optimized_parameters.json"
    fallback_to_overall_best: true

  data_handler_csv:
    csv_file_path: "data/SPY_1min.csv"
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

# Special flag to enable adaptive mode
ENABLE_ADAPTIVE_MODE: true
"""
    
    # Write the config
    with open('config/config_adaptive_mode.yaml', 'w') as f:
        f.write(config_content)
    
    print("\nCreated config/config_adaptive_mode.yaml with EnsembleSignalStrategy")
    print("\nNOTE: You need to modify main.py to:")
    print("1. Register EnsembleSignalStrategy as 'strategy' (not RegimeAdaptiveStrategy)")
    print("2. After setup, call strategy.enable_adaptive_mode(regime_parameters)")
    print("3. The regime_parameters should come from regime_optimized_parameters.json")
    print("\nRun with: python3 main.py --config config/config_adaptive_mode.yaml")

if __name__ == "__main__":
    main()