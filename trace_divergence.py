#!/usr/bin/env python3
"""
Trace the exact divergence between optimizer and production runs.
"""

import sys
import subprocess
import re
from datetime import datetime

sys.path.append('.')

def extract_trade_details(log_file):
    """Extract detailed trade information from log file."""
    trades = []
    
    # Read log file
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except:
        return trades
    
    # Look for trade execution lines
    for i, line in enumerate(lines):
        if "ORDER FILLED" in line or "Trade executed" in line:
            # Extract timestamp, price, quantity, etc.
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            price_match = re.search(r'price[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
            qty_match = re.search(r'quantity[:\s]+(\d+)', line, re.IGNORECASE)
            
            trade = {
                'line': i + 1,
                'timestamp': timestamp_match.group(1) if timestamp_match else 'Unknown',
                'price': float(price_match.group(1)) if price_match else 0.0,
                'quantity': int(qty_match.group(1)) if qty_match else 0,
                'full_line': line.strip()
            }
            trades.append(trade)
    
    return trades

def compare_regime_detection(log_file):
    """Extract regime detection timing from log file."""
    regimes = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except:
        return regimes
    
    for i, line in enumerate(lines):
        if "Regime classification:" in line or "REGIME_CHANGE" in line:
            # Extract regime info
            regime_match = re.search(r'regime=(\w+)', line)
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            
            if regime_match:
                regime = {
                    'line': i + 1,
                    'timestamp': timestamp_match.group(1) if timestamp_match else 'Unknown',
                    'regime': regime_match.group(1),
                    'full_line': line.strip()
                }
                regimes.append(regime)
    
    return regimes

def run_with_detailed_logging():
    """Run both optimizer and production with detailed logging to trace divergence."""
    
    print("="*80)
    print("TRACING DIVERGENCE BETWEEN OPTIMIZER AND PRODUCTION")
    print("="*80)
    
    # Create config with DEBUG logging
    debug_config = """
system:
  name: "ADMF-Debug-Trace"
  version: "0.1.0"

logging:
  level: "DEBUG"  # Enable detailed logging

components:
  # Data handler
  data_handler_csv:
    csv_file_path: "data/1000_1min.csv"
    symbol: "SPY"
    train_test_split_ratio: 0.8
    
  # Portfolio
  basic_portfolio:
    initial_cash: 100000.0
    
  # Risk manager
  basic_risk_manager:
    risk_per_trade: 0.02
    max_portfolio_risk: 0.06
    max_position_count: 3
    
  # Execution handler
  simulated_execution_handler:
    slippage_percent: 0.0
    commission_per_trade: 0.0
    
  # Regime detector with debug
  MyPrimaryRegimeDetector:
    min_regime_duration: 2
    verbose_logging: true  # Enable verbose regime logging
    debug_mode: true       # Enable debug mode
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
      trending_down:
        trend_10_30: {"max": -0.02}
      ranging_high_vol:
        trend_10_30: {"min": -0.02, "max": 0.02}
        atr_20: {"min": 0.15}
      ranging_low_vol:
        trend_10_30: {"min": -0.02, "max": 0.02}
        atr_20: {"max": 0.15}
        
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
      ma_short_period: [10, 10]  # Fixed for testing
      ma_long_period: [20, 20]
      rsi_period: [21, 21]
      rsi_oversold: [30, 30]
      rsi_overbought: [70, 70]
    train_ratio: 0.8
    metric: "sharpe_ratio"
    n_initial_samples: 1  # Just one test
    max_iterations: 1
"""
    
    # Save debug config
    with open('config/config_debug_trace.yaml', 'w') as f:
        f.write(debug_config)
    
    # Run optimizer adaptive test only
    print("\n1. Running Optimizer Adaptive Test with Debug Logging...")
    print("-"*80)
    
    opt_cmd = [
        "python3", "run_adaptive_test_only.py"
    ]
    
    opt_result = subprocess.run(opt_cmd, capture_output=True, text=True)
    
    # Get latest log file
    import os
    log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
    if log_files:
        opt_log = f"logs/{log_files[0]}"
        print(f"Optimizer log: {opt_log}")
        
        # Extract trade info
        opt_trades = extract_trade_details(opt_log)
        opt_regimes = compare_regime_detection(opt_log)
        
        print(f"\nOptimizer Trades: {len(opt_trades)}")
        for t in opt_trades[:5]:  # Show first 5
            print(f"  {t['timestamp']}: {t['full_line'][:80]}...")
            
        print(f"\nOptimizer Regime Changes: {len(opt_regimes)}")
        for r in opt_regimes[:5]:  # Show first 5
            print(f"  {r['timestamp']}: {r['regime']}")
    
    # Run production
    print("\n\n2. Running Production with Debug Logging...")
    print("-"*80)
    
    prod_cmd = [
        "python3", "run_production_backtest_v2.py",
        "--config", "config/config_debug_trace.yaml",
        "--strategy", "regime_adaptive",
        "--dataset", "test",
        "--adaptive-params", "regime_optimized_parameters.json"
    ]
    
    prod_result = subprocess.run(prod_cmd, capture_output=True, text=True)
    
    # Get latest log file
    log_files = sorted([f for f in os.listdir('logs') if f.endswith('.log')], reverse=True)
    if log_files:
        prod_log = f"logs/{log_files[0]}"
        print(f"Production log: {prod_log}")
        
        # Extract trade info
        prod_trades = extract_trade_details(prod_log)
        prod_regimes = compare_regime_detection(prod_log)
        
        print(f"\nProduction Trades: {len(prod_trades)}")
        for t in prod_trades[:5]:  # Show first 5
            print(f"  {t['timestamp']}: {t['full_line'][:80]}...")
            
        print(f"\nProduction Regime Changes: {len(prod_regimes)}")
        for r in prod_regimes[:5]:  # Show first 5
            print(f"  {r['timestamp']}: {r['regime']}")
    
    # Compare
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS")
    print("="*80)
    
    print(f"\nTrade Count Difference: {len(prod_trades)} - {len(opt_trades)} = {len(prod_trades) - len(opt_trades)}")
    print(f"Regime Change Difference: {len(prod_regimes)} - {len(opt_regimes)} = {len(prod_regimes) - len(opt_regimes)}")
    
    print("\nTO INVESTIGATE FURTHER:")
    print(f"1. Compare logs side by side:")
    print(f"   diff {opt_log} {prod_log}")
    print(f"\n2. Search for first SIGNAL divergence:")
    print(f"   grep -n 'SIGNAL' {opt_log} > opt_signals.txt")
    print(f"   grep -n 'SIGNAL' {prod_log} > prod_signals.txt")
    print(f"   diff opt_signals.txt prod_signals.txt")
    print(f"\n3. Check regime detector state at bar 1:")
    print(f"   grep -A5 'Bar #1' {opt_log}")
    print(f"   grep -A5 'Bar #1' {prod_log}")

if __name__ == "__main__":
    run_with_detailed_logging()