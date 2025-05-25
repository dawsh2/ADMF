#!/usr/bin/env python3
"""
Test optimizer vs production with identical configurations.
Focus on finding the exact cause of signal mismatch.
"""

import subprocess
import os
import time
import json
import re

def create_test_config():
    """Create a test config that matches optimizer behavior."""
    config = {
        "system": {
            "name": "ADMF-Test-Match",
            "version": "0.1.0"
        },
        "logging": {
            "level": "INFO"
        },
        "components": {
            "ensemble_strategy": {
                "symbol": "SPY",
                "short_window_default": 10,
                "long_window_default": 20,
                # Match optimizer weights after MA optimization detection
                "ma_rule.weight": 1.0,  # Optimizer uses 1.0 for MA
                "rsi_indicator": {
                    "period": 14
                },
                "rsi_rule": {
                    "oversold_threshold": 30.0,
                    "overbought_threshold": 70.0,
                    "weight": 0.0  # Optimizer uses 0.0 for RSI
                }
            },
            "data_handler_csv": {
                "csv_file_path": "data/1000_1min.csv",
                "symbol": "SPY",
                "timestamp_column": "timestamp",
                "train_test_split_ratio": 0.8,
                "open_column": "Open",
                "high_column": "High",
                "low_column": "Low",
                "close_column": "Close",
                "volume_column": "Volume"
            },
            "basic_portfolio": {
                "initial_cash": 100000.00
            },
            "basic_risk_manager": {
                "target_trade_quantity": 100
            },
            "simulated_execution_handler": {
                "default_quantity": 100,
                "commission_per_trade": 0.005,
                "commission_type": "per_share",
                "passthrough": False,
                "fill_price_logic": "signal_price"
            },
            # Disable regime detector to match optimizer's default regime
            # "MyPrimaryRegimeDetector": {}
        }
    }
    
    config_path = "config/config_test_match.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def run_test_comparison():
    """Run optimizer and production with identical settings."""
    print("OPTIMIZER VS PRODUCTION COMPARISON TEST")
    print("=" * 60)
    
    # Create test config
    config_path = create_test_config()
    print(f"Created test config: {config_path}")
    
    # Run optimizer first
    print("\n1. Running Optimizer...")
    print("-" * 40)
    opt_start = time.time()
    opt_result = subprocess.run(
        ["python", "main.py", "--optimize-ma", "--bars", "1000"],
        capture_output=True, text=True
    )
    opt_time = time.time() - opt_start
    
    if opt_result.returncode != 0:
        print(f"Optimizer error: {opt_result.stderr}")
        return
    
    # Count optimizer signals (in test phase)
    opt_log = find_latest_log("optimizer")
    opt_test_signals = count_test_signals(opt_log)
    print(f"Optimizer completed in {opt_time:.1f}s")
    print(f"Test signals generated: {opt_test_signals}")
    
    # Run production
    print("\n2. Running Production...")
    print("-" * 40)
    prod_start = time.time()
    prod_result = subprocess.run(
        ["python", "main.py", "--config", config_path, "--bars", "1000"],
        capture_output=True, text=True
    )
    prod_time = time.time() - prod_start
    
    if prod_result.returncode != 0:
        print(f"Production error: {prod_result.stderr}")
        return
    
    # Count production signals
    prod_log = find_latest_log("production")
    prod_signals = count_all_signals(prod_log)
    print(f"Production completed in {prod_time:.1f}s")
    print(f"Signals generated: {prod_signals}")
    
    # Detailed comparison
    print("\n3. Detailed Comparison")
    print("-" * 40)
    compare_logs_detailed(opt_log, prod_log)

def find_latest_log(log_type):
    """Find the most recent log file."""
    import glob
    logs = sorted(glob.glob("logs/*.log"), key=os.path.getmtime, reverse=True)
    
    if log_type == "optimizer":
        # Look for optimizer indicators in recent logs
        for log in logs[:5]:
            with open(log, 'r') as f:
                content = f.read(1000)  # Check first 1KB
                if "EnhancedOptimizer" in content or "optimization" in content:
                    return log
    else:
        # Return most recent admf log
        for log in logs[:5]:
            if "admf" in log:
                return log
    
    return logs[0] if logs else None

def count_test_signals(opt_log):
    """Count signals in optimizer test phase."""
    if not opt_log:
        return 0
    
    # Look for test phase markers and count signals
    in_test = False
    signal_count = 0
    
    with open(opt_log, 'r') as f:
        for line in f:
            if "test set" in line.lower() or "evaluating on test" in line.lower():
                in_test = True
            elif "training" in line.lower() and "phase" in line.lower():
                in_test = False
            elif in_test and "SIGNAL GENERATED" in line:
                signal_count += 1
    
    return signal_count

def count_all_signals(log_file):
    """Count all signals in a log file."""
    if not log_file:
        return 0
    
    cmd = f"grep -c 'SIGNAL GENERATED' {log_file} 2>/dev/null || echo 0"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())

def compare_logs_detailed(opt_log, prod_log):
    """Detailed comparison of logs."""
    if not opt_log or not prod_log:
        print("Missing log files for comparison")
        return
    
    print(f"Optimizer log: {opt_log}")
    print(f"Production log: {prod_log}")
    
    # Compare initialization
    print("\nINITIALIZATION COMPARISON:")
    print("-" * 30)
    
    # Check weights
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -m1 'EnsembleSignalStrategy weights' {log}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}: {result.stdout.strip()}")
    
    # Check adaptive mode
    print("\nADAPTIVE MODE:")
    print("-" * 30)
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -m1 'adaptive_mode' {log}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}: {result.stdout.strip()}")
    
    # Compare first few bars
    print("\nFIRST BAR COMPARISON:")
    print("-" * 30)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -m1 'BAR_001.*INDICATORS' {log}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            # Extract key values
            match = re.search(r'Price=([\d.]+).*MA_short=([\d.]+|N/A).*MA_long=([\d.]+|N/A)', result.stdout)
            if match:
                print(f"{name}: Price={match.group(1)}, MA_short={match.group(2)}, MA_long={match.group(3)}")
    
    # Compare first signal
    print("\nFIRST SIGNAL COMPARISON:")
    print("-" * 30)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -m1 -B2 'SIGNAL GENERATED' {log}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "BAR_" in line and "timestamp" in line:
                    match = re.search(r'\[([0-9:\ -]+)\]', line)
                    if match:
                        print(f"{name} first signal at: {match.group(1)}")
                elif "SIGNAL GENERATED" in line:
                    match = re.search(r'Type=(-?\d+), Price=([\d.]+)', line)
                    if match:
                        print(f"  Signal: Type={match.group(1)}, Price={match.group(2)}")

def check_specific_hypothesis():
    """Check specific hypotheses about the mismatch."""
    print("\n\nSPECIFIC HYPOTHESIS CHECKS")
    print("=" * 60)
    
    # Hypothesis 1: RSI weight difference
    print("\nHypothesis 1: RSI weight causing difference")
    print("-" * 40)
    print("Optimizer sets RSI weight to 0.0 during MA optimization")
    print("Production might be using config RSI weight of 0.4")
    print("Solution: Set RSI weight to 0.0 in production config")
    
    # Hypothesis 2: Adaptive mode
    print("\nHypothesis 2: Adaptive mode difference")
    print("-" * 40)
    print("Check if adaptive parameters are being loaded differently")
    
    # Hypothesis 3: Event ordering
    print("\nHypothesis 3: Event subscription order")
    print("-" * 40)
    print("Different component initialization order could affect event processing")

def main():
    """Run the comparison test."""
    # Run comparison
    run_test_comparison()
    
    # Check hypotheses
    check_specific_hypothesis()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. Ensure RSI weight is 0.0 in both runs")
    print("2. Disable adaptive mode in both runs")
    print("3. Check if regime detector affects signals even in default regime")
    print("4. Verify exact component initialization order")

if __name__ == "__main__":
    main()