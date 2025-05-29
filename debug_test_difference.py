#!/usr/bin/env python3
"""
Debug script to understand why optimization test phase and independent test run
produce different results despite using the same data and parameters.
"""

import subprocess
import json
import re
from datetime import datetime
import sys

def run_command(cmd):
    """Run a command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_trades(output):
    """Extract trade information from output."""
    trades = []
    # Look for trade execution messages
    trade_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Executing (BUY|SELL) order.*quantity=(\d+)"
    
    for line in output.split('\n'):
        match = re.search(trade_pattern, line)
        if match:
            trades.append({
                'timestamp': match.group(1),
                'type': match.group(2),
                'quantity': int(match.group(3))
            })
    return trades

def extract_signals(output):
    """Extract signal generation from output."""
    signals = []
    # Look for signal messages
    signal_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Generated (BUY|SELL) signal.*strength=([\d.]+)"
    
    for line in output.split('\n'):
        match = re.search(signal_pattern, line)
        if match:
            signals.append({
                'timestamp': match.group(1),
                'type': match.group(2),
                'strength': float(match.group(3))
            })
    return signals

def extract_metrics(output):
    """Extract final metrics from output."""
    metrics = {}
    
    # Look for return
    return_match = re.search(r"Total Return:\s*([-\d.]+)%", output)
    if return_match:
        metrics['total_return'] = float(return_match.group(1))
    
    # Look for trades
    trades_match = re.search(r"Total Trades:\s*(\d+)", output)
    if trades_match:
        metrics['total_trades'] = int(trades_match.group(1))
        
    # Look for sharpe
    sharpe_match = re.search(r"Sharpe Ratio:\s*([-\d.]+)", output)
    if sharpe_match:
        metrics['sharpe_ratio'] = float(sharpe_match.group(1))
        
    return metrics

def main():
    print("="*80)
    print("DEBUGGING TEST PHASE DIFFERENCES")
    print("="*80)
    
    # Run optimization (just the test phase part)
    print("\n1. Running optimization to capture test phase output...")
    opt_cmd = "cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --optimize --log-level INFO 2>&1"
    
    # We need to capture just the test phase portion
    # Run the full optimization and extract test phase
    opt_output = run_command(opt_cmd)
    
    # Find test phase start
    test_start = opt_output.find("🚀 BEGINNING TEST PHASE 🚀")
    if test_start == -1:
        print("ERROR: Could not find test phase in optimization output")
        return
        
    # Extract just the test phase portion
    test_phase_output = opt_output[test_start:]
    # Find where test phase ends (look for final results)
    test_end = test_phase_output.find("Workflow optimization completed")
    if test_end > 0:
        test_phase_output = test_phase_output[:test_end]
    
    # Save test phase output
    with open('opt_test_phase.log', 'w') as f:
        f.write(test_phase_output)
    
    print("\n2. Running independent test...")
    test_cmd = "cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --dataset test --log-level INFO 2>&1"
    test_output = run_command(test_cmd)
    
    # Save test output
    with open('independent_test.log', 'w') as f:
        f.write(test_output)
    
    print("\n3. Analyzing differences...")
    
    # Extract metrics
    opt_metrics = extract_metrics(test_phase_output)
    test_metrics = extract_metrics(test_output)
    
    print("\nMETRICS COMPARISON:")
    print(f"  Optimization test phase: {opt_metrics}")
    print(f"  Independent test run:    {test_metrics}")
    
    # Extract trades
    opt_trades = extract_trades(test_phase_output)
    test_trades = extract_trades(test_output)
    
    print(f"\nTRADE COUNT:")
    print(f"  Optimization test phase: {len(opt_trades)} trades")
    print(f"  Independent test run:    {len(test_trades)} trades")
    
    # Compare first few trades
    if opt_trades and test_trades:
        print("\nFIRST 5 TRADES COMPARISON:")
        print("  Optimization test phase:")
        for i, trade in enumerate(opt_trades[:5]):
            print(f"    {i+1}. {trade['timestamp']} - {trade['type']} {trade['quantity']}")
        print("  Independent test run:")
        for i, trade in enumerate(test_trades[:5]):
            print(f"    {i+1}. {trade['timestamp']} - {trade['type']} {trade['quantity']}")
    
    # Look for regime changes
    print("\nREGIME CHANGES:")
    regime_pattern = r"REGIME CHANGE: (\w+) -> (\w+)"
    
    opt_regimes = re.findall(regime_pattern, test_phase_output)
    test_regimes = re.findall(regime_pattern, test_output)
    
    print(f"  Optimization test phase: {len(opt_regimes)} regime changes")
    if opt_regimes:
        for r in opt_regimes[:5]:
            print(f"    {r[0]} -> {r[1]}")
    
    print(f"  Independent test run: {len(test_regimes)} regime changes")
    if test_regimes:
        for r in test_regimes[:5]:
            print(f"    {r[0]} -> {r[1]}")
    
    # Check for parameter loading messages
    print("\nPARAMETER LOADING:")
    param_pattern = r"Loading regime-specific parameters"
    opt_param_loads = len(re.findall(param_pattern, test_phase_output))
    test_param_loads = len(re.findall(param_pattern, test_output))
    
    print(f"  Optimization test phase: {opt_param_loads} parameter loads")
    print(f"  Independent test run:    {test_param_loads} parameter loads")
    
    print("\nSaved detailed logs to:")
    print("  - opt_test_phase.log")
    print("  - independent_test.log")
    print("\nYou can diff these files to see exact differences.")

if __name__ == "__main__":
    main()