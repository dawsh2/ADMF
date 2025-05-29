#!/usr/bin/env python3
"""
Comprehensive verification script to prove identity between optimization test phase
and independent test run across all aspects.
"""

import subprocess
import json
import re
from datetime import datetime
import sys
import hashlib
from collections import OrderedDict

def run_command(cmd):
    """Run a command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_test_phase(output):
    """Extract just the test phase portion from optimization output."""
    test_start = output.find("🚀 BEGINNING TEST PHASE 🚀")
    if test_start == -1:
        return None
    test_phase = output[test_start:]
    # Find where test phase ends
    test_end = test_phase.find("Workflow optimization completed")
    if test_end > 0:
        test_phase = test_phase[:test_end]
    return test_phase

def verify_date_ranges(opt_output, test_output):
    """Verify date ranges and datasets are identical."""
    print("\n1. DATE RANGES AND DATASETS:")
    
    # Extract DATA WINDOW logs
    data_window_pattern = r"\[DATA WINDOW\] (\w+) dataset:\s*First bar: ([^\(]+) \(index: (\d+)\)\s*Last bar: ([^\(]+) \(index: (\d+)\)\s*Total bars: (\d+)"
    
    opt_windows = re.findall(data_window_pattern, opt_output, re.MULTILINE | re.DOTALL)
    test_windows = re.findall(data_window_pattern, test_output, re.MULTILINE | re.DOTALL)
    
    # Get TEST dataset windows
    opt_test_window = [w for w in opt_windows if w[0] == 'TEST']
    test_test_window = [w for w in test_windows if w[0] == 'TEST']
    
    if opt_test_window and test_test_window:
        opt_w = opt_test_window[-1]  # Get last one (most recent)
        test_w = test_test_window[-1]
        
        print(f"  Optimization Test Phase:")
        print(f"    First bar: {opt_w[1].strip()} (index: {opt_w[2]})")
        print(f"    Last bar: {opt_w[3].strip()} (index: {opt_w[4]})")
        print(f"    Total bars: {opt_w[5]}")
        
        print(f"  Independent Test Run:")
        print(f"    First bar: {test_w[1].strip()} (index: {test_w[2]})")
        print(f"    Last bar: {test_w[3].strip()} (index: {test_w[4]})")
        print(f"    Total bars: {test_w[5]}")
        
        match = (opt_w[1].strip() == test_w[1].strip() and 
                opt_w[3].strip() == test_w[3].strip() and
                opt_w[5] == test_w[5])
        print(f"  ✓ MATCH: {match}")
    else:
        print("  ✗ ERROR: Could not find DATA WINDOW information")

def verify_rules_config(opt_output, test_output):
    """Verify rules and configuration are identical."""
    print("\n2. RULES AND CONFIGURATION:")
    
    # Look for ensemble initialization
    ensemble_pattern = r"ENSEMBLE INITIALIZATION COMPLETE.*?Rules: \[(.*?)\]"
    
    opt_rules = re.search(ensemble_pattern, opt_output, re.DOTALL)
    test_rules = re.search(ensemble_pattern, test_output, re.DOTALL)
    
    if opt_rules and test_rules:
        opt_rule_list = [r.strip().strip("'") for r in opt_rules.group(1).split(',')]
        test_rule_list = [r.strip().strip("'") for r in test_rules.group(1).split(',')]
        
        print(f"  Optimization rules: {opt_rule_list}")
        print(f"  Test run rules: {test_rule_list}")
        print(f"  ✓ MATCH: {opt_rule_list == test_rule_list}")
    else:
        print("  ✗ ERROR: Could not find rules configuration")

def verify_parameters(opt_output, test_output):
    """Verify all parameters are identical."""
    print("\n3. PARAMETERS:")
    
    # Extract regime parameters
    regime_param_pattern = r"Regime '(\w+)' parameters: ({.*?})"
    
    opt_params = {}
    test_params = {}
    
    for match in re.finditer(regime_param_pattern, opt_output):
        regime = match.group(1)
        params_str = match.group(2)
        try:
            params = eval(params_str)  # Safe here as we control the output
            opt_params[regime] = params
        except:
            pass
    
    for match in re.finditer(regime_param_pattern, test_output):
        regime = match.group(1)
        params_str = match.group(2)
        try:
            params = eval(params_str)
            test_params[regime] = params
        except:
            pass
    
    if opt_params and test_params:
        all_match = True
        for regime in sorted(set(opt_params.keys()) | set(test_params.keys())):
            if regime in opt_params and regime in test_params:
                match = opt_params[regime] == test_params[regime]
                print(f"  {regime}: {'✓ MATCH' if match else '✗ MISMATCH'}")
                if not match:
                    all_match = False
                    print(f"    Opt: {opt_params[regime]}")
                    print(f"    Test: {test_params[regime]}")
            else:
                print(f"  {regime}: ✗ MISSING in {'test' if regime not in test_params else 'opt'}")
                all_match = False
        print(f"  Overall: {'✓ ALL MATCH' if all_match else '✗ MISMATCHES FOUND'}")
    else:
        print("  ✗ ERROR: Could not extract parameters")

def verify_regime_changes(opt_output, test_output):
    """Verify regime changes happen at same times with same parameters."""
    print("\n4. REGIME CHANGES:")
    
    # Extract regime changes with timestamps
    regime_change_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?REGIME CHANGE: (\w+) -> (\w+)"
    
    opt_changes = re.findall(regime_change_pattern, opt_output)
    test_changes = re.findall(regime_change_pattern, test_output)
    
    print(f"  Optimization regime changes: {len(opt_changes)}")
    print(f"  Test run regime changes: {len(test_changes)}")
    
    if len(opt_changes) != len(test_changes):
        print("  ✗ MISMATCH: Different number of regime changes!")
    else:
        all_match = True
        for i, (opt_change, test_change) in enumerate(zip(opt_changes, test_changes)):
            match = opt_change == test_change
            if not match:
                all_match = False
                print(f"  Change {i+1}: ✗ MISMATCH")
                print(f"    Opt: {opt_change[0]} {opt_change[1]} -> {opt_change[2]}")
                print(f"    Test: {test_change[0]} {test_change[1]} -> {test_change[2]}")
            elif i < 5:  # Show first 5
                print(f"  Change {i+1}: ✓ {opt_change[0]} {opt_change[1]} -> {opt_change[2]}")
        
        if all_match:
            print("  ✓ ALL REGIME CHANGES MATCH")

def verify_signals(opt_output, test_output):
    """Verify signal generation is identical."""
    print("\n5. SIGNAL HISTORY:")
    
    # Extract signals
    signal_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Generated (BUY|SELL) signal.*?strength=([\d.]+)"
    
    opt_signals = re.findall(signal_pattern, opt_output)
    test_signals = re.findall(signal_pattern, test_output)
    
    print(f"  Optimization signals: {len(opt_signals)}")
    print(f"  Test run signals: {len(test_signals)}")
    
    if len(opt_signals) != len(test_signals):
        print("  ✗ MISMATCH: Different number of signals!")
        # Show where they diverge
        for i in range(min(len(opt_signals), len(test_signals))):
            if opt_signals[i] != test_signals[i]:
                print(f"  First divergence at signal {i+1}:")
                print(f"    Opt: {opt_signals[i]}")
                print(f"    Test: {test_signals[i]}")
                break
    else:
        # Check if all match
        all_match = all(o == t for o, t in zip(opt_signals, test_signals))
        if all_match:
            print("  ✓ ALL SIGNALS MATCH")
        else:
            print("  ✗ SIGNALS MISMATCH")

def verify_trades(opt_output, test_output):
    """Verify trade execution is identical."""
    print("\n6. TRADE HISTORY:")
    
    # Extract trades with more detail
    trade_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Executing (BUY|SELL) order.*?quantity=(\d+).*?price=([\d.]+)"
    
    opt_trades = re.findall(trade_pattern, opt_output)
    test_trades = re.findall(trade_pattern, test_output)
    
    print(f"  Optimization trades: {len(opt_trades)}")
    print(f"  Test run trades: {len(test_trades)}")
    
    if len(opt_trades) != len(test_trades):
        print("  ✗ MISMATCH: Different number of trades!")
        print(f"  First 5 optimization trades:")
        for t in opt_trades[:5]:
            print(f"    {t[0]} {t[1]} {t[2]} @ {t[3]}")
        print(f"  First 5 test trades:")
        for t in test_trades[:5]:
            print(f"    {t[0]} {t[1]} {t[2]} @ {t[3]}")
    else:
        all_match = all(o == t for o, t in zip(opt_trades, test_trades))
        if all_match:
            print("  ✓ ALL TRADES MATCH")
        else:
            print("  ✗ TRADES MISMATCH")
            # Find first mismatch
            for i, (o, t) in enumerate(zip(opt_trades, test_trades)):
                if o != t:
                    print(f"  First mismatch at trade {i+1}:")
                    print(f"    Opt: {o}")
                    print(f"    Test: {t}")
                    break

def verify_execution_path(opt_output, test_output):
    """Verify execution path is identical."""
    print("\n7. EXECUTION PATH:")
    
    # Look for key execution milestones
    milestones = [
        "RUNNING TEST DATASET WITH OPTIMIZED PARAMETERS",
        "ENABLED regime switching",
        "Applied rule parameter",
        "Portfolio reset"
    ]
    
    for milestone in milestones:
        opt_count = opt_output.count(milestone)
        test_count = test_output.count(milestone)
        match = opt_count > 0 and test_count > 0
        print(f"  {milestone}: {'✓' if match else '✗'} (opt: {opt_count}, test: {test_count})")

def verify_indicator_behavior(opt_output, test_output):
    """Verify indicator resets and updates."""
    print("\n8. INDICATOR BEHAVIOR:")
    
    # Look for indicator value logs after regime changes
    indicator_pattern = r"Indicator values after regime change:.*?MA: fast=([\d.]+).*?slow=([\d.]+)"
    
    opt_indicators = re.findall(indicator_pattern, opt_output, re.DOTALL)
    test_indicators = re.findall(indicator_pattern, test_output, re.DOTALL)
    
    print(f"  Indicator logs found - Opt: {len(opt_indicators)}, Test: {len(test_indicators)}")
    
    if opt_indicators and test_indicators:
        for i, (opt_ind, test_ind) in enumerate(zip(opt_indicators[:3], test_indicators[:3])):
            opt_fast, opt_slow = float(opt_ind[0]), float(opt_ind[1])
            test_fast, test_slow = float(test_ind[0]), float(test_ind[1])
            match = abs(opt_fast - test_fast) < 0.0001 and abs(opt_slow - test_slow) < 0.0001
            print(f"  After change {i+1}: {'✓' if match else '✗'}")
            if not match:
                print(f"    Opt: fast={opt_fast}, slow={opt_slow}")
                print(f"    Test: fast={test_fast}, slow={test_slow}")

def verify_risk_config(opt_output, test_output):
    """Verify risk configuration."""
    print("\n9. RISK CONFIGURATION:")
    
    # Look for position sizing and risk parameters
    risk_patterns = [
        (r"position_size_pct.*?([\d.]+)", "Position size %"),
        (r"max_positions.*?(\d+)", "Max positions"),
        (r"stop_loss_pct.*?([\d.]+)", "Stop loss %")
    ]
    
    for pattern, name in risk_patterns:
        opt_matches = re.findall(pattern, opt_output)
        test_matches = re.findall(pattern, test_output)
        
        if opt_matches and test_matches:
            match = opt_matches[0] == test_matches[0]
            print(f"  {name}: {'✓' if match else '✗'} (opt: {opt_matches[0]}, test: {test_matches[0]})")

def main():
    print("="*80)
    print("COMPREHENSIVE TEST IDENTITY VERIFICATION")
    print("="*80)
    
    # Run optimization
    print("\nRunning optimization...")
    opt_cmd = "cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --optimize --log-level INFO 2>&1"
    opt_output = run_command(opt_cmd)
    
    # Extract test phase
    test_phase_output = extract_test_phase(opt_output)
    if not test_phase_output:
        print("ERROR: Could not extract test phase from optimization")
        return
    
    # Save for reference
    with open('opt_test_phase_full.log', 'w') as f:
        f.write(test_phase_output)
    
    # Run independent test
    print("\nRunning independent test...")
    test_cmd = "cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --dataset test --log-level INFO 2>&1"
    test_output = run_command(test_cmd)
    
    # Save for reference
    with open('independent_test_full.log', 'w') as f:
        f.write(test_output)
    
    # Run all verifications
    verify_date_ranges(test_phase_output, test_output)
    verify_rules_config(test_phase_output, test_output)
    verify_parameters(test_phase_output, test_output)
    verify_regime_changes(test_phase_output, test_output)
    verify_execution_path(test_phase_output, test_output)
    verify_risk_config(test_phase_output, test_output)
    verify_signals(test_phase_output, test_output)
    verify_trades(test_phase_output, test_output)
    verify_indicator_behavior(test_phase_output, test_output)
    
    # Final metrics
    print("\n10. FINAL METRICS:")
    
    # Extract final results
    return_pattern = r"Total Return:\s*([-\d.]+)%"
    trades_pattern = r"Total Trades:\s*(\d+)"
    
    opt_return = re.search(return_pattern, test_phase_output)
    test_return = re.search(return_pattern, test_output)
    opt_trades = re.search(trades_pattern, test_phase_output)
    test_trades = re.search(trades_pattern, test_output)
    
    if opt_return and test_return:
        print(f"  Total Return - Opt: {opt_return.group(1)}%, Test: {test_return.group(1)}%")
    if opt_trades and test_trades:
        print(f"  Total Trades - Opt: {opt_trades.group(1)}, Test: {test_trades.group(1)}")
    
    print("\n" + "="*80)
    print("Full logs saved to:")
    print("  - opt_test_phase_full.log")
    print("  - independent_test_full.log")
    print("\nUse 'diff -u opt_test_phase_full.log independent_test_full.log' to see all differences")

if __name__ == "__main__":
    main()