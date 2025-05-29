#!/usr/bin/env python3
"""Debug script to understand why first regime classification differs."""

import re

def analyze_log(log_file):
    """Extract regime changes and key info from log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find first regime change
    regime_pattern = r"Regime SWITCH from '(\w+)' to '(\w+)'"
    first_change = re.search(regime_pattern, content)
    
    # Find if regime switching is enabled
    enabled = "ENABLED regime switching" in content
    
    # Count total trades
    trades_pattern = r"Number of Trades:\s*(\d+)"
    trades_match = re.search(trades_pattern, content)
    total_trades = int(trades_match.group(1)) if trades_match else 0
    
    # Count signals generated
    signals = content.count("Strategy 'strategy' generated signal")
    
    return {
        'first_change': first_change.groups() if first_change else None,
        'regime_switching_enabled': enabled,
        'total_trades': total_trades,
        'signals_generated': signals
    }

print("="*80)
print("REGIME CLASSIFICATION AND TRADING ANALYSIS")
print("="*80)

# Analyze optimization log
opt_log = "logs/admf_20250528_191224.log"
print(f"\nAnalyzing optimization log: {opt_log}")
opt_results = analyze_log(opt_log)
print(f"- First regime change: {opt_results['first_change']}")
print(f"- Regime switching enabled: {opt_results['regime_switching_enabled']}")
print(f"- Total trades: {opt_results['total_trades']}")
print(f"- Signals generated: {opt_results['signals_generated']}")

# Analyze standalone test log
test_log = "logs/admf_20250528_191354.log"
print(f"\nAnalyzing standalone test log: {test_log}")
test_results = analyze_log(test_log)
print(f"- First regime change: {test_results['first_change']}")
print(f"- Regime switching enabled: {test_results['regime_switching_enabled']}")
print(f"- Total trades: {test_results['total_trades']}")
print(f"- Signals generated: {test_results['signals_generated']}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

if opt_results['first_change'] and test_results['first_change']:
    if opt_results['first_change'] != test_results['first_change']:
        print(f"✗ First regime change DIFFERS:")
        print(f"  - Optimization: {opt_results['first_change'][0]} -> {opt_results['first_change'][1]}")
        print(f"  - Test run: {test_results['first_change'][0]} -> {test_results['first_change'][1]}")
    else:
        print(f"✓ First regime change matches: {opt_results['first_change'][0]} -> {opt_results['first_change'][1]}")

print(f"\n✗ Trade count mismatch:")
print(f"  - Optimization: {opt_results['total_trades']} trades")
print(f"  - Test run: {test_results['total_trades']} trades")

if test_results['total_trades'] == 0 and test_results['signals_generated'] == 0:
    print("\n⚠️  CRITICAL: Test run generated NO SIGNALS despite regime switching being enabled!")
    print("   This suggests the trading rules are not evaluating properly.")

print("\nPossible causes for no trades in test run:")
print("1. Rules not ready due to indicator warmup requirements")
print("2. Parameter application issue preventing rule evaluation")
print("3. Signal aggregation logic filtering out all signals")
print("4. Different rule configurations between runs")