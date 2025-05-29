#!/usr/bin/env python3
"""
Verify that optimized parameters are properly applied during test runs.
"""

import json
import subprocess
import re

def check_parameter_application():
    """Run a small test and verify parameters are applied."""
    
    # First, check what parameters should be applied
    with open('test_regime_parameters.json', 'r') as f:
        params = json.load(f)
    
    print("="*80)
    print("PARAMETER APPLICATION VERIFICATION")
    print("="*80)
    print("\nExpected parameters from test_regime_parameters.json:")
    
    for regime, regime_params in params['regimes'].items():
        print(f"\n{regime.upper()} regime:")
        # Show key differentiating parameters
        if 'strategy_ma_crossover.slow_ma.lookback_period' in regime_params:
            print(f"  MA slow period: {regime_params['strategy_ma_crossover.slow_ma.lookback_period']}")
        if 'strategy_rsi_rule.overbought_threshold' in regime_params:
            print(f"  RSI overbought: {regime_params['strategy_rsi_rule.overbought_threshold']}")
        if 'ma_crossover.weight' in regime_params:
            print(f"  MA weight: {regime_params['ma_crossover.weight']}")
        if 'bb.weight' in regime_params:
            print(f"  BB weight: {regime_params['bb.weight']}")
    
    # Now run a test to see if these are applied
    print("\n" + "-"*80)
    print("Running test to verify parameter application...")
    print("-"*80)
    
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--bars", "500",  # Small test
        "--dataset", "test",
        "--log-level", "DEBUG"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Save output for inspection
    with open("parameter_verification.log", "w") as f:
        f.write(output)
    
    # Look for evidence of parameter application
    print("\nSearching for evidence of parameter application...\n")
    
    # Check if parameters were loaded
    param_load_match = re.search(r"Parameter file: (.+)", output)
    if param_load_match:
        print(f"✅ Parameters loaded from: {param_load_match.group(1)}")
    else:
        print("❌ No parameter file loading found")
    
    # Check if regime parameters were shown
    regime_params_found = re.findall(r"Regime '(\w+)' parameters: ({.+})", output)
    if regime_params_found:
        print(f"✅ Found {len(regime_params_found)} regime parameter sets")
    else:
        print("❌ No regime parameters found in output")
    
    # Check for actual parameter usage in signals
    # Look for different MA periods being used
    ma_periods = re.findall(r"MA.*period.*?(\d+)", output)
    if ma_periods:
        unique_periods = set(ma_periods)
        print(f"✅ Found MA periods in use: {unique_periods}")
    
    # Check for RSI thresholds
    rsi_thresholds = re.findall(r"RSI.*threshold.*?([\d.]+)", output)
    if rsi_thresholds:
        unique_thresholds = set(rsi_thresholds)
        print(f"✅ Found RSI thresholds in use: {unique_thresholds}")
    
    # Check if we see regime changes
    regime_changes = re.findall(r"Market regime changed from '(\w+)' to '(\w+)'", output)
    if regime_changes:
        print(f"✅ Found {len(regime_changes)} regime changes")
        for old, new in regime_changes[:3]:  # Show first 3
            print(f"   {old} → {new}")
    else:
        print("⚠️  No regime changes detected (may be normal for short test)")
    
    # Check for adaptation logs
    adapt_logs = re.findall(r"adapt.*regime|Updating.*parameters.*regime", output, re.IGNORECASE)
    if adapt_logs:
        print(f"✅ Found {len(adapt_logs)} parameter adaptation logs")
    else:
        print("❌ No parameter adaptation logs found")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if not adapt_logs:
        print("❌ Parameters are loaded but may not be dynamically applied on regime changes.")
        print("   Check if RegimeAdaptiveEnsembleComposed._adapt_to_regime() is being called.")
        print("   The strategy should update parameters when regime changes.")
    else:
        print("✅ Parameters appear to be loaded and applied correctly.")
    
    print("\nFull output saved to: parameter_verification.log")
    print("Search the log for 'adapt' or 'regime' to see detailed parameter changes.")

if __name__ == "__main__":
    check_parameter_application()