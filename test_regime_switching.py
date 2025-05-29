#!/usr/bin/env python3
"""
Test that regime switching and parameter adaptation work correctly in test mode.
"""

import subprocess
import re

def test_regime_switching():
    """Run a test backtest and check for regime switching."""
    
    print("="*80)
    print("REGIME SWITCHING TEST")
    print("="*80)
    print("\nRunning test backtest with full event system...")
    
    # Run with test dataset - should use full backtest runner, not isolated
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--dataset", "test",
        "--bars", "5000",
        "--log-level", "INFO"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Save output
    with open("regime_switching_test.log", "w") as f:
        f.write(output)
    
    print("\nAnalyzing results...")
    print("-"*80)
    
    # Check for key indicators
    
    # 1. Check if running in test mode
    test_mode = "TEST DATASET" in output or "test dataset" in output
    print(f"Test dataset mode: {'✅ YES' if test_mode else '❌ NO'}")
    
    # 2. Check if parameters were loaded
    params_loaded = "LOADING REGIME ADAPTIVE ENSEMBLE PARAMETERS" in output
    print(f"Parameters loaded: {'✅ YES' if params_loaded else '❌ NO'}")
    
    # 3. Check if regime switching is enabled
    regime_enabled = "ENABLED regime switching" in output
    regime_disabled = "Regime switching DISABLED" in output
    if regime_enabled and not regime_disabled:
        print("Regime switching: ✅ ENABLED")
    elif regime_disabled:
        print("Regime switching: ❌ DISABLED")
    else:
        print("Regime switching: ⚠️  UNKNOWN")
    
    # 4. Look for regime changes
    regime_changes = re.findall(r"REGIME CHANGE DETECTED: (\w+) -> (\w+)", output)
    if regime_changes:
        print(f"Regime changes detected: ✅ {len(regime_changes)} changes")
        for i, (old, new) in enumerate(regime_changes[:5]):
            print(f"  {i+1}. {old} → {new}")
    else:
        print("Regime changes detected: ❌ NONE")
    
    # 5. Look for parameter updates
    param_updates = re.findall(r"Loading.*parameters:|Current parameters before update:", output)
    if param_updates:
        print(f"Parameter updates: ✅ {len(param_updates)} updates")
    else:
        print("Parameter updates: ❌ NONE")
    
    # 6. Check for classification events
    classification_events = re.findall(r"publishing CLASSIFICATION event|Successfully published CLASSIFICATION", output)
    if classification_events:
        print(f"Classification events: ✅ {len(classification_events)} published")
    else:
        print("Classification events: ❌ NONE")
    
    # 7. Check if running isolated or full backtest
    isolated = "isolated_evaluation" in output or "IsolatedComponentEvaluator" in output
    backtest_runner = "BacktestRunner" in output or "backtest_runner" in output
    if isolated:
        print("Execution mode: ❌ ISOLATED (no regime switching possible)")
    elif backtest_runner:
        print("Execution mode: ✅ FULL BACKTEST (regime switching enabled)")
    else:
        print("Execution mode: ⚠️  UNKNOWN")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if isolated:
        print("❌ The system is running in ISOLATED mode even with --dataset test!")
        print("   This prevents regime switching and parameter adaptation.")
        print("   Check if optimization is still active or if test mode setup is incorrect.")
    elif not regime_changes and backtest_runner:
        print("⚠️  Running in full backtest mode but no regime changes detected.")
        print("   Possible reasons:")
        print("   - Market conditions didn't trigger regime changes in test period")
        print("   - Regime detector thresholds are too strict")
        print("   - Event system not properly connected")
    elif regime_changes and param_updates:
        print("✅ Regime switching and parameter adaptation appear to be working!")
    
    print(f"\nFull output saved to: regime_switching_test.log")
    print("Search for 'REGIME CHANGE' or 'adapt' to see details.")

if __name__ == "__main__":
    test_regime_switching()