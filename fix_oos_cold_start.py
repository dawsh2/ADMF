#!/usr/bin/env python3
"""
Fix for OOS test cold start issue.

This script modifies the BacktestEngine to ensure components are properly reset
between training and test phases to achieve true cold start behavior.
"""

def print_fix_summary():
    """Print summary of the fix needed."""
    print("="*60)
    print("OOS COLD START FIX")
    print("="*60)
    
    print("\nPROBLEM:")
    print("- Optimizer's OOS test has 'memory' from training phase")
    print("- RegimeDetector indicators remain warmed up")
    print("- Production run starts with cold indicators")
    print("- This causes different results")
    
    print("\nSOLUTION IMPLEMENTED:")
    print("1. BacktestEngine now resets RegimeDetector in _reset_components()")
    print("2. RegimeDetector.reset() resets all indicators")
    
    print("\nADDITIONAL FIX NEEDED:")
    print("We need to ensure the optimizer creates fresh component instances")
    print("for the test phase, not reuse the training instances.")
    
    print("\nFILE TO MODIFY:")
    print("src/strategy/optimization/engines/backtest_engine.py")
    
    print("\nCHANGE NEEDED:")
    print("In _resolve_components(), always get fresh instances from container")
    print("This ensures true cold start for each backtest run")

if __name__ == "__main__":
    print_fix_summary()