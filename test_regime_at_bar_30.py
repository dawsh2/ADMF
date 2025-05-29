#!/usr/bin/env python3
"""
Test what regime is detected at bar 30 in both runs.
We know from the logs:
- Test run: trending_up at bar 30 (ma_trend=0.0191, rsi=67.68)
- Optimization: trending_up at bar 34 (ma_trend=0.0104, rsi=55.90)
"""

print("="*80)
print("REGIME DETECTION AT BAR 30")
print("="*80)

print("\nFrom our debug logs:")
print("\nTest run at bar 30:")
print("  ma_trend: 0.0191 (> 0.01 ✓)")
print("  rsi: 67.68 (between 55-80 ✓)")
print("  Result: trending_up")

print("\nOptimization at bar 30:")
print("  ma_trend: ? (need to check)")
print("  rsi: ? (need to check)")
print("  Result: default (since first non-default is at bar 34)")

print("\nKey question: Why does optimization not detect trending_up at bar 30?")
print("Possibilities:")
print("1. Different indicator values at bar 30")
print("2. Different regime detection logic")
print("3. Different indicator warmup/calculation")

print("\nBut wait - compare_regime_changes.py shows:")
print("- Optimization first change: default -> trending_up") 
print("- Test run first change: default -> trending_down")
print("\nThis is DIFFERENT from what our debug logs show!")
print("Our debug logs show both detecting trending_up first.")

print("\nThis suggests the compare_regime_changes.py is using OLD log files.")
print("We need to generate new logs with our current code.")