#!/usr/bin/env python3
"""
Diagnose why optimizer gets different results than production.
"""

print("OPTIMIZER STATE ACCUMULATION DIAGNOSIS")
print("="*60)

print("\nTHE ISSUE:")
print("- Optimizer adaptive test: $100,058.98")
print("- Production (all approaches): $99,870.04")
print("")

print("WHAT WE'VE VERIFIED:")
print("1. Both use BacktestEngine ✓")
print("2. Both use the same data (bars 800-999) ✓")
print("3. Both should have cold start with reset ✓")
print("4. Production results are consistent (~$99,870) ✓")
print("")

print("THE SMOKING GUN:")
print("The optimizer's result ($100,058.98) is DIFFERENT from all")
print("production approaches, which all get ~$99,870.")
print("")

print("HYPOTHESIS:")
print("During --optimize-joint, the optimizer runs MANY backtests:")
print("1. Grid search: Tests 24 parameter combinations")
print("2. Each test supposedly resets components")
print("3. BUT some state is leaking between runs")
print("4. By the time adaptive test runs, components have 'memory'")
print("")

print("POSSIBLE STATE LEAKS:")
print("1. Singleton components not fully resetting")
print("2. Event bus retaining subscriptions or state")
print("3. Container caching component state")
print("4. Static/class variables in indicators")
print("5. Random number generator state")
print("")

print("TO VERIFY:")
print("Run ONLY the adaptive test without the grid search:")
print("1. Load pre-optimized parameters")
print("2. Run adaptive test in isolation")
print("3. Should get $99,870 like production")
print("")

print("COMMAND TO TEST:")
print("-"*60)
print("Create a script that:")
print("1. Loads regime_optimized_parameters.json")
print("2. Runs ONLY the adaptive test (no grid search)")
print("3. Uses the same BacktestEngine approach")
print("")

print("If this gets $99,870, it confirms state accumulation.")
print("If this gets $100,058, the issue is elsewhere.")
print("="*60)