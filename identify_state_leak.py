#!/usr/bin/env python3
"""
Identify what state is leaking between optimizer runs.
"""

print("IDENTIFYING STATE LEAK")
print("="*60)

print("\nPOSSIBLE SOURCES OF STATE LEAK:")
print("-"*60)

print("1. INDICATOR STATE:")
print("   - Moving averages accumulate price history")
print("   - RSI accumulates gain/loss averages")
print("   - ATR accumulates true range values")
print("   Even with reset(), some internal state might persist")
print("")

print("2. EVENT BUS:")
print("   - Subscribers might accumulate")
print("   - Event history might be retained")
print("   - Handlers might have internal state")
print("")

print("3. STATIC/CLASS VARIABLES:")
print("   - Components might use class-level caches")
print("   - Static configuration might change")
print("   - Shared state between instances")
print("")

print("4. RANDOM NUMBER GENERATOR:")
print("   - RNG state affects trade execution")
print("   - Different sequences = different results")
print("   - Grid search might advance RNG differently")
print("")

print("5. PORTFOLIO METRICS:")
print("   - Trade history accumulation")
print("   - Performance statistics")
print("   - Regime transition counts")
print("")

print("TO TEST EACH:")
print("-"*60)

print("1. Log indicator internal state before/after reset")
print("2. Check event bus subscriber count")
print("3. Inspect class.__dict__ for static variables")
print("4. Set fixed random seed and compare")
print("5. Deep inspect portfolio state")
print("")

print("MOST LIKELY CULPRIT:")
print("-"*60)
print("The indicators in RegimeDetector and Strategy are")
print("accumulating state that affects regime detection timing.")
print("Even small differences compound over 200 bars.")
print("="*60)