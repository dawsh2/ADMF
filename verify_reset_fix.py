#!/usr/bin/env python3
"""
Verify that indicator reset is working correctly.
"""

from src.strategy.components.indicators.trend import MovingAverageIndicator

# Create an indicator
ma = MovingAverageIndicator("test_ma", lookback_period=5)
ma._initialize()

# Add some data
for i in range(10):
    ma.update({'close': 100 + i})

print(f"Before reset: buffer has {len(ma._buffer)} bars")
print(f"Buffer content: {ma._buffer}")

# Reset
ma.reset()

print(f"\nAfter reset: buffer has {len(ma._buffer)} bars")
print(f"Buffer content: {ma._buffer}")
print(f"Ready: {ma.ready}")
print(f"Value: {ma.value}")

# Verify reset worked
if len(ma._buffer) == 0 and ma._value is None and not ma._ready:
    print("\n✅ Reset is working correctly!")
else:
    print("\n❌ Reset is NOT working correctly!")