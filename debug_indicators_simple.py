#!/usr/bin/env python3
"""Simple debug to understand regime classification difference."""

# Based on the logs, at the first bar (2024-03-28 13:48:00):
# Optimization sees: ma_trend=0.0553, volume_ratio=1.2087 -> trending_up
# Test run sees: trending_down

print("Analyzing first bar regime classification difference:")
print("\nFrom optimization logs:")
print("  Timestamp: 2024-03-28 13:48:00")
print("  ma_trend: 0.0553")
print("  volume_ratio: 1.2087")
print("  volume_sma: NOT READY")
print("  rsi_value: 60.49")
print("  Classification: trending_up")

print("\nRegime thresholds:")
print("  trending_up: ma_trend > 0.0001 AND volume_ratio > 0.8")
print("  trending_down: ma_trend < -0.0001 AND volume_ratio > 0.8")

print("\nAnalysis:")
print("  ma_trend (0.0553) > 0.0001? YES")
print("  volume_ratio (1.2087) > 0.8? YES")
print("  Expected: trending_up ✓")

print("\nThe test run must be seeing DIFFERENT indicator values!")
print("This suggests the indicators are calculated differently between runs.")
print("\nPossible causes:")
print("1. Different warmup data (train vs no train)")
print("2. Different indicator initialization")
print("3. Different calculation logic")