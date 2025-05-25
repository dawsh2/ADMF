#!/usr/bin/env python3
"""
Analyze the 16 signals from optimizer in detail.
"""

# From the optimizer log, these are the 16 signals
optimizer_signals = [
    ("13:46:00", 523.4800, 1, "default"),      # First bar
    ("13:48:00", 523.3900, -1, "default"),     # Early reversal
    ("14:05:00", 523.4700, 1, "default"),      # Match our 818
    ("14:11:00", 523.2000, -1, "trending_down"), # Different regime!
    ("14:30:00", 523.2300, 1, "default"),      # Match our 843
    ("15:07:00", 523.2500, -1, "ranging_low_vol"),
    ("15:16:00", 523.5000, 1, "trending_down"),
    ("15:18:00", 523.5000, -1, "default"),
    ("15:53:00", 523.5600, 1, "default"),
    ("15:38:00", 523.4250, -1, "ranging_low_vol"),
    ("15:53:00", 523.5600, 1, "ranging_low_vol"),
    ("16:06:00", 523.2490, -1, "ranging_low_vol"),
    ("16:49:00", 523.2000, 1, "ranging_low_vol"),
    ("17:01:00", 523.1450, -1, "ranging_low_vol"),
    ("17:02:00", 523.2200, 1, "ranging_low_vol"),
    ("17:04:00", 523.2440, -1, "ranging_low_vol"),
]

# Count by regime
regime_counts = {}
for _, _, _, regime in optimizer_signals:
    regime_counts[regime] = regime_counts.get(regime, 0) + 1

print("=== OPTIMIZER TEST SIGNALS (16 total) ===")
print("\nBy regime:")
for regime, count in sorted(regime_counts.items()):
    print(f"  {regime}: {count}")

print("\nSignal details:")
for i, (time, price, type, regime) in enumerate(optimizer_signals):
    print(f"{i+1:2d}. {time} - {regime:20s} - {'BUY ' if type == 1 else 'SELL'} at ${price:.2f}")

print("\n=== KEY OBSERVATIONS ===")
print("1. Signals occur in multiple regimes (not just default)")
print("2. The 14:11:00 signal is in trending_down with 5/20 MA windows")
print("3. Many late signals are in ranging_low_vol regime")
print("4. This explains different MA calculations and crossovers")

print("\n=== WHAT WE NEED TO MATCH ===")
print("To get exactly 16 signals, our validation needs:")
print("1. Correct regime detection timing")
print("2. Proper parameter switching (5/20 for trending_down)")
print("3. Same initialization state at test start")
print("4. Possibly same warmup sequence during optimization")