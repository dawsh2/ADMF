#!/usr/bin/env python3
"""
Analyze the differences between our 13 signals and expected 17.
"""

# Our validation signals (bars >= 798)
our_signals = [
    (814, 'SELL', 523.49),
    (818, 'BUY', 523.47),
    (825, 'SELL', 523.05),
    (843, 'BUY', 523.23),
    (880, 'SELL', 523.25),
    (892, 'BUY', 523.56),
    (911, 'SELL', 523.42),
    (926, 'BUY', 523.56),
    (939, 'SELL', 523.25),
    (982, 'BUY', 523.20),
    (994, 'SELL', 523.14),
    (995, 'BUY', 523.22),
    (997, 'SELL', 523.24)
]

# From our earlier analysis, the optimizer's first test signal was around bar 798
# Let's analyze the pattern

print("Our 13 test signals (bars 798-997):")
for i, (bar, type, price) in enumerate(our_signals):
    print(f"{i+1}. Bar {bar} ({bar-798} bars into test), {type} at ${price:.2f}")

print(f"\nSignal frequency:")
print(f"First signal: Bar {our_signals[0][0]} (bar {our_signals[0][0]-798} of test)")
print(f"Last signal: Bar {our_signals[-1][0]} (bar {our_signals[-1][0]-798} of test)")
print(f"Total test bars: 200 (bars 798-997)")
print(f"Bars with signals: {len(our_signals)}")

# Check for potential missing signals at the start
print("\nPotential missing signals:")
print("- Optimizer might generate signals immediately at test start (bar 798)")
print("- Our first signal is at bar 814 (16 bars into test)")
print("- This gap suggests we might be missing 2-4 early signals")

# Pattern analysis
buy_bars = [bar for bar, type, _ in our_signals if type == 'BUY']
sell_bars = [bar for bar, type, _ in our_signals if type == 'SELL']

print(f"\nSignal pattern:")
print(f"BUY signals: {len(buy_bars)} at bars {buy_bars}")
print(f"SELL signals: {len(sell_bars)} at bars {sell_bars}")

# Check alternation
alternating = True
last_type = None
for _, type, _ in our_signals:
    if last_type and type == last_type:
        alternating = False
        break
    last_type = type
        
print(f"Signals alternate properly: {alternating}")

print("\nHypothesis for missing 4 signals:")
print("1. Missing early signals around bars 798-813")
print("2. Different MA calculation or rounding")
print("3. Different initialization state at test start")