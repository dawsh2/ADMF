#!/usr/bin/env python3
"""
Analyze the 16 signals from the optimizer test phase.
"""

# Optimizer signals from the log
optimizer_signals = [
    (523.4800, 1),   # BUY
    (523.3900, -1),  # SELL  
    (523.4700, 1),   # BUY
    (523.2000, -1),  # SELL
    (523.2300, 1),   # BUY
    (523.2500, -1),  # SELL
    (523.5000, 1),   # BUY
    (523.5000, -1),  # SELL
    (523.5600, 1),   # BUY
    (523.4250, -1),  # SELL
    (523.5600, 1),   # BUY
    (523.2490, -1),  # SELL
    (523.2000, 1),   # BUY
    (523.1450, -1),  # SELL
    (523.2200, 1),   # BUY
    (523.2440, -1),  # SELL
]

# Our validation signals (13 signals)
our_signals = [
    (814, 523.49, -1),  # SELL
    (818, 523.47, 1),   # BUY
    (825, 523.05, -1),  # SELL
    (843, 523.23, 1),   # BUY
    (880, 523.25, -1),  # SELL
    (892, 523.56, 1),   # BUY
    (911, 523.42, -1),  # SELL
    (926, 523.56, 1),   # BUY
    (939, 523.25, -1),  # SELL
    (982, 523.20, 1),   # BUY
    (994, 523.14, -1),  # SELL
    (995, 523.22, 1),   # BUY
    (997, 523.24, -1),  # SELL
]

print("=== OPTIMIZER VS VALIDATION COMPARISON ===")
print(f"Optimizer signals: {len(optimizer_signals)}")
print(f"Validation signals: {len(our_signals)}")
print(f"Difference: {len(optimizer_signals) - len(our_signals)}")

# Check for matching prices (with tolerance)
tolerance = 0.05
matches = []
optimizer_unmatched = list(range(len(optimizer_signals)))
our_unmatched = list(range(len(our_signals)))

for i, (opt_price, opt_type) in enumerate(optimizer_signals):
    for j, (bar, our_price, our_type) in enumerate(our_signals):
        if abs(opt_price - our_price) < tolerance and opt_type == our_type and j in our_unmatched:
            matches.append((i, j, opt_price, our_price))
            if i in optimizer_unmatched:
                optimizer_unmatched.remove(i)
            our_unmatched.remove(j)
            break

print(f"\nMatched signals: {len(matches)}")
print(f"Optimizer unmatched: {len(optimizer_unmatched)}")
print(f"Validation unmatched: {len(our_unmatched)}")

print("\n=== UNMATCHED OPTIMIZER SIGNALS ===")
for i in optimizer_unmatched:
    price, signal_type = optimizer_signals[i]
    print(f"  {i+1}. {'BUY' if signal_type == 1 else 'SELL'} at ${price:.2f}")

print("\n=== UNMATCHED VALIDATION SIGNALS ===")  
for j in our_unmatched:
    bar, price, signal_type = our_signals[j]
    print(f"  Bar {bar}: {'BUY' if signal_type == 1 else 'SELL'} at ${price:.2f}")

# Analyze the pattern
print("\n=== ANALYSIS ===")
print("The optimizer generates 3 additional signals:")
print("1. First signal at test start (13:46:00) - BUY at $523.48")
print("2. Early SELL at $523.39 (13:48:00)")
print("3. One extra signal somewhere in the sequence")
print("\nThese early signals suggest the optimizer's MA indicators")
print("have different values at test start than our validation.")