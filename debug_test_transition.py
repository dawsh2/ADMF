#!/usr/bin/env python3
"""
Debug the exact state at test transition to understand missing signals.
"""

import pandas as pd

# Load data
data = pd.read_csv('data/1000_1min.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Get data around test transition
test_start = 798
context_bars = 30  # bars before and after

start = max(0, test_start - context_bars)
end = min(len(data), test_start + context_bars)

print(f"Data around test transition (bar {test_start}):")
print(f"Showing bars {start} to {end}")
print("\n")

# Calculate MAs manually
prices = []
for i in range(start, end):
    row = data.iloc[i]
    prices.append(row['Close'])
    
    # Calculate MAs if we have enough data
    if len(prices) >= 20:
        short_ma = sum(prices[-10:]) / 10
        long_ma = sum(prices[-20:]) / 20
        
        # Check for crossover
        if len(prices) > 20:
            prev_short = sum(prices[-11:-1]) / 10
            prev_long = sum(prices[-21:-1]) / 20
            
            # Detect crossover
            crossover = None
            if prev_short <= prev_long and short_ma > long_ma:
                crossover = "BULLISH"
            elif prev_short >= prev_long and short_ma < long_ma:
                crossover = "BEARISH"
        else:
            prev_short = None
            prev_long = None
            crossover = None
    else:
        short_ma = None
        long_ma = None
        prev_short = None
        prev_long = None
        crossover = None
    
    # Mark test transition
    marker = " <-- TEST START" if i == test_start else ""
    
    # Print bar info
    print(f"Bar {i:3d}: {row['timestamp']} Close={row['Close']:.2f}")
    if short_ma and long_ma:
        print(f"         MA10={short_ma:.4f}, MA20={long_ma:.4f}, Diff={short_ma-long_ma:+.4f}")
        if prev_short and prev_long:
            print(f"         Prev: MA10={prev_short:.4f}, MA20={prev_long:.4f}, Diff={prev_short-prev_long:+.4f}")
        if crossover:
            print(f"         >>> {crossover} CROSSOVER! <<<")
    print(f"{marker}")
    
    # Extra spacing for important bars
    if i == test_start or crossover:
        print()

# Focus on test period crossovers
print("\n=== ANALYZING TEST PERIOD CROSSOVERS ===")
print("Looking for crossovers in bars 798-820...")

# Reset and process just test start area
prices = list(data['Close'].iloc[:test_start])  # Pre-load with training data

for i in range(test_start, min(len(data), test_start + 23)):
    row = data.iloc[i]
    prices.append(row['Close'])
    
    short_ma = sum(prices[-10:]) / 10
    long_ma = sum(prices[-20:]) / 20
    prev_short = sum(prices[-11:-1]) / 10
    prev_long = sum(prices[-21:-1]) / 20
    
    # Check crossover
    if (prev_short <= prev_long and short_ma > long_ma) or \
       (prev_short >= prev_long and short_ma < long_ma):
        signal_type = "BUY" if short_ma > long_ma else "SELL"
        print(f"Bar {i}: {signal_type} signal at ${row['Close']:.2f}")
        print(f"  MA10: {prev_short:.4f} -> {short_ma:.4f}")
        print(f"  MA20: {prev_long:.4f} -> {long_ma:.4f}")
        print()