#!/usr/bin/env python3
"""
Final validation test - count signals exactly as optimizer would.
"""

import pandas as pd
import numpy as np

def calculate_ma_signals(data, test_start=798):
    """Calculate MA crossover signals exactly as the strategy would."""
    signals = []
    prices = []
    
    for i, row in data.iterrows():
        prices.append(row['Close'])
        
        # Need at least 20 bars for long MA
        if len(prices) >= 20:
            # Calculate current MAs
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-20:])
            
            # Need at least 21 bars to check previous MAs
            if len(prices) > 20:
                # Calculate previous MAs
                prev_short_ma = np.mean(prices[-11:-1])
                prev_long_ma = np.mean(prices[-21:-1])
                
                # Check for crossover
                if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                    # Bullish crossover
                    signals.append({
                        'bar': i,
                        'timestamp': row['timestamp'],
                        'type': 'BUY',
                        'price': row['Close'],
                        'short_ma': short_ma,
                        'long_ma': long_ma
                    })
                elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                    # Bearish crossover
                    signals.append({
                        'bar': i,
                        'timestamp': row['timestamp'],
                        'type': 'SELL',
                        'price': row['Close'],
                        'short_ma': short_ma,
                        'long_ma': long_ma
                    })
    
    return signals

def main():
    # Load data
    data = pd.read_csv('data/1000_1min.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Limit to 998 bars like optimizer
    data = data.iloc[:998]
    
    print(f"Total bars: {len(data)}")
    print(f"Test start: bar 798")
    print(f"Test bars: 798-997 (200 bars)")
    
    # Calculate all signals
    all_signals = calculate_ma_signals(data)
    
    # Split into training and test
    train_signals = [s for s in all_signals if s['bar'] < 798]
    test_signals = [s for s in all_signals if s['bar'] >= 798]
    
    print(f"\n=== SIGNAL SUMMARY ===")
    print(f"Total signals: {len(all_signals)}")
    print(f"Training signals: {len(train_signals)}")
    print(f"Test signals: {len(test_signals)}")
    
    print(f"\n=== TEST SIGNALS ===")
    for i, sig in enumerate(test_signals):
        print(f"{i+1}. Bar {sig['bar']}, {sig['timestamp']}, {sig['type']} at ${sig['price']:.2f}")
        print(f"   MA10={sig['short_ma']:.4f}, MA20={sig['long_ma']:.4f}")
    
    # Check for any issues
    print(f"\n=== ANALYSIS ===")
    print(f"Expected test signals: 17")
    print(f"Actual test signals: {len(test_signals)}")
    print(f"Difference: {17 - len(test_signals)}")
    
    if len(test_signals) < 17:
        print("\nPossible reasons for missing signals:")
        print("1. Different weight adjustment in optimizer (MA weight might not be 1.0)")
        print("2. Different indicator calculation (e.g., using EMA instead of SMA)")
        print("3. Additional signal filtering in optimizer")
        print("4. Different data preprocessing")
    
    # Double-check our first few test signals
    print("\n=== VERIFICATION ===")
    print("First test bar prices:")
    for i in range(798, min(803, len(data))):
        print(f"  Bar {i}: {data.iloc[i]['timestamp']}, Close=${data.iloc[i]['Close']:.2f}")
    
    return 0

if __name__ == "__main__":
    main()