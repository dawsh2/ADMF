#!/usr/bin/env python3
"""
Pre-warm production indicators to match optimizer behavior.

The optimizer warms up indicators on training data before evaluating test data.
This script replicates that behavior for production runs.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

def prewarm_indicators(data_file='data/1000_1min.csv', split_ratio=0.8):
    """
    Pre-warm MA indicators on training data to match optimizer state.
    
    Returns the warmed-up indicator states that can be used to initialize
    production indicators.
    """
    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate split point
    split_idx = int(len(df) * split_ratio)
    train_data = df[:split_idx]
    test_data = df[split_idx:]
    
    print(f"Total bars: {len(df)}")
    print(f"Training bars: {len(train_data)} (0 to {split_idx-1})")
    print(f"Test bars: {len(test_data)} ({split_idx} to {len(df)-1})")
    print(f"Test period starts at: {test_data.iloc[0]['timestamp']}")
    
    # Initialize MA indicators
    short_window = 10
    long_window = 20
    
    # Warm up indicators on training data
    prices = train_data['Close'].values
    
    # Calculate final MA values after training period
    if len(prices) >= long_window:
        final_short_ma = np.mean(prices[-short_window:])
        final_long_ma = np.mean(prices[-long_window:])
        
        print(f"\nPre-warmed indicator states at end of training:")
        print(f"MA_short (last {short_window} prices): {final_short_ma:.4f}")
        print(f"MA_long (last {long_window} prices): {final_long_ma:.4f}")
        print(f"Last price in training: {prices[-1]:.4f}")
        
        # Show what happens at test start
        test_prices = test_data['Close'].values[:5]
        print(f"\nFirst 5 test prices: {test_prices}")
        
        # Calculate MAs for first test bar (like optimizer would)
        ma_short_prices = list(prices[-9:]) + [test_prices[0]]
        ma_long_prices = list(prices[-19:]) + [test_prices[0]]
        
        first_test_ma_short = np.mean(ma_short_prices)
        first_test_ma_long = np.mean(ma_long_prices)
        
        print(f"\nMA values at first test bar:")
        print(f"MA_short: {first_test_ma_short:.4f}")
        print(f"MA_long: {first_test_ma_long:.4f}")
        print(f"First test price: {test_prices[0]:.4f}")
        
        # Check if this would generate a signal
        if test_prices[0] > first_test_ma_short > first_test_ma_long:
            print("Would generate BUY signal at first test bar!")
        elif test_prices[0] < first_test_ma_short < first_test_ma_long:
            print("Would generate SELL signal at first test bar!")
        else:
            print("No signal at first test bar")
            
        return {
            'ma_short_buffer': list(prices[-9:]),
            'ma_long_buffer': list(prices[-19:]),
            'last_training_price': prices[-1],
            'split_index': split_idx
        }
    else:
        print("Not enough training data to warm up indicators!")
        return None

if __name__ == "__main__":
    print("PRODUCTION INDICATOR PRE-WARMING ANALYSIS")
    print("=" * 60)
    
    # Analyze with standard 80% split
    prewarm_indicators()