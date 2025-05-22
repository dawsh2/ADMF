#!/usr/bin/env python3

"""
Debug script to understand why 10000-bar optimization produces identical results
"""

import sys
import pandas as pd
sys.path.append('/Users/daws/ADMF')

def analyze_spy_data():
    """Analyze SPY data to understand the difference between first 1000 vs 10000 bars"""
    
    # Load SPY data
    data = pd.read_csv('/Users/daws/ADMF/data/SPY_1min.csv')
    
    print("SPY Data Analysis:")
    print("=" * 50)
    
    # Basic stats
    print(f"Total bars in dataset: {len(data)}")
    print(f"Date range: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    
    # First 1000 bars
    data_1000 = data.head(1000)
    print(f"\nFirst 1000 bars:")
    print(f"  Date range: {data_1000['timestamp'].iloc[0]} to {data_1000['timestamp'].iloc[-1]}")
    print(f"  Price range: {data_1000['Close'].min():.2f} to {data_1000['Close'].max():.2f}")
    print(f"  Price change: {data_1000['Close'].iloc[-1] - data_1000['Close'].iloc[0]:.2f}")
    
    # First 10000 bars
    data_10000 = data.head(10000)
    print(f"\nFirst 10000 bars:")
    print(f"  Date range: {data_10000['timestamp'].iloc[0]} to {data_10000['timestamp'].iloc[-1]}")
    print(f"  Price range: {data_10000['Close'].min():.2f} to {data_10000['Close'].max():.2f}")
    print(f"  Price change: {data_10000['Close'].iloc[-1] - data_10000['Close'].iloc[0]:.2f}")
    
    # Calculate simple MA crossovers for comparison
    for window_combo in [(5, 20), (10, 20)]:
        short_window, long_window = window_combo
        
        print(f"\nMA({short_window}, {long_window}) crossover analysis:")
        
        # 1000 bars
        data_1000_copy = data_1000.copy()
        data_1000_copy['MA_short'] = data_1000_copy['Close'].rolling(window=short_window).mean()
        data_1000_copy['MA_long'] = data_1000_copy['Close'].rolling(window=long_window).mean()
        data_1000_copy['Signal'] = 0
        data_1000_copy.loc[data_1000_copy['MA_short'] > data_1000_copy['MA_long'], 'Signal'] = 1
        data_1000_copy.loc[data_1000_copy['MA_short'] < data_1000_copy['MA_long'], 'Signal'] = -1
        
        # Count crossovers in 1000 bars
        signal_changes_1000 = (data_1000_copy['Signal'].diff() != 0).sum()
        
        # 10000 bars
        data_10000_copy = data_10000.copy()
        data_10000_copy['MA_short'] = data_10000_copy['Close'].rolling(window=short_window).mean()
        data_10000_copy['MA_long'] = data_10000_copy['Close'].rolling(window=long_window).mean()
        data_10000_copy['Signal'] = 0
        data_10000_copy.loc[data_10000_copy['MA_short'] > data_10000_copy['MA_long'], 'Signal'] = 1
        data_10000_copy.loc[data_10000_copy['MA_short'] < data_10000_copy['MA_long'], 'Signal'] = -1
        
        # Count crossovers in 10000 bars
        signal_changes_10000 = (data_10000_copy['Signal'].diff() != 0).sum()
        
        print(f"  1000 bars: {signal_changes_1000} signal changes")
        print(f"  10000 bars: {signal_changes_10000} signal changes")
        
        # Check if the pattern is the same
        first_1000_of_10000 = data_10000_copy.head(1000)['Signal']
        signals_match = data_1000_copy['Signal'].equals(first_1000_of_10000)
        print(f"  First 1000 signals match: {signals_match}")

if __name__ == "__main__":
    analyze_spy_data()