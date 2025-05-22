#!/usr/bin/env python3
"""
Simple RSI signal frequency analysis without framework dependencies
"""

import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """Calculate RSI values"""
    prices = np.array(prices, dtype=float)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi_values = []
    
    # Initial SMA for first RSI calculation
    if len(gains) >= period:
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(prices)):
            if i == period:
                # First RSI value using SMA
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
            else:
                # Subsequent RSI values using EMA
                current_gain = gains[i-1]
                current_loss = losses[i-1]
                avg_gain = (avg_gain * (period - 1) + current_gain) / period
                avg_loss = (avg_loss * (period - 1) + current_loss) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
    
    return np.array(rsi_values)

def count_rsi_signals(prices, oversold=30.0, overbought=70.0):
    """Count RSI crossing signals"""
    rsi_values = calculate_rsi(prices)
    
    buy_signals = 0
    sell_signals = 0
    
    for i in range(1, len(rsi_values)):
        prev_rsi = rsi_values[i-1]
        curr_rsi = rsi_values[i]
        
        # BUY: RSI crosses above oversold threshold
        if prev_rsi <= oversold and curr_rsi > oversold:
            buy_signals += 1
            
        # SELL: RSI crosses below overbought threshold  
        if prev_rsi >= overbought and curr_rsi < overbought:
            sell_signals += 1
    
    return buy_signals, sell_signals, buy_signals + sell_signals

def analyze_bar_counts():
    """Analyze RSI signals for different bar counts"""
    
    # Load data
    df = pd.read_csv('data/SPY_1min.csv')
    print(f"Loaded {len(df)} total bars from SPY data")
    
    bar_counts = [500, 1000, 2000, 5000, 10000]
    oversold, overbought = 30.0, 70.0
    
    print(f"\nRSI Analysis with thresholds: {oversold}/{overbought}")
    print("=" * 80)
    print(f"{'Bars':<6} {'Train':<6} {'Test':<6} | {'Train Signals':<15} {'Test Signals':<15} {'Freq%':<6}")
    print("-" * 80)
    
    for bar_count in bar_counts:
        # Take first N bars
        subset = df.head(bar_count)
        
        # 80/20 train/test split
        train_size = int(len(subset) * 0.8)
        train_df = subset.iloc[:train_size]
        test_df = subset.iloc[train_size:]
        
        # Count signals in training data
        train_buy, train_sell, train_total = count_rsi_signals(train_df['Close'].values, oversold, overbought)
        
        # Count signals in test data  
        test_buy, test_sell, test_total = count_rsi_signals(test_df['Close'].values, oversold, overbought)
        
        # Calculate frequency
        train_freq = (train_total / len(train_df) * 100) if len(train_df) > 0 else 0
        test_freq = (test_total / len(test_df) * 100) if len(test_df) > 0 else 0
        
        print(f"{bar_count:<6} {len(train_df):<6} {len(test_df):<6} | {train_buy}B+{train_sell}S={train_total:<6} {test_buy}B+{test_sell}S={test_total:<6} {train_freq:<6.2f}")
        
        # Show RSI stats for training data
        if len(train_df) > 20:  # Need enough data for RSI
            rsi_vals = calculate_rsi(train_df['Close'].values)
            if len(rsi_vals) > 0:
                rsi_min, rsi_max = np.min(rsi_vals), np.max(rsi_vals)
                oversold_bars = np.sum(rsi_vals <= oversold)
                overbought_bars = np.sum(rsi_vals >= overbought)
                print(f"       RSI range: {rsi_min:.1f}-{rsi_max:.1f}, ≤{oversold}: {oversold_bars}, ≥{overbought}: {overbought_bars}")
    
    print("\n" + "=" * 80)
    print("🔍 ANALYSIS:")
    print("- With 1000 bars (800 train), you get very few RSI signals")
    print("- This makes optimization results highly variable")  
    print("- With 10000 bars (8000 train), you get consistent signal counts")
    print("- Recommendation: Use more bars or adjust RSI thresholds for optimization")
    
    # Test different thresholds
    print(f"\n📊 SENSITIVITY TEST - More Aggressive Thresholds (40/60):")
    print("-" * 60)
    
    oversold, overbought = 40.0, 60.0
    for bar_count in [1000, 10000]:
        subset = df.head(bar_count)
        train_size = int(len(subset) * 0.8)
        train_df = subset.iloc[:train_size]
        
        train_buy, train_sell, train_total = count_rsi_signals(train_df['Close'].values, oversold, overbought)
        train_freq = (train_total / len(train_df) * 100) if len(train_df) > 0 else 0
        
        print(f"{bar_count} bars: {train_total} signals ({train_freq:.2f}% frequency)")

if __name__ == "__main__":
    analyze_bar_counts()