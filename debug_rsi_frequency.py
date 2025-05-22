#!/usr/bin/env python3
"""
Debug script to analyze RSI signal frequency with different bar counts
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus

def analyze_rsi_signals(data_file, bar_counts, oversold_threshold=30.0, overbought_threshold=70.0):
    """Analyze RSI signal frequency for different bar counts"""
    
    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} total bars from {data_file}")
    print(f"RSI thresholds: oversold={oversold_threshold}, overbought={overbought_threshold}")
    print("\n" + "="*80)
    
    results = {}
    
    for bar_count in bar_counts:
        print(f"\nAnalyzing with {bar_count} bars:")
        print("-" * 40)
        
        # Take first bar_count bars
        subset_df = df.head(bar_count).copy()
        
        # Calculate train/test split (80/20)
        train_size = int(len(subset_df) * 0.8)
        train_df = subset_df.iloc[:train_size]
        test_df = subset_df.iloc[train_size:]
        
        print(f"Train bars: {len(train_df)}, Test bars: {len(test_df)}")
        
        # Analyze signals in training data
        train_signals = count_rsi_signals(train_df['Close'].values, oversold_threshold, overbought_threshold)
        test_signals = count_rsi_signals(test_df['Close'].values, oversold_threshold, overbought_threshold)
        
        results[bar_count] = {
            'train_bars': len(train_df),
            'test_bars': len(test_df),
            'train_signals': train_signals,
            'test_signals': test_signals
        }
        
        print(f"Training RSI signals: {train_signals['total']} total")
        print(f"  - BUY signals: {train_signals['buy']}")
        print(f"  - SELL signals: {train_signals['sell']}")
        print(f"Test RSI signals: {test_signals['total']} total")
        print(f"  - BUY signals: {test_signals['buy']}")
        print(f"  - SELL signals: {test_signals['sell']}")
        
        # Calculate signal frequency
        train_freq = train_signals['total'] / len(train_df) * 100 if len(train_df) > 0 else 0
        test_freq = test_signals['total'] / len(test_df) * 100 if len(test_df) > 0 else 0
        print(f"Signal frequency: Train={train_freq:.2f}%, Test={test_freq:.2f}%")
        
        # Show RSI range in training data for context
        rsi_values = calculate_rsi_values(train_df['Close'].values)
        rsi_min, rsi_max = np.min(rsi_values), np.max(rsi_values)
        oversold_touches = np.sum(rsi_values <= oversold_threshold)
        overbought_touches = np.sum(rsi_values >= overbought_threshold)
        
        print(f"Training RSI range: {rsi_min:.1f} - {rsi_max:.1f}")
        print(f"Bars touching oversold ({oversold_threshold}): {oversold_touches}")  
        print(f"Bars touching overbought ({overbought_threshold}): {overbought_touches}")
        
    return results

def calculate_rsi_values(prices, period=14):
    """Calculate RSI values for a price series"""
    prices = np.array(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi_values = []
    
    for i in range(period, len(prices)):
        if i == period:
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
        else:
            # Smoothed averages
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return np.array(rsi_values)

def count_rsi_signals(prices, oversold_threshold, overbought_threshold, period=14):
    """Count RSI crossing signals in a price series"""
    rsi_values = calculate_rsi_values(prices, period)
    
    buy_signals = 0
    sell_signals = 0
    
    for i in range(1, len(rsi_values)):
        prev_rsi = rsi_values[i-1]
        curr_rsi = rsi_values[i]
        
        # BUY signal: RSI crosses up from oversold
        if prev_rsi <= oversold_threshold and curr_rsi > oversold_threshold:
            buy_signals += 1
            
        # SELL signal: RSI crosses down from overbought  
        if prev_rsi >= overbought_threshold and curr_rsi < overbought_threshold:
            sell_signals += 1
    
    return {
        'buy': buy_signals,
        'sell': sell_signals,
        'total': buy_signals + sell_signals
    }

def main():
    print("RSI Signal Frequency Analysis")
    print("="*80)
    
    # Test different bar counts
    bar_counts = [500, 1000, 2000, 5000, 10000]
    data_file = "data/SPY_1min.csv"
    
    # Test with default RSI thresholds
    print("\n📊 DEFAULT RSI THRESHOLDS (30/70)")
    results_default = analyze_rsi_signals(data_file, bar_counts, 30.0, 70.0)
    
    # Test with more sensitive thresholds  
    print("\n\n📊 SENSITIVE RSI THRESHOLDS (40/60)")
    results_sensitive = analyze_rsi_signals(data_file, bar_counts, 40.0, 60.0)
    
    # Summary comparison
    print("\n\n" + "="*80)
    print("SUMMARY: Signal Frequency by Bar Count")
    print("="*80)
    print(f"{'Bars':<8} {'Train':<8} {'Signals':<8} {'Freq%':<8} | {'Test':<8} {'Freq%':<8}")
    print("-" * 60)
    
    for bar_count in bar_counts:
        train_bars = results_default[bar_count]['train_bars']
        train_signals = results_default[bar_count]['train_signals']['total']
        train_freq = train_signals / train_bars * 100 if train_bars > 0 else 0
        
        test_bars = results_default[bar_count]['test_bars'] 
        test_signals = results_default[bar_count]['test_signals']['total']
        test_freq = test_signals / test_bars * 100 if test_bars > 0 else 0
        
        print(f"{bar_count:<8} {train_bars:<8} {train_signals:<8} {train_freq:<8.2f} | {test_bars:<8} {test_freq:<8.2f}")
    
    print("\n🔍 DIAGNOSIS:")
    print("- Small datasets (1000 bars) may have 0-3 RSI signals in training")  
    print("- This causes high variance in optimization results")
    print("- Large datasets (10000 bars) have 20+ RSI signals for stable optimization")
    print("- Consider using more sensitive RSI thresholds for small datasets")

if __name__ == "__main__":
    main()