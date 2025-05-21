#!/usr/bin/env python3
"""
Analyze RSI behavior in SPY_1min.csv dataset to understand why RSI signals aren't being generated
during optimization with period=21.
"""

import pandas as pd
import numpy as np
from typing import List

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI for a series of prices."""
    if len(prices) < period + 1:
        return [None] * len(prices)
    
    rsi_values = [None] * period  # First 'period' values are None
    
    # Calculate initial average gain and loss
    price_changes = [prices[i] - prices[i-1] for i in range(1, period + 1)]
    gains = [max(0, change) for change in price_changes]
    losses = [max(0, -change) for change in price_changes]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    # Calculate first RSI value
    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))
    
    # Calculate subsequent RSI values using smoothed averages
    for i in range(period + 1, len(prices)):
        price_change = prices[i] - prices[i-1]
        gain = max(0, price_change)
        loss = max(0, -price_change)
        
        # Smoothed averages (Wilder's smoothing)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    
    return rsi_values

def analyze_rsi_signals(prices: List[float], period: int, oversold: float = 20.0, overbought: float = 60.0):
    """Analyze RSI signals for given parameters."""
    rsi_values = calculate_rsi(prices, period)
    
    # Remove None values for analysis
    valid_rsi = [rsi for rsi in rsi_values if rsi is not None]
    
    if not valid_rsi:
        return {
            'period': period,
            'total_bars': len(prices),
            'rsi_ready_after_bar': None,
            'valid_rsi_count': 0,
            'min_rsi': None,
            'max_rsi': None,
            'avg_rsi': None,
            'above_overbought_pct': 0,
            'below_oversold_pct': 0,
            'overbought_crossings': 0,
            'oversold_crossings': 0
        }
    
    rsi_ready_after_bar = period + 1  # RSI becomes ready after period+1 bars
    
    # Count threshold crossings
    overbought_crossings = 0
    oversold_crossings = 0
    
    for i in range(1, len(rsi_values)):
        if rsi_values[i-1] is not None and rsi_values[i] is not None:
            # Overbought crossing (RSI falls below overbought threshold)
            if rsi_values[i-1] >= overbought and rsi_values[i] < overbought:
                overbought_crossings += 1
            # Oversold crossing (RSI rises above oversold threshold)
            if rsi_values[i-1] <= oversold and rsi_values[i] > oversold:
                oversold_crossings += 1
    
    return {
        'period': period,
        'total_bars': len(prices),
        'rsi_ready_after_bar': rsi_ready_after_bar,
        'valid_rsi_count': len(valid_rsi),
        'min_rsi': min(valid_rsi),
        'max_rsi': max(valid_rsi),
        'avg_rsi': sum(valid_rsi) / len(valid_rsi),
        'above_overbought_pct': (sum(1 for rsi in valid_rsi if rsi >= overbought) / len(valid_rsi)) * 100,
        'below_oversold_pct': (sum(1 for rsi in valid_rsi if rsi <= oversold) / len(valid_rsi)) * 100,
        'overbought_crossings': overbought_crossings,
        'oversold_crossings': oversold_crossings
    }

def main():
    # Load the dataset
    print("Loading SPY_1min.csv dataset...")
    df = pd.read_csv('data/SPY_1min.csv')
    
    # Get first 1000 bars (matching optimization test)
    prices_1000 = df['Close'].iloc[:1000].tolist()
    
    # Also get first 50 bars for comparison
    prices_50 = df['Close'].iloc[:50].tolist()
    
    print(f"Dataset loaded: {len(df)} total bars")
    print(f"Analyzing first 1000 bars: {prices_1000[0]:.2f} to {prices_1000[-1]:.2f}")
    print(f"Analyzing first 50 bars: {prices_50[0]:.2f} to {prices_50[-1]:.2f}")
    print()
    
    # Test different scenarios matching optimization parameters
    scenarios = [
        ("50 bars, period=9", prices_50, 9, 20.0, 60.0),
        ("50 bars, period=21", prices_50, 21, 20.0, 60.0),
        ("1000 bars, period=9", prices_1000, 9, 20.0, 60.0),
        ("1000 bars, period=21", prices_1000, 21, 20.0, 60.0),
        ("1000 bars, period=9, threshold=70", prices_1000, 9, 20.0, 70.0),
        ("1000 bars, period=21, threshold=70", prices_1000, 21, 20.0, 70.0),
        ("1000 bars, period=21, threshold=30-60", prices_1000, 21, 30.0, 60.0),
        ("1000 bars, period=21, threshold=30-70", prices_1000, 21, 30.0, 70.0),
    ]
    
    print("=" * 80)
    print("RSI SIGNAL ANALYSIS")
    print("=" * 80)
    
    for scenario_name, prices, period, oversold, overbought in scenarios:
        result = analyze_rsi_signals(prices, period, oversold, overbought)
        
        print(f"\nScenario: {scenario_name}")
        print(f"  RSI Period: {result['period']}")
        print(f"  Total Bars: {result['total_bars']}")
        print(f"  RSI Ready After Bar: {result['rsi_ready_after_bar']}")
        print(f"  Valid RSI Values: {result['valid_rsi_count']}")
        
        if result['valid_rsi_count'] > 0:
            print(f"  RSI Range: {result['min_rsi']:.2f} to {result['max_rsi']:.2f}")
            print(f"  Average RSI: {result['avg_rsi']:.2f}")
            print(f"  Above Overbought ({overbought}): {result['above_overbought_pct']:.1f}%")
            print(f"  Below Oversold ({oversold}): {result['below_oversold_pct']:.1f}%")
            print(f"  Overbought Crossings (SELL signals): {result['overbought_crossings']}")
            print(f"  Oversold Crossings (BUY signals): {result['oversold_crossings']}")
            print(f"  Total Signal Opportunities: {result['overbought_crossings'] + result['oversold_crossings']}")
        else:
            print("  No valid RSI values calculated!")
        
        print("-" * 60)

if __name__ == "__main__":
    main()