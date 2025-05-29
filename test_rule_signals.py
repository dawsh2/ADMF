#!/usr/bin/env python3
"""
Test if rules are generating any signals with the current parameters.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def test_ma_crossover_params():
    """Test MA crossover with regime parameters."""
    print("\nTesting MA Crossover parameters...")
    
    # Load parameters
    with open('test_regime_parameters.json', 'r') as f:
        params = json.load(f)
    
    # Test for each regime
    for regime in ['default', 'trending_up', 'trending_down']:
        regime_params = params['regimes'][regime]
        fast_period = regime_params.get('strategy_ma_crossover.fast_ma.lookback_period', 10)
        slow_period = regime_params.get('strategy_ma_crossover.slow_ma.lookback_period', 20)
        min_sep = regime_params.get('strategy_ma_crossover.min_separation', 0.001)
        
        print(f"\n{regime}: fast={fast_period}, slow={slow_period}, min_sep={min_sep}")
        
        # Generate some test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        # Calculate MAs
        fast_ma = pd.Series(prices).rolling(fast_period).mean()
        slow_ma = pd.Series(prices).rolling(slow_period).mean()
        
        # Check for crossovers
        crossovers = 0
        for i in range(slow_period + 1, len(prices)):
            if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                # Check min separation
                separation = abs(fast_ma.iloc[i] - slow_ma.iloc[i]) / slow_ma.iloc[i]
                if separation >= min_sep:
                    crossovers += 1
                    
        print(f"  Potential crossovers: {crossovers}")

def test_rsi_params():
    """Test RSI with regime parameters."""
    print("\n\nTesting RSI parameters...")
    
    # Load parameters
    with open('test_regime_parameters.json', 'r') as f:
        params = json.load(f)
    
    # Test for each regime  
    for regime in ['default', 'trending_up', 'trending_down']:
        regime_params = params['regimes'][regime]
        period = regime_params.get('strategy_rsi_rule.rsi_indicator.lookback_period', 14)
        oversold = regime_params.get('strategy_rsi_rule.oversold_threshold', 30)
        overbought = regime_params.get('strategy_rsi_rule.overbought_threshold', 70)
        
        print(f"\n{regime}: period={period}, oversold={oversold}, overbought={overbought}")
        
        # Generate test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        # Simple RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Count signals
        oversold_signals = (rsi < oversold).sum()
        overbought_signals = (rsi > overbought).sum()
        
        print(f"  Oversold signals: {oversold_signals}")
        print(f"  Overbought signals: {overbought_signals}")

def check_actual_data():
    """Check with actual SPY data."""
    print("\n\nChecking with actual SPY data...")
    
    try:
        # Load SPY data
        df = pd.read_csv('data/SPY_1min.csv')
        df = df.tail(200)  # Last 200 bars like test set
        
        print(f"Data shape: {df.shape}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"Price volatility: {df['close'].pct_change().std():.4f}")
        
        # Calculate indicators
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_40'] = df['close'].rolling(40).mean()
        
        # Count potential crossovers
        df['cross_10_20'] = ((df['ma_10'].shift(1) <= df['ma_20'].shift(1)) & 
                             (df['ma_10'] > df['ma_20'])).astype(int)
        df['cross_10_40'] = ((df['ma_10'].shift(1) <= df['ma_40'].shift(1)) & 
                             (df['ma_10'] > df['ma_40'])).astype(int)
                             
        print(f"\nMA(10) x MA(20) crossovers: {df['cross_10_20'].sum()}")
        print(f"MA(10) x MA(40) crossovers: {df['cross_10_40'].sum()}")
        
        # Check volatility
        print(f"\nFirst valid MA values at index {df['ma_40'].first_valid_index()}:")
        first_valid = df.loc[df['ma_40'].first_valid_index():]
        print(f"Bars with valid MA(40): {len(first_valid)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")

def main():
    print("="*80)
    print("TESTING RULE SIGNAL GENERATION")
    print("="*80)
    
    test_ma_crossover_params()
    test_rsi_params()
    check_actual_data()
    
    print("\n\nCONCLUSION:")
    print("If we're seeing 0 trades, possible reasons:")
    print("1. Indicator warmup period consuming too many bars (MA(40) needs 40 bars)")
    print("2. min_separation filter blocking MA crossover signals")
    print("3. All rules voting in opposite directions, canceling out")
    print("4. Position sizing or risk management blocking trades")

if __name__ == "__main__":
    main()