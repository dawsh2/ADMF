#!/usr/bin/env python3
"""
Minimal test to match optimizer behavior exactly.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy

def main():
    """Run minimal test matching optimizer."""
    # Load config
    config = SimpleConfigLoader('config/config.yaml')
    setup_logging(config, cmd_log_level='INFO')
    
    # Load data
    data = pd.read_csv('data/1000_1min.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Limit to 998 bars like optimizer
    data = data.iloc[:998]
    
    print(f"Total bars: {len(data)}")
    print(f"Test start bar: 798")
    print(f"Test start timestamp: {data.iloc[798]['timestamp']}")
    
    # Create strategy with MA-only mode
    # The key is to simulate the optimizer's behavior:
    # 1. Process training data (bars 0-797) 
    # 2. Process test data (bars 798-997)
    # 3. MA-only mode (RSI disabled)
    # 4. No adaptive mode
    
    # Initialize counters
    training_signals = 0
    test_signals = 0
    
    # Simulate MA crossover logic
    prices = []
    for i, row in data.iterrows():
        prices.append(row['Close'])
        
        # Need at least 20 bars for long MA
        if len(prices) >= 20:
            short_ma = sum(prices[-10:]) / 10
            long_ma = sum(prices[-20:]) / 20
            
            # Check for crossover
            if len(prices) > 20:
                prev_short_ma = sum(prices[-11:-1]) / 10
                prev_long_ma = sum(prices[-21:-1]) / 20
                
                # Bullish crossover
                if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                    if i < 798:
                        training_signals += 1
                    else:
                        test_signals += 1
                        print(f"Test signal {test_signals}: Bar {i}, {row['timestamp']}, BUY at {row['Close']:.2f}")
                        
                # Bearish crossover
                elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                    if i < 798:
                        training_signals += 1
                    else:
                        test_signals += 1
                        print(f"Test signal {test_signals}: Bar {i}, {row['timestamp']}, SELL at {row['Close']:.2f}")
    
    print(f"\nResults:")
    print(f"Training signals: {training_signals}")
    print(f"Test signals: {test_signals}")
    print(f"Expected test signals: 17")
    print(f"Match: {'YES' if test_signals == 17 else 'NO'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())