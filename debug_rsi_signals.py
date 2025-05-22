#!/usr/bin/env python3

"""
Debug script to test if RSI signals are being generated during optimization
"""

import pandas as pd
import sys
sys.path.append('/Users/daws/ADMF')

from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule

def test_rsi_signal_generation():
    """Test if RSI generates signals with different parameter combinations"""
    
    # Load some SPY data
    data = pd.read_csv('/Users/daws/ADMF/data/SPY_1min.csv')
    prices = data['Close'].head(200)  # First 200 bars for testing
    
    # Test different RSI parameter combinations from your optimization
    test_combinations = [
        {"period": 9, "oversold_threshold": 20.0, "overbought_threshold": 60.0},
        {"period": 9, "oversold_threshold": 20.0, "overbought_threshold": 70.0},
        {"period": 9, "oversold_threshold": 30.0, "overbought_threshold": 60.0},
        {"period": 9, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
        {"period": 14, "oversold_threshold": 20.0, "overbought_threshold": 60.0},
        {"period": 21, "oversold_threshold": 30.0, "overbought_threshold": 70.0},
    ]
    
    print("Testing RSI signal generation with different parameter combinations:")
    print("=" * 80)
    
    for i, params in enumerate(test_combinations):
        print(f"\nTest {i+1}: Period={params['period']}, OS={params['oversold_threshold']}, OB={params['overbought_threshold']}")
        
        # Create RSI indicator and rule
        rsi_indicator = RSIIndicator(
            instance_name=f"test_rsi_{i}",
            config_loader=None,
            event_bus=None,
            component_config_key=None,
            parameters={"period": params["period"]}
        )
        
        rsi_rule = RSIRule(
            instance_name=f"test_rule_{i}",
            config_loader=None,
            event_bus=None,
            component_config_key=None,
            rsi_indicator=rsi_indicator,
            parameters={
                "oversold_threshold": params["oversold_threshold"],
                "overbought_threshold": params["overbought_threshold"],
                "weight": 1.0
            }
        )
        
        # Process price data and count signals
        buy_signals = 0
        sell_signals = 0
        rsi_values = []
        
        for price in prices:
            # Update RSI
            rsi_value = rsi_indicator.update(price)
            if rsi_value is not None:
                rsi_values.append(rsi_value)
                
                # Check for signals
                triggered, strength, signal_type = rsi_rule.evaluate()
                if triggered:
                    if signal_type == "BUY":
                        buy_signals += 1
                        print(f"  📈 BUY signal at price {price:.2f}, RSI {rsi_value:.2f}")
                    elif signal_type == "SELL":
                        sell_signals += 1
                        print(f"  📉 SELL signal at price {price:.2f}, RSI {rsi_value:.2f}")
        
        # Summary stats
        if rsi_values:
            min_rsi = min(rsi_values)
            max_rsi = max(rsi_values)
            avg_rsi = sum(rsi_values) / len(rsi_values)
            print(f"  RSI range: {min_rsi:.2f} - {max_rsi:.2f} (avg: {avg_rsi:.2f})")
        else:
            print("  No RSI values calculated")
            
        print(f"  Signals generated: {buy_signals} BUY, {sell_signals} SELL")
        
        # Check if RSI ever hits thresholds
        if rsi_values:
            hits_oversold = any(rsi <= params["oversold_threshold"] for rsi in rsi_values)
            hits_overbought = any(rsi >= params["overbought_threshold"] for rsi in rsi_values)
            print(f"  Threshold hits: Oversold={hits_oversold}, Overbought={hits_overbought}")
        
    print("\n" + "=" * 80)
    print("Analysis complete. If no signals are generated, that explains identical optimization results.")

if __name__ == "__main__":
    test_rsi_signal_generation()