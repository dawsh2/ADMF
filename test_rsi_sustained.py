#!/usr/bin/env python3
"""
Test the sustained RSI signals to verify they're working
"""

import pandas as pd
import numpy as np
from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus

def test_rsi_sustained_signals():
    """Test that sustained RSI signals are working correctly"""
    
    # Load some data
    df = pd.read_csv('data/SPY_1min.csv')
    prices = df['Close'].head(1000).values
    
    print(f"Testing RSI sustained signals with {len(prices)} prices")
    print("=" * 60)
    
    # Test different thresholds
    thresholds = [
        (20.0, 60.0),
        (20.0, 70.0), 
        (30.0, 60.0),
        (30.0, 70.0)
    ]
    
    for oversold, overbought in thresholds:
        print(f"\nTesting thresholds: {oversold}/{overbought}")
        print("-" * 40)
        
        # Create RSI components
        config_loader = SimpleConfigLoader("config/config.yaml")
        event_bus = EventBus()
        
        rsi_indicator = RSIIndicator(
            instance_name="TestRSI",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key=None,
            parameters={'period': 14}
        )
        
        rsi_rule = RSIRule(
            instance_name="TestRSIRule",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="test",
            rsi_indicator=rsi_indicator,
            parameters={
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'weight': 1.0
            }
        )
        
        # Setup components
        rsi_indicator.setup()
        rsi_rule.setup()
        rsi_indicator.start()
        rsi_rule.start()
        
        # Track signals
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # Feed prices and count signals
        for i, price in enumerate(prices):
            # Update RSI
            rsi_indicator.update(price)
            
            # Evaluate rule
            triggered, strength, signal_type = rsi_rule.evaluate({'close': price})
            
            if triggered:
                total_signals += 1
                if signal_type == "BUY":
                    buy_signals += 1
                elif signal_type == "SELL":
                    sell_signals += 1
                    
            # Show first few signals for verification
            if triggered and total_signals <= 3:
                rsi_val = rsi_indicator.value
                print(f"  Signal {total_signals}: {signal_type} at price {price:.2f}, RSI={rsi_val:.1f}")
        
        print(f"Total signals: {total_signals} (BUY: {buy_signals}, SELL: {sell_signals})")
        print(f"Signal frequency: {total_signals/len(prices)*100:.2f}%")
        
        # Show RSI range for context
        rsi_values = []
        rsi_indicator.reset_state()
        for price in prices:
            rsi_val = rsi_indicator.update(price)
            if rsi_val is not None:
                rsi_values.append(rsi_val)
                
        if rsi_values:
            rsi_min, rsi_max = min(rsi_values), max(rsi_values)
            oversold_count = sum(1 for r in rsi_values if r <= oversold)
            overbought_count = sum(1 for r in rsi_values if r >= overbought)
            print(f"RSI range: {rsi_min:.1f} - {rsi_max:.1f}")
            print(f"Bars ≤{oversold}: {oversold_count}, Bars ≥{overbought}: {overbought_count}")
            
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS:")
    print("If sustained signals are working, you should see:")
    print("- Much higher signal counts than before")
    print("- Signals roughly equal to bars touching thresholds")
    print("- Different signal counts for different thresholds")

if __name__ == "__main__":
    test_rsi_sustained_signals()