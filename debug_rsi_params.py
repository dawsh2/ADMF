#!/usr/bin/env python3
"""
Debug RSI parameter application
"""

from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus

def debug_rsi_parameters():
    """Test if RSI parameters are being applied correctly"""
    
    print("Testing RSI Parameter Application")
    print("=" * 50)
    
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    
    # Test different parameter sets
    test_params = [
        {'oversold_threshold': 20.0, 'overbought_threshold': 60.0, 'weight': 0.8},
        {'oversold_threshold': 30.0, 'overbought_threshold': 70.0, 'weight': 0.4},
    ]
    
    for i, params in enumerate(test_params, 1):
        print(f"\nTest {i}: Parameters = {params}")
        print("-" * 30)
        
        # Create RSI indicator
        rsi_indicator = RSIIndicator(
            instance_name=f"TestRSI_{i}",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key=None,
            parameters={'period': 14}
        )
        
        # Create RSI rule with specific parameters
        rsi_rule = RSIRule(
            instance_name=f"TestRSIRule_{i}",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="test",  # This causes the warning
            rsi_indicator=rsi_indicator,
            parameters=params
        )
        
        # Check what parameters were actually applied
        print(f"Applied oversold_threshold: {rsi_rule.oversold_threshold}")
        print(f"Applied overbought_threshold: {rsi_rule.overbought_threshold}")
        print(f"Applied weight: {rsi_rule._weight}")
        
        # Verify get_parameters() returns what we set
        retrieved_params = rsi_rule.get_parameters()
        print(f"Retrieved parameters: {retrieved_params}")
        
        # Test parameter setting
        new_params = {'oversold_threshold': 25.0, 'overbought_threshold': 65.0, 'weight': 0.9}
        print(f"Setting new parameters: {new_params}")
        rsi_rule.set_parameters(new_params)
        
        print(f"After set_parameters:")
        print(f"  oversold_threshold: {rsi_rule.oversold_threshold}")
        print(f"  overbought_threshold: {rsi_rule.overbought_threshold}")
        print(f"  weight: {rsi_rule._weight}")
        
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS:")
    print("If parameters are working correctly, you should see:")
    print("- Applied parameters match what was passed in")
    print("- set_parameters() correctly updates the values")
    print("- The config warning is just about the test key, not the parameters")

if __name__ == "__main__":
    debug_rsi_parameters()