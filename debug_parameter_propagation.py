#!/usr/bin/env python3

"""
Debug script to test parameter propagation in EnsembleSignalStrategy
"""

import sys
import datetime
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append('/Users/daws/ADMF')

from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy

def test_parameter_propagation():
    """Test if parameters set via set_parameters() are preserved after setup()"""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create basic dependencies
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Create strategy instance
    strategy = EnsembleSignalStrategy(
        instance_name="test_strategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    # Test parameter set 1 - same as optimization run
    test_params_1 = {
        "short_window": 5,
        "long_window": 20,
        "rsi_indicator.period": 9,
        "rsi_rule.oversold_threshold": 20.0,
        "rsi_rule.overbought_threshold": 60.0,
    }
    
    print("=== Testing Parameter Set 1 ===")
    print(f"Setting parameters: {test_params_1}")
    
    # Step 1: Set parameters (what optimizer does)
    success = strategy.set_parameters(test_params_1)
    print(f"set_parameters() returned: {success}")
    
    # Check parameters after set_parameters
    print("\nParameters after set_parameters():")
    print(f"  MA: short_window={strategy._short_window}, long_window={strategy._long_window}")
    print(f"  RSI indicator period: {strategy.rsi_indicator.period}")
    print(f"  RSI rule oversold: {strategy.rsi_rule.oversold_threshold}")
    print(f"  RSI rule overbought: {strategy.rsi_rule.overbought_threshold}")
    
    # Step 2: Call setup (what optimizer does next)
    print("\nCalling setup()...")
    strategy.setup()
    
    # Check parameters after setup
    print("\nParameters after setup():")
    print(f"  MA: short_window={strategy._short_window}, long_window={strategy._long_window}")
    print(f"  RSI indicator period: {strategy.rsi_indicator.period}")
    print(f"  RSI rule oversold: {strategy.rsi_rule.oversold_threshold}")
    print(f"  RSI rule overbought: {strategy.rsi_rule.overbought_threshold}")
    
    # Test parameter set 2 - different RSI thresholds
    test_params_2 = {
        "short_window": 5,
        "long_window": 20,
        "rsi_indicator.period": 9,
        "rsi_rule.oversold_threshold": 30.0,  # Changed from 20.0
        "rsi_rule.overbought_threshold": 70.0,  # Changed from 60.0
    }
    
    print("\n=== Testing Parameter Set 2 ===")
    print(f"Setting parameters: {test_params_2}")
    
    # Step 1: Set parameters (what optimizer does)
    success = strategy.set_parameters(test_params_2)
    print(f"set_parameters() returned: {success}")
    
    # Check parameters after set_parameters
    print("\nParameters after set_parameters():")
    print(f"  MA: short_window={strategy._short_window}, long_window={strategy._long_window}")
    print(f"  RSI indicator period: {strategy.rsi_indicator.period}")
    print(f"  RSI rule oversold: {strategy.rsi_rule.oversold_threshold}")
    print(f"  RSI rule overbought: {strategy.rsi_rule.overbought_threshold}")
    
    # Step 2: Call setup (what optimizer does next)
    print("\nCalling setup()...")
    strategy.setup()
    
    # Check parameters after setup
    print("\nParameters after setup():")
    print(f"  MA: short_window={strategy._short_window}, long_window={strategy._long_window}")
    print(f"  RSI indicator period: {strategy.rsi_indicator.period}")
    print(f"  RSI rule oversold: {strategy.rsi_rule.oversold_threshold}")
    print(f"  RSI rule overbought: {strategy.rsi_rule.overbought_threshold}")
    
    # Check if the two parameter sets produce different internal states
    print("\n=== Comparison ===")
    if (strategy.rsi_rule.oversold_threshold == 30.0 and 
        strategy.rsi_rule.overbought_threshold == 70.0):
        print("✅ Parameters from set 2 are correctly applied")
    else:
        print("❌ Parameters not correctly applied - still using old values")
        
    print("\nParameter propagation test complete.")

if __name__ == "__main__":
    test_parameter_propagation()