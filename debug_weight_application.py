#!/usr/bin/env python3

"""
Test script to verify that weight parameters are actually affecting strategy behavior.
This will create a simple test to see if different weights produce different results.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy

def test_weight_application():
    """Test if different weights actually affect strategy parameters"""
    
    # Setup basic infrastructure
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Create strategy instance
    strategy = EnsembleSignalStrategy(
        instance_name="TestStrategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    print("=== Testing Weight Parameter Application ===")
    
    # Test 1: Check initial parameters
    print("\n1. Initial parameters:")
    initial_params = strategy.get_parameters()
    print(f"   Initial: {initial_params}")
    
    # Test 2: Set new weights (scenario 1)
    print("\n2. Setting weights to MA=0.8, RSI=0.2")
    test_params_1 = {
        "ma_rule.weight": 0.8,
        "rsi_rule.weight": 0.2
    }
    success_1 = strategy.set_parameters(test_params_1)
    print(f"   Success: {success_1}")
    
    # Check if parameters were applied
    params_after_1 = strategy.get_parameters()
    print(f"   After set: {params_after_1}")
    
    # Check internal weight attributes
    ma_weight_1 = getattr(strategy, '_ma_weight', 'NOT_FOUND')
    rsi_weight_1 = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
    print(f"   Internal weights: MA={ma_weight_1}, RSI={rsi_weight_1}")
    
    # Test 3: Set different weights (scenario 2)
    print("\n3. Setting weights to MA=0.3, RSI=0.7")
    test_params_2 = {
        "ma_rule.weight": 0.3,
        "rsi_rule.weight": 0.7
    }
    success_2 = strategy.set_parameters(test_params_2)
    print(f"   Success: {success_2}")
    
    # Check if parameters were applied
    params_after_2 = strategy.get_parameters()
    print(f"   After set: {params_after_2}")
    
    # Check internal weight attributes
    ma_weight_2 = getattr(strategy, '_ma_weight', 'NOT_FOUND')
    rsi_weight_2 = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
    print(f"   Internal weights: MA={ma_weight_2}, RSI={rsi_weight_2}")
    
    # Test 4: Verify parameters are different
    print("\n4. Verification:")
    weights_changed = (ma_weight_1 != ma_weight_2) and (rsi_weight_1 != rsi_weight_2)
    print(f"   Weights actually changed: {weights_changed}")
    print(f"   MA: {ma_weight_1} -> {ma_weight_2}")
    print(f"   RSI: {rsi_weight_1} -> {rsi_weight_2}")
    
    # Test 5: Check if RSI rule weight is also updated
    if hasattr(strategy, 'rsi_rule') and strategy.rsi_rule:
        rsi_rule_weight = getattr(strategy.rsi_rule, 'weight', 'NOT_FOUND')
        rsi_rule_weight_private = getattr(strategy.rsi_rule, '_weight', 'NOT_FOUND')
        print(f"   RSI Rule weight: {rsi_rule_weight}")
        print(f"   RSI Rule _weight: {rsi_rule_weight_private}")
    
    if weights_changed:
        print("\n✅ SUCCESS: Weight parameters are being applied correctly!")
        return True
    else:
        print("\n❌ FAILURE: Weight parameters are NOT being applied!")
        return False

if __name__ == "__main__":
    test_weight_application()