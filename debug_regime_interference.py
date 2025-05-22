#!/usr/bin/env python3

"""
Debug script to check if regime-specific parameter loading is interfering with genetic optimization.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy

def debug_regime_interference():
    """Debug if regime parameter loading interferes with weight setting"""
    
    print("=== Debugging Regime Interference ===")
    
    # Setup infrastructure
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Register basic components
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create strategy
    strategy = EnsembleSignalStrategy(
        instance_name="TestStrategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    print("\n1. Initial Strategy State:")
    initial_params = strategy.get_parameters()
    print(f"Initial parameters: {initial_params}")
    
    print("\n2. Setting extreme weights manually:")
    test_weights = {
        "ma_rule.weight": 0.95,
        "rsi_rule.weight": 0.05,
        "short_window": 10,
        "long_window": 20
    }
    
    success = strategy.set_parameters(test_weights)
    print(f"Set parameters success: {success}")
    print(f"Parameters after manual set: {strategy.get_parameters()}")
    
    # Check internal weights
    ma_weight = getattr(strategy, '_ma_weight', 'NOT_FOUND')
    rsi_weight = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
    print(f"Internal weights: MA={ma_weight}, RSI={rsi_weight}")
    
    print("\n3. Simulating regime change (like what happens during optimization):")
    
    # Trigger a regime change event manually (simulate what happens during optimization)
    from src.core.event import Event, EventType
    import datetime
    
    regime_event = Event(EventType.CLASSIFICATION, {
        'classification': 'default',
        'timestamp': datetime.datetime.now(datetime.timezone.utc),
        'classifier_name': 'TestDetector'
    })
    
    print("Triggering regime change event...")
    strategy.on_classification_change(regime_event)
    
    print(f"Parameters after regime event: {strategy.get_parameters()}")
    
    # Check internal weights again
    ma_weight_after = getattr(strategy, '_ma_weight', 'NOT_FOUND')
    rsi_weight_after = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
    print(f"Internal weights after regime event: MA={ma_weight_after}, RSI={rsi_weight_after}")
    
    print("\n4. Analysis:")
    if ma_weight != ma_weight_after or rsi_weight != rsi_weight_after:
        print("❌ PROBLEM: Regime change event is overriding manually set weights!")
        print("This could explain why genetic algorithm weights don't stick during optimization")
    else:
        print("✅ SUCCESS: Regime change event does not override manually set weights")
    
    print("\n5. Testing with optimization environment simulation:")
    
    # Check if there are any optimization-related flags that might affect behavior
    optimization_mode = any(opt in sys.argv for opt in ['--optimize', '--genetic-optimize'])
    print(f"Optimization mode detected: {optimization_mode}")
    
    # Check if there's adaptive mode interference
    adaptive_mode = getattr(strategy, '_adaptive_mode_enabled', False)
    print(f"Adaptive mode enabled: {adaptive_mode}")
    
    # Check if there are regime parameters in memory
    regime_params = getattr(strategy, '_regime_best_parameters', {})
    print(f"Regime parameters in memory: {regime_params}")
    
    if regime_params:
        print("⚠️  WARNING: Strategy has regime parameters in memory that might override weights")

if __name__ == "__main__":
    debug_regime_interference()