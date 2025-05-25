#!/usr/bin/env python3
"""
Quick test of cold start fix - runs a single optimization iteration
"""

import sys
sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.strategy.optimization.engines.backtest_engine import BacktestEngine

def test_cold_start():
    """Test that regime detector properly resets between runs."""
    
    # Setup
    config_loader = SimpleConfigLoader('config/config_debug_comparison.yaml')
    event_bus = EventBus()
    container = Container()
    
    # Register components (simplified)
    from src.strategy.regime_detector import RegimeDetector
    container.register_type(
        "MyPrimaryRegimeDetector",
        RegimeDetector,
        True,  # Singleton
        constructor_kwargs={
            "instance_name": "TestRegimeDetector",
            "config_loader": config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.MyPrimaryRegimeDetector"
        }
    )
    
    # Create backtest engine
    engine = BacktestEngine(container, config_loader, event_bus)
    
    print("Testing component reset behavior...")
    print("="*60)
    
    # Get regime detector instance
    regime_detector = container.resolve("MyPrimaryRegimeDetector")
    
    # Check initial state
    print(f"Initial state: {regime_detector._current_classification}")
    print(f"Initial duration: {regime_detector._current_regime_duration}")
    
    # Simulate some classification to change state
    regime_detector._current_classification = "test_regime"
    regime_detector._current_regime_duration = 10
    
    print(f"\nAfter simulation:")
    print(f"Classification: {regime_detector._current_classification}")
    print(f"Duration: {regime_detector._current_regime_duration}")
    
    # Now test reset
    regime_detector.reset()
    
    print(f"\nAfter reset:")
    print(f"Classification: {regime_detector._current_classification}")
    print(f"Duration: {regime_detector._current_regime_duration}")
    
    if regime_detector._current_classification is None and regime_detector._current_regime_duration == 0:
        print("\n✅ Reset working correctly!")
    else:
        print("\n❌ Reset not working properly!")

if __name__ == "__main__":
    test_cold_start()