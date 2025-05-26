#!/usr/bin/env python3
"""
Test bootstrap system with minimal dependencies.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.core.bootstrap import Bootstrap, RunMode
from src.core.event_bus import EventBus
from src.core.container import Container


class MockConfig:
    """Mock config loader without yaml dependency."""
    def __init__(self, config_dict):
        self.config = config_dict
        
    def get_config(self):
        return self.config
        
    def get(self, key, default=None):
        return self.config.get(key, default)


def test_bootstrap():
    """Test the bootstrap system."""
    print("Testing Bootstrap System")
    print("=" * 50)
    
    # Create mock config
    config = MockConfig({
        'components': {
            'data_handler_csv': {
                'file_path': 'data/SPY_1min.csv',
                'symbol': 'SPY'
            },
            'basic_portfolio': {
                'initial_cash': 100000
            },
            'basic_risk_manager': {},
            'simulated_execution_handler': {},
            'regime_adaptive_strategy': {
                'regime_params_file': 'adaptive_regime_parameters.json'
            },
            'MyPrimaryRegimeDetector': {},
            'dummy_service': {}
        }
    })
    
    # Create bootstrap
    bootstrap = Bootstrap()
    
    try:
        # Create event bus and container manually
        event_bus = EventBus()
        container = Container()
        
        # Create minimal context
        context = {
            'config': config,
            'event_bus': event_bus,
            'container': container,
            'run_mode': RunMode.BACKTEST
        }
        
        print("\nTesting component creation...")
        
        # Test creating a simple component
        from src.core.dummy_component import DummyComponent
        dummy = DummyComponent('test_dummy', 'components.dummy_service')
        print(f"Created: {dummy} (state: {dummy.state})")
        
        # Initialize it
        init_context = {
            'config': config.get_config(),
            'event_bus': event_bus,
            'container': container
        }
        dummy.initialize(init_context)
        print(f"Initialized: {dummy} (state: {dummy.state})")
        
        # Start it
        dummy.start()
        print(f"Started: {dummy} (state: {dummy.state})")
        
        # Stop it
        dummy.stop()
        print(f"Stopped: {dummy} (state: {dummy.state})")
        
        # Dispose it
        dummy.dispose()
        print(f"Disposed: {dummy} (state: {dummy.state})")
        
        print("\n✅ Bootstrap test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(test_bootstrap())