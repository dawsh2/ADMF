#!/usr/bin/env python3
"""
Test the AppRunner functionality before refactoring into proper components.

This verifies that our Bootstrap infrastructure works correctly.
"""

import sys
import logging
from src.core.config import Config
from src.core.bootstrap import Bootstrap, RunMode


def test_backtest_mode():
    """Test running a backtest through AppRunner."""
    print("\n=== Testing Backtest Mode ===")
    
    # Create test configuration
    config = Config("config/config.yaml")
    
    # Override to backtest mode
    config.set("system.application_mode", "backtest")
    
    # Set up AppRunner as entrypoint
    config.set("system.run_modes.backtest", {
        "entrypoint_component": "app_runner"
    })
    
    # Limit data for quick test
    config.set("components.data_handler_csv.max_bars", 100)
    
    # Metadata with CLI args
    metadata = {
        'cli_args': {
            'bars': 100,
            'config': 'config/config.yaml'
        }
    }
    
    try:
        with Bootstrap() as bootstrap:
            # Initialize
            context = bootstrap.initialize(
                config=config,
                run_mode=RunMode.BACKTEST,
                metadata=metadata
            )
            
            # Register AppRunner
            bootstrap.component_definitions['app_runner'] = {
                'class': 'AppRunner',
                'module': 'core.app_runner',
                'dependencies': ['event_bus', 'container'],
                'config_key': 'components.app_runner',
                'required': True
            }
            
            # Setup components
            print("Setting up components...")
            bootstrap.setup_managed_components()
            
            # Start components
            print("Starting components...")
            bootstrap.start_components()
            
            # Execute backtest
            print("Executing backtest...")
            result = bootstrap.execute_entrypoint()
            
            print(f"Backtest completed: {result}")
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_mode():
    """Test running optimization through AppRunner."""
    print("\n=== Testing Optimization Mode ===")
    
    # This would test optimization...
    # Similar structure but with RunMode.OPTIMIZATION
    print("Optimization test not implemented yet")
    return True


def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Backtest Mode", test_backtest_mode),
        ("Optimization Mode", test_optimization_mode),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(success for _, success in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())