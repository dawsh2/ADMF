#!/usr/bin/env python3
"""Test script for RegimeAdaptiveEnsembleComposed strategy."""

import sys
sys.path.append('/Users/daws/ADMF')

from src.core.app_runner import AppRunner
import yaml

def test_ensemble_strategy():
    """Test the ensemble strategy with basic configuration."""
    
    # Load test configuration
    config_path = "/Users/daws/ADMF/config/test_ensemble_optimization.yaml"
    
    print(f"Loading configuration from {config_path}")
    
    # Create app runner
    runner = AppRunner(config_path)
    
    # Run with optimization enabled to test workflow
    runner.args.optimize = True
    runner.args.mode = "backtest"
    
    print("Starting optimization workflow test...")
    
    try:
        runner.run()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = test_ensemble_strategy()
    sys.exit(0 if success else 1)