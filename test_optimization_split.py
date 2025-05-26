#!/usr/bin/env python3
"""
Test script to verify train/test split optimization with CleanBacktestEngine.
Run this with your virtual environment activated.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the optimization with train/test split
if __name__ == "__main__":
    from src.core.application_launcher import ApplicationLauncher
    
    # Set up test arguments
    test_args = [
        '--config', 'config/config_optimization_train_test.yaml',
        '--bars', '100',
        '--optimize',
        '--log-level', 'INFO'
    ]
    
    print("Testing optimization with train/test split...")
    print(f"Arguments: {test_args}")
    
    launcher = ApplicationLauncher()
    result = launcher.launch(test_args)
    
    print(f"\nTest completed with result: {result}")