#!/usr/bin/env python3
"""
Quick test to verify if regime switching is working by running a short adaptive test
with verbose logging enabled
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    print("=== TESTING REGIME SWITCHING WITH VERBOSE LOGGING ===")
    print("Running short adaptive test to verify parameter switching...")
    
    # Override sys.argv to run adaptive test only
    original_argv = sys.argv.copy()
    sys.argv = [
        "test_regime_switching.py",
        "--config", "config/config.yaml", 
        "--adaptive-test-only"  # Run only the adaptive test phase
    ]
    
    try:
        main()
    except SystemExit:
        pass  # Expected
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        sys.argv = original_argv
    
    print("\n=== Check the latest log file for regime switching evidence ===")
    print("Look for:")
    print("  - 'REGIME CHANGED:' messages")
    print("  - 'ADAPTIVE TEST: Applying regime-specific parameters' messages")  
    print("  - 'Updated MA weight to:' or 'Updated RSI' messages")