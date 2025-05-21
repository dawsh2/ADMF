#!/usr/bin/env python3
# test_logging_optimization.py
# A simple test script to verify our logging optimizations

import subprocess
import sys
import os

def run_test():
    """Run a test of the optimized logging system."""
    print("=== Testing Logging Optimization ===")
    
    # Run a standard backtest with limited bars
    print("\n1. Running standard backtest with 200 bars...")
    result = subprocess.run(
        ["python", "main.py", "--bars", "200"],
        capture_output=True,
        text=True
    )
    
    # Check for regime detection summary in the output
    if "=== Regime Detection Summary ===" in result.stdout:
        print("✅ Regime Detection Summary generated successfully")
    else:
        print("❌ Regime Detection Summary not found in output")
    
    # Check for periodic summaries
    if "Regime detection summary:" in result.stdout:
        print("✅ Periodic regime detection summaries found")
    else:
        print("❌ Periodic summaries not found in output")
    
    # Check for excessive OPTIMIZATION DEBUG messages
    if "OPTIMIZATION DEBUG" in result.stdout:
        print("⚠️ OPTIMIZATION DEBUG messages still present - these should be at DEBUG level")
    else:
        print("✅ No OPTIMIZATION DEBUG messages at INFO level")
    
    # Run with verbose logging enabled (modify config temporarily)
    print("\n2. Temporarily enabling verbose_logging in config...")
    
    # Read config
    with open("config/config.yaml", "r") as f:
        config_content = f.read()
    
    # Modify verbose_logging to true
    modified_config = config_content.replace("verbose_logging: false", "verbose_logging: true")
    
    # Save to a temporary file
    temp_config_path = "config/temp_verbose_config.yaml"
    with open(temp_config_path, "w") as f:
        f.write(modified_config)
    
    # Run with verbose config
    print("3. Running with verbose logging...")
    verbose_result = subprocess.run(
        ["python", "main.py", "--bars", "100", "--config", temp_config_path],
        capture_output=True,
        text=True
    )
    
    # Check for more detailed output
    detailed_output_count = verbose_result.stdout.count("RegimeDet")
    print(f"Detected {detailed_output_count} RegimeDet log lines with verbose logging")
    
    # Clean up temporary file
    os.remove(temp_config_path)
    print("\nTemporary config file removed")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    run_test()