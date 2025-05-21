#!/usr/bin/env python3
# test_top_n_optimization.py
# Test script for top-N optimization feature

import subprocess
import sys
import os
import json
import argparse

def run_test():
    """Run a test of the top-N optimization feature."""
    parser = argparse.ArgumentParser(description="Test top-N optimization feature")
    parser.add_argument("--bars", type=int, default=100, help="Number of bars to process")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top performers to test")
    args = parser.parse_args()
    
    bars = args.bars
    top_n = args.top_n
    
    print(f"=== Testing Top-{top_n} Optimization with {bars} bars ===")
    
    # Create temporary config file with desired top_n value
    with open("config/config.yaml", "r") as f:
        config_content = f.read()
    
    # Set top_n_to_test value
    if "top_n_to_test:" in config_content:
        config_content = config_content.replace("top_n_to_test: 3", f"top_n_to_test: {top_n}")
    else:
        print("Warning: Could not find top_n_to_test in config.yaml. Make sure it's properly set.")
    
    # Save to a temporary file
    temp_config_path = "config/temp_optimization_config.yaml"
    with open(temp_config_path, "w") as f:
        f.write(config_content)
    
    # Run optimization with specified bars
    print(f"\nRunning optimization with {bars} bars and top-{top_n} testing...")
    result = subprocess.run(
        ["python", "main.py", "--bars", str(bars), "--optimize", "--config", temp_config_path],
        capture_output=True,
        text=True
    )
    
    # Check if optimization was successful
    if "Optimization results saved to regime_optimized_parameters.json" in result.stdout:
        print("✅ Optimization completed successfully")
    else:
        print("❌ Optimization failed")
        print("\nOutput:")
        print(result.stdout[-1000:])  # Show last part of output
        return
    
    # Check if multiple performers were tested
    top_n_pattern = f"Top {top_n} Performers on Test Set"
    if top_n_pattern in result.stdout:
        print(f"✅ Top-{top_n} performers were tested successfully")
        
        # Extract and display the results
        start_idx = result.stdout.find(top_n_pattern)
        end_idx = result.stdout.find("---", start_idx + len(top_n_pattern))
        if end_idx == -1:
            end_idx = len(result.stdout)
        
        top_n_results = result.stdout[start_idx:end_idx].strip()
        print("\nTop-N Results:")
        print(top_n_results)
    else:
        print(f"❌ Top-{top_n} testing not found in output")
    
    # Check memory optimization
    if "Clearing regime performance data to free memory" in result.stdout:
        print("✅ Memory optimization applied successfully")
    else:
        print("❌ Memory optimization not applied")
    
    # Clean up temporary file
    os.remove(temp_config_path)
    print("\nTemporary config file removed")
    
    # Check results file
    if os.path.exists("regime_optimized_parameters.json"):
        try:
            with open("regime_optimized_parameters.json", "r") as f:
                results = json.load(f)
            
            print("\nResults file content:")
            print(f"Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"Best overall metric: {results.get('overall_best_metric', {}).get('value', 'N/A')}")
            print(f"Regimes encountered: {results.get('regimes_encountered', [])}")
            
            # Count regime-specific parameters
            regimes_with_params = len(results.get("regime_best_parameters", {}))
            print(f"Regimes with optimized parameters: {regimes_with_params}")
            
        except json.JSONDecodeError:
            print("❌ Results file is not valid JSON")
    else:
        print("❌ Results file not found")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    run_test()