#!/usr/bin/env python3
"""
Compare data usage between optimization mode and production mode
"""

import subprocess
import sys
import time
import re
from datetime import datetime

def run_command(cmd, description):
    """Run a command and capture relevant output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    # Create a unique log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/compare_{description.replace(' ', '_')}_{timestamp}.log"
    
    # Run the command
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Collect output
    output_lines = []
    split_info = None
    first_bar_timestamp = None
    dataset_switches = []
    
    for line in process.stdout:
        output_lines.append(line)
        print(line.rstrip())
        
        # Capture split information
        if "SPLIT_DEBUG:" in line:
            split_info = line.strip()
            
        # Capture first BAR event to see what data is being used
        if "Publishing event: Event(type=BAR" in line and not first_bar_timestamp:
            match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
            if match:
                first_bar_timestamp = match.group(1)
                
        # Capture dataset switches
        if "SWITCHING TO" in line and "DATASET" in line:
            dataset_switches.append(line.strip())
    
    process.wait()
    duration = time.time() - start_time
    
    # Save full output to log file
    with open(log_file, 'w') as f:
        f.writelines(output_lines)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY for: {description}")
    print(f"{'='*60}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Log saved to: {log_file}")
    if split_info:
        print(f"Split info: {split_info}")
    if first_bar_timestamp:
        print(f"First BAR timestamp: {first_bar_timestamp}")
    if dataset_switches:
        print(f"Dataset switches: {len(dataset_switches)}")
        for switch in dataset_switches[:3]:  # Show first 3
            print(f"  - {switch}")
    print(f"{'='*60}\n")
    
    return {
        'description': description,
        'duration': duration,
        'log_file': log_file,
        'split_info': split_info,
        'first_bar_timestamp': first_bar_timestamp,
        'dataset_switches': dataset_switches
    }

def main():
    print("Comparing optimization mode vs production mode data usage")
    print("="*60)
    
    # Test 1: Optimization mode with config.yaml
    print("\nTest 1: Optimization mode (should use TRAINING data)")
    result1 = run_command(
        "python3 main.py --config config/config.yaml --optimize --log-level WARNING --bars 1000",
        "optimization_mode"
    )
    
    # Wait a bit between runs
    time.sleep(2)
    
    # Test 2: Production mode with config_production.yaml
    print("\nTest 2: Production mode (should use ALL/TEST data)")
    result2 = run_command(
        "python3 main.py --config config/config_production.yaml --log-level WARNING --bars 1000",
        "production_mode"
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nOptimization mode:")
    print(f"  - First data point: {result1['first_bar_timestamp']}")
    print(f"  - Split info: {result1['split_info']}")
    print(f"  - Dataset switches: {len(result1['dataset_switches'])}")
    
    print(f"\nProduction mode:")
    print(f"  - First data point: {result2['first_bar_timestamp']}")
    print(f"  - Split info: {result2['split_info']}")
    print(f"  - Dataset switches: {len(result2['dataset_switches'])}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if result1['first_bar_timestamp'] and result2['first_bar_timestamp']:
        if result1['first_bar_timestamp'] < "2025":
            print("✓ Optimization mode correctly uses training data (2024)")
        else:
            print("✗ ERROR: Optimization mode is using test data (2025)!")
            
        if result2['first_bar_timestamp'] < "2025":
            print("✓ Production mode uses full/training data (2024)")
        else:
            print("✗ Production mode uses test data (2025)")
    
    print(f"\nFull logs saved to:")
    print(f"  - Optimization: {result1['log_file']}")
    print(f"  - Production: {result2['log_file']}")

if __name__ == "__main__":
    main()