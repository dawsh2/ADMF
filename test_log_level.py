#!/usr/bin/env python3
# test_log_level.py - Test the log level command-line argument

import subprocess
import sys
import os
import argparse

def main():
    """Test different log levels."""
    parser = argparse.ArgumentParser(description="Test log level command-line argument")
    parser.add_argument("--bars", type=int, default=10, help="Number of bars to use for testing")
    args = parser.parse_args()
    
    bars = args.bars
    print(f"=== Testing Log Level Command-Line Argument with {bars} bars ===\n")
    
    # Test different log levels
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for level in levels:
        print(f"\nTesting with --log-level={level}")
        cmd = ["python", "main.py", "--bars", str(bars), "--log-level", level]
        
        # Run with output to a temp file
        temp_output = f"log_level_test_{level.lower()}.log"
        with open(temp_output, "w") as outfile:
            result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT)
            
        # Read and count lines by log level
        with open(temp_output, "r") as infile:
            stdout = infile.read()
        
        # Count lines by log level
        debug_count = stdout.count(" - DEBUG - ")
        info_count = stdout.count(" - INFO - ")
        warning_count = stdout.count(" - WARNING - ")
        error_count = stdout.count(" - ERROR - ")
        critical_count = stdout.count(" - CRITICAL - ")
        
        # Count total lines
        line_count = len(stdout.strip().split('\n'))
        
        print(f"Log line counts with --log-level={level}:")
        print(f"  DEBUG:    {debug_count}")
        print(f"  INFO:     {info_count}")
        print(f"  WARNING:  {warning_count}")
        print(f"  ERROR:    {error_count}")
        print(f"  CRITICAL: {critical_count}")
        print(f"  Total lines: {line_count}")
        print(f"  Output in: {temp_output}")
        
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()