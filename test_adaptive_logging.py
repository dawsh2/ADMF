#!/usr/bin/env python3

"""
Test to debug adaptive test logging issue.
"""

import sys
import subprocess
import re

def test_adaptive_logging():
    """Test if adaptive test logging is working."""
    print("Testing adaptive test execution and logging...")
    
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--config", "config/config.yaml", 
            "--optimize", "--genetic-optimize", 
            "--bars", "200"
        ], capture_output=True, text=True, timeout=120)
        
        output = result.stdout + result.stderr
        lines = output.split('\n')
        
        # Look for specific log messages
        adaptive_messages = []
        genetic_messages = []
        error_messages = []
        
        for i, line in enumerate(lines):
            if 'adaptive' in line.lower():
                adaptive_messages.append((i, line.strip()))
            if 'genetic' in line.lower() and ('optimization' in line.lower() or 'complete' in line.lower()):
                genetic_messages.append((i, line.strip()))
            if 'error' in line.lower() or 'exception' in line.lower():
                error_messages.append((i, line.strip()))
        
        print(f"Found {len(genetic_messages)} genetic optimization messages")
        print(f"Found {len(adaptive_messages)} adaptive-related messages")  
        print(f"Found {len(error_messages)} error messages")
        
        if genetic_messages:
            print("\nLast few genetic optimization messages:")
            for line_no, msg in genetic_messages[-3:]:
                print(f"  Line {line_no}: {msg}")
        
        if adaptive_messages:
            print("\nAll adaptive-related messages:")
            for line_no, msg in adaptive_messages:
                print(f"  Line {line_no}: {msg}")
        
        if error_messages:
            print("\nError messages:")
            for line_no, msg in error_messages[-3:]:
                print(f"  Line {line_no}: {msg}")
        
        # Check if the program completed normally
        final_lines = lines[-10:]
        program_completed = any('================================================================================' in line for line in final_lines)
        print(f"\nProgram completed normally: {program_completed}")
        
        # Look for the specific adaptive test section
        has_adaptive_results = any('ADAPTIVE GA ENSEMBLE STRATEGY TEST RESULTS' in line for line in lines)
        print(f"Has adaptive test results section: {has_adaptive_results}")
        
        return has_adaptive_results
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_adaptive_logging()
    if not success:
        print("\n🔧 Investigation needed: Adaptive test section not appearing")