#!/usr/bin/env python3

"""
Test script to check if the adaptive test is being called after genetic optimization.
"""

import sys
import subprocess

def test_adaptive_call():
    """Test if adaptive test section appears in genetic optimization output."""
    print("Testing genetic optimization with adaptive test...")
    
    # Run the command and capture output
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--config", "config/config.yaml", 
            "--optimize", "--genetic-optimize", 
            "--bars", "300"
        ], capture_output=True, text=True, timeout=180)
        
        output = result.stdout + result.stderr
        
        # Check for key indicators
        has_genetic_section = "=== PER-REGIME GENETIC OPTIMIZATION ===" in output
        has_adaptive_section = "ADAPTIVE GA ENSEMBLE STRATEGY TEST RESULTS" in output
        has_adaptive_mode = "ADAPTIVE MODE STATUS" in output
        has_regime_weights = "Optimized weights for" in output
        
        print(f"✅ Has genetic optimization section: {has_genetic_section}")
        print(f"❓ Has adaptive test results section: {has_adaptive_section}")  
        print(f"✅ Has adaptive mode status: {has_adaptive_mode}")
        print(f"✅ Has regime weights summary: {has_regime_weights}")
        
        if not has_adaptive_section:
            print("\n🔍 Looking for adaptive test related messages...")
            adaptive_lines = [line for line in output.split('\n') if 'adaptive' in line.lower()]
            if adaptive_lines:
                print("Found adaptive-related lines:")
                for line in adaptive_lines[-5:]:  # Show last 5 adaptive-related lines
                    print(f"  {line}")
            else:
                print("No adaptive-related messages found")
                
            print("\n🔍 Looking for error messages...")
            error_lines = [line for line in output.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
            if error_lines:
                print("Found error-related lines:")
                for line in error_lines[-3:]:  # Show last 3 error lines
                    print(f"  {line}")
            else:
                print("No error messages found")
        
        return has_adaptive_section
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 180 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_adaptive_call()
    if success:
        print("\n✅ Adaptive test section found!")
    else:
        print("\n❌ Adaptive test section missing - needs investigation")