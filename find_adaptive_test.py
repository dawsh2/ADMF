#!/usr/bin/env python3
"""
Find where the adaptive test begins in the optimization log
"""

import sys

def find_adaptive_test_start(filename):
    """Find the line number where adaptive test starts"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    print(f"Total lines in file: {len(lines)}")
    
    # Look for adaptive test markers
    adaptive_markers = [
        "ENABLING ADAPTIVE MODE",
        "Running Regime-Adaptive Strategy Test",
        "Running regime-adaptive test",
        "ADAPTIVE GA ENSEMBLE STRATEGY TEST",
        "=== Running Regime-Adaptive Strategy Test on Test Set ===",
        "_run_regime_adaptive_test"
    ]
    
    found_markers = []
    
    for i, line in enumerate(lines):
        for marker in adaptive_markers:
            if marker in line:
                found_markers.append((i+1, marker, line.strip()))
                
    print(f"\nFound {len(found_markers)} adaptive test markers:")
    for line_num, marker, content in found_markers[-10:]:  # Show last 10
        print(f"  Line {line_num}: {marker}")
        print(f"    Content: {content[:100]}...")
        
    # Look for the last major section before results
    if found_markers:
        last_marker_line = found_markers[-1][0]
        print(f"\nAdaptive test likely starts around line {last_marker_line}")
        print(f"That's {len(lines) - last_marker_line} lines from the end")
        
        # Count signals after this point
        signals_after = 0
        for i in range(last_marker_line, len(lines)):
            if i < len(lines) and ("Publishing event: Event(type=SIGNAL" in lines[i] or "Publishing SIGNAL event" in lines[i]):
                signals_after += 1
                
        print(f"Signals after adaptive test start: {signals_after}")
        
    return found_markers

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_adaptive_test.py <optimization_log_file>")
        sys.exit(1)
        
    find_adaptive_test_start(sys.argv[1])