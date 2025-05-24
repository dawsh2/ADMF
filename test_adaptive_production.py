#!/usr/bin/env python3
"""
Test script to verify dynamic parameter loading and regime switching
"""

import subprocess
import re
import time

def run_adaptive_test():
    """Run the adaptive production config and monitor parameter changes"""
    
    print("="*60)
    print("Testing Regime-Adaptive Production Mode")
    print("="*60)
    print("\nThis test will:")
    print("1. Use the test dataset (last 20% of data)")
    print("2. Load optimized parameters from regime_optimized_parameters.json")
    print("3. Dynamically switch parameters based on detected regimes")
    print("="*60)
    
    # Command to run
    cmd = "python3 main.py --config config/config_adaptive_production.yaml --log-level INFO --bars 5000"
    
    print(f"\nRunning: {cmd}")
    print("="*60)
    
    # Track regime changes and parameter updates
    regime_changes = []
    parameter_updates = []
    portfolio_values = []
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Monitor output
    for line in process.stdout:
        # Print all output
        print(line.rstrip())
        
        # Track regime changes
        if "REGIME CHANGE DETECTED" in line or "Current regime:" in line:
            regime_changes.append(line.strip())
            
        # Track parameter updates
        if "Applying parameters for regime" in line or "Updated parameters" in line:
            parameter_updates.append(line.strip())
            
        # Track portfolio values
        if "Final Portfolio Value:" in line or "get_final_portfolio_value" in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                portfolio_values.append(float(match.group(1)))
    
    process.wait()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nRegime Changes Detected: {len(regime_changes)}")
    if regime_changes:
        print("First 5 regime changes:")
        for change in regime_changes[:5]:
            print(f"  - {change}")
    
    print(f"\nParameter Updates: {len(parameter_updates)}")
    if parameter_updates:
        print("First 5 parameter updates:")
        for update in parameter_updates[:5]:
            print(f"  - {update}")
    
    if portfolio_values:
        print(f"\nFinal Portfolio Value: ${portfolio_values[-1]:,.2f}")
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if len(regime_changes) > 0 and len(parameter_updates) > 0:
        print("✓ SUCCESS: Regime detection and parameter switching are working!")
        print(f"  - Detected {len(regime_changes)} regime changes")
        print(f"  - Made {len(parameter_updates)} parameter updates")
    elif len(regime_changes) > 0:
        print("⚠ WARNING: Regimes detected but no parameter updates")
        print("  - Check if regime_optimized_parameters.json has parameters for detected regimes")
    else:
        print("✗ ERROR: No regime changes detected")
        print("  - Check if RegimeDetector is properly configured")
        print("  - Verify market conditions in test data trigger regime changes")
    
    return {
        'regime_changes': len(regime_changes),
        'parameter_updates': len(parameter_updates),
        'final_value': portfolio_values[-1] if portfolio_values else None
    }

if __name__ == "__main__":
    result = run_adaptive_test()
    
    print("\nTest complete!")
    print(f"Results: {result}")