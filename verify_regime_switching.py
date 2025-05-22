#!/usr/bin/env python3
"""
Quick test to verify regime parameter switching is working
"""

import json
import re
from pathlib import Path

def analyze_log_for_regime_switching():
    """
    Analyze the log file for evidence of regime switching
    """
    log_files = list(Path("logs").glob("admf_*.log"))
    if not log_files:
        print("No log files found in logs/ directory")
        return
    
    # Get the most recent log file
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"Analyzing log file: {latest_log}")
    
    regime_changes = []
    parameter_applications = []
    
    with open(latest_log, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Look for regime changes
            if "REGIME CHANGED:" in line:
                regime_changes.append((line_num, line.strip()))
            
            # Look for parameter applications
            if "ADAPTIVE TEST: Applying regime-specific parameters" in line:
                parameter_applications.append((line_num, line.strip()))
            
            # Look for parameter updates
            if "Updated MA weight to:" in line or "Updated RSI" in line:
                parameter_applications.append((line_num, line.strip()))
    
    print(f"\n=== REGIME SWITCHING ANALYSIS ===")
    print(f"Found {len(regime_changes)} regime changes")
    print(f"Found {len(parameter_applications)} parameter applications")
    
    if regime_changes:
        print(f"\nFirst few regime changes:")
        for i, (line_num, line) in enumerate(regime_changes[:5]):
            print(f"  Line {line_num}: {line}")
    
    if parameter_applications:
        print(f"\nFirst few parameter applications:")
        for i, (line_num, line) in enumerate(parameter_applications[:5]):
            print(f"  Line {line_num}: {line}")
    
    # Check if regime changes are followed by parameter applications
    if regime_changes and parameter_applications:
        print(f"\n=== TIMING ANALYSIS ===")
        print("Checking if regime changes trigger parameter updates...")
        
        for regime_line, regime_content in regime_changes[:3]:
            # Look for parameter applications within 10 lines of regime change
            nearby_params = [
                (param_line, param_content) for param_line, param_content in parameter_applications
                if abs(param_line - regime_line) <= 10
            ]
            if nearby_params:
                print(f"✅ Regime change at line {regime_line} had parameter update nearby")
            else:
                print(f"❌ Regime change at line {regime_line} had NO parameter update nearby")

def check_saved_parameters():
    """
    Check the saved parameters file for regime-specific parameters
    """
    params_file = Path("regime_optimized_parameters.json")
    if not params_file.exists():
        print(f"❌ Parameters file {params_file} not found")
        return
    
    with open(params_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== SAVED PARAMETERS ANALYSIS ===")
    
    if 'regime_best_parameters' in data:
        regimes = data['regime_best_parameters']
        print(f"Found parameters for {len(regimes)} regimes:")
        
        for regime, regime_data in regimes.items():
            if 'parameters' in regime_data:
                params = regime_data['parameters']
                print(f"  {regime}: {params}")
            else:
                print(f"  {regime}: No parameters found")
        
        # Check if regimes have different parameters
        param_sets = []
        for regime, regime_data in regimes.items():
            if 'parameters' in regime_data:
                param_sets.append(str(sorted(regime_data['parameters'].items())))
        
        unique_param_sets = set(param_sets)
        if len(unique_param_sets) > 1:
            print(f"✅ Found {len(unique_param_sets)} different parameter sets - regime switching should work!")
        else:
            print(f"❌ All regimes have identical parameters - regime switching may not be effective")
    else:
        print("❌ No regime_best_parameters found in file")

if __name__ == "__main__":
    print("=== REGIME SWITCHING VERIFICATION ===")
    analyze_log_for_regime_switching()
    check_saved_parameters()