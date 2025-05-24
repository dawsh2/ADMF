#!/usr/bin/env python3
"""
Compare strategy types and parameters between optimization and production logs
"""

import sys
import re

def extract_strategy_info(filename, start_line=0):
    """Extract strategy and parameter information from log"""
    
    strategy_info = {
        'strategy_type': None,
        'parameters': [],
        'weights': [],
        'regime_changes': [],
        'first_signals': []
    }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    signal_count = 0
    
    for i in range(start_line, len(lines)):
        line = lines[i]
        
        # Find strategy type
        if "RegimeAdaptiveStrategy" in line and "Setting up" in line:
            strategy_info['strategy_type'] = 'RegimeAdaptiveStrategy'
        elif "EnsembleSignalStrategy" in line and "Setting up" in line:
            strategy_info['strategy_type'] = 'EnsembleSignalStrategy'
            
        # Find parameter applications
        if "Applying parameters for regime" in line or "Updated parameters" in line:
            strategy_info['parameters'].append(line.strip())
            
        # Find weight information
        if "WEIGHTS AFTER APPLICATION" in line or "MA=" in line and "RSI=" in line:
            strategy_info['weights'].append(line.strip())
            
        # Find regime changes
        if "REGIME CHANGE DETECTED" in line or "Current regime:" in line:
            strategy_info['regime_changes'].append(line.strip())
            
        # Capture first few signals
        if signal_count < 5 and ("Publishing event: Event(type=SIGNAL" in line or "Signal:" in line):
            strategy_info['first_signals'].append(f"Line {i+1}: {line.strip()[:150]}...")
            signal_count += 1
            
    return strategy_info

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_strategies.py <log1> <log2> [start_line_for_log2]")
        sys.exit(1)
        
    log1 = sys.argv[1]
    log2 = sys.argv[2]
    start_line = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    print(f"\n{'='*60}")
    print(f"Comparing {log1} vs {log2}")
    if start_line > 0:
        print(f"Starting from line {start_line} in second log")
    print(f"{'='*60}")
    
    info1 = extract_strategy_info(log1)
    info2 = extract_strategy_info(log2, start_line)
    
    print(f"\n{log1}:")
    print(f"  Strategy: {info1['strategy_type']}")
    print(f"  Parameter updates: {len(info1['parameters'])}")
    print(f"  Weight updates: {len(info1['weights'])}")
    print(f"  Regime changes: {len(info1['regime_changes'])}")
    
    if info1['weights']:
        print(f"  Sample weights: {info1['weights'][0][:100]}...")
        
    print(f"\n{log2} (from line {start_line}):")
    print(f"  Strategy: {info2['strategy_type']}")
    print(f"  Parameter updates: {len(info2['parameters'])}")
    print(f"  Weight updates: {len(info2['weights'])}")
    print(f"  Regime changes: {len(info2['regime_changes'])}")
    
    if info2['weights']:
        print(f"  Sample weights: {info2['weights'][0][:100]}...")
        
    print(f"\nFirst signals comparison:")
    print(f"\n{log1}:")
    for sig in info1['first_signals'][:3]:
        print(f"  {sig}")
        
    print(f"\n{log2}:")
    for sig in info2['first_signals'][:3]:
        print(f"  {sig}")

if __name__ == "__main__":
    main()