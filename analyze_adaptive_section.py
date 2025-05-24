#!/usr/bin/env python3
"""
Analyze just the adaptive test section of the optimization log
"""

import sys

def analyze_adaptive_section(filename, start_line=2602000):
    """Analyze the adaptive test portion of the log"""
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    print(f"Analyzing from line {start_line} to end (total: {len(lines)} lines)")
    
    # Track what happens in adaptive section
    adaptive_start = None
    signals = 0
    trades = 0
    portfolio_updates = 0
    regime_changes = 0
    
    for i in range(start_line, len(lines)):
        line = lines[i]
        
        # Find adaptive mode start
        if "ENABLING ADAPTIVE MODE" in line and not adaptive_start:
            adaptive_start = i
            print(f"\nAdaptive mode starts at line {i+1}:")
            print(f"  {line.strip()[:150]}...")
            
        # Count events after adaptive start
        if adaptive_start and i > adaptive_start:
            if "Publishing event: Event(type=SIGNAL" in line or "Publishing SIGNAL event" in line:
                signals += 1
                if signals <= 3:
                    print(f"  Signal {signals} at line {i+1}: {line.strip()[:100]}...")
                    
            if "Trade logged" in line or "FILL event" in line:
                trades += 1
                
            if "Portfolio value" in line or "Final Portfolio Value" in line:
                portfolio_updates += 1
                if "Final Portfolio Value" in line:
                    print(f"\n  FINAL RESULT: {line.strip()}")
                    
            if "REGIME CHANGE" in line or "Regime changed" in line:
                regime_changes += 1
    
    print(f"\nAdaptive test summary (from line {adaptive_start if adaptive_start else start_line}):")
    print(f"  Signals: {signals}")
    print(f"  Trades: {trades}")
    print(f"  Portfolio updates: {portfolio_updates}")
    print(f"  Regime changes: {regime_changes}")
    
    # Look for strategy type
    print(f"\nSearching for strategy information...")
    for i in range(max(0, start_line-1000), len(lines)):
        if i < len(lines) and ("RegimeAdaptiveStrategy" in lines[i] or "EnsembleSignalStrategy" in lines[i]):
            if "Setting up" in lines[i] or "Starting" in lines[i]:
                print(f"  Found at line {i+1}: {lines[i].strip()[:150]}...")
                break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_adaptive_section.py <optimization_log>")
        sys.exit(1)
        
    filename = sys.argv[1]
    # Start analysis a bit before the first adaptive marker
    analyze_adaptive_section(filename, 2602000)