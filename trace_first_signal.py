#!/usr/bin/env python3
"""
Trace why the first signal differs between runs.
"""

import re

def trace_signal_generation(log_file, is_optimization=False):
    """Trace the signal generation process."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to start
    if is_optimization:
        # Find test phase start
        start_idx = 0
        for i, line in enumerate(lines):
            if "TEST PHASE BACKTEST..." in line:
                start_idx = i
                break
    else:
        start_idx = 0
    
    # Track state
    first_bar_found = False
    indicators_ready = {}
    rules_evaluated = []
    signal_components = []
    
    # Process lines
    for i in range(start_idx, min(start_idx + 1000, len(lines))):
        line = lines[i]
        
        # First bar
        if not first_bar_found and "[BAR STREAM DEBUG] Streamed bar 1/" in line:
            first_bar_found = True
            print(f"First bar: {line.strip()}")
            print()
        
        # Indicator readiness
        if "now READY" in line and "Indicator" in line:
            match = re.search(r"Indicator (\w+) now READY", line)
            if match:
                ind_name = match.group(1)
                if ind_name not in indicators_ready:
                    indicators_ready[ind_name] = line.strip()
        
        # Rule evaluations
        if "rule" in line.lower() and ("evaluating" in line.lower() or "evaluated" in line.lower()):
            rules_evaluated.append(line.strip())
        
        # Signal components
        if "signal strength" in line.lower() or "weight" in line.lower() and "signal" in line.lower():
            signal_components.append(line.strip())
        
        # First signal generated
        if "generated signal:" in line:
            match = re.search(r"generated signal: ([-\d]+)", line)
            if match:
                signal = int(match.group(1))
                print(f"FIRST SIGNAL: {signal} {'(SELL)' if signal == -1 else '(BUY)'}")
                print(f"Full line: {line.strip()}")
                
                # Show context
                print("\nINDICATORS READY:")
                for ind, ready_line in indicators_ready.items():
                    print(f"  {ind}")
                
                if rules_evaluated:
                    print("\nRULES EVALUATED:")
                    for rule in rules_evaluated[:5]:
                        print(f"  {rule}")
                
                if signal_components:
                    print("\nSIGNAL COMPONENTS:")
                    for comp in signal_components[:5]:
                        print(f"  {comp}")
                
                # Look for classification
                for j in range(max(0, i-20), i):
                    if "CLASSIFICATION" in lines[j] and "regime" in lines[j]:
                        print(f"\nREGIME: {lines[j].strip()}")
                        break
                
                return signal
    
    return None

print("="*80)
print("FIRST SIGNAL TRACE - OPTIMIZATION")
print("="*80)
opt_signal = trace_signal_generation('opt_new.log', is_optimization=True)

print("\n" + "="*80)
print("FIRST SIGNAL TRACE - TEST RUN")
print("="*80)
test_signal = trace_signal_generation('test_new.log')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Optimization first signal: {opt_signal}")
print(f"Test run first signal: {test_signal}")

if opt_signal != test_signal:
    print("\n⚠️  Different first signals explain the trade offset!")
    print("Need to investigate why the same data produces different signals.")