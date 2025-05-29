#!/usr/bin/env python3
"""
Debug signals by tracking bar processing events.
"""
import re

def analyze_bar_signals(log_file, is_optimization=False):
    """Extract bar processing and signal generation."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # If optimization, extract test phase only
    if is_optimization:
        test_start = content.find("EXECUTING TEST PHASE")
        if test_start > 0:
            content = content[test_start:]
    
    # Track bars and their associated events
    bar_events = {}
    current_bar_time = None
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Detect bar streaming
        bar_match = re.search(r'\[BAR STREAM DEBUG\] Streamed bar \d+/\d+: timestamp=([^,]+)', line)
        if bar_match:
            current_bar_time = bar_match.group(1)
            if current_bar_time not in bar_events:
                bar_events[current_bar_time] = {
                    'signals': [],
                    'regime': None,
                    'parameters': None
                }
        
        # Detect signals
        if current_bar_time and "Strategy 'strategy' generated signal:" in line:
            signal_match = re.search(r"generated signal: ([-\d]+)", line)
            if signal_match:
                bar_events[current_bar_time]['signals'].append(int(signal_match.group(1)))
        
        # Detect regime
        if current_bar_time and "Portfolio Update at" in line and current_bar_time in line:
            regime_match = re.search(r'\[([^\]]+)\]:', line)
            if regime_match:
                bar_events[current_bar_time]['regime'] = regime_match.group(1)
        
        # Detect parameter application
        if current_bar_time and "Applied parameters" in line:
            param_match = re.search(r"MA: (\d+)/(\d+)", line)
            if param_match:
                bar_events[current_bar_time]['parameters'] = f"MA {param_match.group(1)}/{param_match.group(2)}"
    
    return bar_events

# Analyze both runs
print("="*80)
print("BAR-BY-BAR SIGNAL ANALYSIS")
print("="*80)

opt_bars = analyze_bar_signals('logs/admf_20250528_191224.log', is_optimization=True)
test_bars = analyze_bar_signals('logs/admf_20250528_191354.log')

# Find bars with signals
opt_signal_bars = [(time, data) for time, data in opt_bars.items() if data['signals']]
test_signal_bars = [(time, data) for time, data in test_bars.items() if data['signals']]

print(f"\nOPTIMIZATION TEST PHASE:")
print(f"  Total bars processed: {len(opt_bars)}")
print(f"  Bars with signals: {len(opt_signal_bars)}")
if opt_signal_bars:
    print(f"\n  First 5 bars with signals:")
    for i, (bar_time, data) in enumerate(opt_signal_bars[:5]):
        signals_str = ','.join(map(str, data['signals']))
        print(f"    {bar_time}: signals=[{signals_str}] regime={data['regime']} params={data['parameters']}")

print(f"\nSTANDALONE TEST RUN:")
print(f"  Total bars processed: {len(test_bars)}")
print(f"  Bars with signals: {len(test_signal_bars)}")
if test_signal_bars:
    print(f"\n  First 5 bars with signals:")
    for i, (bar_time, data) in enumerate(test_signal_bars[:5]):
        signals_str = ','.join(map(str, data['signals']))
        print(f"    {bar_time}: signals=[{signals_str}] regime={data['regime']} params={data['parameters']}")

# Find when signals start
if opt_signal_bars and test_signal_bars:
    print("\n" + "="*80)
    print("SIGNAL TIMING COMPARISON:")
    print("="*80)
    
    opt_first_bar = opt_signal_bars[0][0]
    test_first_bar = test_signal_bars[0][0]
    
    print(f"\nFirst bar with signals:")
    print(f"  Optimization: {opt_first_bar}")
    print(f"  Test Run:     {test_first_bar}")
    
    # Find common bars
    opt_bar_times = set(opt_bars.keys())
    test_bar_times = set(test_bars.keys())
    common_bars = opt_bar_times & test_bar_times
    
    print(f"\nCommon bars: {len(common_bars)} out of {len(opt_bars)}/{len(test_bars)}")
    
    # Check signals on common bars
    matching_signals = 0
    different_signals = 0
    
    for bar_time in sorted(common_bars):
        opt_sigs = opt_bars[bar_time]['signals']
        test_sigs = test_bars[bar_time]['signals']
        
        if opt_sigs and test_sigs:
            if opt_sigs == test_sigs:
                matching_signals += 1
            else:
                different_signals += 1
                if different_signals <= 3:  # Show first 3 differences
                    print(f"\n  Signal mismatch at {bar_time}:")
                    print(f"    Opt:  {opt_sigs}")
                    print(f"    Test: {test_sigs}")