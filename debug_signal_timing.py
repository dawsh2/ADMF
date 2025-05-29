#!/usr/bin/env python3
"""
Debug why optimization and test runs generate different signals at the start.
"""

import re

def analyze_signal_timing(log_file, is_optimization=False):
    """Extract signal timing and context."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # If optimization, extract test phase only
    if is_optimization:
        test_start = content.find("EXECUTING TEST PHASE")
        if test_start > 0:
            content = content[test_start:]
    
    results = {
        'first_bar_time': None,
        'first_signal_time': None,
        'first_signal_value': None,
        'parameter_changes': [],
        'regime_at_first_signal': None,
        'indicators_at_first_signal': {}
    }
    
    # Find first bar
    bar_match = re.search(r"\[BAR STREAM DEBUG\] Streamed bar 1/\d+: timestamp=([^,]+)", content)
    if bar_match:
        results['first_bar_time'] = bar_match.group(1)
    
    # Find first signal
    signal_match = re.search(r"Strategy 'strategy' generated signal: ([-\d]+)", content)
    if signal_match:
        results['first_signal_value'] = int(signal_match.group(1))
        
        # Find the portfolio update just before this signal
        before_signal = content[:signal_match.start()]
        portfolio_matches = list(re.finditer(r"Portfolio Update at ([^+\n]+)\+[^[]+\[([^\]]+)\]", before_signal))
        if portfolio_matches:
            last_portfolio = portfolio_matches[-1]
            results['first_signal_time'] = last_portfolio.group(1)
            results['regime_at_first_signal'] = last_portfolio.group(2)
    
    # Find parameter applications before first signal
    param_pattern = r"Applied parameters - MA: (\d+)/(\d+), RSI: (\d+)"
    for match in re.finditer(param_pattern, content):
        if signal_match and match.start() > signal_match.start():
            break
        results['parameter_changes'].append({
            'fast_ma': int(match.group(1)),
            'slow_ma': int(match.group(2)),
            'rsi': int(match.group(3))
        })
    
    # Find indicator values near first signal
    if signal_match:
        # Look for indicator values within 500 chars before first signal
        search_start = max(0, signal_match.start() - 1000)
        search_content = content[search_start:signal_match.start()]
        
        # MA values
        ma_match = re.search(r"MA: fast=([\d.]+).*?slow=([\d.]+)", search_content)
        if ma_match:
            results['indicators_at_first_signal']['fast_ma'] = float(ma_match.group(1))
            results['indicators_at_first_signal']['slow_ma'] = float(ma_match.group(2))
        
        # RSI value
        rsi_match = re.search(r"RSI: ([\d.]+)", search_content)
        if rsi_match:
            results['indicators_at_first_signal']['rsi'] = float(rsi_match.group(1))
    
    return results

# Analyze both logs
print("="*80)
print("SIGNAL GENERATION TIMING ANALYSIS")
print("="*80)

opt_results = analyze_signal_timing('logs/admf_20250528_191224.log', is_optimization=True)
test_results = analyze_signal_timing('logs/admf_20250528_191354.log', is_optimization=False)

print("\nOPTIMIZATION TEST PHASE:")
print(f"  First bar: {opt_results['first_bar_time']}")
print(f"  First signal: {opt_results['first_signal_time']} (value: {opt_results['first_signal_value']})")
print(f"  Regime: {opt_results['regime_at_first_signal']}")
print(f"  Parameter changes before signal: {len(opt_results['parameter_changes'])}")
if opt_results['parameter_changes']:
    print(f"  Last params: MA {opt_results['parameter_changes'][-1]['fast_ma']}/{opt_results['parameter_changes'][-1]['slow_ma']}")
print(f"  Indicators: {opt_results['indicators_at_first_signal']}")

print("\nSTANDALONE TEST RUN:")
print(f"  First bar: {test_results['first_bar_time']}")
print(f"  First signal: {test_results['first_signal_time']} (value: {test_results['first_signal_value']})")
print(f"  Regime: {test_results['regime_at_first_signal']}")
print(f"  Parameter changes before signal: {len(test_results['parameter_changes'])}")
if test_results['parameter_changes']:
    print(f"  Last params: MA {test_results['parameter_changes'][-1]['fast_ma']}/{test_results['parameter_changes'][-1]['slow_ma']}")
print(f"  Indicators: {test_results['indicators_at_first_signal']}")

print("\nKEY DIFFERENCES:")
# Calculate time difference
if opt_results['first_signal_time'] and test_results['first_signal_time']:
    opt_time = opt_results['first_signal_time'].split()[1]
    test_time = test_results['first_signal_time'].split()[1]
    print(f"  Signal timing: Opt at {opt_time}, Test at {test_time}")

print(f"  Signal values: Opt={opt_results['first_signal_value']}, Test={test_results['first_signal_value']}")

if opt_results['indicators_at_first_signal'] and test_results['indicators_at_first_signal']:
    opt_ind = opt_results['indicators_at_first_signal']
    test_ind = test_results['indicators_at_first_signal']
    if 'fast_ma' in opt_ind and 'fast_ma' in test_ind:
        print(f"  Fast MA: Opt={opt_ind['fast_ma']:.2f}, Test={test_ind['fast_ma']:.2f}")
    if 'slow_ma' in opt_ind and 'slow_ma' in test_ind:
        print(f"  Slow MA: Opt={opt_ind['slow_ma']:.2f}, Test={test_ind['slow_ma']:.2f}")