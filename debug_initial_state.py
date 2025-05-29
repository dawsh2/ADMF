#!/usr/bin/env python3
"""
Debug the initial state differences between optimization and test runs.
"""

import re

def analyze_initial_conditions(log_file, is_optimization=False):
    """Extract initial conditions from log."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # If optimization, find test phase section
    if is_optimization:
        test_start = content.find("TEST PHASE BACKTEST...")
        if test_start > 0:
            # Also look a bit before for setup
            content = content[max(0, test_start - 5000):]
    
    results = {
        'first_bar': None,
        'first_regime': None,
        'first_signal': None,
        'first_parameters': None,
        'regime_switching_enabled': False,
        'initial_indicators': {}
    }
    
    # Find first bar
    bar_match = re.search(r'\[BAR STREAM DEBUG\] Streamed bar 1/\d+: timestamp=([^,]+), close=([\d.]+)', content)
    if bar_match:
        results['first_bar'] = {
            'timestamp': bar_match.group(1),
            'close': float(bar_match.group(2))
        }
    
    # Find regime switching status
    if "ENABLED regime switching" in content:
        results['regime_switching_enabled'] = True
    
    # Find first regime classification
    regime_match = re.search(r"Current regime: '([^']+)'", content)
    if regime_match:
        results['first_regime'] = regime_match.group(1)
    
    # Find first signal
    signal_match = re.search(r"generated signal: ([-\d]+)", content)
    if signal_match:
        results['first_signal'] = int(signal_match.group(1))
    
    # Find first parameter application
    param_match = re.search(r"Applied parameters - MA: (\d+)/(\d+), RSI: (\d+)", content)
    if param_match:
        results['first_parameters'] = {
            'fast_ma': int(param_match.group(1)),
            'slow_ma': int(param_match.group(2)),
            'rsi': int(param_match.group(3))
        }
    
    # Look for indicator ready messages
    for match in re.finditer(r"Indicator (\w+) now READY with (\d+) bars", content):
        if match.start() > 1000:  # Don't go too far
            break
        results['initial_indicators'][match.group(1)] = int(match.group(2))
    
    return results

print("="*80)
print("INITIAL STATE COMPARISON")
print("="*80)

# Analyze both logs
opt_state = analyze_initial_conditions('opt_new.log', is_optimization=True)
test_state = analyze_initial_conditions('test_new.log')

# Compare
print("\nFIRST BAR:")
if opt_state['first_bar'] and test_state['first_bar']:
    print(f"  Optimization: {opt_state['first_bar']['timestamp']} @ ${opt_state['first_bar']['close']}")
    print(f"  Test Run:     {test_state['first_bar']['timestamp']} @ ${test_state['first_bar']['close']}")
    if opt_state['first_bar']['close'] == test_state['first_bar']['close']:
        print("  ✓ Same starting price")
    else:
        print("  ✗ Different starting prices!")

print("\nREGIME SWITCHING:")
print(f"  Optimization: {'Enabled' if opt_state['regime_switching_enabled'] else 'Disabled'}")
print(f"  Test Run:     {'Enabled' if test_state['regime_switching_enabled'] else 'Disabled'}")

print("\nFIRST REGIME:")
print(f"  Optimization: {opt_state['first_regime']}")
print(f"  Test Run:     {test_state['first_regime']}")

print("\nFIRST SIGNAL:")
print(f"  Optimization: {opt_state['first_signal']} {'(SELL)' if opt_state['first_signal'] == -1 else '(BUY)' if opt_state['first_signal'] == 1 else ''}")
print(f"  Test Run:     {test_state['first_signal']} {'(SELL)' if test_state['first_signal'] == -1 else '(BUY)' if test_state['first_signal'] == 1 else ''}")

print("\nFIRST PARAMETERS:")
if opt_state['first_parameters'] and test_state['first_parameters']:
    opt_p = opt_state['first_parameters']
    test_p = test_state['first_parameters']
    print(f"  Optimization: MA {opt_p['fast_ma']}/{opt_p['slow_ma']}, RSI {opt_p['rsi']}")
    print(f"  Test Run:     MA {test_p['fast_ma']}/{test_p['slow_ma']}, RSI {test_p['rsi']}")
    if opt_p == test_p:
        print("  ✓ Same parameters")
    else:
        print("  ✗ Different parameters!")

print("\nINITIAL INDICATOR READINESS:")
all_indicators = set(opt_state['initial_indicators'].keys()) | set(test_state['initial_indicators'].keys())
for ind in sorted(all_indicators):
    opt_bars = opt_state['initial_indicators'].get(ind, 'N/A')
    test_bars = test_state['initial_indicators'].get(ind, 'N/A')
    match = "✓" if opt_bars == test_bars else "✗"
    print(f"  {ind}: Opt={opt_bars}, Test={test_bars} {match}")

# Key insight
print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)

if opt_state['first_signal'] != test_state['first_signal']:
    print("1. Different first signals explain the one-trade offset")
    
if opt_state['first_parameters'] != test_state['first_parameters']:
    print("2. Different parameters suggest different initial regime")
    
if opt_state['initial_indicators'] != test_state['initial_indicators']:
    print("3. Different indicator readiness timing")