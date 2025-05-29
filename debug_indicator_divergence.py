#!/usr/bin/env python3
"""
Debug why indicators produce different signals at the same bar.
"""
import re

def extract_rule_evaluations(log_file, is_optimization=False):
    """Extract rule evaluations and indicator states."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # If optimization, extract test phase only
    if is_optimization:
        test_start = content.find("EXECUTING TEST PHASE")
        if test_start > 0:
            content = content[test_start:]
    
    # Look for patterns around first signal generation
    first_signal_pos = content.find("Strategy 'strategy' generated signal:")
    if first_signal_pos == -1:
        return None
    
    # Get 5000 chars before first signal
    context_start = max(0, first_signal_pos - 5000)
    context = content[context_start:first_signal_pos + 200]
    
    results = {
        'first_signal': None,
        'ma_values': [],
        'rsi_values': [],
        'rule_evaluations': [],
        'indicator_updates': []
    }
    
    # Get the first signal value
    signal_match = re.search(r"generated signal: ([-\d]+)", context[first_signal_pos - context_start:])
    if signal_match:
        results['first_signal'] = int(signal_match.group(1))
    
    # Extract MA values
    for match in re.finditer(r"MA: fast=([\d.]+).*?slow=([\d.]+)", context):
        results['ma_values'].append({
            'fast': float(match.group(1)),
            'slow': float(match.group(2))
        })
    
    # Extract RSI values
    for match in re.finditer(r"RSI: ([\d.]+)", context):
        results['rsi_values'].append(float(match.group(1)))
    
    # Extract rule evaluations
    for match in re.finditer(r"Rule '([^']+)'.*?(True|False)", context):
        results['rule_evaluations'].append({
            'rule': match.group(1),
            'result': match.group(2) == 'True'
        })
    
    # Extract indicator updates
    for match in re.finditer(r"Indicator (\w+).*?value[=:]?\s*([\d.]+)", context):
        results['indicator_updates'].append({
            'name': match.group(1),
            'value': float(match.group(2))
        })
    
    return results

print("="*80)
print("INDICATOR STATE AT FIRST SIGNAL")
print("="*80)

# Analyze both logs
opt_state = extract_rule_evaluations('logs/admf_20250528_191224.log', is_optimization=True)
test_state = extract_rule_evaluations('logs/admf_20250528_191354.log')

if opt_state:
    print("\nOPTIMIZATION TEST PHASE:")
    print(f"  First signal: {opt_state['first_signal']}")
    if opt_state['ma_values']:
        last_ma = opt_state['ma_values'][-1]
        print(f"  Last MA before signal: fast={last_ma['fast']:.2f}, slow={last_ma['slow']:.2f}")
        print(f"  MA Cross: {'BULLISH' if last_ma['fast'] > last_ma['slow'] else 'BEARISH'}")
    if opt_state['rsi_values']:
        print(f"  Last RSI: {opt_state['rsi_values'][-1]:.2f}")
    print(f"  Rule evaluations found: {len(opt_state['rule_evaluations'])}")
    for rule in opt_state['rule_evaluations'][-5:]:
        print(f"    {rule['rule']}: {rule['result']}")

if test_state:
    print("\nSTANDALONE TEST RUN:")
    print(f"  First signal: {test_state['first_signal']}")
    if test_state['ma_values']:
        last_ma = test_state['ma_values'][-1]
        print(f"  Last MA before signal: fast={last_ma['fast']:.2f}, slow={last_ma['slow']:.2f}")
        print(f"  MA Cross: {'BULLISH' if last_ma['fast'] > last_ma['slow'] else 'BEARISH'}")
    if test_state['rsi_values']:
        print(f"  Last RSI: {test_state['rsi_values'][-1]:.2f}")
    print(f"  Rule evaluations found: {len(test_state['rule_evaluations'])}")
    for rule in test_state['rule_evaluations'][-5:]:
        print(f"    {rule['rule']}: {rule['result']}")

# Compare
if opt_state and test_state and opt_state['ma_values'] and test_state['ma_values']:
    print("\n" + "="*80)
    print("INDICATOR COMPARISON:")
    print("="*80)
    
    opt_ma = opt_state['ma_values'][-1] if opt_state['ma_values'] else None
    test_ma = test_state['ma_values'][-1] if test_state['ma_values'] else None
    
    if opt_ma and test_ma:
        print(f"\nMA Values:")
        print(f"  Optimization: fast={opt_ma['fast']:.2f}, slow={opt_ma['slow']:.2f}")
        print(f"  Test Run:     fast={test_ma['fast']:.2f}, slow={test_ma['slow']:.2f}")
        print(f"  Difference:   fast={abs(opt_ma['fast'] - test_ma['fast']):.2f}, slow={abs(opt_ma['slow'] - test_ma['slow']):.2f}")
        
        # This is the key - are the MAs showing different crosses?
        opt_bullish = opt_ma['fast'] > opt_ma['slow']
        test_bullish = test_ma['fast'] > test_ma['slow']
        
        if opt_bullish != test_bullish:
            print(f"\n🚨 CRITICAL: MA crossover signals are OPPOSITE!")
            print(f"  Optimization: {'BULLISH (fast > slow)' if opt_bullish else 'BEARISH (fast < slow)'}")
            print(f"  Test Run:     {'BULLISH (fast > slow)' if test_bullish else 'BEARISH (fast < slow)'}")

# Now let's trace back to see why the MAs are different
print("\n" + "="*80)
print("SEARCHING FOR ROOT CAUSE...")
print("="*80)

# Check for initialization differences
print("\nChecking for warm-up differences in the logs...")