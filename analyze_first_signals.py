#!/usr/bin/env python3
"""
Analyze the first signals by bar timestamp, not system time.
"""
import re

def extract_first_signal_context(log_file, is_optimization=False):
    """Extract context around first signal with bar timestamps."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # If optimization, extract test phase only
    if is_optimization:
        test_start = content.find("EXECUTING TEST PHASE")
        if test_start > 0:
            content = content[test_start:]
    
    # Find all signals with their associated bar times
    signals = []
    
    # Pattern to find portfolio update followed by signal
    pattern = r"Portfolio Update at ([^+\n]+)\+[^[]+\[([^\]]+)\][^\n]+\n[^\n]*?Strategy 'strategy' generated signal: ([-\d]+)"
    
    for match in re.finditer(pattern, content):
        bar_time = match.group(1).strip()
        regime = match.group(2)
        signal = int(match.group(3))
        
        # Skip system timestamps (they contain more than just date/time)
        if not re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', bar_time):
            continue
            
        signals.append({
            'bar_time': bar_time,
            'regime': regime,
            'signal': signal
        })
    
    # Also look for trades to see what actually executed
    trades = []
    trade_pattern = r"TRADE: (BUY|SELL) ([0-9.]+) (\w+) @ \$([0-9.]+)"
    for match in re.finditer(trade_pattern, content):
        trades.append({
            'action': match.group(1),
            'quantity': float(match.group(2)),
            'symbol': match.group(3),
            'price': float(match.group(4))
        })
    
    return signals, trades

print("="*80)
print("FIRST SIGNALS BY BAR TIMESTAMP")
print("="*80)

# Analyze optimization
opt_signals, opt_trades = extract_first_signal_context('logs/admf_20250528_191224.log', is_optimization=True)
print(f"\nOPTIMIZATION TEST PHASE:")
print(f"Total signals found: {len(opt_signals)}")
print(f"Total trades executed: {len(opt_trades)}")
if opt_signals:
    print(f"\nFirst 5 signals:")
    for i, sig in enumerate(opt_signals[:5]):
        print(f"  {i+1}. Bar {sig['bar_time']} [{sig['regime']}]: Signal={sig['signal']}")
if opt_trades:
    print(f"\nFirst trade: {opt_trades[0]['action']} {opt_trades[0]['quantity']} @ ${opt_trades[0]['price']}")

# Analyze test run
test_signals, test_trades = extract_first_signal_context('logs/admf_20250528_191354.log')
print(f"\nSTANDALONE TEST RUN:")
print(f"Total signals found: {len(test_signals)}")
print(f"Total trades executed: {len(test_trades)}")
if test_signals:
    print(f"\nFirst 5 signals:")
    for i, sig in enumerate(test_signals[:5]):
        print(f"  {i+1}. Bar {sig['bar_time']} [{sig['regime']}]: Signal={sig['signal']}")
if test_trades:
    print(f"\nFirst trade: {test_trades[0]['action']} {test_trades[0]['quantity']} @ ${test_trades[0]['price']}")

# Compare
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

if opt_signals and test_signals:
    opt_first = opt_signals[0]
    test_first = test_signals[0]
    
    print(f"\nFirst signal timing:")
    print(f"  Optimization: {opt_first['bar_time']} (Signal={opt_first['signal']})")
    print(f"  Test Run:     {test_first['bar_time']} (Signal={test_first['signal']})")
    
    # Calculate time difference
    opt_time_parts = opt_first['bar_time'].split(' ')[1].split(':')
    test_time_parts = test_first['bar_time'].split(' ')[1].split(':')
    opt_minutes = int(opt_time_parts[0]) * 60 + int(opt_time_parts[1])
    test_minutes = int(test_time_parts[0]) * 60 + int(test_time_parts[1])
    diff_minutes = opt_minutes - test_minutes
    
    print(f"\nTime difference: {diff_minutes} minutes")
    print(f"Optimization starts generating signals {abs(diff_minutes)} minutes {'later' if diff_minutes > 0 else 'earlier'}")
    
    # Check if signals align after the offset
    print(f"\nChecking if signals align after offset...")
    test_offset = abs(diff_minutes) if diff_minutes > 0 else 0
    opt_offset = abs(diff_minutes) if diff_minutes < 0 else 0
    
    matches = 0
    for i in range(min(10, len(opt_signals) - opt_offset, len(test_signals) - test_offset)):
        opt_sig = opt_signals[i + opt_offset]
        test_sig = test_signals[i + test_offset]
        if opt_sig['signal'] == test_sig['signal']:
            matches += 1
    
    print(f"Signals matching after offset: {matches}/10")