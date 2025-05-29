#!/usr/bin/env python3
"""
Compare trades between new optimization and test runs.
"""

import re

def extract_trades(filename):
    """Extract trades from log file."""
    trades = []
    with open(filename, 'r') as f:
        for line in f:
            if 'TRADE:' in line and 'SPY' in line:
                match = re.search(r'TRADE: (BUY|SELL) ([0-9.]+) SPY @ \$([0-9.]+)', line)
                if match:
                    trades.append({
                        'action': match.group(1),
                        'quantity': float(match.group(2)),
                        'price': float(match.group(3))
                    })
    return trades

# Extract trades from test phase of optimization
opt_trades = []
test_phase_started = False
with open('opt_new.log', 'r') as f:
    for line in f:
        if 'TEST PHASE BACKTEST...' in line:
            test_phase_started = True
        if test_phase_started and 'TRADE:' in line and 'SPY' in line:
            match = re.search(r'TRADE: (BUY|SELL) ([0-9.]+) SPY @ \$([0-9.]+)', line)
            if match:
                opt_trades.append({
                    'action': match.group(1),
                    'quantity': float(match.group(2)),
                    'price': float(match.group(3))
                })

# Extract trades from test run
test_trades = extract_trades('test_new.log')

print("="*60)
print("TRADE COMPARISON - AFTER FIX")
print("="*60)

print(f"\nOptimization Test Phase: {len(opt_trades)} trades")
print(f"Standalone Test Run: {len(test_trades)} trades")

print("\nFirst 5 trades comparison:")
print("-"*60)
print(f"{'#':<3} {'Optimization':<25} {'Test Run':<25} {'Match'}")
print("-"*60)

for i in range(min(5, max(len(opt_trades), len(test_trades)))):
    opt_str = ""
    test_str = ""
    
    if i < len(opt_trades):
        t = opt_trades[i]
        opt_str = f"{t['action']} {t['quantity']:.0f} @ ${t['price']:.2f}"
    
    if i < len(test_trades):
        t = test_trades[i]
        test_str = f"{t['action']} {t['quantity']:.0f} @ ${t['price']:.2f}"
    
    match = "✓" if opt_str == test_str else "✗"
    print(f"{i+1:<3} {opt_str:<25} {test_str:<25} {match}")

# Check if all trades match
all_match = True
if len(opt_trades) == len(test_trades):
    for i in range(len(opt_trades)):
        if (opt_trades[i]['action'] != test_trades[i]['action'] or
            opt_trades[i]['quantity'] != test_trades[i]['quantity'] or
            abs(opt_trades[i]['price'] - test_trades[i]['price']) > 0.01):
            all_match = False
            break
else:
    all_match = False

print("\n" + "="*60)
if all_match:
    print("✅ SUCCESS! All trades match between optimization and test run!")
else:
    print("❌ ISSUE: Trades still don't match")
    if opt_trades and test_trades:
        print(f"\nPrice ranges:")
        opt_prices = [t['price'] for t in opt_trades]
        test_prices = [t['price'] for t in test_trades]
        print(f"  Optimization: ${min(opt_prices):.2f} - ${max(opt_prices):.2f}")
        print(f"  Test Run: ${min(test_prices):.2f} - ${max(test_prices):.2f}")
        
        if abs(min(opt_prices) - min(test_prices)) > 2:
            print("\n⚠️  CRITICAL: Different price ranges suggest different data!")