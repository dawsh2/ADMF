#!/usr/bin/env python3
"""
Compare trades between optimization test phase and standalone test run.
"""

import re
from datetime import datetime

def extract_trades(log_content):
    """Extract all trades from log content."""
    trades = []
    
    # Pattern for trade execution
    trade_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?TRADE: (BUY|SELL) ([\d.]+) (\w+) @ \$([\d.]+), Portfolio Value: \$([\d.]+)"
    
    for match in re.finditer(trade_pattern, log_content):
        trades.append({
            'timestamp': match.group(1),
            'action': match.group(2),
            'quantity': float(match.group(3)),
            'symbol': match.group(4),
            'price': float(match.group(5)),
            'portfolio_value': float(match.group(6))
        })
    
    return trades

def compare_trade_logs():
    """Compare trades from both logs."""
    
    # Read optimization log (test phase section)
    with open('logs/admf_20250528_191224.log', 'r') as f:
        opt_content = f.read()
    
    # Extract just the test phase
    test_start = opt_content.find("EXECUTING TEST PHASE BACKTEST")
    test_end = opt_content.find("TEST PHASE BACKTEST COMPLETE")
    opt_test_content = opt_content[test_start:test_end] if test_start >= 0 and test_end > test_start else ""
    
    # Read standalone test log
    with open('logs/admf_20250528_191354.log', 'r') as f:
        test_content = f.read()
    
    # Extract trades
    opt_trades = extract_trades(opt_test_content)
    test_trades = extract_trades(test_content)
    
    print("="*80)
    print("TRADE COMPARISON")
    print("="*80)
    print(f"\nOptimization Test Phase: {len(opt_trades)} trades")
    print(f"Standalone Test Run: {len(test_trades)} trades")
    
    # Compare first 10 trades
    print("\nFIRST 10 TRADES COMPARISON:")
    print("-"*80)
    print(f"{'#':<3} {'Timestamp':<20} {'Opt Trade':<25} {'Test Trade':<25} {'Match'}")
    print("-"*80)
    
    for i in range(max(10, min(len(opt_trades), len(test_trades)))):
        opt_str = ""
        test_str = ""
        timestamp = ""
        
        if i < len(opt_trades):
            t = opt_trades[i]
            opt_str = f"{t['action']} {t['quantity']:.0f} @ ${t['price']:.2f}"
            timestamp = t['timestamp'].split(' ')[1] if ' ' in t['timestamp'] else t['timestamp']
        
        if i < len(test_trades):
            t = test_trades[i]
            test_str = f"{t['action']} {t['quantity']:.0f} @ ${t['price']:.2f}"
            if not timestamp:
                timestamp = t['timestamp'].split(' ')[1] if ' ' in t['timestamp'] else t['timestamp']
        
        # Check if trades match
        match = "✓" if opt_str == test_str else "✗"
        
        print(f"{i+1:<3} {timestamp:<20} {opt_str:<25} {test_str:<25} {match}")
    
    # Find where they diverge
    print("\nDIVERGENCE ANALYSIS:")
    print("-"*60)
    
    divergence_point = None
    for i in range(min(len(opt_trades), len(test_trades))):
        opt_t = opt_trades[i]
        test_t = test_trades[i]
        
        if (opt_t['action'] != test_t['action'] or 
            opt_t['quantity'] != test_t['quantity'] or 
            abs(opt_t['price'] - test_t['price']) > 0.01):
            divergence_point = i
            break
    
    if divergence_point is not None:
        print(f"✗ Trades diverge at position #{divergence_point + 1}")
        if divergence_point == 0:
            print("\n⚠️  CRITICAL: The VERY FIRST trade is different!")
            print(f"  Optimization: {opt_trades[0]['action']} {opt_trades[0]['quantity']:.0f} @ ${opt_trades[0]['price']:.2f}")
            print(f"  Test Run: {test_trades[0]['action']} {test_trades[0]['quantity']:.0f} @ ${test_trades[0]['price']:.2f}")
            print("\nThis suggests different initial conditions or signals!")
    else:
        print("✓ All trades match!")
    
    # Portfolio value comparison
    print("\nPORTFOLIO VALUE PROGRESSION:")
    print("-"*60)
    for i in range(min(5, len(opt_trades), len(test_trades))):
        opt_val = opt_trades[i]['portfolio_value']
        test_val = test_trades[i]['portfolio_value']
        diff = abs(opt_val - test_val)
        
        print(f"After trade {i+1}:")
        print(f"  Opt:  ${opt_val:,.2f}")
        print(f"  Test: ${test_val:,.2f}")
        if diff > 0.01:
            print(f"  Diff: ${diff:,.2f} ✗")
        else:
            print(f"  Diff: ${diff:,.2f} ✓")
    
    # Calculate final difference
    if opt_trades and test_trades:
        final_opt = opt_trades[-1]['portfolio_value']
        final_test = test_trades[-1]['portfolio_value']
        print(f"\nFINAL PORTFOLIO VALUES:")
        print(f"  Optimization: ${final_opt:,.2f}")
        print(f"  Test Run: ${final_test:,.2f}")
        print(f"  Difference: ${abs(final_opt - final_test):,.2f}")

if __name__ == "__main__":
    compare_trade_logs()