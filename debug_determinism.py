#!/usr/bin/env python3
"""
Debug script to find sources of non-determinism between optimization test phase 
and standalone test runs.
"""

import re
from datetime import datetime

def extract_detailed_info(log_file):
    """Extract detailed information from log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    info = {
        'regime_changes': [],
        'trades': [],
        'signals': [],
        'indicator_states': [],
        'first_bars': [],
        'parameter_changes': []
    }
    
    # Extract regime changes with timestamps
    regime_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Regime SWITCH from '(\w+)' to '(\w+)'"
    for match in re.finditer(regime_pattern, content):
        info['regime_changes'].append({
            'timestamp': match.group(1),
            'from': match.group(2),
            'to': match.group(3)
        })
    
    # Extract trades
    trade_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?TRADE: (BUY|SELL) ([\d.]+) (\w+) @ \$([\d.]+)"
    for match in re.finditer(trade_pattern, content):
        info['trades'].append({
            'timestamp': match.group(1),
            'action': match.group(2),
            'quantity': float(match.group(3)),
            'symbol': match.group(4),
            'price': float(match.group(5))
        })
    
    # Extract signals
    signal_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Strategy.*generated signal: ([-\d]+)"
    for match in re.finditer(signal_pattern, content):
        info['signals'].append({
            'timestamp': match.group(1),
            'signal': int(match.group(2))
        })
    
    # Extract first 5 bars
    first_bar_pattern = r"\[REGIME DETECTOR FIRST BARS\] Bar #(\d+): timestamp=([^,]+), open=\$([^,]+), high=\$([^,]+), low=\$([^,]+), close=\$([\d.]+)"
    for match in re.finditer(first_bar_pattern, content):
        if int(match.group(1)) <= 5:
            info['first_bars'].append({
                'bar_num': int(match.group(1)),
                'timestamp': match.group(2),
                'close': float(match.group(6))
            })
    
    # Extract parameter changes
    param_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Applied parameters.*?MA: (\d+)/(\d+).*?RSI: (\d+)"
    for match in re.finditer(param_pattern, content):
        info['parameter_changes'].append({
            'timestamp': match.group(1),
            'fast_ma': int(match.group(2)),
            'slow_ma': int(match.group(3)),
            'rsi': int(match.group(4))
        })
    
    # Extract final metrics
    return_match = re.search(r"Total Return: ([-\d.]+)%", content)
    info['total_return'] = float(return_match.group(1)) if return_match else None
    
    # Get trade counts from regime performance
    regime_trades = {}
    regime_perf_pattern = r"(\w+): PnL=\$[-\d.]+, Trades=(\d+)"
    for match in re.finditer(regime_perf_pattern, content):
        regime_trades[match.group(1)] = int(match.group(2))
    info['regime_trades'] = regime_trades
    info['actual_trade_count'] = sum(regime_trades.values())
    
    return info

def compare_runs(opt_log, test_log):
    """Compare optimization and test runs in detail."""
    print("="*80)
    print("DETERMINISM ANALYSIS")
    print("="*80)
    
    opt_info = extract_detailed_info(opt_log)
    test_info = extract_detailed_info(test_log)
    
    # 1. Check if first bars are identical
    print("\n1. FIRST BARS COMPARISON:")
    print("-"*60)
    if opt_info['first_bars'] and test_info['first_bars']:
        for i in range(min(5, len(opt_info['first_bars']), len(test_info['first_bars']))):
            opt_bar = opt_info['first_bars'][i] if i < len(opt_info['first_bars']) else None
            test_bar = test_info['first_bars'][i] if i < len(test_info['first_bars']) else None
            
            if opt_bar and test_bar:
                if opt_bar['close'] == test_bar['close']:
                    print(f"✓ Bar {i+1}: Both see ${opt_bar['close']:.2f}")
                else:
                    print(f"✗ Bar {i+1}: Opt=${opt_bar['close']:.2f}, Test=${test_bar['close']:.2f}")
    
    # 2. First regime change
    print("\n2. FIRST REGIME CHANGE:")
    print("-"*60)
    if opt_info['regime_changes'] and test_info['regime_changes']:
        opt_first = opt_info['regime_changes'][0]
        test_first = test_info['regime_changes'][0]
        
        if opt_first['from'] == test_first['from'] and opt_first['to'] == test_first['to']:
            print(f"✓ Both: {opt_first['from']} -> {opt_first['to']}")
        else:
            print(f"✗ DIFFERENT!")
            print(f"  Opt:  {opt_first['from']} -> {opt_first['to']} at {opt_first['timestamp']}")
            print(f"  Test: {test_first['from']} -> {test_first['to']} at {test_first['timestamp']}")
    
    # 3. Trade comparison
    print("\n3. TRADE ANALYSIS:")
    print("-"*60)
    print(f"Optimization: {opt_info['actual_trade_count']} trades")
    print(f"Test run: {test_info['actual_trade_count']} trades")
    
    if opt_info['regime_trades'] and test_info['regime_trades']:
        print("\nBy regime:")
        all_regimes = set(opt_info['regime_trades'].keys()) | set(test_info['regime_trades'].keys())
        for regime in sorted(all_regimes):
            opt_count = opt_info['regime_trades'].get(regime, 0)
            test_count = test_info['regime_trades'].get(regime, 0)
            if opt_count == test_count:
                print(f"  ✓ {regime}: {opt_count} trades")
            else:
                print(f"  ✗ {regime}: Opt={opt_count}, Test={test_count}")
    
    # 4. Signal comparison (first 10)
    print("\n4. FIRST 10 SIGNALS:")
    print("-"*60)
    for i in range(min(10, max(len(opt_info['signals']), len(test_info['signals'])))):
        opt_sig = opt_info['signals'][i] if i < len(opt_info['signals']) else None
        test_sig = test_info['signals'][i] if i < len(test_info['signals']) else None
        
        if opt_sig and test_sig:
            if opt_sig['signal'] == test_sig['signal']:
                print(f"✓ Signal {i+1}: {opt_sig['signal']} at {opt_sig['timestamp']}")
            else:
                print(f"✗ Signal {i+1}: Opt={opt_sig['signal']}, Test={test_sig['signal']}")
        elif opt_sig:
            print(f"✗ Signal {i+1}: Only in Opt: {opt_sig['signal']}")
        elif test_sig:
            print(f"✗ Signal {i+1}: Only in Test: {test_sig['signal']}")
    
    # 5. Parameter changes
    print("\n5. PARAMETER CHANGES:")
    print("-"*60)
    print(f"Optimization: {len(opt_info['parameter_changes'])} changes")
    print(f"Test run: {len(test_info['parameter_changes'])} changes")
    
    # 6. Final results
    print("\n6. FINAL RESULTS:")
    print("-"*60)
    print(f"Return: Opt={opt_info['total_return']:.2f}%, Test={test_info['total_return']:.2f}%")
    print(f"Difference: {abs(opt_info['total_return'] - test_info['total_return']):.2f}%")
    
    # 7. Regime change count
    print("\n7. REGIME CHANGES:")
    print("-"*60)
    print(f"Optimization: {len(opt_info['regime_changes'])} changes")
    print(f"Test run: {len(test_info['regime_changes'])} changes")
    
    if len(opt_info['regime_changes']) != len(test_info['regime_changes']):
        print("\n✗ DIFFERENT NUMBER OF REGIME CHANGES!")
        print("\nFinding divergence point...")
        for i in range(min(len(opt_info['regime_changes']), len(test_info['regime_changes']))):
            opt_ch = opt_info['regime_changes'][i]
            test_ch = test_info['regime_changes'][i]
            if opt_ch['from'] != test_ch['from'] or opt_ch['to'] != test_ch['to']:
                print(f"\nDiverges at change #{i+1}:")
                print(f"  Opt:  {opt_ch['from']} -> {opt_ch['to']}")
                print(f"  Test: {test_ch['from']} -> {test_ch['to']}")
                break

if __name__ == "__main__":
    # Use the specific log files
    opt_log = "logs/admf_20250528_191224.log"
    test_log = "logs/admf_20250528_191354.log"
    
    print(f"Comparing:")
    print(f"  Optimization: {opt_log}")
    print(f"  Test run: {test_log}")
    
    compare_runs(opt_log, test_log)