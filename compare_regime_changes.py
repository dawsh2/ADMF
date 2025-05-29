#!/usr/bin/env python3
"""
Compare regime changes between optimization test phase and independent test run.
"""

import re
from datetime import datetime

def extract_regime_changes(log_file):
    """Extract regime changes from log file."""
    changes = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern for regime changes
    regime_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?REGIME CHANGE: (\w+) -> (\w+)"
    
    for match in re.finditer(regime_pattern, content):
        timestamp = match.group(1)
        from_regime = match.group(2)
        to_regime = match.group(3)
        changes.append({
            'timestamp': timestamp,
            'from': from_regime,
            'to': to_regime
        })
    
    return changes

def extract_indicator_values(log_file):
    """Extract indicator values after regime changes."""
    values = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern for indicator values
    indicator_pattern = r"Indicator values after regime change:.*?MA: fast=([\d.]+).*?slow=([\d.]+)"
    
    for match in re.finditer(indicator_pattern, content, re.DOTALL):
        fast_ma = float(match.group(1))
        slow_ma = float(match.group(2))
        values.append({'fast_ma': fast_ma, 'slow_ma': slow_ma})
    
    return values

def compare_logs():
    """Compare the two log files."""
    print("="*80)
    print("REGIME CHANGE COMPARISON")
    print("="*80)
    
    # Extract from both logs
    opt_changes = extract_regime_changes('opt_test_phase_full.log')
    test_changes = extract_regime_changes('independent_test_full.log')
    
    opt_indicators = extract_indicator_values('opt_test_phase_full.log')
    test_indicators = extract_indicator_values('independent_test_full.log')
    
    print(f"\nOptimization test phase: {len(opt_changes)} regime changes")
    print(f"Independent test run: {len(test_changes)} regime changes")
    print(f"Difference: {len(opt_changes) - len(test_changes)} changes")
    
    # Show first 10 changes from each WITH TIMESTAMPS
    print("\n" + "-"*80)
    print("FIRST 10 REGIME CHANGES WITH TIMESTAMPS:")
    print("-"*80)
    print(f"{'#':<3} {'Timestamp':<20} {'Optimization':<25} {'Test Run':<25}")
    print("-"*80)
    
    for i in range(min(10, max(len(opt_changes), len(test_changes)))):
        opt_str = ""
        test_str = ""
        timestamp = ""
        
        if i < len(opt_changes):
            o = opt_changes[i]
            opt_str = f"{o['from']} -> {o['to']}"
            # Extract just the time part for readability
            timestamp = o['timestamp'].split(' ')[1] if ' ' in o['timestamp'] else o['timestamp']
        
        if i < len(test_changes):
            t = test_changes[i]
            test_str = f"{t['from']} -> {t['to']}"
            if not timestamp and ' ' in t['timestamp']:
                timestamp = t['timestamp'].split(' ')[1]
            
        match = "✓" if opt_str == test_str else "✗"
        
        # Check if timestamps match too
        time_match = ""
        if i < len(opt_changes) and i < len(test_changes):
            opt_time = opt_changes[i]['timestamp'].split(' ')[1] if ' ' in opt_changes[i]['timestamp'] else opt_changes[i]['timestamp']
            test_time = test_changes[i]['timestamp'].split(' ')[1] if ' ' in test_changes[i]['timestamp'] else test_changes[i]['timestamp']
            if opt_time != test_time:
                time_match = f" (time diff: opt={opt_time}, test={test_time})"
        
        print(f"{i+1:<3} {timestamp:<20} {opt_str:<25} {test_str:<25} {match}{time_match}")
    
    # Find where they diverge
    divergence_point = None
    for i in range(min(len(opt_changes), len(test_changes))):
        if (opt_changes[i]['from'] != test_changes[i]['from'] or 
            opt_changes[i]['to'] != test_changes[i]['to']):
            divergence_point = i
            break
    
    if divergence_point is not None:
        print(f"\n⚠️  DIVERGENCE at change #{divergence_point + 1}")
        print(f"Optimization: {opt_changes[divergence_point]['from']} -> {opt_changes[divergence_point]['to']} at {opt_changes[divergence_point]['timestamp']}")
        print(f"Test run: {test_changes[divergence_point]['from']} -> {test_changes[divergence_point]['to']} at {test_changes[divergence_point]['timestamp']}")
    
    # Show timing analysis
    print("\n" + "-"*60)
    print("TIMING ANALYSIS:")
    print("-"*60)
    
    # Calculate time between changes
    def get_time_diffs(changes):
        diffs = []
        for i in range(1, len(changes)):
            # Parse timestamps (assuming format "YYYY-MM-DD HH:MM:SS")
            t1 = changes[i-1]['timestamp']
            t2 = changes[i]['timestamp']
            # Simple minute calculation (assuming same day)
            if ' ' in t1 and ' ' in t2:
                time1 = t1.split(' ')[1]
                time2 = t2.split(' ')[1]
                h1, m1, s1 = map(int, time1.split(':'))
                h2, m2, s2 = map(int, time2.split(':'))
                diff_minutes = (h2*60 + m2) - (h1*60 + m1)
                diffs.append(diff_minutes)
        return diffs
    
    opt_diffs = get_time_diffs(opt_changes[:10])
    test_diffs = get_time_diffs(test_changes[:10])
    
    print("Minutes between first 10 changes:")
    print(f"Optimization: {opt_diffs}")
    print(f"Test run: {test_diffs}")
    
    # Check if first change happens at same relative position
    if opt_changes and test_changes:
        print(f"\nFirst change timing:")
        print(f"Optimization: {opt_changes[0]['timestamp']}")
        print(f"Test run: {test_changes[0]['timestamp']}")
    
    # Check for patterns
    print("\n" + "-"*60)
    print("REGIME FREQUENCY:")
    print("-"*60)
    
    def count_regimes(changes):
        counts = {}
        for c in changes:
            counts[c['to']] = counts.get(c['to'], 0) + 1
        return counts
    
    opt_counts = count_regimes(opt_changes)
    test_counts = count_regimes(test_changes)
    
    all_regimes = set(opt_counts.keys()) | set(test_counts.keys())
    for regime in sorted(all_regimes):
        opt_count = opt_counts.get(regime, 0)
        test_count = test_counts.get(regime, 0)
        print(f"{regime:<15} Opt: {opt_count:>3}  Test: {test_count:>3}  Diff: {opt_count - test_count:>+3}")
    
    # Check indicator values
    if opt_indicators and test_indicators:
        print("\n" + "-"*60)
        print("INDICATOR VALUE COMPARISON (first 5):")
        print("-"*60)
        
        for i in range(min(5, len(opt_indicators), len(test_indicators))):
            opt_fast = opt_indicators[i]['fast_ma']
            opt_slow = opt_indicators[i]['slow_ma']
            test_fast = test_indicators[i]['fast_ma']
            test_slow = test_indicators[i]['slow_ma']
            
            fast_diff = abs(opt_fast - test_fast)
            slow_diff = abs(opt_slow - test_slow)
            
            print(f"Change {i+1}:")
            print(f"  Fast MA - Opt: {opt_fast:.4f}, Test: {test_fast:.4f}, Diff: {fast_diff:.4f}")
            print(f"  Slow MA - Opt: {opt_slow:.4f}, Test: {test_slow:.4f}, Diff: {slow_diff:.4f}")
    
    # Look for MyPrimaryRegimeDetector
    print("\n" + "-"*60)
    print("CHECKING FOR MULTIPLE DETECTORS:")
    print("-"*60)
    
    with open('independent_test_full.log', 'r') as f:
        test_content = f.read()
        
    primary_count = test_content.count('MyPrimaryRegimeDetector')
    regime_count = test_content.count('regime_detector')
    
    print(f"MyPrimaryRegimeDetector mentions: {primary_count}")
    print(f"regime_detector mentions: {regime_count}")
    
    # Check which detector is publishing
    primary_publishes = len(re.findall(r"MyPrimaryRegimeDetector.*?publishing CLASSIFICATION", test_content))
    regime_publishes = len(re.findall(r"regime_detector.*?publishing CLASSIFICATION", test_content))
    
    print(f"\nCLASSIFICATION events published:")
    print(f"  By MyPrimaryRegimeDetector: {primary_publishes}")
    print(f"  By regime_detector: {regime_publishes}")

if __name__ == "__main__":
    compare_logs()