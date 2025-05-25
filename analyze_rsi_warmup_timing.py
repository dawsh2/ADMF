#!/usr/bin/env python3
"""
Analyze RSI warmup timing between production and optimizer runs.

This script:
1. Extracts the complete timeline of RSI values (not just at signal points)
2. Identifies when RSI transitions from N/A to valid values
3. Compares the RSI warmup timing between production and optimizer runs
4. Shows the first 50 bars of RSI values for both runs
5. Identifies any RSI period changes during the runs
"""

import re
from datetime import datetime
from collections import defaultdict
import json

def parse_log_file(filepath):
    """Parse log file and extract RSI-related information."""
    rsi_timeline = []
    rsi_period_changes = []
    signal_events = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Extract RSI values from BAR indicator updates
            # Pattern: 📊 BAR_123 [timestamp] INDICATORS: ... RSI=value
            bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS:.*RSI=([^,]+)', line)
            if bar_match:
                bar_num = int(bar_match.group(1))
                timestamp_str = bar_match.group(2)
                rsi_value_str = bar_match.group(3).strip()
                
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except:
                    continue
                
                # Convert string to float or None
                if rsi_value_str in ['N/A', 'nan', 'None']:
                    rsi_val = None
                else:
                    try:
                        rsi_val = float(rsi_value_str)
                    except:
                        rsi_val = None
                
                rsi_timeline.append({
                    'timestamp': timestamp,
                    'rsi_value': rsi_val,
                    'bar_number': bar_num
                })
            
            # Look for RSI period changes
            period_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*RSI.*period[:\s]*(\d+)', line, re.IGNORECASE)
            if state_match and not rsi_calc_match:  # Avoid duplicates
                timestamp_str = state_match.group(1)
                rsi_value = state_match.group(2)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                if rsi_value in ['N/A', 'nan', 'None']:
                    rsi_val = None
                else:
                    try:
                        rsi_val = float(rsi_value)
                    except:
                        rsi_val = None
                
                rsi_timeline.append({
                    'timestamp': timestamp,
                    'rsi_value': rsi_val,
                    'bar_number': len(rsi_timeline) + 1
                })
            
            # Pattern 3: Check for RSI period changes
            period_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*RSI.*period[:\s]+(\d+)', line, re.IGNORECASE)
            if period_match:
                timestamp_str = period_match.group(1)
                period = int(period_match.group(2))
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                rsi_period_changes.append({
                    'timestamp': timestamp,
                    'period': period
                })
            
            # Pattern 4: Signal generation events with RSI
            signal_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*signal.*RSI[:\s]+([\d.]+)', line, re.IGNORECASE)
            if signal_match:
                timestamp_str = signal_match.group(1)
                rsi_value = float(signal_match.group(2))
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                signal_events.append({
                    'timestamp': timestamp,
                    'rsi_value': rsi_value
                })
    
    # Sort by timestamp to ensure chronological order
    rsi_timeline.sort(key=lambda x: x['timestamp'])
    
    # Renumber bars after sorting
    for i, entry in enumerate(rsi_timeline):
        entry['bar_number'] = i + 1
    
    return rsi_timeline, rsi_period_changes, signal_events

def find_warmup_transition(rsi_timeline):
    """Find when RSI transitions from N/A to valid values."""
    first_valid_idx = None
    for i, entry in enumerate(rsi_timeline):
        if entry['rsi_value'] is not None:
            first_valid_idx = i
            break
    
    if first_valid_idx is None:
        return None, 0
    
    return rsi_timeline[first_valid_idx], first_valid_idx

def analyze_rsi_warmup():
    """Main analysis function."""
    print("RSI Warmup Timing Analysis")
    print("=" * 80)
    
    # Analyze production log
    print("\n1. PRODUCTION RUN ANALYSIS")
    print("-" * 40)
    
    prod_timeline, prod_period_changes, prod_signals = parse_log_file('logs/admf_20250524_203019.log')
    
    if not prod_timeline:
        print("No RSI data found in production log")
        return
    
    print(f"Total RSI data points found: {len(prod_timeline)}")
    
    # Find warmup transition
    first_valid, warmup_bars = find_warmup_transition(prod_timeline)
    if first_valid:
        print(f"\nRSI Warmup completed at:")
        print(f"  Bar #{first_valid['bar_number']}")
        print(f"  Timestamp: {first_valid['timestamp']}")
        print(f"  First valid RSI value: {first_valid['rsi_value']:.2f}")
        print(f"  Warmup period: {warmup_bars} bars")
    
    # Show RSI period changes
    if prod_period_changes:
        print(f"\nRSI Period Changes:")
        for change in prod_period_changes:
            print(f"  {change['timestamp']}: Period = {change['period']}")
    else:
        print("\nNo RSI period changes detected")
    
    # Show first 50 bars
    print(f"\nFirst 50 bars of RSI values:")
    print(f"{'Bar':>5} {'Timestamp':>20} {'RSI Value':>10}")
    print("-" * 40)
    for entry in prod_timeline[:50]:
        rsi_str = f"{entry['rsi_value']:.2f}" if entry['rsi_value'] is not None else "N/A"
        print(f"{entry['bar_number']:>5} {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {rsi_str:>10}")
    
    # Analyze optimizer log
    print("\n\n2. OPTIMIZER RUN ANALYSIS")
    print("-" * 40)
    
    opt_timeline, opt_period_changes, opt_signals = parse_log_file('logs/admf_20250523_230532.log')
    
    if not opt_timeline:
        print("No RSI data found in optimizer log")
        return
    
    print(f"Total RSI data points found: {len(opt_timeline)}")
    
    # Find warmup transition
    first_valid_opt, warmup_bars_opt = find_warmup_transition(opt_timeline)
    if first_valid_opt:
        print(f"\nRSI Warmup completed at:")
        print(f"  Bar #{first_valid_opt['bar_number']}")
        print(f"  Timestamp: {first_valid_opt['timestamp']}")
        print(f"  First valid RSI value: {first_valid_opt['rsi_value']:.2f}")
        print(f"  Warmup period: {warmup_bars_opt} bars")
    
    # Show RSI period changes
    if opt_period_changes:
        print(f"\nRSI Period Changes:")
        for change in opt_period_changes:
            print(f"  {change['timestamp']}: Period = {change['period']}")
    else:
        print("\nNo RSI period changes detected")
    
    # Show first 50 bars
    print(f"\nFirst 50 bars of RSI values:")
    print(f"{'Bar':>5} {'Timestamp':>20} {'RSI Value':>10}")
    print("-" * 40)
    for entry in opt_timeline[:50]:
        rsi_str = f"{entry['rsi_value']:.2f}" if entry['rsi_value'] is not None else "N/A"
        print(f"{entry['bar_number']:>5} {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {rsi_str:>10}")
    
    # Comparison
    print("\n\n3. WARMUP COMPARISON")
    print("-" * 40)
    
    if first_valid and first_valid_opt:
        print(f"Production warmup: {warmup_bars} bars")
        print(f"Optimizer warmup: {warmup_bars_opt} bars")
        print(f"Difference: {abs(warmup_bars - warmup_bars_opt)} bars")
        
        # Compare first valid values
        print(f"\nFirst valid RSI values:")
        print(f"  Production: {first_valid['rsi_value']:.2f} at {first_valid['timestamp']}")
        print(f"  Optimizer: {first_valid_opt['rsi_value']:.2f} at {first_valid_opt['timestamp']}")
        
        # Compare RSI values at same bar numbers
        print(f"\nRSI Value Comparison (first 20 valid bars):")
        print(f"{'Bar':>5} {'Production':>12} {'Optimizer':>12} {'Difference':>12}")
        print("-" * 50)
        
        # Get valid bars for both
        prod_valid = [e for e in prod_timeline if e['rsi_value'] is not None][:20]
        opt_valid = [e for e in opt_timeline if e['rsi_value'] is not None][:20]
        
        for i in range(min(len(prod_valid), len(opt_valid))):
            prod_val = prod_valid[i]['rsi_value']
            opt_val = opt_valid[i]['rsi_value']
            diff = prod_val - opt_val
            print(f"{i+1:>5} {prod_val:>12.2f} {opt_val:>12.2f} {diff:>12.2f}")
    
    # Signal timing analysis
    print(f"\n\n4. SIGNAL GENERATION TIMING")
    print("-" * 40)
    
    if prod_signals:
        print(f"Production - First signal at: {prod_signals[0]['timestamp']} (RSI: {prod_signals[0]['rsi_value']:.2f})")
    else:
        print("Production - No signals found")
    
    if opt_signals:
        print(f"Optimizer - First signal at: {opt_signals[0]['timestamp']} (RSI: {opt_signals[0]['rsi_value']:.2f})")
    else:
        print("Optimizer - No signals found")

if __name__ == "__main__":
    analyze_rsi_warmup()