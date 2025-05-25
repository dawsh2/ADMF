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
                    # Handle timestamps with timezone
                    if '+' in timestamp_str:
                        timestamp_str = timestamp_str.split('+')[0]
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
            if period_match:
                timestamp_str = period_match.group(1)
                period = int(period_match.group(2))
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                rsi_period_changes.append({
                    'timestamp': timestamp,
                    'period': period
                })
            
            # Look for signal events with RSI values
            signal_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*SIGNAL.*RSI[:\s]+([\d.]+|N/A|nan)', line, re.IGNORECASE)
            if signal_match:
                timestamp_str = signal_match.group(1)
                rsi_value = signal_match.group(2)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                if rsi_value in ['N/A', 'nan', 'None']:
                    rsi_val = None
                else:
                    try:
                        rsi_val = float(rsi_value)
                    except:
                        rsi_val = None
                
                signal_events.append({
                    'timestamp': timestamp,
                    'rsi_value': rsi_val
                })
    
    return rsi_timeline, rsi_period_changes, signal_events

def find_warmup_transition(rsi_timeline):
    """Find when RSI transitions from N/A to valid values."""
    first_valid = None
    warmup_bars = 0
    
    for entry in rsi_timeline:
        if entry['rsi_value'] is not None:
            first_valid = entry
            warmup_bars = entry['bar_number']
            break
    
    return first_valid, warmup_bars

def display_first_n_bars(rsi_timeline, n=50):
    """Display the first N bars of RSI values."""
    print(f"{'Bar':>5} {'Timestamp':>20} {'RSI Value':>10}")
    print("-" * 40)
    for entry in rsi_timeline[:n]:
        rsi_str = f"{entry['rsi_value']:.2f}" if entry['rsi_value'] is not None else "N/A"
        print(f"{entry['bar_number']:>5} {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {rsi_str:>10}")

def compare_warmup_timing(prod_timeline, opt_timeline):
    """Compare warmup timing between production and optimizer."""
    print("\n\n3. WARMUP TIMING COMPARISON")
    print("-" * 40)
    
    # Find first valid RSI in each
    prod_first, prod_warmup = find_warmup_transition(prod_timeline)
    opt_first, opt_warmup = find_warmup_transition(opt_timeline)
    
    if prod_first and opt_first:
        print(f"\nProduction warmup: {prod_warmup} bars")
        print(f"Optimizer warmup:  {opt_warmup} bars")
        print(f"Difference: {abs(prod_warmup - opt_warmup)} bars")
        
        # Compare timestamps
        time_diff = prod_first['timestamp'] - opt_first['timestamp']
        print(f"\nFirst valid RSI timestamps:")
        print(f"  Production: {prod_first['timestamp']} (Bar #{prod_first['bar_number']})")
        print(f"  Optimizer:  {opt_first['timestamp']} (Bar #{opt_first['bar_number']})")
        print(f"  Time difference: {time_diff}")
    
    # Compare RSI values at matching timestamps
    print("\n\nRSI VALUES AT MATCHING TIMESTAMPS (first 20 matches):")
    print(f"{'Timestamp':>20} {'Prod RSI':>10} {'Opt RSI':>10} {'Difference':>10}")
    print("-" * 60)
    
    # Create timestamp lookup for optimizer
    opt_by_time = {entry['timestamp']: entry for entry in opt_timeline}
    
    matches = 0
    for prod_entry in prod_timeline:
        if prod_entry['timestamp'] in opt_by_time:
            opt_entry = opt_by_time[prod_entry['timestamp']]
            
            prod_val = prod_entry['rsi_value']
            opt_val = opt_entry['rsi_value']
            
            if prod_val is not None and opt_val is not None:
                diff = abs(prod_val - opt_val)
                print(f"{prod_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {prod_val:>10.2f} {opt_val:>10.2f} {diff:>10.2f}")
            elif prod_val is None and opt_val is None:
                print(f"{prod_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            else:
                prod_str = f"{prod_val:.2f}" if prod_val is not None else "N/A"
                opt_str = f"{opt_val:.2f}" if opt_val is not None else "N/A"
                print(f"{prod_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):>20} {prod_str:>10} {opt_str:>10} {'MISMATCH':>10}")
            
            matches += 1
            if matches >= 20:
                break

def analyze_rsi_warmup():
    print("RSI Warmup Timing Analysis")
    print("=" * 80)
    
    # Parse both log files
    prod_timeline, prod_period_changes, prod_signals = parse_log_file('logs/admf_20250524_203019.log')
    
    if not prod_timeline:
        print("No RSI data found in production log")
        return
    
    # Analyze production log
    print("\n1. PRODUCTION RUN ANALYSIS")
    print("-" * 40)
    
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
    print("\nFirst 50 RSI values:")
    display_first_n_bars(prod_timeline, 50)
    
    # Analyze optimizer log
    print("\n\n2. OPTIMIZER RUN ANALYSIS")
    print("-" * 40)
    
    opt_timeline, opt_period_changes, opt_signals = parse_log_file('logs/admf_20250523_230532.log')
    
    if not opt_timeline:
        print("No RSI data found in optimizer log")
        return
    
    print(f"Total RSI data points found: {len(opt_timeline)}")
    
    # Find warmup transition
    first_valid, warmup_bars = find_warmup_transition(opt_timeline)
    if first_valid:
        print(f"\nRSI Warmup completed at:")
        print(f"  Bar #{first_valid['bar_number']}")
        print(f"  Timestamp: {first_valid['timestamp']}")
        print(f"  First valid RSI value: {first_valid['rsi_value']:.2f}")
        print(f"  Warmup period: {warmup_bars} bars")
    
    # Show RSI period changes
    if opt_period_changes:
        print(f"\nRSI Period Changes:")
        for change in opt_period_changes:
            print(f"  {change['timestamp']}: Period = {change['period']}")
    else:
        print("\nNo RSI period changes detected")
    
    # Show first 50 bars
    print("\nFirst 50 RSI values:")
    display_first_n_bars(opt_timeline, 50)
    
    # Compare warmup timing
    compare_warmup_timing(prod_timeline, opt_timeline)

if __name__ == "__main__":
    analyze_rsi_warmup()