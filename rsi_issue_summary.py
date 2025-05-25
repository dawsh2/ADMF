#!/usr/bin/env python3
"""
Analyze RSI issues in optimizer vs production runs.

This script:
1. Summarizes the RSI issues found
2. Shows that production RSI warms up normally
3. Shows that optimizer RSI never becomes valid due to bar counter resets
4. Extracts evidence from logs showing the issue
5. Provides recommendations for fixing
"""

import re
import json
from datetime import datetime
from collections import defaultdict
import os

def analyze_log_file(log_file):
    """Analyze a log file for RSI issues."""
    production_data = []
    optimizer_data = []
    bar_counter_resets = []
    rsi_validity_changes = []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        return None, None, None
    
    for i, line in enumerate(lines):
        # Extract BAR indicator updates
        bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS:.*RSI=([^,]+)', line)
        if bar_match:
            bar_num = int(bar_match.group(1))
            timestamp_str = bar_match.group(2)
            rsi_value_str = bar_match.group(3).strip()
            
            # Parse timestamp
            if '+' in timestamp_str:
                timestamp_str = timestamp_str.split('+')[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                continue
            
            # Parse RSI value
            is_valid = rsi_value_str not in ['N/A', 'nan', 'None']
            rsi_val = None
            if is_valid:
                try:
                    rsi_val = float(rsi_value_str)
                except:
                    is_valid = False
            
            data_entry = {
                'timestamp': timestamp,
                'bar_number': bar_num,
                'rsi_value': rsi_val,
                'is_valid': is_valid,
                'bar_counter': bar_num,
                'line_num': i + 1
            }
            
            # Determine if this is production or optimizer
            if 'OPTIMIZER' in line or 'adaptive test' in line.lower():
                data_entry['mode'] = 'optimizer'
                optimizer_data.append(data_entry)
            else:
                data_entry['mode'] = 'production'
                production_data.append(data_entry)
            
            # Check for bar counter resets
            if bar_num == 1 and i > 0:
                # Look back for previous bar number
                prev_bar_num = None
                for j in range(i-1, max(0, i-100), -1):
                    prev_match = re.search(r'📊 BAR_(\d+)', lines[j])
                    if prev_match:
                        prev_bar_num = int(prev_match.group(1))
                        break
                
                if prev_bar_num and prev_bar_num > 1:
                    bar_counter_resets.append({
                        'timestamp': timestamp,
                        'line_num': i + 1,
                        'prev_counter': prev_bar_num,
                        'mode': data_entry['mode'],
                        'message': line.strip()
                    })
        
        # Check for RSI validity changes
        if 'RSI' in line and ('first valid value' in line or 'warmup complete' in line.lower()):
            rsi_validity_changes.append({
                'line_num': i + 1,
                'message': line.strip()
            })
    
    return {
        'production': production_data,
        'optimizer': optimizer_data
    }, bar_counter_resets, rsi_validity_changes

def print_summary(rsi_data, bar_counter_resets, rsi_validity_changes):
    """Print a summary of RSI issues."""
    print("=" * 80)
    print("RSI ISSUE SUMMARY REPORT")
    print("=" * 80)
    
    production_data = rsi_data['production']
    optimizer_data = rsi_data['optimizer']
    
    # 1. Production RSI behavior
    print("\n1. PRODUCTION RSI BEHAVIOR:")
    print("-" * 40)
    if production_data:
        first_valid = next((d for d in production_data if d['is_valid']), None)
        if first_valid:
            print(f"RSI becomes valid at bar #{first_valid['bar_number']}")
            print(f"Timestamp: {first_valid['timestamp']}")
            print(f"First valid value: {first_valid['rsi_value']:.2f}")
        else:
            print("RSI never becomes valid in production")
        
        # Count valid vs invalid
        valid_count = sum(1 for d in production_data if d['is_valid'])
        invalid_count = len(production_data) - valid_count
        print(f"\nTotal bars: {len(production_data)}")
        print(f"Valid RSI: {valid_count} ({valid_count/len(production_data)*100:.1f}%)")
        print(f"Invalid RSI: {invalid_count} ({invalid_count/len(production_data)*100:.1f}%)")
    else:
        print("No production RSI data found")
    
    # 2. Optimizer RSI behavior
    print("\n2. OPTIMIZER RSI BEHAVIOR:")
    print("-" * 40)
    if optimizer_data:
        first_valid = next((d for d in optimizer_data if d['is_valid']), None)
        if first_valid:
            print(f"RSI becomes valid at bar #{first_valid['bar_number']}")
            print(f"Timestamp: {first_valid['timestamp']}")
            print(f"First valid value: {first_valid['rsi_value']:.2f}")
        else:
            print("RSI NEVER becomes valid in optimizer!")
        
        # Count valid vs invalid
        valid_count = sum(1 for d in optimizer_data if d['is_valid'])
        invalid_count = len(optimizer_data) - valid_count
        print(f"\nTotal bars: {len(optimizer_data)}")
        print(f"Valid RSI: {valid_count} ({valid_count/len(optimizer_data)*100:.1f}%)")
        print(f"Invalid RSI: {invalid_count} ({invalid_count/len(optimizer_data)*100:.1f}%)")
        
        # Check bar counter pattern
        bar_counts = [d['bar_counter'] for d in optimizer_data]
        unique_counts = sorted(set(bar_counts))
        if unique_counts == [1]:
            print("\n⚠️  CRITICAL: All bars show as BAR_001 - counter is being reset!")
    else:
        print("No optimizer RSI data found")
    
    # 3. Bar counter reset evidence
    print("\n3. BAR COUNTER RESET EVIDENCE:")
    print("-" * 40)
    if bar_counter_resets:
        print(f"Found {len(bar_counter_resets)} bar counter resets")
        
        # Show first 5 resets
        for i, reset in enumerate(bar_counter_resets[:5]):
            print(f"\n  Reset #{i+1}:")
            print(f"  Timestamp: {reset['timestamp']}")
            print(f"  Mode: {reset['mode']}")
            print(f"  Counter went from {reset['prev_counter']} → 1")
            print(f"  Message: {reset['message'][:100]}...")
        
        if len(bar_counter_resets) > 5:
            print(f"\n  ... and {len(bar_counter_resets) - 5} more resets")
        
        # Group resets by mode
        resets_by_mode = defaultdict(int)
        for reset in bar_counter_resets:
            resets_by_mode[reset['mode']] += 1
        
        print("\n  Resets by mode:")
        for mode, count in resets_by_mode.items():
            print(f"    {mode}: {count} resets")
    else:
        print("No bar counter resets detected")
    
    # 4. RSI validity changes
    print("\n4. RSI VALIDITY CHANGES:")
    print("-" * 40)
    if rsi_validity_changes:
        for change in rsi_validity_changes[:3]:
            print(f"  Line {change['line_num']}: {change['message'][:100]}...")
    else:
        print("No RSI validity changes detected")
    
    # 5. Root cause analysis
    print("\n5. ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    print("The issue is that the optimizer's bar counter keeps resetting to 1, preventing")
    print("the RSI from accumulating the required 15 bars of data to become valid.")
    print()
    print("Evidence:")
    
    # Calculate reset frequency for optimizer
    optimizer_resets = [r for r in bar_counter_resets if r['mode'] == 'optimizer']
    if optimizer_resets and optimizer_data:
        print(f"- Optimizer had {len(optimizer_resets)} bar counter resets")
        print(f"- Total optimizer RSI checks: {len(optimizer_data)}")
        reset_ratio = len(optimizer_resets) / len(optimizer_data) * 100
        print(f"- Reset frequency: {reset_ratio:.1f}%")
    
    # Check max consecutive bar count in optimizer
    if optimizer_data:
        max_consecutive = 0
        current_consecutive = 0
        for entry in optimizer_data:
            counter = entry.get('bar_counter', 0)
            if counter == 1:
                current_consecutive = 1
            else:
                current_consecutive = counter
            max_consecutive = max(max_consecutive, current_consecutive)
        
        print(f"- Maximum consecutive bar count in optimizer: {max_consecutive}")
        if max_consecutive < 15:
            print(f"  ⚠️  This is less than the 15 bars needed for RSI warmup!")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS FOR FIXING:")
    print("-" * 40)
    print("1. **Indicator State Persistence**: The optimizer needs to maintain indicator")
    print("   state across evaluations. Currently, indicators are being reset between")
    print("   optimization iterations.")
    print()
    print("2. **Shared Indicator Instances**: Use the same indicator instances throughout")
    print("   the optimization process rather than creating new ones for each evaluation.")
    print()
    print("3. **Warm-up Period**: Implement a proper warm-up period in the optimizer to")
    print("   ensure indicators have sufficient data before evaluation begins.")
    print()
    print("4. **State Management**: Add proper state management in the optimizer to track")
    print("   and preserve indicator states between parameter evaluations.")
    print()
    print("5. **Quick Fix**: As a temporary solution, pre-warm indicators with historical")
    print("   data before starting the optimization process.")

def main():
    print("Analyzing RSI issues in ADMF logs...")
    
    # Log files to analyze
    production_log = "logs/admf_20250524_203019.log"
    optimizer_log = "logs/admf_20250523_230532.log"
    
    # Analyze production log
    print(f"\nAnalyzing production log: {production_log}")
    if os.path.exists(production_log):
        prod_data, prod_resets, prod_changes = analyze_log_file(production_log)
    else:
        print(f"Production log not found: {production_log}")
        prod_data = {'production': [], 'optimizer': []}
        prod_resets = []
        prod_changes = []
    
    # Analyze optimizer log
    print(f"Analyzing optimizer log: {optimizer_log}")
    if os.path.exists(optimizer_log):
        opt_data, opt_resets, opt_changes = analyze_log_file(optimizer_log)
    else:
        print(f"Optimizer log not found: {optimizer_log}")
        opt_data = {'production': [], 'optimizer': []}
        opt_resets = []
        opt_changes = []
    
    # Combine data
    combined_data = {
        'production': prod_data['production'] + opt_data['production'],
        'optimizer': prod_data['optimizer'] + opt_data['optimizer']
    }
    combined_resets = prod_resets + opt_resets
    combined_changes = prod_changes + opt_changes
    
    # Print summary
    print_summary(combined_data, combined_resets, combined_changes)
    
    # Save summary data
    summary_data = {
        'production_entries': len(combined_data['production']),
        'optimizer_entries': len(combined_data['optimizer']),
        'bar_counter_resets': len(combined_resets),
        'rsi_validity_changes': len(combined_changes),
        'production_first_valid': next((d for d in combined_data['production'] if d['is_valid']), None),
        'optimizer_first_valid': next((d for d in combined_data['optimizer'] if d['is_valid']), None)
    }
    
    with open('rsi_issue_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("Summary data saved to: rsi_issue_summary.json")

if __name__ == "__main__":
    main()