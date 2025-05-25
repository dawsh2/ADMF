#!/usr/bin/env python3
"""
Analyze regime alignment between production and optimizer runs at common timestamps
"""
import re
from datetime import datetime
from collections import defaultdict

def extract_regime_timeline(log_file_path, source_name):
    """Extract complete regime timeline from log"""
    regime_timeline = []
    current_regime = None
    regime_changes = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Track regime changes
            if "REGIME CHANGED:" in line:
                match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", line)
                if match:
                    from_regime = match.group(1)
                    to_regime = match.group(2)
                    timestamp = match.group(3)
                    
                    regime_changes.append({
                        'timestamp': timestamp,
                        'from': from_regime,
                        'to': to_regime,
                        'line_num': line_num
                    })
                    current_regime = to_regime
            
            # Track bar processing with regime
            if '📊 BAR_' in line and 'INDICATORS:' in line and current_regime:
                bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+).*Regime=(\w+)', line)
                if bar_match:
                    timestamp = bar_match.group(2)
                    price = float(bar_match.group(3))
                    bar_regime = bar_match.group(4)
                    
                    regime_timeline.append({
                        'timestamp': timestamp,
                        'price': price,
                        'regime': bar_regime,
                        'line_num': line_num
                    })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return [], []
    
    return regime_timeline, regime_changes

def compare_regimes_at_timestamps(prod_timeline, opt_timeline, prod_changes, opt_changes):
    """Compare regimes at common timestamps"""
    # Create dictionaries for easy lookup
    prod_by_time = {entry['timestamp']: entry for entry in prod_timeline}
    opt_by_time = {entry['timestamp']: entry for entry in opt_timeline}
    
    # Find common timestamps
    common_timestamps = sorted(set(prod_by_time.keys()) & set(opt_by_time.keys()))
    
    print("=" * 140)
    print("REGIME ALIGNMENT ANALYSIS")
    print("=" * 140)
    
    print(f"\n📊 REGIME CHANGE SUMMARY:")
    print(f"  Production: {len(prod_changes)} regime changes")
    print(f"  Optimizer:  {len(opt_changes)} regime changes")
    
    print(f"\n📅 TIMELINE COVERAGE:")
    print(f"  Production bars: {len(prod_timeline)}")
    print(f"  Optimizer bars:  {len(opt_timeline)}")
    print(f"  Common timestamps: {len(common_timestamps)}")
    
    # Show regime changes for both
    print(f"\n🔄 PRODUCTION REGIME CHANGES:")
    for i, change in enumerate(prod_changes[:10]):  # First 10
        print(f"  {i+1}. {change['timestamp']}: {change['from']} → {change['to']}")
    if len(prod_changes) > 10:
        print(f"  ... and {len(prod_changes) - 10} more")
    
    print(f"\n🔄 OPTIMIZER REGIME CHANGES:")
    for i, change in enumerate(opt_changes[:10]):  # First 10
        print(f"  {i+1}. {change['timestamp']}: {change['from']} → {change['to']}")
    if len(opt_changes) > 10:
        print(f"  ... and {len(opt_changes) - 10} more")
    
    # Analyze regime alignment
    print(f"\n📊 REGIME ALIGNMENT AT COMMON TIMESTAMPS:")
    print(f"{'Timestamp':^20} | {'Production':^20} | {'Optimizer':^20} | {'Match':^8}")
    print("-" * 75)
    
    matches = 0
    mismatches = 0
    sample_size = min(50, len(common_timestamps))  # Show first 50
    
    for ts in common_timestamps[:sample_size]:
        prod_regime = prod_by_time[ts]['regime']
        opt_regime = opt_by_time[ts]['regime']
        match = prod_regime == opt_regime
        
        if match:
            matches += 1
        else:
            mismatches += 1
        
        # Show mismatches and every 10th match
        if not match or (matches % 10 == 0 and match):
            match_str = "✓" if match else "✗"
            print(f"{ts[:19]:^20} | {prod_regime:^20} | {opt_regime:^20} | {match_str:^8}")
    
    # Count total matches/mismatches
    total_matches = 0
    total_mismatches = 0
    regime_mismatch_counts = defaultdict(int)
    
    for ts in common_timestamps:
        prod_regime = prod_by_time[ts]['regime']
        opt_regime = opt_by_time[ts]['regime']
        
        if prod_regime == opt_regime:
            total_matches += 1
        else:
            total_mismatches += 1
            mismatch_key = f"{prod_regime} vs {opt_regime}"
            regime_mismatch_counts[mismatch_key] += 1
    
    print(f"\n📊 ALIGNMENT STATISTICS:")
    print(f"  Total matches: {total_matches} ({total_matches/len(common_timestamps)*100:.1f}%)")
    print(f"  Total mismatches: {total_mismatches} ({total_mismatches/len(common_timestamps)*100:.1f}%)")
    
    print(f"\n🔍 TOP MISMATCHES:")
    sorted_mismatches = sorted(regime_mismatch_counts.items(), key=lambda x: x[1], reverse=True)
    for mismatch, count in sorted_mismatches[:5]:
        print(f"  {mismatch}: {count} times")
    
    # Find periods of sustained mismatch
    print(f"\n⚠️  SUSTAINED MISMATCH PERIODS:")
    mismatch_start = None
    mismatch_count = 0
    
    for i, ts in enumerate(common_timestamps):
        prod_regime = prod_by_time[ts]['regime']
        opt_regime = opt_by_time[ts]['regime']
        
        if prod_regime != opt_regime:
            if mismatch_start is None:
                mismatch_start = ts
                mismatch_count = 1
            else:
                mismatch_count += 1
        else:
            if mismatch_start and mismatch_count >= 5:  # At least 5 consecutive mismatches
                print(f"  {mismatch_start[:19]} to {common_timestamps[i-1][:19]} ({mismatch_count} bars)")
            mismatch_start = None
            mismatch_count = 0

def main():
    print("Regime Alignment Analysis")
    
    # Extract regime timelines
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print(f"\nExtracting production regimes: {production_file}")
    prod_timeline, prod_changes = extract_regime_timeline(production_file, "PRODUCTION")
    
    print(f"Extracting optimizer regimes: {optimizer_file}")
    opt_timeline, opt_changes = extract_regime_timeline(optimizer_file, "OPTIMIZER")
    
    # Compare regimes
    compare_regimes_at_timestamps(prod_timeline, opt_timeline, prod_changes, opt_changes)

if __name__ == "__main__":
    main()