#!/usr/bin/env python3
"""
Analyze detailed signal timing differences between optimization and production
"""

import re
from datetime import datetime, timedelta
from collections import defaultdict

def extract_detailed_signals(log_file, is_optimization=False):
    """Extract signals with detailed timing analysis"""
    signals = []
    
    # For optimization, we need to identify the test phase
    in_test_phase = False
    test_start_time = datetime.strptime("2025-01-27 18:06:00", "%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check if we're in test phase for optimization logs
            if is_optimization:
                if "ADAPTIVE TEST" in line or "!!! ADAPTIVE TEST !!!" in line:
                    in_test_phase = True
                elif "Optimization complete" in line or "Enhanced Grid Search with Train/Test Ended" in line:
                    in_test_phase = False
            
            # Look for SIGNAL events
            if "Publishing event: Event(type=SIGNAL" in line:
                # Extract timestamp
                ts_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                if ts_match:
                    signal_time = ts_match.group(1).split('+')[0]
                    signal_dt = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase signals
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include signals from test period
                    if signal_dt < test_start_time:
                        continue
                    
                    # Extract signal details
                    signal_type_match = re.search(r"'signal_type': (-?[0-9]+)", line)
                    strength_match = re.search(r"'signal_strength': ([0-9.]+)", line)
                    price_match = re.search(r"'price_at_signal': ([0-9.]+)", line)
                    reason_match = re.search(r"'reason': '([^']+)'", line)
                    
                    if signal_type_match:
                        signal_type = int(signal_type_match.group(1))
                        action = "BUY" if signal_type == 1 else "SELL" if signal_type == -1 else "NEUTRAL"
                        
                        signal = {
                            'timestamp': signal_time,
                            'datetime': signal_dt,
                            'action': action,
                            'signal_type': signal_type,
                            'strength': float(strength_match.group(1)) if strength_match else None,
                            'price': float(price_match.group(1)) if price_match else None,
                            'reason': reason_match.group(1) if reason_match else None,
                        }
                        signals.append(signal)
    
    return signals

def analyze_signal_patterns(signals, name):
    """Analyze signal patterns"""
    print(f"\n=== {name.upper()} SIGNAL ANALYSIS ===")
    print(f"Total signals: {len(signals)}")
    
    if not signals:
        return
    
    # First and last signals
    first_signal = min(signals, key=lambda s: s['datetime'])
    last_signal = max(signals, key=lambda s: s['datetime'])
    
    print(f"First signal: {first_signal['timestamp']} ({first_signal['action']})")
    print(f"Last signal:  {last_signal['timestamp']} ({last_signal['action']})")
    
    # Signal frequency analysis
    signal_times = [s['datetime'] for s in signals]
    if len(signal_times) > 1:
        intervals = []
        for i in range(1, len(signal_times)):
            interval = (signal_times[i] - signal_times[i-1]).total_seconds() / 60.0  # minutes
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        print(f"Signal intervals: avg={avg_interval:.1f}min, min={min_interval:.1f}min, max={max_interval:.1f}min")
    
    # Action distribution
    actions = [s['action'] for s in signals]
    buy_count = actions.count('BUY')
    sell_count = actions.count('SELL')
    print(f"Actions: {buy_count} BUY, {sell_count} SELL")
    
    # Hourly distribution (test period should be across multiple days)
    hourly_dist = defaultdict(int)
    for signal in signals:
        hour = signal['datetime'].hour
        hourly_dist[hour] += 1
    
    print(f"Hourly distribution (top 5):")
    for hour, count in sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {hour:02d}:xx - {count} signals")

def find_signal_gaps(opt_signals, prod_signals):
    """Find where one system generates signals but the other doesn't"""
    print(f"\n=== SIGNAL GAP ANALYSIS ===")
    
    # Group signals by time windows (5-minute buckets)
    def get_time_bucket(dt, bucket_minutes=5):
        minutes = (dt.minute // bucket_minutes) * bucket_minutes
        return dt.replace(minute=minutes, second=0, microsecond=0)
    
    opt_buckets = defaultdict(list)
    prod_buckets = defaultdict(list)
    
    for signal in opt_signals:
        bucket = get_time_bucket(signal['datetime'])
        opt_buckets[bucket].append(signal)
    
    for signal in prod_signals:
        bucket = get_time_bucket(signal['datetime'])
        prod_buckets[bucket].append(signal)
    
    # Find buckets with signal differences
    all_buckets = set(opt_buckets.keys()) | set(prod_buckets.keys())
    differences = []
    
    for bucket in sorted(all_buckets):
        opt_count = len(opt_buckets.get(bucket, []))
        prod_count = len(prod_buckets.get(bucket, []))
        
        if opt_count != prod_count:
            differences.append({
                'bucket': bucket,
                'opt_count': opt_count,
                'prod_count': prod_count,
                'opt_signals': opt_buckets.get(bucket, []),
                'prod_signals': prod_buckets.get(bucket, [])
            })
    
    print(f"Found {len(differences)} time buckets with different signal counts")
    print(f"\nFirst 10 significant differences:")
    
    for i, diff in enumerate(differences[:10]):
        bucket_str = diff['bucket'].strftime("%Y-%m-%d %H:%M")
        print(f"  {bucket_str}: Opt={diff['opt_count']}, Prod={diff['prod_count']}")
        
        # Show actual signals in this bucket
        if diff['opt_signals']:
            opt_times = [s['timestamp'] for s in diff['opt_signals']]
            print(f"    Opt signals: {', '.join(opt_times)}")
        if diff['prod_signals']:
            prod_times = [s['timestamp'] for s in diff['prod_signals']]
            print(f"    Prod signals: {', '.join(prod_times)}")

def main():
    print("Analyzing detailed signal timing differences...")
    
    # Extract signals
    print("\nExtracting signals from optimization log...")
    opt_signals = extract_detailed_signals('logs/admf_20250523_202211.log', is_optimization=True)
    
    print("Extracting signals from production log...")
    prod_signals = extract_detailed_signals('logs/admf_20250523_202911.log', is_optimization=False)
    
    # Analyze patterns
    analyze_signal_patterns(opt_signals, "optimization")
    analyze_signal_patterns(prod_signals, "production")
    
    # Find gaps
    find_signal_gaps(opt_signals, prod_signals)

if __name__ == "__main__":
    main()