#!/usr/bin/env python3
"""
Debug BAR event timestamps to verify they're being processed correctly.
"""

import re
from collections import defaultdict

def extract_bar_events(log_file, limit=50):
    """Extract BAR event timestamps and related info."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for BAR event timestamps
    # Multiple patterns to catch different log formats
    patterns = [
        # Pattern 1: "Publishing BAR event for timestamp X"
        r"Publishing BAR event.*?timestamp[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})",
        # Pattern 2: "BAR event.*timestamp.*2024"
        r"BAR event.*?timestamp.*?([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})",
        # Pattern 3: Market data timestamps in regime detector
        r"Bar #\d+ at ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})",
        # Pattern 4: Strategy processing
        r"Processing bar.*?([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})",
    ]
    
    all_timestamps = []
    for pattern in patterns:
        matches = re.findall(pattern, content)
        all_timestamps.extend(matches)
    
    # Sort and deduplicate
    unique_timestamps = sorted(list(set(all_timestamps)))
    
    return unique_timestamps[:limit]

def analyze_timestamp_spacing(timestamps):
    """Analyze the spacing between timestamps."""
    if len(timestamps) < 2:
        return []
    
    spacings = []
    for i in range(1, len(timestamps)):
        # Parse timestamps
        t1 = timestamps[i-1]
        t2 = timestamps[i]
        
        # Extract time portions
        if ' ' in t1 and ' ' in t2:
            time1 = t1.split(' ')[1]
            time2 = t2.split(' ')[1]
            
            # Calculate difference in seconds
            h1, m1, s1 = map(int, time1.split(':'))
            h2, m2, s2 = map(int, time2.split(':'))
            
            total_seconds1 = h1 * 3600 + m1 * 60 + s1
            total_seconds2 = h2 * 3600 + m2 * 60 + s2
            
            diff = total_seconds2 - total_seconds1
            spacings.append(diff)
    
    return spacings

def main():
    print("="*80)
    print("BAR EVENT TIMESTAMP ANALYSIS")
    print("="*80)
    
    # Extract timestamps from both logs
    print("\nExtracting BAR event timestamps...")
    
    opt_timestamps = extract_bar_events('opt_test_phase_full.log')
    test_timestamps = extract_bar_events('independent_test_full.log')
    
    print(f"\nFound {len(opt_timestamps)} unique timestamps in optimization log")
    print(f"Found {len(test_timestamps)} unique timestamps in test log")
    
    # Show first 10 timestamps
    print("\n" + "-"*60)
    print("FIRST 10 BAR TIMESTAMPS:")
    print("-"*60)
    print(f"{'#':<5} {'Optimization':<25} {'Test Run':<25}")
    print("-"*60)
    
    for i in range(min(10, max(len(opt_timestamps), len(test_timestamps)))):
        opt_ts = opt_timestamps[i] if i < len(opt_timestamps) else "N/A"
        test_ts = test_timestamps[i] if i < len(test_timestamps) else "N/A"
        
        # Extract just times for comparison
        opt_time = opt_ts.split(' ')[1] if ' ' in opt_ts else opt_ts
        test_time = test_ts.split(' ')[1] if ' ' in test_ts else test_ts
        
        match = "✓" if opt_time == test_time else "✗"
        print(f"{i+1:<5} {opt_time:<25} {test_time:<25} {match}")
    
    # Analyze spacing
    print("\n" + "-"*60)
    print("TIMESTAMP SPACING ANALYSIS:")
    print("-"*60)
    
    opt_spacings = analyze_timestamp_spacing(opt_timestamps[:20])
    test_spacings = analyze_timestamp_spacing(test_timestamps[:20])
    
    print(f"\nOptimization timestamp spacings (seconds): {opt_spacings[:10]}")
    print(f"Test run timestamp spacings (seconds): {test_spacings[:10]}")
    
    # Check for 1-minute intervals
    opt_minute_gaps = [s for s in opt_spacings if s == 60]
    test_minute_gaps = [s for s in test_spacings if s == 60]
    
    print(f"\nOptimization: {len(opt_minute_gaps)} out of {len(opt_spacings)} are 60-second gaps")
    print(f"Test run: {len(test_minute_gaps)} out of {len(test_spacings)} are 60-second gaps")
    
    # Check for anomalies
    opt_zero_gaps = [s for s in opt_spacings if s == 0]
    test_zero_gaps = [s for s in test_spacings if s == 0]
    
    if opt_zero_gaps:
        print(f"\n⚠️  WARNING: Optimization has {len(opt_zero_gaps)} zero-second gaps!")
    if test_zero_gaps:
        print(f"⚠️  WARNING: Test run has {len(test_zero_gaps)} zero-second gaps!")
    
    # Look for actual BAR event processing
    print("\n" + "-"*60)
    print("CHECKING BAR EVENT PROCESSING:")
    print("-"*60)
    
    with open('opt_test_phase_full.log', 'r') as f:
        opt_content = f.read()
    
    # Count various event types
    bar_events = opt_content.count("EventType.BAR")
    classification_events = opt_content.count("EventType.CLASSIFICATION")
    published_bars = opt_content.count("Published BAR event")
    
    print(f"\nOptimization log:")
    print(f"  EventType.BAR mentions: {bar_events}")
    print(f"  EventType.CLASSIFICATION mentions: {classification_events}")
    print(f"  Published BAR events: {published_bars}")

if __name__ == "__main__":
    main()