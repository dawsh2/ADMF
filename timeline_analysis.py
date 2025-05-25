#!/usr/bin/env python3
"""
Detailed timeline analysis of regime changes and surrounding events
"""
import re
from datetime import datetime

def extract_timeline_events(log_file_path, source_name, max_events=100):
    """Extract regime changes and surrounding events with timestamps"""
    events = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Extract timestamp from log line
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})', line)
                if not timestamp_match:
                    continue
                    
                timestamp = timestamp_match.group(1)
                
                # Look for regime-related events
                event_type = None
                event_details = None
                
                if 'REGIME CHANGED:' in line:
                    regime_match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)'", line)
                    if regime_match:
                        event_type = "REGIME_CHANGE"
                        event_details = f"{regime_match.group(1)} → {regime_match.group(2)}"
                
                elif 'REGIME PARAMETER UPDATE:' in line:
                    param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", line)
                    if param_match:
                        event_type = "PARAM_UPDATE"
                        event_details = f"'{param_match.group(1)}': {param_match.group(2)}"
                
                elif '🚨 SIGNAL GENERATED' in line:
                    signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+)', line)
                    if signal_match:
                        event_type = "SIGNAL"
                        event_details = f"#{signal_match.group(1)} Type={signal_match.group(2)} Price={signal_match.group(3)} Regime={signal_match.group(4)}"
                
                elif 'on_bar' in line and 'Processing bar:' in line:
                    bar_match = re.search(r'Processing bar: ([^,]+)', line)
                    if bar_match:
                        event_type = "BAR_PROCESS"
                        event_details = f"Processing {bar_match.group(1)}"
                
                elif 'initialized' in line and source_name in line:
                    event_type = "INITIALIZATION"
                    event_details = "Strategy initialized"
                
                elif 'enable_adaptive_mode' in line or 'ADAPTIVE MODE' in line:
                    event_type = "ADAPTIVE_MODE"
                    event_details = "Adaptive mode enabled"
                
                # Store event if we found one
                if event_type:
                    events.append({
                        'source': source_name,
                        'line_num': line_num,
                        'timestamp': timestamp,
                        'event_type': event_type,
                        'event_details': event_details,
                        'full_line': line.strip()
                    })
                
                # Limit events to prevent overflow
                if len(events) >= max_events:
                    break
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return events

def compare_timelines(production_events, optimizer_events, focus_period_start="2024-03-28 13:30:00", focus_period_end="2024-03-28 15:30:00"):
    """Compare event timelines between production and optimizer"""
    
    print("=" * 120)
    print("DETAILED TIMELINE ANALYSIS")
    print("=" * 120)
    
    # Filter events to focus period
    def in_focus_period(timestamp_str):
        try:
            ts = datetime.fromisoformat(timestamp_str.replace('+00:00', ''))
            start = datetime.fromisoformat(focus_period_start)
            end = datetime.fromisoformat(focus_period_end)
            return start <= ts <= end
        except:
            return False
    
    prod_focus = [e for e in production_events if in_focus_period(e['timestamp'])]
    opt_focus = [e for e in optimizer_events if in_focus_period(e['timestamp'])]
    
    print(f"\nFocus Period: {focus_period_start} to {focus_period_end}")
    print(f"Production Events: {len(prod_focus)}")
    print(f"Optimizer Events:  {len(opt_focus)}")
    
    # Combine and sort all events by timestamp
    all_events = []
    for event in prod_focus:
        event['display_source'] = 'PROD'
        all_events.append(event)
    for event in opt_focus:
        event['display_source'] = 'OPT '
        all_events.append(event)
    
    # Sort by timestamp
    all_events.sort(key=lambda x: x['timestamp'])
    
    print(f"\nCombined Timeline (showing first 50 events):")
    print("-" * 120)
    print(f"{'Time':<20} {'Source':<6} {'Event Type':<15} {'Details':<70}")
    print("-" * 120)
    
    for i, event in enumerate(all_events[:50]):
        time_only = event['timestamp'].split(' ')[1].split('+')[0]  # Extract HH:MM:SS
        event_type = event['event_type']
        details = event['event_details'][:68] + "..." if len(event['event_details']) > 68 else event['event_details']
        
        print(f"{time_only:<20} {event['display_source']:<6} {event_type:<15} {details:<70}")
    
    # Focus on regime changes only
    print(f"\n" + "=" * 120)
    print("REGIME CHANGES ONLY")
    print("=" * 120)
    
    prod_regime_changes = [e for e in prod_focus if e['event_type'] == 'REGIME_CHANGE']
    opt_regime_changes = [e for e in opt_focus if e['event_type'] == 'REGIME_CHANGE']
    
    print(f"\nProduction Regime Changes ({len(prod_regime_changes)}):")
    print("-" * 80)
    for i, event in enumerate(prod_regime_changes[:20]):
        time_only = event['timestamp'].split(' ')[1].split('+')[0]
        print(f"  {i+1:2d}. {time_only} - {event['event_details']}")
    
    print(f"\nOptimizer Regime Changes ({len(opt_regime_changes)}):")
    print("-" * 80)
    for i, event in enumerate(opt_regime_changes[:20]):
        time_only = event['timestamp'].split(' ')[1].split('+')[0]
        print(f"  {i+1:2d}. {time_only} - {event['event_details']}")
    
    # Analyze the time difference
    if prod_regime_changes and opt_regime_changes:
        print(f"\nTime Difference Analysis:")
        print("-" * 50)
        
        first_prod = datetime.fromisoformat(prod_regime_changes[0]['timestamp'].replace('+00:00', ''))
        first_opt = datetime.fromisoformat(opt_regime_changes[0]['timestamp'].replace('+00:00', ''))
        
        time_diff = (first_prod - first_opt).total_seconds() / 60  # difference in minutes
        
        print(f"First Production regime change: {prod_regime_changes[0]['timestamp']}")
        print(f"First Optimizer regime change:  {opt_regime_changes[0]['timestamp']}")
        print(f"Time difference: {time_diff:.1f} minutes (Production is {'ahead' if time_diff > 0 else 'behind'})")
        
        # Check if this offset is consistent
        if len(prod_regime_changes) >= 3 and len(opt_regime_changes) >= 3:
            print(f"\nConsistency check (first 3 changes):")
            for i in range(min(3, len(prod_regime_changes), len(opt_regime_changes))):
                prod_time = datetime.fromisoformat(prod_regime_changes[i]['timestamp'].replace('+00:00', ''))
                opt_time = datetime.fromisoformat(opt_regime_changes[i]['timestamp'].replace('+00:00', ''))
                diff = (prod_time - opt_time).total_seconds() / 60
                print(f"  Change {i+1}: {diff:.1f} minute difference")
    
    print(f"\n" + "=" * 120)

def main():
    print("Timeline Analysis: Detailed Event Sequencing")
    
    # Extract timeline events from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nExtracting timeline from production log: {production_file}")
    production_events = extract_timeline_events(production_file, "PRODUCTION", max_events=200)
    
    print(f"Extracting timeline from optimizer log: {optimizer_file}")
    optimizer_events = extract_timeline_events(optimizer_file, "OPTIMIZER", max_events=200)
    
    # Compare the timelines
    compare_timelines(production_events, optimizer_events)

if __name__ == "__main__":
    main()