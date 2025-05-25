#!/usr/bin/env python3
"""
Analyze why event processing order differs between production and optimizer
"""
import re

def trace_event_flow(log_file, timestamp, source_name):
    """Trace the exact flow of events for a specific bar"""
    events = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    capturing = False
    bar_count = 0
    
    for i, line in enumerate(lines):
        # Start capturing when we see the target timestamp
        if timestamp in line and '📊 BAR_' in line:
            capturing = True
            bar_count = 0
        
        if capturing:
            # Extract relevant events
            if 'Publishing event: Event(type=BAR' in line:
                events.append(f"[{i:6d}] 1. BAR EVENT PUBLISHED")
            
            elif '📊 BAR_' in line:
                match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\].*Regime=(\w+)', line)
                if match:
                    events.append(f"[{i:6d}] 2. STRATEGY processes bar - Regime: {match.group(3)}")
            
            elif 'RegimeDet.*received BAR event' in line:
                events.append(f"[{i:6d}] 3. REGIME DETECTOR receives BAR event")
            
            elif 'Regime classification:' in line:
                match = re.search(r'→ regime=(\w+)', line)
                if match:
                    events.append(f"[{i:6d}] 4. REGIME DETECTOR classifies: {match.group(1)}")
            
            elif 'SIGNAL GENERATED' in line:
                match = re.search(r'Type=([+-]?\d+).*Regime=(\w+)', line)
                if match:
                    events.append(f"[{i:6d}] 5. SIGNAL GENERATED Type={match.group(1)} in Regime={match.group(2)}")
            
            elif 'Publishing event: Event(type=SIGNAL' in line:
                events.append(f"[{i:6d}] 6. SIGNAL EVENT PUBLISHED")
            
            elif 'REGIME CHANGED:' in line and 'strategy' in line.lower():
                match = re.search(r"'([^']+)' → '([^']+)'", line)
                if match:
                    events.append(f"[{i:6d}] 7. STRATEGY notified of regime change: {match.group(1)} → {match.group(2)}")
            
            elif 'RegimeDet.*Stabilization.*SWITCH' in line:
                match = re.search(r"from '([^']+)' to '([^']+)'", line)
                if match:
                    events.append(f"[{i:6d}] 8. REGIME DETECTOR switches: {match.group(1)} → {match.group(2)}")
            
            elif 'Publishing event: Event(type=CLASSIFICATION' in line:
                events.append(f"[{i:6d}] 9. CLASSIFICATION EVENT PUBLISHED")
            
            # Stop after processing the bar and its effects
            bar_count += 1
            if bar_count > 50 and 'BAR_' in line:
                break
    
    return events

def analyze_subscription_order(log_file):
    """Analyze the order of event subscriptions"""
    subscriptions = []
    
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if 'subscribed to' in line.lower() or 'subscribe' in line:
                if 'BAR' in line:
                    component = "Unknown"
                    if 'strategy' in line.lower():
                        component = "Strategy"
                    elif 'regime' in line.lower():
                        component = "RegimeDetector"
                    elif 'portfolio' in line.lower():
                        component = "Portfolio"
                    
                    subscriptions.append(f"Line {i:6d}: {component} subscribed to BAR events")
    
    return subscriptions

def main():
    print("EVENT PROCESSING ORDER ANALYSIS")
    print("=" * 80)
    
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_002523.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    # Analyze the 14:30:00 case
    timestamp = "2024-03-28 14:30:00"
    
    print(f"\n1. EVENT FLOW FOR BAR AT {timestamp}")
    print("-" * 80)
    
    print("\nPRODUCTION EVENT FLOW:")
    prod_events = trace_event_flow(prod_file, timestamp, "Production")
    for event in prod_events[:15]:  # Show first 15 events
        print(f"  {event}")
    
    print("\nOPTIMIZER EVENT FLOW:")
    opt_events = trace_event_flow(opt_file, timestamp, "Optimizer")
    for event in opt_events[:15]:  # Show first 15 events
        print(f"  {event}")
    
    print("\n2. SUBSCRIPTION ORDER ANALYSIS")
    print("-" * 80)
    
    print("\nPRODUCTION SUBSCRIPTIONS:")
    prod_subs = analyze_subscription_order(prod_file)
    for sub in prod_subs[:10]:
        print(f"  {sub}")
    
    print("\nOPTIMIZER SUBSCRIPTIONS:")
    opt_subs = analyze_subscription_order(opt_file)
    for sub in opt_subs[:10]:
        print(f"  {sub}")
    
    print("\n3. KEY INSIGHTS")
    print("-" * 80)
    print("\nThe event processing order differs because:")
    print("1. Components may subscribe to events in different order")
    print("2. Event bus processes subscribers in order of subscription")
    print("3. Production vs Optimizer may initialize components differently")
    print("4. Timing of regime change detection can shift the sequence")
    
    print("\n4. RACE CONDITION EXPLANATION")
    print("-" * 80)
    print("\nAt regime boundaries:")
    print("- If Strategy processes BAR before RegimeDetector changes regime:")
    print("  → Signal uses old regime")
    print("- If RegimeDetector changes regime before Strategy processes BAR:")
    print("  → Signal uses new regime")
    print("\nThis depends on:")
    print("- Subscription order")
    print("- Processing speed")
    print("- Prior state (pending regime changes)")

if __name__ == "__main__":
    main()