#!/usr/bin/env python3
"""Debug event ordering differences between optimization and production"""

import re
from datetime import datetime

def extract_events_at_timestamp(log_file, target_timestamp, context_before=10, context_after=10):
    """Extract all events around a specific timestamp"""
    events = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if target_timestamp in line:
            # Get context lines
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after + 1)
            
            for j in range(start, end):
                # Extract event type and key info
                if 'Publishing event:' in lines[j]:
                    event_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}).*Event\(type=(\w+).*', lines[j])
                    if event_match:
                        log_time = event_match.group(1)
                        event_type = event_match.group(2)
                        
                        # Extract additional context based on event type
                        context = ""
                        if event_type == "SIGNAL":
                            sig_match = re.search(r"signal_type': ([+-]?\d+).*Regime: (\w+)", lines[j])
                            if sig_match:
                                context = f"signal={sig_match.group(1)}, regime={sig_match.group(2)}"
                        elif event_type == "CLASSIFICATION":
                            class_match = re.search(r"classification': '(\w+)'.*previous_classification': '(\w+)'", lines[j])
                            if class_match:
                                context = f"{class_match.group(2)} -> {class_match.group(1)}"
                            else:
                                class_match = re.search(r"classification': '(\w+)'", lines[j])
                                if class_match:
                                    context = f"regime={class_match.group(1)}"
                        elif event_type == "BAR":
                            context = "price data"
                            
                        events.append((log_time, event_type, context, j - i))
                        
    return events

print("EVENT ORDERING ANALYSIS")
print("=" * 80)
print("\nAnalyzing events around 2024-04-24 16:52:00 (first divergence point)")
print("-" * 80)

# Extract events from both logs
prod_events = extract_events_at_timestamp('/Users/daws/ADMF/logs/admf_20250522_225839.log', '2024-04-24 16:52:00')
opt_events = extract_events_at_timestamp('/Users/daws/ADMF/logs/admf_20250522_225752.log', '2024-04-24 16:52:00')

print("\nPRODUCTION EVENT SEQUENCE:")
print(f"{'Time':<12} {'Event Type':<15} {'Details':<40} {'Offset'}")
print("-" * 80)
for time, event_type, context, offset in prod_events:
    print(f"{time:<12} {event_type:<15} {context:<40} {offset:>3}")

print("\nOPTIMIZATION EVENT SEQUENCE:")
print(f"{'Time':<12} {'Event Type':<15} {'Details':<40} {'Offset'}")
print("-" * 80)
for time, event_type, context, offset in opt_events:
    print(f"{time:<12} {event_type:<15} {context:<40} {offset:>3}")

# Find key differences
print("\nKEY OBSERVATIONS:")
print("-" * 80)

# Check if SIGNAL comes before or after CLASSIFICATION
prod_signal_idx = next((i for i, (_, t, _, _) in enumerate(prod_events) if t == "SIGNAL" and "16:52" in prod_events[i][2]), None)
prod_class_idx = next((i for i, (_, t, c, _) in enumerate(prod_events) if t == "CLASSIFICATION" and "trending_down -> default" in c), None)

opt_signal_idx = next((i for i, (_, t, _, _) in enumerate(opt_events) if t == "SIGNAL" and "16:52" in opt_events[i][2]), None)
opt_class_idx = next((i for i, (_, t, c, _) in enumerate(opt_events) if t == "CLASSIFICATION" and "trending_down -> default" in c), None)

if prod_signal_idx is not None and prod_class_idx is not None:
    if prod_signal_idx < prod_class_idx:
        print("Production: SIGNAL published BEFORE regime change (uses old regime)")
    else:
        print("Production: SIGNAL published AFTER regime change (uses new regime)")
        
if opt_signal_idx is not None and opt_class_idx is not None:
    if opt_signal_idx < opt_class_idx:
        print("Optimization: SIGNAL published BEFORE regime change (uses old regime)")
    else:
        print("Optimization: SIGNAL published AFTER regime change (uses new regime)")
elif opt_signal_idx is None:
    print("Optimization: NO SIGNAL generated at 16:52 (possibly due to regime already changed)")

print("\nThis event ordering difference explains why the signals diverge!")