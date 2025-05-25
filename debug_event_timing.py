#!/usr/bin/env python3
"""
Debug why event timing differs between production and optimizer
"""
import re

def trace_single_bar_processing(log_file, timestamp="2024-03-28 14:30:00"):
    """Trace complete processing of a single bar"""
    events = []
    in_target_bar = False
    bar_count = 0
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Detect when we're processing the target timestamp
        if timestamp in line:
            if 'Publishing event: Event(type=BAR' in line:
                in_target_bar = True
                bar_count = 0
                events.append(f"\n{'='*60}")
                events.append(f"BAR EVENT FOR {timestamp}")
                events.append(f"{'='*60}")
        
        if in_target_bar:
            # Track all processing for this bar
            if 'Publishing event:' in line:
                event_match = re.search(r'Publishing event: Event\(type=(\w+)', line)
                if event_match:
                    event_type = event_match.group(1)
                    events.append(f"{bar_count:3d}. PUBLISH {event_type}")
                    if event_type != 'BAR':
                        bar_count += 1
            
            elif 'Processing event type' in line:
                proc_match = re.search(r"Processing event type '(\w+)' with handler", line)
                if proc_match:
                    events.append(f"     → Processing {proc_match.group(1)} event")
            
            elif 'Dispatching event type' in line:
                disp_match = re.search(r"Dispatching event type '(\w+)' to handler '([^']+)'", line)
                if disp_match:
                    events.append(f"     → Dispatch {disp_match.group(1)} to {disp_match.group(2)}")
            
            elif 'received BAR event' in line:
                comp_match = re.search(r'(\w+).*received BAR event', line)
                if comp_match:
                    events.append(f"     → {comp_match.group(1)} receives BAR")
            
            elif '📊 BAR_' in line and 'INDICATORS:' in line:
                events.append(f"     → Strategy processes indicators")
            
            elif 'SIGNAL GENERATED' in line:
                sig_match = re.search(r'Type=([+-]?\d+).*Regime=(\w+)', line)
                if sig_match:
                    events.append(f"     → SIGNAL: type={sig_match.group(1)} regime={sig_match.group(2)}")
            
            elif 'Regime classification:' in line:
                reg_match = re.search(r'→ regime=(\w+)', line)
                if reg_match:
                    events.append(f"     → RegimeDetector classifies: {reg_match.group(1)}")
            
            elif 'REGIME CHANGED:' in line:
                change_match = re.search(r"'([^']+)' → '([^']+)'", line)
                if change_match:
                    events.append(f"     → REGIME CHANGE: {change_match.group(1)} → {change_match.group(2)}")
            
            elif 'RegimeDet.*Stabilization' in line and 'SWITCH' in line:
                events.append(f"     → Regime stabilization complete")
            
            # Check if we've moved to the next bar
            if 'Publishing event: Event(type=BAR' in line and bar_count > 0:
                in_target_bar = False
                events.append(f"\nEND OF BAR PROCESSING")
                break
    
    return events

def analyze_processing_difference():
    """Analyze why processing differs"""
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_203019.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print("DETAILED BAR PROCESSING ANALYSIS")
    print("=" * 80)
    
    # Analyze a bar where regime changes
    timestamp = "2024-03-28 14:30:00"
    
    print(f"\nPRODUCTION processing of {timestamp}:")
    print("-" * 80)
    prod_events = trace_single_bar_processing(prod_file, timestamp)
    for event in prod_events:
        print(event)
    
    print(f"\n\nOPTIMIZER processing of {timestamp}:")
    print("-" * 80)
    opt_events = trace_single_bar_processing(opt_file, timestamp)
    for event in opt_events:
        print(event)
    
    # Check component states
    print("\n\nKEY INSIGHTS:")
    print("-" * 80)
    print("1. In Production, RegimeDetector processes BEFORE publishing CLASSIFICATION")
    print("2. In Optimizer, CLASSIFICATION is published BEFORE Strategy processes")
    print("3. This suggests different event handling or component states")

def check_component_states(log_file):
    """Check component states during initialization"""
    states = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            # Track state changes
            if "State of" in line and "after" in line:
                match = re.search(r"State of '([^']+)' after (\w+): (\w+)", line)
                if match:
                    component = match.group(1)
                    action = match.group(2)
                    state = match.group(3)
                    if component not in states:
                        states[component] = []
                    states[component].append((action, state))
    
    return states

def main():
    analyze_processing_difference()
    
    # Also check component states
    print("\n\nCOMPONENT STATE ANALYSIS:")
    print("=" * 80)
    
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_203019.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    prod_states = check_component_states(prod_file)
    opt_states = check_component_states(opt_file)
    
    print("\nProduction component states:")
    for comp, states in prod_states.items():
        if 'Regime' in comp or 'Strategy' in comp:
            print(f"  {comp}:")
            for action, state in states:
                print(f"    after {action}: {state}")
    
    print("\nOptimizer component states:")
    for comp, states in list(opt_states.items())[:10]:  # First 10
        if 'Regime' in comp or 'Strategy' in comp:
            print(f"  {comp}:")
            for action, state in states:
                print(f"    after {action}: {state}")

if __name__ == "__main__":
    main()