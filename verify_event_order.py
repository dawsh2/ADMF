#!/usr/bin/env python3
"""
Verify that events are processed in identical order between production and optimizer
"""
import re

def extract_event_sequence(log_file, target_timestamp="2024-03-28 14:30:00"):
    """Extract the exact sequence of events around a specific timestamp"""
    events = []
    capturing = False
    lines_captured = 0
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Start capturing when we see the target timestamp
        if target_timestamp in line and ('📊 BAR_' in line or 'Publishing event: Event(type=BAR' in line):
            capturing = True
            lines_captured = 0
        
        if capturing:
            # Extract different event types with their exact order
            if 'Publishing event: Event(type=' in line:
                event_match = re.search(r'Publishing event: Event\(type=(\w+)', line)
                if event_match:
                    event_type = event_match.group(1)
                    events.append(f"[{i:6d}] 1. PUBLISH {event_type} EVENT")
            
            elif 'subscribed for event type' in line and 'BAR' in line:
                events.append(f"[{i:6d}] 2. SUBSCRIBER COUNT for BAR")
            
            elif 'on_bar_event from' in line or 'on_bar from' in line:
                component_match = re.search(r'(on_bar(?:_event)?) from ([\w.]+)', line)
                if component_match:
                    method = component_match.group(1)
                    component = component_match.group(2)
                    events.append(f"[{i:6d}] 3. CALL {method} on {component}")
            
            elif '📊 BAR_' in line and 'INDICATORS:' in line:
                regime_match = re.search(r'Regime=(\w+)', line)
                regime = regime_match.group(1) if regime_match else 'unknown'
                events.append(f"[{i:6d}] 4. STRATEGY processes BAR (regime={regime})")
            
            elif 'RegimeDet.*received BAR event' in line:
                events.append(f"[{i:6d}] 5. REGIME DETECTOR receives BAR")
            
            elif 'Regime classification:' in line:
                regime_match = re.search(r'→ regime=(\w+)', line)
                regime = regime_match.group(1) if regime_match else 'unknown'
                events.append(f"[{i:6d}] 6. REGIME DETECTOR classifies: {regime}")
            
            elif 'SIGNAL GENERATED' in line:
                signal_match = re.search(r'Type=([+-]?\d+).*Regime=(\w+)', line)
                if signal_match:
                    signal_type = signal_match.group(1)
                    regime = signal_match.group(2)
                    events.append(f"[{i:6d}] 7. SIGNAL GENERATED type={signal_type} regime={regime}")
            
            elif 'Publishing event: Event(type=SIGNAL' in line:
                events.append(f"[{i:6d}] 8. PUBLISH SIGNAL EVENT")
            
            elif 'REGIME CHANGED:' in line and 'strategy' in line.lower():
                regime_match = re.search(r"'([^']+)' → '([^']+)'", line)
                if regime_match:
                    from_regime = regime_match.group(1)
                    to_regime = regime_match.group(2)
                    events.append(f"[{i:6d}] 9. STRATEGY notified: {from_regime} → {to_regime}")
            
            elif 'Publishing event: Event(type=CLASSIFICATION' in line:
                events.append(f"[{i:6d}] 10. PUBLISH CLASSIFICATION EVENT")
            
            elif 'Handler.*subscribed to event type' in line:
                handler_match = re.search(r"Handler '([^']+)' subscribed to event type '(\w+)'", line)
                if handler_match:
                    handler = handler_match.group(1)
                    event_type = handler_match.group(2)
                    events.append(f"[{i:6d}] SUBSCRIBE: {handler} to {event_type}")
            
            # Stop after capturing enough context
            lines_captured += 1
            if lines_captured > 100:
                break
    
    return events

def compare_event_sequences(prod_events, opt_events):
    """Compare event sequences to find differences"""
    print("\nEVENT SEQUENCE COMPARISON")
    print("=" * 100)
    
    # Normalize events to compare just the sequence, not line numbers
    prod_normalized = [e.split('] ', 1)[1] if '] ' in e else e for e in prod_events]
    opt_normalized = [e.split('] ', 1)[1] if '] ' in e else e for e in opt_events]
    
    # Find common prefix
    common_length = 0
    for i in range(min(len(prod_normalized), len(opt_normalized))):
        if prod_normalized[i] == opt_normalized[i]:
            common_length += 1
        else:
            break
    
    print(f"Events match for first {common_length} events")
    
    if common_length < min(len(prod_normalized), len(opt_normalized)):
        print(f"\nFIRST DIVERGENCE at event {common_length + 1}:")
        print(f"  Production: {prod_events[common_length] if common_length < len(prod_events) else 'N/A'}")
        print(f"  Optimizer:  {opt_events[common_length] if common_length < len(opt_events) else 'N/A'}")
    
    # Show side by side comparison
    print("\nDETAILED SEQUENCE (first 20 events):")
    print("-" * 100)
    print(f"{'PRODUCTION':^45} | {'OPTIMIZER':^45}")
    print("-" * 100)
    
    max_events = max(len(prod_events), len(opt_events))
    for i in range(min(20, max_events)):
        prod_event = prod_normalized[i] if i < len(prod_normalized) else "---"
        opt_event = opt_normalized[i] if i < len(opt_normalized) else "---"
        
        match = "✓" if prod_event == opt_event else "✗"
        print(f"{prod_event:^45} | {opt_event:^45} {match}")

def check_subscription_order(log_file):
    """Extract BAR event subscription order"""
    subscriptions = []
    
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if "subscribed to event type 'BAR'" in line or "subscribed to BAR events" in line:
                # Extract component/handler name
                handler_match = re.search(r"Handler '([^']+)'", line)
                component_match = re.search(r"(\w+) '([^']+)' subscribed", line)
                
                if handler_match:
                    handler = handler_match.group(1)
                    subscriptions.append((i, handler))
                elif component_match:
                    component_type = component_match.group(1)
                    component_name = component_match.group(2)
                    subscriptions.append((i, f"{component_type} '{component_name}'"))
    
    return subscriptions

def main():
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_203019.log"  # Latest production with matched order
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Optimizer log
    
    print("VERIFYING EVENT PROCESSING ORDER")
    print("=" * 100)
    print(f"Production: {prod_file}")
    print(f"Optimizer:  {opt_file}")
    
    # Check subscription order
    print("\n1. BAR EVENT SUBSCRIPTION ORDER:")
    print("-" * 100)
    
    prod_subs = check_subscription_order(prod_file)
    opt_subs = check_subscription_order(opt_file)
    
    print("Production subscriptions:")
    for line_num, handler in prod_subs[:10]:
        print(f"  Line {line_num:6d}: {handler}")
    
    print("\nOptimizer subscriptions:")
    for line_num, handler in opt_subs[:10]:
        print(f"  Line {line_num:6d}: {handler}")
    
    # Check event processing at a critical timestamp
    timestamp = "2024-03-28 14:30:00"  # Where we had regime mismatch
    
    print(f"\n2. EVENT PROCESSING SEQUENCE AT {timestamp}:")
    print("-" * 100)
    
    prod_events = extract_event_sequence(prod_file, timestamp)
    opt_events = extract_event_sequence(opt_file, timestamp)
    
    compare_event_sequences(prod_events[:30], opt_events[:30])
    
    # Check another timestamp
    timestamp2 = "2024-03-28 14:05:00"
    print(f"\n3. EVENT PROCESSING SEQUENCE AT {timestamp2}:")
    print("-" * 100)
    
    prod_events2 = extract_event_sequence(prod_file, timestamp2)
    opt_events2 = extract_event_sequence(opt_file, timestamp2)
    
    compare_event_sequences(prod_events2[:20], opt_events2[:20])

if __name__ == "__main__":
    main()