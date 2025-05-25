#!/usr/bin/env python3
"""
Comprehensive diagnostic to find the root cause of signal mismatch.
Focus on event ordering, component initialization, and state differences.
"""

import subprocess
import os
import json
import re
from collections import OrderedDict, defaultdict

def extract_detailed_execution_flow(log_file, source_name):
    """Extract detailed execution flow including event ordering."""
    flow_data = {
        'component_init_order': [],
        'event_subscriptions': defaultdict(list),
        'event_publications': [],
        'component_setup_order': [],
        'component_start_order': [],
        'first_bar_processing': {},
        'signal_generation_context': [],
        'parameters_applied': defaultdict(dict),
        'state_at_first_signal': {}
    }
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Component creation order
        if "Component.*created. State: CREATED" in line:
            match = re.search(r"component\.(\w+).*Component '([^']+)' created", line)
            if match:
                comp_name = match.group(2)
                if comp_name not in flow_data['component_init_order']:
                    flow_data['component_init_order'].append(comp_name)
        
        # Event subscriptions
        if "subscribed to" in line and "events" in line:
            match = re.search(r"'([^']+)'.*subscribed to (\w+) events", line)
            if match:
                comp_name = match.group(1)
                event_type = match.group(2)
                flow_data['event_subscriptions'][event_type].append(comp_name)
        
        # Component setup order
        if "setup complete" in line:
            match = re.search(r"([^']+).*setup complete", line)
            if match:
                comp_name = match.group(1).split("'")[0].strip()
                if comp_name not in flow_data['component_setup_order']:
                    flow_data['component_setup_order'].append(comp_name)
        
        # Component start order
        if "started" in line and "State:" not in line:
            match = re.search(r"([^']+).*started", line)
            if match:
                comp_name = match.group(1).split("'")[0].strip()
                if comp_name not in flow_data['component_start_order']:
                    flow_data['component_start_order'].append(comp_name)
        
        # First BAR event processing
        if "BAR_001" in line or "BAR_1 " in line:
            flow_data['first_bar_processing']['line'] = line.strip()
            # Look for subsequent processing
            for j in range(i, min(i+20, len(lines))):
                if "handle" in lines[j] or "process" in lines[j]:
                    flow_data['first_bar_processing']['handlers'] = flow_data['first_bar_processing'].get('handlers', [])
                    flow_data['first_bar_processing']['handlers'].append(lines[j].strip())
        
        # Parameter settings
        if "parameters updated" in line or "Setting parameters" in line:
            match = re.search(r"'([^']+)'.*parameters.*: (.+)", line)
            if match:
                comp_name = match.group(1)
                params_str = match.group(2)
                flow_data['parameters_applied'][comp_name][line.split()[0]] = params_str
        
        # Signal generation context
        if "SIGNAL GENERATED" in line:
            context = {
                'line': line.strip(),
                'timestamp': line.split()[0],
                'signal_num': re.search(r'#(\d+)', line).group(1) if re.search(r'#(\d+)', line) else None
            }
            # Get surrounding context
            for j in range(max(0, i-5), min(i+5, len(lines))):
                if "Price=" in lines[j] or "MA_" in lines[j] or "regime" in lines[j].lower():
                    context['context'] = context.get('context', [])
                    context['context'].append(lines[j].strip())
            flow_data['signal_generation_context'].append(context)
    
    return flow_data

def compare_execution_flows():
    """Compare optimizer vs production execution flows."""
    print("COMPARING EXECUTION FLOWS")
    print("=" * 60)
    
    # Find recent optimizer and production logs
    import glob
    
    # Get most recent enhanced optimizer log
    opt_logs = sorted(glob.glob("logs/enhanced*.log"), key=os.path.getmtime, reverse=True)
    if not opt_logs:
        print("No optimizer logs found")
        return
    
    # Get most recent production log
    prod_logs = sorted(glob.glob("logs/admf*.log"), key=os.path.getmtime, reverse=True)
    if not prod_logs:
        print("No production logs found")
        return
    
    opt_log = opt_logs[0]
    prod_log = prod_logs[0]
    
    print(f"Optimizer log: {opt_log}")
    print(f"Production log: {prod_log}")
    
    # Extract flows
    opt_flow = extract_detailed_execution_flow(opt_log, "optimizer")
    prod_flow = extract_detailed_execution_flow(prod_log, "production")
    
    # Compare component initialization order
    print("\nCOMPONENT INITIALIZATION ORDER:")
    print("-" * 40)
    print(f"Optimizer: {opt_flow['component_init_order'][:5]}...")
    print(f"Production: {prod_flow['component_init_order'][:5]}...")
    
    # Compare event subscriptions
    print("\nEVENT SUBSCRIPTIONS:")
    print("-" * 40)
    for event_type in ['BAR', 'SIGNAL', 'CLASSIFICATION']:
        opt_subs = opt_flow['event_subscriptions'].get(event_type, [])
        prod_subs = prod_flow['event_subscriptions'].get(event_type, [])
        if opt_subs != prod_subs:
            print(f"{event_type} subscribers differ!")
            print(f"  Optimizer: {opt_subs}")
            print(f"  Production: {prod_subs}")
    
    # Compare first signals
    print("\nFIRST SIGNAL CONTEXT:")
    print("-" * 40)
    if opt_flow['signal_generation_context']:
        print(f"Optimizer first signal: {opt_flow['signal_generation_context'][0]['line'][:100]}...")
    if prod_flow['signal_generation_context']:
        print(f"Production first signal: {prod_flow['signal_generation_context'][0]['line'][:100]}...")
    
    return opt_flow, prod_flow

def check_event_ordering():
    """Check if event ordering could cause the issue."""
    print("\nCHECKING EVENT ORDERING")
    print("=" * 60)
    
    # Create a test to verify event order
    test_code = '''
import logging
from src.core.event_bus import EventBus
from src.core.event import Event, EventType

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test event ordering
bus = EventBus()

order = []

def handler1(event):
    order.append(f"Handler1: {event.event_type.name}")
    logger.info(f"Handler1 processed {event.event_type.name}")

def handler2(event):
    order.append(f"Handler2: {event.event_type.name}")
    logger.info(f"Handler2 processed {event.event_type.name}")

# Subscribe in different orders
bus.subscribe(EventType.BAR, handler1)
bus.subscribe(EventType.BAR, handler2)

# Publish event
event = Event(EventType.BAR, {"test": "data"})
bus.publish(event)

print(f"Event processing order: {order}")
'''
    
    with open("test_event_order.py", 'w') as f:
        f.write(test_code)
    
    # Run test
    result = subprocess.run(["python", "test_event_order.py"], capture_output=True, text=True)
    print("Event order test output:")
    print(result.stdout)
    
    os.remove("test_event_order.py")

def check_component_state_differences():
    """Check for component state differences at critical points."""
    print("\nCHECKING COMPONENT STATE DIFFERENCES")
    print("=" * 60)
    
    # Look for differences in:
    # 1. Strategy parameters
    # 2. Indicator states
    # 3. Regime detector state
    # 4. Event bus state
    
    checks = [
        ("Strategy parameters", "parameters updated", "parameters"),
        ("Indicator states", "indicator.*value", "indicators"),
        ("Regime state", "regime.*current", "regime"),
        ("Weights", "weight.*=", "weights")
    ]
    
    for check_name, pattern, key in checks:
        print(f"\n{check_name}:")
        print("-" * 30)
        
        # Check in recent logs
        cmd = f"grep -i '{pattern}' logs/enhanced*.log | tail -3"
        opt_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        cmd = f"grep -i '{pattern}' logs/admf*.log | tail -3"
        prod_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if opt_result.stdout or prod_result.stdout:
            print(f"Optimizer: {opt_result.stdout[:200]}")
            print(f"Production: {prod_result.stdout[:200]}")

def find_exact_divergence_point():
    """Find the exact point where signals diverge."""
    print("\nFINDING EXACT DIVERGENCE POINT")
    print("=" * 60)
    
    # Extract bar-by-bar comparison
    print("Extracting bar processing from both runs...")
    
    # Get first 10 bars from each
    opt_bars = subprocess.run(
        "grep 'BAR_' logs/enhanced*.log | grep 'INDICATORS' | head -10",
        shell=True, capture_output=True, text=True
    ).stdout.strip().split('\n')
    
    prod_bars = subprocess.run(
        "grep 'BAR_' logs/admf*.log | grep 'INDICATORS' | head -10",
        shell=True, capture_output=True, text=True
    ).stdout.strip().split('\n')
    
    print("\nFirst few bars comparison:")
    for i, (opt, prod) in enumerate(zip(opt_bars, prod_bars)):
        # Extract key values
        opt_match = re.search(r'Price=([\d.]+).*MA_short=([\d.]+|N/A).*MA_long=([\d.]+|N/A)', opt)
        prod_match = re.search(r'Price=([\d.]+).*MA_short=([\d.]+|N/A).*MA_long=([\d.]+|N/A)', prod)
        
        if opt_match and prod_match:
            if opt_match.groups() != prod_match.groups():
                print(f"\nDIVERGENCE at bar {i+1}:")
                print(f"  Optimizer: Price={opt_match.group(1)}, MA_short={opt_match.group(2)}, MA_long={opt_match.group(3)}")
                print(f"  Production: Price={prod_match.group(1)}, MA_short={prod_match.group(2)}, MA_long={prod_match.group(3)}")
                break
        else:
            print(f"Bar {i+1}: {'MATCH' if opt == prod else 'DIFFER'}")

def main():
    """Run comprehensive diagnostics."""
    
    # Compare execution flows
    opt_flow, prod_flow = compare_execution_flows()
    
    # Check event ordering
    check_event_ordering()
    
    # Check component states
    check_component_state_differences()
    
    # Find divergence point
    find_exact_divergence_point()
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    print("Potential causes to investigate:")
    print("1. Event subscription order - causing different processing sequence")
    print("2. Component initialization order - affecting initial states")
    print("3. Default parameter differences - hidden configuration mismatches")
    print("4. State persistence - optimizer might have residual state")
    print("5. Floating point precision - accumulating differences")
    print("6. Random seeds - if any components use randomness")
    print("7. Threading/timing - race conditions in event processing")
    
    print("\nNext steps:")
    print("- Add more detailed logging at signal generation points")
    print("- Force identical component initialization order")
    print("- Reset all state between runs")
    print("- Use fixed random seeds if applicable")

if __name__ == "__main__":
    main()