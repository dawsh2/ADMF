#!/usr/bin/env python3
"""
Analyze parameter changes during regime transitions
"""

import re
from datetime import datetime
from collections import defaultdict

def extract_parameter_applications(log_file, is_optimization=False):
    """Extract parameter application events from log file"""
    parameter_events = []
    
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
            
            # Look for parameter application events
            if any(keyword in line for keyword in [
                "Applying parameters for current regime",
                "!!! ADAPTIVE TEST !!!",
                "set_parameters",
                "parameters updated",
                "regime-specific parameters",
                "Parameter update for regime"
            ]):
                # Extract timestamp if available
                timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase events
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include events from test period
                    if timestamp_dt < test_start_time:
                        continue
                    
                    # Extract regime and parameters from the line
                    regime_match = re.search(r"regime['\s]*[:\s]*['\s]*(\w+)", line, re.IGNORECASE)
                    
                    parameter_event = {
                        'timestamp': timestamp_str,
                        'datetime': timestamp_dt,
                        'regime': regime_match.group(1) if regime_match else None,
                        'line': line.strip(),
                        'type': 'parameter_application'
                    }
                    parameter_events.append(parameter_event)
    
    return parameter_events

def extract_regime_changes_with_parameters(log_file, is_optimization=False):
    """Extract regime changes and any associated parameter updates"""
    events = []
    
    # For optimization, we need to identify the test phase
    in_test_phase = False
    test_start_time = datetime.strptime("2025-01-27 18:06:00", "%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Check if we're in test phase for optimization logs
        if is_optimization:
            if "ADAPTIVE TEST" in line or "!!! ADAPTIVE TEST !!!" in line:
                in_test_phase = True
            elif "Optimization complete" in line or "Enhanced Grid Search with Train/Test Ended" in line:
                in_test_phase = False
        
        # Look for regime changes (CLASSIFICATION events where current != previous)
        if "Publishing event: Event(type=CLASSIFICATION" in line:
            # Extract timestamp
            ts_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
            if ts_match:
                classification_time = ts_match.group(1).split('+')[0]
                classification_dt = datetime.strptime(classification_time, "%Y-%m-%d %H:%M:%S")
                
                # For optimization, only include test phase events
                if is_optimization and not in_test_phase:
                    continue
                
                # Only include events from test period
                if classification_dt < test_start_time:
                    continue
                
                # Extract classification details
                current_match = re.search(r"'classification': '([^']+)'", line)
                previous_match = re.search(r"'previous_classification': '([^']+)'", line)
                
                if current_match and previous_match:
                    current_regime = current_match.group(1)
                    previous_regime = previous_match.group(1)
                    
                    # Only record actual regime changes
                    if current_regime != previous_regime:
                        # Look for parameter applications in the next few lines
                        parameter_updates = []
                        for j in range(i+1, min(i+20, len(lines))):  # Look ahead 20 lines
                            next_line = lines[j]
                            if any(keyword in next_line for keyword in [
                                "Applying parameters",
                                "set_parameters",
                                "parameters updated",
                                "Parameter update"
                            ]):
                                parameter_updates.append(next_line.strip())
                        
                        event = {
                            'timestamp': classification_time,
                            'datetime': classification_dt,
                            'from_regime': previous_regime,
                            'to_regime': current_regime,
                            'parameter_updates': parameter_updates,
                            'type': 'regime_change'
                        }
                        events.append(event)
    
    return events

def extract_weight_changes(log_file, is_optimization=False):
    """Extract weight change events"""
    weight_events = []
    
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
            
            # Look for weight-related events
            if any(keyword in line for keyword in [
                "MA weight",
                "RSI weight", 
                "weight changed",
                "EnsembleSignalStrategy weights:",
                "WEIGHTS AFTER APPLICATION"
            ]):
                # Extract timestamp if available
                timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase events
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include events from test period
                    if timestamp_dt < test_start_time:
                        continue
                    
                    # Extract weight values
                    ma_match = re.search(r"MA[=:\s]+([\d.]+)", line)
                    rsi_match = re.search(r"RSI[=:\s]+([\d.]+)", line)
                    
                    weight_event = {
                        'timestamp': timestamp_str,
                        'datetime': timestamp_dt,
                        'ma_weight': float(ma_match.group(1)) if ma_match else None,
                        'rsi_weight': float(rsi_match.group(1)) if rsi_match else None,
                        'line': line.strip(),
                        'type': 'weight_change'
                    }
                    weight_events.append(weight_event)
    
    return weight_events

def analyze_parameter_behavior(param_events, regime_events, weight_events, name):
    """Analyze parameter behavior"""
    print(f"\n=== {name.upper()} PARAMETER ANALYSIS ===")
    print(f"Parameter application events: {len(param_events)}")
    print(f"Regime change events: {len(regime_events)}")
    print(f"Weight change events: {len(weight_events)}")
    
    # Show first few parameter applications
    if param_events:
        print(f"\nFirst 5 parameter applications:")
        for event in param_events[:5]:
            print(f"  {event['timestamp']}: Regime={event['regime']}")
            print(f"    {event['line'][:100]}...")
    
    # Show first few regime changes with parameter updates
    if regime_events:
        print(f"\nFirst 5 regime changes:")
        for event in regime_events[:5]:
            print(f"  {event['timestamp']}: {event['from_regime']} → {event['to_regime']}")
            for update in event['parameter_updates']:
                print(f"    Parameter update: {update[:80]}...")
    
    # Show weight changes
    if weight_events:
        print(f"\nWeight changes:")
        for event in weight_events:
            ma_str = f"MA={event['ma_weight']}" if event['ma_weight'] else ""
            rsi_str = f"RSI={event['rsi_weight']}" if event['rsi_weight'] else ""
            print(f"  {event['timestamp']}: {ma_str} {rsi_str}")

def main():
    print("Analyzing parameter changes during regime transitions...")
    
    # Extract events from optimization log
    print("\nExtracting events from optimization log...")
    opt_param_events = extract_parameter_applications('logs/admf_20250523_202211.log', is_optimization=True)
    opt_regime_events = extract_regime_changes_with_parameters('logs/admf_20250523_202211.log', is_optimization=True)
    opt_weight_events = extract_weight_changes('logs/admf_20250523_202211.log', is_optimization=True)
    
    # Extract events from production log
    print("Extracting events from production log...")
    prod_param_events = extract_parameter_applications('logs/admf_20250523_202911.log', is_optimization=False)
    prod_regime_events = extract_regime_changes_with_parameters('logs/admf_20250523_202911.log', is_optimization=False)
    prod_weight_events = extract_weight_changes('logs/admf_20250523_202911.log', is_optimization=False)
    
    # Analyze both
    analyze_parameter_behavior(opt_param_events, opt_regime_events, opt_weight_events, "optimization")
    analyze_parameter_behavior(prod_param_events, prod_regime_events, prod_weight_events, "production")

if __name__ == "__main__":
    main()