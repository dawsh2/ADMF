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

def find_latest_logs():
    """Find the most recent optimizer and production logs."""
    import glob
    
    # Find optimizer log (look for "enhanced_optimizer" in filename or content)
    all_logs = sorted(glob.glob("logs/*.log"), key=os.path.getmtime, reverse=True)
    
    opt_log = None
    prod_log = None
    
    for log in all_logs[:10]:  # Check recent logs
        # Check if it's an optimizer log
        check_cmd = f"grep -l 'EnhancedOptimizer\\|optimization' {log} 2>/dev/null"
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            if not opt_log:
                opt_log = log
        else:
            # It's a production log
            if not prod_log and "admf" in log:
                prod_log = log
        
        if opt_log and prod_log:
            break
    
    return opt_log, prod_log

def extract_signals_and_context(log_file, source_name):
    """Extract all signals with full context."""
    signals = []
    
    # Get all signal lines with context
    cmd = f"grep -B5 -A5 'SIGNAL GENERATED' {log_file} 2>/dev/null"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if not result.stdout:
        return signals
    
    # Parse the output
    blocks = result.stdout.split('--\n')
    
    for block in blocks:
        if 'SIGNAL GENERATED' in block:
            signal_data = {
                'source': source_name,
                'block': block,
                'signal_line': None,
                'bar_info': None,
                'price': None,
                'ma_values': None,
                'regime': None,
                'timestamp': None
            }
            
            lines = block.strip().split('\n')
            for line in lines:
                if 'SIGNAL GENERATED' in line:
                    signal_data['signal_line'] = line
                    # Extract signal type and price
                    match = re.search(r'Type=(-?\d+), Price=([\d.]+)', line)
                    if match:
                        signal_data['type'] = int(match.group(1))
                        signal_data['price'] = float(match.group(2))
                    # Extract regime
                    match = re.search(r'Regime=(\w+)', line)
                    if match:
                        signal_data['regime'] = match.group(1)
                
                elif 'BAR_' in line and 'INDICATORS' in line:
                    signal_data['bar_info'] = line
                    # Extract timestamp
                    match = re.search(r'\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})', line)
                    if match:
                        signal_data['timestamp'] = match.group(1)
                    # Extract MA values
                    match = re.search(r'MA_short=([\d.]+|N/A), MA_long=([\d.]+|N/A)', line)
                    if match:
                        signal_data['ma_values'] = (match.group(1), match.group(2))
            
            signals.append(signal_data)
    
    return signals

def compare_signals():
    """Compare signals between optimizer and production."""
    print("SIGNAL COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Find logs
    opt_log, prod_log = find_latest_logs()
    
    if not opt_log or not prod_log:
        print(f"Could not find both logs. Optimizer: {opt_log}, Production: {prod_log}")
        return
    
    print(f"Optimizer log: {opt_log}")
    print(f"Production log: {prod_log}")
    
    # Extract signals
    opt_signals = extract_signals_and_context(opt_log, "optimizer")
    prod_signals = extract_signals_and_context(prod_log, "production")
    
    print(f"\nOptimizer signals: {len(opt_signals)}")
    print(f"Production signals: {len(prod_signals)}")
    
    # Compare first few signals
    print("\nFIRST 5 SIGNALS COMPARISON:")
    print("-" * 60)
    
    for i in range(min(5, max(len(opt_signals), len(prod_signals)))):
        print(f"\nSignal #{i+1}:")
        
        if i < len(opt_signals):
            opt = opt_signals[i]
            print(f"  Optimizer: {opt.get('timestamp', 'N/A')} - Type={opt.get('type', 'N/A')}, Price={opt.get('price', 'N/A')}, Regime={opt.get('regime', 'N/A')}")
            if opt.get('ma_values'):
                print(f"    MA values: short={opt['ma_values'][0]}, long={opt['ma_values'][1]}")
        else:
            print("  Optimizer: No signal")
        
        if i < len(prod_signals):
            prod = prod_signals[i]
            print(f"  Production: {prod.get('timestamp', 'N/A')} - Type={prod.get('type', 'N/A')}, Price={prod.get('price', 'N/A')}, Regime={prod.get('regime', 'N/A')}")
            if prod.get('ma_values'):
                print(f"    MA values: short={prod['ma_values'][0]}, long={prod['ma_values'][1]}")
        else:
            print("  Production: No signal")
    
    # Find where they diverge
    print("\nDIVERGENCE ANALYSIS:")
    print("-" * 60)
    
    for i in range(min(len(opt_signals), len(prod_signals))):
        opt = opt_signals[i]
        prod = prod_signals[i]
        
        if (opt.get('timestamp') != prod.get('timestamp') or 
            opt.get('type') != prod.get('type') or 
            abs(opt.get('price', 0) - prod.get('price', 0)) > 0.01):
            
            print(f"\nDivergence found at signal #{i+1}:")
            print(f"  Timestamp: Opt={opt.get('timestamp')} vs Prod={prod.get('timestamp')}")
            print(f"  Type: Opt={opt.get('type')} vs Prod={prod.get('type')}")
            print(f"  Price: Opt={opt.get('price')} vs Prod={prod.get('price')}")
            print(f"  Regime: Opt={opt.get('regime')} vs Prod={prod.get('regime')}")
            break
    
    return opt_signals, prod_signals

def check_initialization_differences():
    """Check for differences in component initialization."""
    print("\n\nINITIALIZATION ANALYSIS")
    print("=" * 60)
    
    opt_log, prod_log = find_latest_logs()
    
    if not opt_log or not prod_log:
        return
    
    # Check weights
    print("\nWEIGHT CONFIGURATION:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep 'weights:' {log} | head -3"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            print(result.stdout)
    
    # Check regime detector
    print("\nREGIME DETECTOR CONFIG:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep 'RegimeDetector.*initialized' {log} | head -3"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            print(result.stdout)
    
    # Check adaptive mode
    print("\nADAPTIVE MODE:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -i 'adaptive.*mode\\|ADAPTIVE' {log} | head -3"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            print(result.stdout)

def check_data_handling():
    """Check how data is being processed."""
    print("\n\nDATA HANDLING ANALYSIS")
    print("=" * 60)
    
    opt_log, prod_log = find_latest_logs()
    
    if not opt_log or not prod_log:
        return
    
    # Check dataset info
    print("\nDATASET INFO:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -E 'train.*test.*split|dataset.*type|active.*dataset' {log} | head -5"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            print(result.stdout)
    
    # Check first bar
    print("\nFIRST BAR PROCESSED:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep -m1 'BAR_.*INDICATORS' {log}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            print(result.stdout)

def check_event_flow():
    """Check event flow differences."""
    print("\n\nEVENT FLOW ANALYSIS")
    print("=" * 60)
    
    opt_log, prod_log = find_latest_logs()
    
    if not opt_log or not prod_log:
        return
    
    # Check event subscriptions
    print("\nEVENT SUBSCRIPTIONS ORDER:")
    print("-" * 40)
    
    for log, name in [(opt_log, "Optimizer"), (prod_log, "Production")]:
        cmd = f"grep 'subscribed to.*events' {log} | head -10"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"{name}:")
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines[:5]):
                # Extract component and event type
                match = re.search(r"'([^']+)'.*subscribed to (\w+) events", line)
                if match:
                    print(f"  {i+1}. {match.group(1)} -> {match.group(2)}")

def main():
    """Run comprehensive diagnostics."""
    
    # Compare signals
    opt_signals, prod_signals = compare_signals()
    
    # Check initialization
    check_initialization_differences()
    
    # Check data handling
    check_data_handling()
    
    # Check event flow
    check_event_flow()
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print("\nPotential root causes:")
    print("1. Warmup difference - Optimizer processes training data first")
    print("2. Adaptive mode - May be enabled differently") 
    print("3. Weight differences - Check if weights match exactly")
    print("4. Regime detection - Different regimes = different parameters")
    print("5. Event ordering - Different subscription order could affect processing")
    print("6. Data split handling - Optimizer vs production may handle differently")
    
    print("\nRecommended next steps:")
    print("1. Force identical weights in both runs")
    print("2. Disable adaptive mode in both runs")
    print("3. Log exact indicator values at signal points")
    print("4. Trace the exact execution path for first signal")

if __name__ == "__main__":
    main()