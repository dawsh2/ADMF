#!/usr/bin/env python3
"""
Analyze and compare regime detection patterns between production and optimizer
"""
import re
from datetime import datetime

def extract_regime_changes(log_file_path, source_name):
    """Extract all regime changes from a log file"""
    regime_changes = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Look for regime change messages
                if 'REGIME CHANGED:' in line:
                    # Parse the regime change details
                    regime_match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", line)
                    if regime_match:
                        from_regime = regime_match.group(1)
                        to_regime = regime_match.group(2)
                        timestamp = regime_match.group(3)
                        
                        regime_changes.append({
                            'source': source_name,
                            'line_num': line_num,
                            'from_regime': from_regime,
                            'to_regime': to_regime,
                            'timestamp': timestamp,
                            'full_line': line.strip()
                        })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return regime_changes

def compare_regime_changes(production_changes, optimizer_changes, max_compare=20):
    """Compare regime changes between production and optimizer runs"""
    
    print("=" * 100)
    print("REGIME CHANGE COMPARISON ANALYSIS")
    print("=" * 100)
    
    print(f"\nRegime Change Counts:")
    print(f"  Production: {len(production_changes)} changes")
    print(f"  Optimizer:  {len(optimizer_changes)} changes")
    
    if len(production_changes) != len(optimizer_changes):
        print(f"  ❌ DIFFERENT REGIME CHANGE COUNTS! Difference: {abs(len(production_changes) - len(optimizer_changes))}")
    else:
        print(f"  ✅ Same regime change count")
    
    # Compare regime changes by timestamp/order
    print(f"\nRegime Change Comparison (showing first {max_compare}):")
    print("-" * 100)
    
    max_changes = min(max_compare, len(production_changes), len(optimizer_changes))
    differences = []
    
    for i in range(max_changes):
        prod_change = production_changes[i] if i < len(production_changes) else None
        opt_change = optimizer_changes[i] if i < len(optimizer_changes) else None
        
        if prod_change and opt_change:
            # Compare regime change details
            timestamp_match = prod_change['timestamp'] == opt_change['timestamp']
            from_regime_match = prod_change['from_regime'] == opt_change['from_regime']
            to_regime_match = prod_change['to_regime'] == opt_change['to_regime']
            
            if timestamp_match and from_regime_match and to_regime_match:
                status = "✅"
            else:
                status = "❌"
                differences.append(i + 1)
            
            print(f"  Change {i+1:2d}: {status}")
            print(f"    Timestamp: {prod_change['timestamp']} | {opt_change['timestamp']} {'✅' if timestamp_match else '❌'}")
            print(f"    Prod:      {prod_change['from_regime']:20} → {prod_change['to_regime']:20}")
            print(f"    Opt:       {opt_change['from_regime']:20} → {opt_change['to_regime']:20} {'✅' if from_regime_match and to_regime_match else '❌'}")
            print()
            
        elif prod_change and not opt_change:
            print(f"  Change {i+1:2d}: ❌ Production has change, Optimizer missing")
            print(f"    Prod: {prod_change['from_regime']} → {prod_change['to_regime']} at {prod_change['timestamp']}")
            differences.append(i + 1)
            print()
        elif opt_change and not prod_change:
            print(f"  Change {i+1:2d}: ❌ Optimizer has change, Production missing")
            print(f"    Opt: {opt_change['from_regime']} → {opt_change['to_regime']} at {opt_change['timestamp']}")
            differences.append(i + 1)
            print()
    
    print(f"Summary:")
    if differences:
        print(f"  ❌ Found {len(differences)} regime change differences at positions: {differences}")
        print(f"  Regime detection is NOT identical")
    else:
        print(f"  ✅ All regime changes match! Regime detection is identical")
    
    # Show regime frequency analysis
    print(f"\nRegime Frequency Analysis:")
    print("-" * 50)
    
    # Count regime transitions
    prod_regimes = {}
    opt_regimes = {}
    
    for change in production_changes:
        from_regime = change['from_regime']
        to_regime = change['to_regime']
        transition = f"{from_regime} → {to_regime}"
        prod_regimes[transition] = prod_regimes.get(transition, 0) + 1
    
    for change in optimizer_changes:
        from_regime = change['from_regime']
        to_regime = change['to_regime']
        transition = f"{from_regime} → {to_regime}"
        opt_regimes[transition] = opt_regimes.get(transition, 0) + 1
    
    all_transitions = set(prod_regimes.keys()) | set(opt_regimes.keys())
    
    print("Transition frequencies:")
    for transition in sorted(all_transitions):
        prod_count = prod_regimes.get(transition, 0)
        opt_count = opt_regimes.get(transition, 0)
        match_status = "✅" if prod_count == opt_count else "❌"
        print(f"  {transition:40} | Prod: {prod_count:2d} | Opt: {opt_count:2d} {match_status}")
    
    print(f"\n" + "=" * 100)
    
    return len(differences) == 0

def main():
    print("Regime Analysis: Comparing Production vs Optimizer Regime Detection")
    
    # Extract regime changes from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nExtracting regime changes from production log: {production_file}")
    production_changes = extract_regime_changes(production_file, "PRODUCTION")
    
    print(f"Extracting regime changes from optimizer log: {optimizer_file}")
    optimizer_changes = extract_regime_changes(optimizer_file, "OPTIMIZER")
    
    # Compare the regime changes
    regimes_match = compare_regime_changes(production_changes, optimizer_changes, max_compare=15)
    
    if regimes_match:
        print(f"\n🎯 CONCLUSION: Regime detection is IDENTICAL between production and optimizer!")
        print(f"   The regime detection differences are NOT the root cause.")
        print(f"   The signal differences must be due to other factors.")
    else:
        print(f"\n⚠️  CONCLUSION: Regime detection DIFFERS between production and optimizer!")
        print(f"   This explains the different signal generation patterns.")
        print(f"   Different regimes lead to different MA/RSI configurations.")

if __name__ == "__main__":
    main()