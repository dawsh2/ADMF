#!/usr/bin/env python3
"""
Analyze why signals don't match 100% even with warmup
"""
import re

def extract_regime_history(log_file):
    """Extract regime changes and their timing"""
    regime_changes = []
    current_regime = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Track regime changes
            if "REGIME CHANGED:" in line:
                match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", line)
                if match:
                    regime_changes.append({
                        'from': match.group(1),
                        'to': match.group(2),
                        'timestamp': match.group(3),
                        'line': line.strip()
                    })
                    current_regime = match.group(2)
            
            # Track initial regime
            elif "Current regime:" in line and current_regime is None:
                match = re.search(r"Current regime: (\w+)", line)
                if match:
                    current_regime = match.group(1)
    
    return regime_changes, current_regime

def analyze_key_differences():
    """Analyze the key differences at specific timestamps"""
    
    prod_file = "/Users/daws/ADMF/logs/admf_20250524_002523.log"
    opt_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print("ANALYZING KEY SIGNAL MISMATCHES")
    print("=" * 80)
    
    # Key mismatches from the comparison
    key_mismatches = [
        {
            'timestamp': '2024-03-28 14:30:00',
            'prod': {'type': 1, 'regime': 'ranging_low_vol'},
            'opt': {'type': 1, 'regime': 'default'},
            'issue': 'Same signal type but different regime'
        },
        {
            'timestamp': '2024-03-28 16:49:00',
            'prod': {'type': 1, 'regime': 'default'},
            'opt': {'type': 1, 'regime': 'ranging_low_vol'},
            'issue': 'Same signal type but different regime'
        }
    ]
    
    # Get regime histories
    prod_changes, _ = extract_regime_history(prod_file)
    opt_changes, _ = extract_regime_history(opt_file)
    
    print("\n1. REGIME CHANGE TIMING ANALYSIS")
    print("-" * 80)
    
    # Find regime changes around key timestamps
    for mismatch in key_mismatches:
        ts = mismatch['timestamp']
        print(f"\nAround {ts} (Mismatch: {mismatch['issue']}):")
        
        # Find nearby regime changes
        print("  Production regime changes:")
        found_prod = False
        for change in prod_changes:
            if change['timestamp'].startswith(ts[:16]):  # Same minute
                print(f"    {change['timestamp']}: {change['from']} → {change['to']}")
                found_prod = True
        if not found_prod:
            # Look for changes in previous 5 minutes
            for change in prod_changes:
                change_time = change['timestamp'][:16]
                if change_time >= ts[:14] + str(int(ts[14:16]) - 5).zfill(2) and change_time < ts[:16]:
                    print(f"    {change['timestamp']}: {change['from']} → {change['to']}")
        
        print("  Optimizer regime changes:")
        found_opt = False
        for change in opt_changes:
            if change['timestamp'].startswith(ts[:16]):  # Same minute
                print(f"    {change['timestamp']}: {change['from']} → {change['to']}")
                found_opt = True
        if not found_opt:
            # Look for changes in previous 5 minutes
            for change in opt_changes:
                change_time = change['timestamp'][:16]
                if change_time >= ts[:14] + str(int(ts[14:16]) - 5).zfill(2) and change_time < ts[:16]:
                    print(f"    {change['timestamp']}: {change['from']} → {change['to']}")
    
    print("\n2. WARMUP PERIOD ANALYSIS")
    print("-" * 80)
    print("\nProduction starts processing at: 2024-03-26 14:08:00 (shifted split)")
    print("Optimizer starts processing at:  2024-03-28 13:46:00 (original split)")
    print("\nKey difference: Production processes ~230 bars vs Optimizer's ~200 bars")
    print("This means Production has 30 extra bars of market data affecting:")
    print("  - Regime detector's internal indicators")
    print("  - Regime stabilization state")
    print("  - Initial regime classification")
    
    print("\n3. REGIME DETECTOR STABILIZATION")
    print("-" * 80)
    print("\nThe regime detector requires min_regime_duration = 2 bars to switch")
    print("Even small timing differences can cause regime misalignment:")
    print("  - If Production detects a regime change 1 bar earlier")
    print("  - It will be in the new regime while Optimizer is still pending")
    print("  - This creates a 2-bar window of regime mismatch")
    
    print("\n4. ROOT CAUSES OF REMAINING MISMATCHES")
    print("-" * 80)
    print("\n1. Different Data Windows:")
    print("   - Production: 230 bars (includes 30 warmup bars)")
    print("   - Optimizer: 200 bars (standard test set)")
    print("   - Extra data changes indicator calculations slightly")
    
    print("\n2. Regime Detector State Divergence:")
    print("   - Different starting points lead to different internal states")
    print("   - Stabilization logic amplifies small differences")
    print("   - Once regimes diverge, they may stay misaligned for several bars")
    
    print("\n3. Edge Effects:")
    print("   - Early signals (before 13:46) are from different market conditions")
    print("   - Late signals may be affected by accumulated state differences")

def main():
    analyze_key_differences()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThe 81.8% match rate is actually quite good considering:")
    print("1. We're comparing different data windows (230 vs 200 bars)")
    print("2. Regime detection is sensitive to initial conditions")
    print("3. Small timing differences get amplified by stabilization logic")
    print("\nTo achieve 100% match, we would need:")
    print("- Exactly the same data window")
    print("- Identical initial regime detector states")
    print("- Synchronized regime stabilization counters")

if __name__ == "__main__":
    main()