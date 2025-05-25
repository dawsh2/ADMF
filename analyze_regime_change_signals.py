#!/usr/bin/env python3
"""
Analyze signal generation around regime changes
"""

import re
from datetime import datetime, timedelta

def analyze_signals_around_regime_changes(log_file, start_line=0):
    """Check signal patterns around regime changes"""
    
    regime_changes = []
    signals = []
    current_regime = None
    
    with open(log_file, 'r') as f:
        # Skip to start line
        for _ in range(start_line):
            f.readline()
            
        for line in f:
            # Only analyze test period
            if '2025-01-27' not in line and '2025-01' not in line and '2025-02' not in line:
                continue
                
            # Track regime changes
            if "Market regime changed from" in line:
                match = re.search(r"regime changed from '(\w+)' to '(\w+)' at ([^']+?)(?:\s+for\s+|$)", line)
                if match:
                    from_regime, to_regime, timestamp = match.groups()
                    regime_changes.append({
                        'timestamp': timestamp.strip(),
                        'from': from_regime,
                        'to': to_regime
                    })
                    current_regime = to_regime
                    
            # Track CLASSIFICATION events that might indicate regime changes
            elif "Publishing event: Event(type=CLASSIFICATION" in line and "'previous_classification'" in line:
                prev_match = re.search(r"'previous_classification': '(\w+)'", line)
                curr_match = re.search(r"'classification': '(\w+)'", line)
                time_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                
                if prev_match and curr_match and time_match and prev_match.group(1) != curr_match.group(1):
                    regime_changes.append({
                        'timestamp': time_match.group(1),
                        'from': prev_match.group(1),
                        'to': curr_match.group(1)
                    })
                    current_regime = curr_match.group(1)
                    
            # Track signals
            elif "Publishing event: Event(type=SIGNAL" in line:
                time_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                type_match = re.search(r"'signal_type': ([+-]?\d+)", line)
                regime_match = re.search(r"Regime:\s*(\w+)", line)
                
                if time_match and type_match:
                    signals.append({
                        'timestamp': time_match.group(1),
                        'type': int(type_match.group(1)),
                        'regime': regime_match.group(1) if regime_match else current_regime
                    })
    
    return regime_changes, signals

def analyze_signal_gaps(regime_changes, signals):
    """Find gaps in signal generation around regime changes"""
    
    print(f"\nAnalyzing {len(regime_changes)} regime changes and {len(signals)} signals")
    
    gaps = []
    
    for i, change in enumerate(regime_changes[:10]):  # First 10 regime changes
        change_time = datetime.fromisoformat(change['timestamp'].replace('+0000', '+00:00'))
        
        # Find signals before and after this regime change
        before_signals = []
        after_signals = []
        
        for sig in signals:
            sig_time = datetime.fromisoformat(sig['timestamp'].replace('+0000', '+00:00'))
            time_diff = (sig_time - change_time).total_seconds() / 60  # Minutes
            
            if -30 <= time_diff <= -1:  # 30 mins before to 1 min before
                before_signals.append((time_diff, sig))
            elif 1 <= time_diff <= 30:  # 1 min after to 30 mins after
                after_signals.append((time_diff, sig))
                
        # Check for gaps
        last_before = max(before_signals, key=lambda x: x[0])[0] if before_signals else None
        first_after = min(after_signals, key=lambda x: x[0])[0] if after_signals else None
        
        if last_before and first_after:
            gap = first_after - last_before
            if gap > 5:  # More than 5 minute gap
                gaps.append({
                    'change': change,
                    'gap_minutes': gap,
                    'last_signal_before': last_before,
                    'first_signal_after': first_after
                })
                
        print(f"\n{i+1}. Regime change at {change['timestamp']}: {change['from']} → {change['to']}")
        print(f"   Signals before: {len(before_signals)} (last at {last_before:.1f} min)" if last_before else "   No signals before")
        print(f"   Signals after: {len(after_signals)} (first at {first_after:.1f} min)" if first_after else "   No signals after")
        if last_before and first_after:
            print(f"   Gap: {gap:.1f} minutes")
            
    return gaps

def main():
    print("="*80)
    print("Analyzing Signal Generation Around Regime Changes")
    print("="*80)
    
    # Analyze optimization
    print("\nOPTIMIZATION ADAPTIVE TEST:")
    opt_changes, opt_signals = analyze_signals_around_regime_changes('logs/admf_20250523_175845.log', 2602000)
    opt_gaps = analyze_signal_gaps(opt_changes, opt_signals)
    
    # Analyze production
    print("\n\nPRODUCTION (ENSEMBLE):")
    prod_changes, prod_signals = analyze_signals_around_regime_changes('logs/admf_20250523_183637.log')
    prod_gaps = analyze_signal_gaps(prod_changes, prod_signals)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nOptimization:")
    print(f"  Total signals: {len(opt_signals)}")
    print(f"  Regime changes: {len(opt_changes)}")
    print(f"  Significant gaps (>5 min): {len(opt_gaps)}")
    if opt_gaps:
        avg_gap = sum(g['gap_minutes'] for g in opt_gaps) / len(opt_gaps)
        print(f"  Average gap: {avg_gap:.1f} minutes")
        
    print(f"\nProduction:")
    print(f"  Total signals: {len(prod_signals)}")
    print(f"  Regime changes: {len(prod_changes)}")
    print(f"  Significant gaps (>5 min): {len(prod_gaps)}")
    if prod_gaps:
        avg_gap = sum(g['gap_minutes'] for g in prod_gaps) / len(prod_gaps)
        print(f"  Average gap: {avg_gap:.1f} minutes")
        
    if len(prod_gaps) > len(opt_gaps):
        print(f"\n⚠️ Production has MORE signal gaps around regime changes!")
        print(f"This suggests indicators are being reset on regime changes in production")
    elif len(prod_gaps) < len(opt_gaps):
        print(f"\n✓ Production has fewer gaps than optimization")
    else:
        print(f"\n✓ Similar gap patterns in both runs")

if __name__ == "__main__":
    main()