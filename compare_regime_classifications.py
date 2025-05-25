#!/usr/bin/env python3
"""
Compare regime classifications between optimization and production runs
"""

import re
from datetime import datetime
from collections import Counter

def extract_regime_classifications(log_file, is_optimization=False):
    """Extract regime classification events from log file"""
    classifications = []
    
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
            
            # Look for CLASSIFICATION events
            if "Publishing event: Event(type=CLASSIFICATION" in line:
                # Extract timestamp
                ts_match = re.search(r"'timestamp': Timestamp\('([^']+)'", line)
                if ts_match:
                    classification_time = ts_match.group(1).split('+')[0]
                    classification_dt = datetime.strptime(classification_time, "%Y-%m-%d %H:%M:%S")
                    
                    # For optimization, only include test phase classifications
                    if is_optimization and not in_test_phase:
                        continue
                    
                    # Only include classifications from test period
                    if classification_dt < test_start_time:
                        continue
                    
                    # Extract classification details
                    current_match = re.search(r"'classification': '([^']+)'", line)
                    previous_match = re.search(r"'previous_classification': '([^']+)'", line)
                    price_match = re.search(r"'bar_close_price': ([0-9.]+)", line)
                    
                    if current_match:
                        classification = {
                            'timestamp': classification_time,
                            'current': current_match.group(1),
                            'previous': previous_match.group(1) if previous_match else None,
                            'price': float(price_match.group(1)) if price_match else None,
                            'line': line.strip()
                        }
                        classifications.append(classification)
    
    return classifications

def analyze_regime_transitions(classifications):
    """Analyze regime transitions and durations"""
    transitions = []
    regime_durations = {}
    current_regime = None
    regime_start = None
    
    for i, cls in enumerate(classifications):
        if current_regime != cls['current']:
            # Regime change detected
            if current_regime is not None:
                # Calculate duration of previous regime
                duration = i - regime_start if regime_start is not None else 0
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(duration)
                
                transitions.append({
                    'from': current_regime,
                    'to': cls['current'],
                    'timestamp': cls['timestamp'],
                    'duration': duration
                })
            
            current_regime = cls['current']
            regime_start = i
    
    return transitions, regime_durations

def compare_classifications(opt_classifications, prod_classifications):
    """Compare classifications between optimization and production"""
    print(f"\n=== REGIME CLASSIFICATION COMPARISON ===")
    print(f"Optimization classifications in test period: {len(opt_classifications)}")
    print(f"Production classifications in test period: {len(prod_classifications)}")
    
    # Analyze regime distributions
    opt_regimes = Counter(cls['current'] for cls in opt_classifications)
    prod_regimes = Counter(cls['current'] for cls in prod_classifications)
    
    print(f"\n=== REGIME DISTRIBUTION ===")
    print("\nOptimization regime counts:")
    for regime, count in sorted(opt_regimes.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(opt_classifications)
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    print("\nProduction regime counts:")
    for regime, count in sorted(prod_regimes.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(prod_classifications)
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    # Compare classifications at same timestamps
    opt_by_time = {cls['timestamp']: cls for cls in opt_classifications}
    prod_by_time = {cls['timestamp']: cls for cls in prod_classifications}
    
    common_times = set(opt_by_time.keys()) & set(prod_by_time.keys())
    opt_only_times = set(opt_by_time.keys()) - set(prod_by_time.keys())
    prod_only_times = set(prod_by_time.keys()) - set(opt_by_time.keys())
    
    print(f"\n=== TIMESTAMP COMPARISON ===")
    print(f"Common classification times: {len(common_times)}")
    print(f"Optimization-only times: {len(opt_only_times)}")
    print(f"Production-only times: {len(prod_only_times)}")
    
    # Check for differences in common classifications
    differences = []
    for time in sorted(common_times)[:20]:  # First 20 for analysis
        opt_cls = opt_by_time[time]
        prod_cls = prod_by_time[time]
        
        if opt_cls['current'] != prod_cls['current']:
            differences.append({
                'time': time,
                'opt': opt_cls['current'],
                'prod': prod_cls['current']
            })
    
    if differences:
        print(f"\n=== CLASSIFICATION DIFFERENCES (first {len(differences)}) ===")
        for diff in differences:
            print(f"  {diff['time']}: Opt={diff['opt']}, Prod={diff['prod']}")
    else:
        print(f"\n✓ All {len(common_times)} common classifications match!")
    
    # Analyze regime transitions
    print(f"\n=== REGIME TRANSITIONS ===")
    opt_transitions, opt_durations = analyze_regime_transitions(opt_classifications)
    prod_transitions, prod_durations = analyze_regime_transitions(prod_classifications)
    
    print(f"Optimization transitions: {len(opt_transitions)}")
    print(f"Production transitions: {len(prod_transitions)}")
    
    # Show first few transitions
    print(f"\nFirst 5 optimization transitions:")
    for trans in opt_transitions[:5]:
        print(f"  {trans['timestamp']}: {trans['from']} → {trans['to']} (duration: {trans['duration']})")
    
    print(f"\nFirst 5 production transitions:")
    for trans in prod_transitions[:5]:
        print(f"  {trans['timestamp']}: {trans['from']} → {trans['to']} (duration: {trans['duration']})")
    
    # Show regime timing differences
    if opt_only_times:
        print(f"\n=== FIRST 10 OPTIMIZATION-ONLY CLASSIFICATIONS ===")
        for time in sorted(opt_only_times)[:10]:
            cls = opt_by_time[time]
            print(f"  {time}: {cls['current']}")
    
    if prod_only_times:
        print(f"\n=== FIRST 10 PRODUCTION-ONLY CLASSIFICATIONS ===")
        for time in sorted(prod_only_times)[:10]:
            cls = prod_by_time[time]
            print(f"  {time}: {cls['current']}")

def main():
    print("Comparing regime classifications between optimization and production runs...")
    
    # Extract classifications
    print("\nExtracting classifications from optimization log...")
    opt_classifications = extract_regime_classifications('logs/admf_20250523_202211.log', is_optimization=True)
    
    print("Extracting classifications from production log...")
    prod_classifications = extract_regime_classifications('logs/admf_20250523_202911.log', is_optimization=False)
    
    # Compare classifications
    compare_classifications(opt_classifications, prod_classifications)

if __name__ == "__main__":
    main()