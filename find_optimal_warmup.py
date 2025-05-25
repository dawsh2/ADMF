#!/usr/bin/env python3
"""
Find the optimal warmup bars by testing incremental values
"""
import subprocess
import time
import re
import os

def run_with_split(split_ratio):
    """Run with a specific split ratio and return signal count"""
    # Calculate approximate warmup bars
    warmup_bars = int((0.80 - split_ratio) * 1000)
    
    print(f"\nTesting split={split_ratio:.3f} (~{warmup_bars} warmup bars)")
    
    # Read base config
    with open('config/config_adaptive_production_shifted.yaml', 'r') as f:
        config = f.read()
    
    # Update split ratio
    import re
    config = re.sub(r'train_test_split_ratio: \d+\.\d+', 
                    f'train_test_split_ratio: {split_ratio}', config)
    
    # Write config
    config_path = f'config/config_test_{split_ratio}.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    # Get timestamp before run
    before_time = time.time()
    
    # Run silently
    result = subprocess.run(
        ['python', 'main.py', '--config', config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Find log created after our start time
    latest_log = None
    for log in sorted(os.listdir('logs'), reverse=True):
        if log.startswith('admf_') and log.endswith('.log'):
            log_path = os.path.join('logs', log)
            if os.path.getmtime(log_path) > before_time:
                latest_log = log_path
                break
    
    if not latest_log:
        print("  ERROR: No log file found")
        return None
    
    # Count signals
    try:
        with open(latest_log, 'r') as f:
            content = f.read()
            signal_count = content.count('SIGNAL GENERATED')
        
        print(f"  Signals: {signal_count}")
        print(f"  Log: {os.path.basename(latest_log)}")
        
        # Clean up temp config
        os.remove(config_path)
        
        return signal_count
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("Finding optimal warmup bars to match optimizer (16 signals)")
    print("="*60)
    
    # Start with coarse search
    coarse_results = []
    for split in [0.80, 0.795, 0.79, 0.785, 0.78, 0.775, 0.77]:
        signals = run_with_split(split)
        if signals is not None:
            coarse_results.append((split, signals))
        time.sleep(1)  # Brief pause between runs
    
    print("\n" + "="*60)
    print("COARSE SEARCH RESULTS")
    print("="*60)
    print(f"{'Split':>6} | {'Warmup':>7} | {'Signals':>7} | {'Diff from 16':>12}")
    print("-"*60)
    
    for split, signals in coarse_results:
        warmup = int((0.80 - split) * 1000)
        diff = signals - 16
        print(f"{split:>6.3f} | {warmup:>7} | {signals:>7} | {diff:>+12}")
    
    # Find the splits that bracket 16 signals
    below_16 = [(s, sig) for s, sig in coarse_results if sig < 16]
    above_16 = [(s, sig) for s, sig in coarse_results if sig >= 16]
    
    if below_16 and above_16:
        # Get the closest ones
        closest_below = max(below_16, key=lambda x: x[1])
        closest_above = min(above_16, key=lambda x: x[1])
        
        print(f"\nSignals jump from {closest_below[1]} to {closest_above[1]}")
        print(f"Between splits {closest_below[0]:.3f} and {closest_above[0]:.3f}")
        
        # Fine search between these values
        print("\n" + "="*60)
        print("FINE SEARCH")
        print("="*60)
        
        fine_results = []
        split_low = closest_below[0]
        split_high = closest_above[0]
        
        # Test 3 intermediate values
        for i in range(1, 4):
            split = split_low + (split_high - split_low) * i / 4
            signals = run_with_split(round(split, 4))
            if signals is not None:
                fine_results.append((split, signals))
            time.sleep(1)
        
        # Final results
        all_results = sorted(coarse_results + fine_results)
        
        print("\n" + "="*60)
        print("ALL RESULTS (sorted by split)")
        print("="*60)
        print(f"{'Split':>6} | {'Warmup':>7} | {'Signals':>7} | {'Match?':>7}")
        print("-"*60)
        
        for split, signals in all_results:
            warmup = int((0.80 - split) * 1000)
            match = "YES!" if signals == 16 else ""
            print(f"{split:>6.3f} | {warmup:>7} | {signals:>7} | {match:>7}")
        
        # Find exact match or closest
        exact_matches = [(s, sig) for s, sig in all_results if sig == 16]
        if exact_matches:
            print(f"\n✓ EXACT MATCH FOUND!")
            for split, _ in exact_matches:
                warmup = int((0.80 - split) * 1000)
                print(f"  Split ratio: {split:.4f} ({warmup} warmup bars)")
        else:
            closest = min(all_results, key=lambda x: abs(x[1] - 16))
            warmup = int((0.80 - closest[0]) * 1000)
            print(f"\n✗ No exact match. Closest: {closest[1]} signals at split {closest[0]:.4f} ({warmup} warmup bars)")

if __name__ == "__main__":
    main()