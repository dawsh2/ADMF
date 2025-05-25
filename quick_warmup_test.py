#!/usr/bin/env python3
"""
Quick test of a few warmup values
"""
import subprocess
import time

def test_split(split_ratio, warmup_bars):
    """Quick test of one split ratio"""
    print(f"\nTesting {split_ratio} split (~{warmup_bars} warmup bars)...")
    
    # Modify the existing shifted config
    with open('config/config_adaptive_production_shifted.yaml', 'r') as f:
        config = f.read()
    
    # Update split ratio
    config = config.replace('train_test_split_ratio: 0.77', f'train_test_split_ratio: {split_ratio}')
    
    # Write temporary config
    with open('config/config_temp_test.yaml', 'w') as f:
        f.write(config)
    
    # Run test
    subprocess.run(['python', 'main.py', '--config', 'config/config_temp_test.yaml'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(2)
    
    # Get latest log
    result = subprocess.run(['ls', '-t', 'logs/admf_*.log'], 
                          capture_output=True, text=True, shell=True)
    latest_log = result.stdout.strip().split('\n')[0]
    
    # Count signals
    result = subprocess.run(['grep', '-c', 'SIGNAL GENERATED', latest_log], 
                          capture_output=True, text=True)
    signals = int(result.stdout.strip())
    
    print(f"  Signals: {signals}")
    return signals

# Test key values
print("Finding optimal warmup bars...")
print("Target: 16 signals (optimizer baseline)")

results = []
for split, warmup in [(0.80, 0), (0.79, 10), (0.78, 20), (0.77, 30)]:
    signals = test_split(split, warmup)
    results.append((split, warmup, signals))

print(f"\nResults:")
for split, warmup, signals in results:
    diff = signals - 16
    print(f"  {warmup:2d} warmup bars: {signals} signals ({diff:+d})")