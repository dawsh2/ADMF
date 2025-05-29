#!/usr/bin/env python3
"""
Verify regime classifier configuration between optimization and test runs.
"""

import subprocess
import re

def run_command(cmd):
    """Run a command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_classifier_config(output):
    """Extract classifier configuration from output."""
    config = {}
    
    # Look for regime detector initialization
    detector_patterns = [
        (r"RegimeDetector initialized with thresholds:.*?trending_up_threshold=([\d.]+).*?trending_down_threshold=([-\d.]+)", 
         "thresholds"),
        (r"Trend indicator lookback period: (\d+)", "trend_lookback"),
        (r"Volatility indicator lookback period: (\d+)", "volatility_lookback"),
        (r"RegimeDetector.*?trend_lookback_period.*?(\d+)", "trend_period_alt"),
        (r"RegimeDetector.*?volatility_lookback_period.*?(\d+)", "vol_period_alt"),
        (r"Volatility Threshold: ([\d.]+)", "volatility_threshold"),
        (r"trend_filter_enabled.*?(True|False)", "trend_filter"),
        (r"regime_change_cooldown.*?(\d+)", "cooldown"),
        (r"indicator_history_size.*?(\d+)", "history_size")
    ]
    
    for pattern, key in detector_patterns:
        matches = re.findall(pattern, output)
        if matches:
            config[key] = matches[0]
    
    return config

def extract_initial_values(output):
    """Extract initial indicator values that affect classification."""
    values = {}
    
    # Look for initial trend and volatility values
    patterns = [
        (r"Initial trend value: ([-\d.]+)", "initial_trend"),
        (r"Initial volatility value: ([\d.]+)", "initial_volatility"),
        (r"First classification:.*?trend=([-\d.]+).*?volatility=([\d.]+)", "first_classification"),
        (r"Bar #1.*?Current trend: ([-\d.]+)", "bar1_trend"),
        (r"Bar #1.*?Current volatility: ([\d.]+)", "bar1_volatility")
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, output)
        if match:
            values[key] = match.groups()
    
    return values

def extract_classification_sequence(output):
    """Extract the sequence of classifications."""
    # Pattern to match classification events
    classification_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Bar #(\d+).*?trend=([-\d.]+).*?volatility=([\d.]+).*?classification=(\w+)"
    
    classifications = []
    for match in re.finditer(classification_pattern, output):
        classifications.append({
            'timestamp': match.group(1),
            'bar': int(match.group(2)),
            'trend': float(match.group(3)),
            'volatility': float(match.group(4)),
            'regime': match.group(5)
        })
    
    return classifications

def main():
    print("="*80)
    print("REGIME CLASSIFIER CONFIGURATION VERIFICATION")
    print("="*80)
    
    # Run both with extra debug logging for regime detector
    opt_cmd = """cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --optimize --log-level DEBUG 2>&1 | grep -E "(RegimeDetector|REGIME|classification|trend.*value|volatility.*value)" | tail -1000"""
    
    test_cmd = """cd /Users/daws/ADMF && source venv/bin/activate && python main_ultimate.py --config config/test_ensemble_optimization.yaml --bars 1000 --dataset test --log-level DEBUG 2>&1 | grep -E "(RegimeDetector|REGIME|classification|trend.*value|volatility.*value)" | head -1000"""
    
    print("\n1. Extracting optimization classifier info...")
    opt_output = run_command(opt_cmd)
    
    print("\n2. Extracting test run classifier info...")
    test_output = run_command(test_cmd)
    
    # Extract configurations
    opt_config = extract_classifier_config(opt_output)
    test_config = extract_classifier_config(test_output)
    
    print("\n3. CLASSIFIER CONFIGURATION:")
    print("  Optimization:")
    for key, value in sorted(opt_config.items()):
        print(f"    {key}: {value}")
    
    print("\n  Test Run:")
    for key, value in sorted(test_config.items()):
        print(f"    {key}: {value}")
    
    print("\n  Configuration Match:")
    all_match = True
    for key in set(opt_config.keys()) | set(test_config.keys()):
        opt_val = opt_config.get(key, "NOT FOUND")
        test_val = test_config.get(key, "NOT FOUND")
        match = opt_val == test_val
        if not match:
            all_match = False
            print(f"    {key}: {'✓' if match else '✗'} (opt: {opt_val}, test: {test_val})")
    
    if all_match:
        print("    ✓ ALL CONFIGURATION MATCHES")
    
    # Extract initial values
    opt_values = extract_initial_values(opt_output)
    test_values = extract_initial_values(test_output)
    
    print("\n4. INITIAL VALUES:")
    print("  Optimization:", opt_values)
    print("  Test Run:", test_values)
    
    # Extract first few classifications
    opt_classifications = extract_classification_sequence(opt_output)
    test_classifications = extract_classification_sequence(test_output)
    
    print("\n5. FIRST 5 CLASSIFICATIONS:")
    print("  Optimization:")
    for c in opt_classifications[:5]:
        print(f"    Bar {c['bar']}: trend={c['trend']:.4f}, vol={c['volatility']:.4f} -> {c['regime']}")
    
    print("\n  Test Run:")
    for c in test_classifications[:5]:
        print(f"    Bar {c['bar']}: trend={c['trend']:.4f}, vol={c['volatility']:.4f} -> {c['regime']}")
    
    # Check for warmup differences
    print("\n6. WARMUP BEHAVIOR:")
    
    # Look for warmup-related messages
    warmup_pattern = r"(warmup|not ready|insufficient data)"
    opt_warmup = re.findall(warmup_pattern, opt_output, re.IGNORECASE)
    test_warmup = re.findall(warmup_pattern, test_output, re.IGNORECASE)
    
    print(f"  Optimization warmup messages: {len(opt_warmup)}")
    print(f"  Test run warmup messages: {len(test_warmup)}")
    
    # Save detailed logs
    with open('opt_classifier_debug.log', 'w') as f:
        f.write(opt_output)
    
    with open('test_classifier_debug.log', 'w') as f:
        f.write(test_output)
    
    print("\nDetailed classifier logs saved to:")
    print("  - opt_classifier_debug.log")
    print("  - test_classifier_debug.log")

if __name__ == "__main__":
    main()