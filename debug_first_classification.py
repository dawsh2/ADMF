#!/usr/bin/env python3
"""
Debug why the first regime classification is opposite between runs.
"""

import re

def extract_first_bars_and_classifications(log_file):
    """Extract the first few bars and their classifications."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find first few BAR events with indicator values
    bar_pattern = r"Bar #(\d+).*?trend=([-\d.]+).*?volatility=([\d.]+).*?classification=(\w+)"
    bars = []
    
    for match in re.finditer(bar_pattern, content):
        bar_num = int(match.group(1))
        trend = float(match.group(2))
        volatility = float(match.group(3))
        classification = match.group(4)
        
        bars.append({
            'bar': bar_num,
            'trend': trend,
            'volatility': volatility,
            'classification': classification
        })
        
        if len(bars) >= 10:  # Get first 10
            break
    
    # Also look for threshold information
    threshold_pattern = r"trending_up.*?ma_trend.*?min.*?([\d.]+)"
    threshold_match = re.search(threshold_pattern, content)
    up_threshold = float(threshold_match.group(1)) if threshold_match else None
    
    threshold_pattern = r"trending_down.*?ma_trend.*?max.*?([-\d.]+)"
    threshold_match = re.search(threshold_pattern, content)
    down_threshold = float(threshold_match.group(1)) if threshold_match else None
    
    return bars, up_threshold, down_threshold

def analyze_classification_logic():
    """Analyze why classifications differ."""
    print("="*80)
    print("FIRST CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Get data from both logs
    opt_bars, opt_up, opt_down = extract_first_bars_and_classifications('opt_test_phase_full.log')
    test_bars, test_up, test_down = extract_first_bars_and_classifications('independent_test_full.log')
    
    print(f"\nThresholds:")
    print(f"  Trending UP: trend > {opt_up}")
    print(f"  Trending DOWN: trend < {opt_down}")
    print(f"  Default: {opt_down} <= trend <= {opt_up}")
    
    print("\n" + "-"*60)
    print("FIRST 5 BARS COMPARISON:")
    print("-"*60)
    print(f"{'Bar':<5} {'Opt Trend':<12} {'Test Trend':<12} {'Opt Class':<15} {'Test Class':<15}")
    print("-"*60)
    
    for i in range(min(5, len(opt_bars), len(test_bars))):
        opt = opt_bars[i] if i < len(opt_bars) else None
        test = test_bars[i] if i < len(test_bars) else None
        
        if opt and test:
            print(f"{opt['bar']:<5} {opt['trend']:<12.6f} {test['trend']:<12.6f} {opt['classification']:<15} {test['classification']:<15}")
    
    # Look for the actual first regime detector classification
    print("\n" + "-"*60)
    print("SEARCHING FOR INITIAL CLASSIFICATIONS:")
    print("-"*60)
    
    with open('opt_test_phase_full.log', 'r') as f:
        opt_content = f.read()
    
    with open('independent_test_full.log', 'r') as f:
        test_content = f.read()
    
    # Find first real classification from regime_detector
    regime_class_pattern = r"regime_detector.*?classification.*?trend=([-\d.]+).*?volatility=([\d.]+).*?→ regime=(\w+)"
    
    opt_first = re.search(regime_class_pattern, opt_content)
    test_first = re.search(regime_class_pattern, test_content)
    
    if opt_first:
        print(f"\nOptimization first classification:")
        print(f"  Trend: {opt_first.group(1)}")
        print(f"  Volatility: {opt_first.group(2)}")
        print(f"  Result: {opt_first.group(3)}")
    
    if test_first:
        print(f"\nTest run first classification:")
        print(f"  Trend: {test_first.group(1)}")
        print(f"  Volatility: {test_first.group(2)}")
        print(f"  Result: {test_first.group(3)}")
    
    # Check for warmup period differences
    print("\n" + "-"*60)
    print("CHECKING WARMUP BEHAVIOR:")
    print("-"*60)
    
    # Count "not ready" messages
    opt_not_ready = opt_content.count("not ready")
    test_not_ready = test_content.count("not ready")
    
    print(f"Optimization 'not ready' count: {opt_not_ready}")
    print(f"Test run 'not ready' count: {test_not_ready}")
    
    # Look for first valid trend calculation
    first_trend_pattern = r"ma_trend.*?value.*?([-\d.]+)"
    
    opt_trends = re.findall(first_trend_pattern, opt_content)[:5]
    test_trends = re.findall(first_trend_pattern, test_content)[:5]
    
    print(f"\nFirst few trend values:")
    print(f"Optimization: {opt_trends}")
    print(f"Test run: {test_trends}")

if __name__ == "__main__":
    analyze_classification_logic()