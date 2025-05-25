#!/usr/bin/env python3

import re
from collections import defaultdict

def extract_signals_from_log(log_file, label=""):
    """Extract signal generation details from log file"""
    signals = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if "Signal generated:" in line:
                # Extract timestamp
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                
                # Extract signal details
                signal_type_match = re.search(r"'signal_type': (-?[0-9]+)", line)
                strength_match = re.search(r"'signal_strength': ([0-9.]+)", line)
                price_match = re.search(r"'price_at_signal': ([0-9.]+)", line)
                
                if all([timestamp_match, signal_type_match, strength_match]):
                    signals.append({
                        'timestamp': timestamp_match.group(1),
                        'signal_type': int(signal_type_match.group(1)),
                        'signal_strength': float(strength_match.group(1)),
                        'price': float(price_match.group(1)) if price_match else None,
                        'line_num': line_num
                    })
    
    print(f"\n{label} Signals Summary:")
    print(f"Total signals: {len(signals)}")
    
    if signals:
        print(f"First signal: {signals[0]['timestamp']} (type: {signals[0]['signal_type']}, strength: {signals[0]['signal_strength']})")
        print(f"Last signal: {signals[-1]['timestamp']} (type: {signals[-1]['signal_type']}, strength: {signals[-1]['signal_strength']})")
        
        # Signal strength distribution
        strength_counts = defaultdict(int)
        for signal in signals:
            strength_counts[signal['signal_strength']] += 1
        
        print(f"Signal strength distribution:")
        for strength, count in sorted(strength_counts.items()):
            print(f"  {strength}: {count} signals")
    
    return signals

def compare_signal_timing(opt_signals, prod_signals):
    """Compare signal timing between optimization and production"""
    print(f"\n=== SIGNAL TIMING COMPARISON ===")
    
    # Look at first 10 signals from each
    print(f"\nFirst 10 signals comparison:")
    print(f"{'Index':<5} {'Optimization Time':<20} {'Production Time':<20} {'Opt Strength':<12} {'Prod Strength':<12}")
    print("-" * 85)
    
    min_count = min(len(opt_signals), len(prod_signals), 10)
    for i in range(min_count):
        opt_time = opt_signals[i]['timestamp']
        prod_time = prod_signals[i]['timestamp']
        opt_strength = opt_signals[i]['signal_strength']
        prod_strength = prod_signals[i]['signal_strength']
        
        print(f"{i:<5} {opt_time:<20} {prod_time:<20} {opt_strength:<12} {prod_strength:<12}")

def extract_weight_adjustments(log_file, label=""):
    """Extract weight adjustment events from log file"""
    adjustments = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if "Weights adjusted to" in line:
                # Extract timestamp
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                
                # Extract weights
                ma_weight_match = re.search(r"ma_weight=([0-9.]+)", line)
                rsi_weight_match = re.search(r"rsi_weight=([0-9.]+)", line)
                
                if all([timestamp_match, ma_weight_match, rsi_weight_match]):
                    adjustments.append({
                        'timestamp': timestamp_match.group(1),
                        'ma_weight': float(ma_weight_match.group(1)),
                        'rsi_weight': float(rsi_weight_match.group(1)),
                        'line_num': line_num
                    })
    
    print(f"\n{label} Weight Adjustments:")
    print(f"Total weight adjustments: {len(adjustments)}")
    
    if adjustments:
        print(f"First adjustment: {adjustments[0]['timestamp']} (MA: {adjustments[0]['ma_weight']}, RSI: {adjustments[0]['rsi_weight']})")
        print(f"Last adjustment: {adjustments[-1]['timestamp']} (MA: {adjustments[-1]['ma_weight']}, RSI: {adjustments[-1]['rsi_weight']})")
        
        # Weight distribution
        weight_combinations = defaultdict(int)
        for adj in adjustments:
            combo = f"MA:{adj['ma_weight']}, RSI:{adj['rsi_weight']}"
            weight_combinations[combo] += 1
        
        print(f"Weight combination distribution:")
        for combo, count in sorted(weight_combinations.items()):
            print(f"  {combo}: {count} times")
    
    return adjustments

def main():
    print("=== COMPARING OPTIMIZATION TEST PHASE vs ADAPTIVE PRODUCTION ===")
    
    # Extract signals from both logs
    print("\nExtracting signals from optimization log...")
    opt_signals = extract_signals_from_log('rsi.out', "OPTIMIZATION")
    
    print("\nExtracting signals from production log...")
    prod_signals = extract_signals_from_log('logs/admf_20250523_205431.log', "PRODUCTION")
    
    # Compare timing
    compare_signal_timing(opt_signals, prod_signals)
    
    # Extract weight adjustments
    print("\n" + "=" * 80)
    print("WEIGHT ADJUSTMENT ANALYSIS")
    print("=" * 80)
    
    opt_weights = extract_weight_adjustments('rsi.out', "OPTIMIZATION")
    prod_weights = extract_weight_adjustments('logs/admf_20250523_205431.log', "PRODUCTION")
    
    # Compare first few weight adjustments
    print(f"\nFirst 5 weight adjustments comparison:")
    print(f"{'Index':<5} {'Optimization Time':<20} {'Production Time':<20} {'Opt MA':<8} {'Opt RSI':<8} {'Prod MA':<8} {'Prod RSI':<8}")
    print("-" * 90)
    
    min_count = min(len(opt_weights), len(prod_weights), 5)
    for i in range(min_count):
        opt_time = opt_weights[i]['timestamp']
        prod_time = prod_weights[i]['timestamp']
        opt_ma = opt_weights[i]['ma_weight']
        opt_rsi = opt_weights[i]['rsi_weight']
        prod_ma = prod_weights[i]['ma_weight']
        prod_rsi = prod_weights[i]['rsi_weight']
        
        print(f"{i:<5} {opt_time:<20} {prod_time:<20} {opt_ma:<8} {opt_rsi:<8} {prod_ma:<8} {prod_rsi:<8}")

if __name__ == "__main__":
    main()