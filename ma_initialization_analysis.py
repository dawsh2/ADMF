#!/usr/bin/env python3
"""
Analyze MA initialization and calculation history differences
"""
import re
from datetime import datetime

def extract_ma_calculation_history(log_file_path, source_name, max_bars=50):
    """Extract MA calculation history from the beginning"""
    ma_history = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        # Look for the first bar processing and MA calculations
        for line_num, line in enumerate(lines):
            # Find bar indicators with MA values
            if '📊 BAR_' in line and 'INDICATORS:' in line:
                bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=([^,]+), Weights=\(MA:([^,]+),RSI:([^)]+)\)', line)
                if bar_match:
                    bar_num = int(bar_match.group(1))
                    bar_timestamp = bar_match.group(2)
                    price = float(bar_match.group(3))
                    ma_short = bar_match.group(4).strip()
                    ma_long = bar_match.group(5).strip()
                    rsi = bar_match.group(6).strip()
                    rsi_thresholds = bar_match.group(7)
                    regime = bar_match.group(8).strip()
                    ma_weight = bar_match.group(9).strip()
                    rsi_weight = bar_match.group(10).strip()
                    
                    # Convert MA values
                    try:
                        ma_short_val = float(ma_short) if ma_short != "N/A" and ma_short != "None" else None
                        ma_long_val = float(ma_long) if ma_long != "N/A" and ma_long != "None" else None
                    except:
                        ma_short_val = None
                        ma_long_val = None
                    
                    ma_history.append({
                        'source': source_name,
                        'bar_num': bar_num,
                        'timestamp': bar_timestamp,
                        'price': price,
                        'ma_short': ma_short_val,
                        'ma_long': ma_long_val,
                        'rsi': rsi,
                        'regime': regime,
                        'line_num': line_num
                    })
                    
                    if len(ma_history) >= max_bars:
                        break
            
            # Also look for any indicator initialization messages
            elif 'indicator' in line.lower() and ('initialized' in line.lower() or 'reset' in line.lower() or 'created' in line.lower()):
                ma_history.append({
                    'source': source_name,
                    'bar_num': 'INIT',
                    'timestamp': 'INIT',
                    'price': None,
                    'ma_short': None,
                    'ma_long': None,
                    'rsi': None,
                    'regime': 'INIT',
                    'line_num': line_num,
                    'init_message': line.strip()
                })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return ma_history

def find_data_start_points(log_file_path, source_name):
    """Find when data processing actually starts"""
    start_events = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Look for data loading, processing start, or component initialization
            if any(keyword in line.lower() for keyword in ['loading data', 'processing data', 'csv loaded', 'data_handler', 'started processing', 'begin']):
                start_events.append({
                    'source': source_name,
                    'line_num': line_num,
                    'event': line.strip()
                })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return start_events

def compare_ma_initialization(production_history, optimizer_history):
    """Compare MA initialization and early calculations"""
    
    print("=" * 120)
    print("MA INITIALIZATION AND WARM-UP ANALYSIS")
    print("=" * 120)
    
    print(f"\nMA History Lengths:")
    print(f"  Production: {len(production_history)} bars")
    print(f"  Optimizer:  {len(optimizer_history)} bars")
    
    # Show first 20 bars side by side
    print(f"\nFirst 20 Bars MA Calculation Comparison:")
    print("-" * 120)
    print(f"{'Bar':<4} {'Production':<50} {'Optimizer':<50} {'Match':<10}")
    print(f"{'#':<4} {'Timestamp | Price | MA_short | MA_long':<50} {'Timestamp | Price | MA_short | MA_long':<50} {'Status':<10}")
    print("-" * 120)
    
    max_compare = min(20, len(production_history), len(optimizer_history))
    
    for i in range(max_compare):
        prod_bar = production_history[i] if i < len(production_history) else None
        opt_bar = optimizer_history[i] if i < len(optimizer_history) else None
        
        if prod_bar and opt_bar:
            # Format the bar info
            if prod_bar['bar_num'] == 'INIT':
                prod_info = "INIT: " + prod_bar.get('init_message', '')[:40]
            else:
                prod_ts = prod_bar['timestamp'].split(' ')[1][:8] if ' ' in prod_bar['timestamp'] else prod_bar['timestamp'][:8]
                prod_info = f"{prod_ts} | {prod_bar['price']:.4f} | {prod_bar['ma_short'] or 'N/A'} | {prod_bar['ma_long'] or 'N/A'}"
            
            if opt_bar['bar_num'] == 'INIT':
                opt_info = "INIT: " + opt_bar.get('init_message', '')[:40]
            else:
                opt_ts = opt_bar['timestamp'].split(' ')[1][:8] if ' ' in opt_bar['timestamp'] else opt_bar['timestamp'][:8]
                opt_info = f"{opt_ts} | {opt_bar['price']:.4f} | {opt_bar['ma_short'] or 'N/A'} | {opt_bar['ma_long'] or 'N/A'}"
            
            # Check if they match
            if prod_bar['bar_num'] != 'INIT' and opt_bar['bar_num'] != 'INIT':
                price_match = abs(prod_bar['price'] - opt_bar['price']) < 0.01 if prod_bar['price'] and opt_bar['price'] else False
                ma_short_match = (prod_bar['ma_short'] is None and opt_bar['ma_short'] is None) or \
                               (prod_bar['ma_short'] is not None and opt_bar['ma_short'] is not None and abs(prod_bar['ma_short'] - opt_bar['ma_short']) < 0.01)
                ma_long_match = (prod_bar['ma_long'] is None and opt_bar['ma_long'] is None) or \
                              (prod_bar['ma_long'] is not None and opt_bar['ma_long'] is not None and abs(prod_bar['ma_long'] - opt_bar['ma_long']) < 0.01)
                
                if price_match and ma_short_match and ma_long_match:
                    match_status = "✅"
                else:
                    match_status = "❌"
            else:
                match_status = "INIT"
            
            bar_num = prod_bar['bar_num'] if prod_bar['bar_num'] != 'INIT' else opt_bar['bar_num']
            print(f"{str(bar_num):<4} {prod_info:<50} {opt_info:<50} {match_status:<10}")
        
        elif prod_bar and not opt_bar:
            prod_info = f"Bar {prod_bar['bar_num']}: {prod_bar['timestamp']} | {prod_bar['price']:.4f}"
            print(f"{str(prod_bar['bar_num']):<4} {prod_info:<50} {'[Missing]':<50} {'❌':<10}")
        
        elif opt_bar and not prod_bar:
            opt_info = f"Bar {opt_bar['bar_num']}: {opt_bar['timestamp']} | {opt_bar['price']:.4f}"
            print(f"{str(opt_bar['bar_num']):<4} {'[Missing]':<50} {opt_info:<50} {'❌':<10}")
    
    # Find first divergence point
    print(f"\n🔍 DIVERGENCE ANALYSIS:")
    print("-" * 60)
    
    first_divergence = None
    for i in range(min(len(production_history), len(optimizer_history))):
        prod_bar = production_history[i]
        opt_bar = optimizer_history[i]
        
        if prod_bar['bar_num'] != 'INIT' and opt_bar['bar_num'] != 'INIT':
            if prod_bar['ma_short'] is not None and opt_bar['ma_short'] is not None:
                if abs(prod_bar['ma_short'] - opt_bar['ma_short']) > 0.01:
                    first_divergence = i
                    break
    
    if first_divergence is not None:
        print(f"First MA divergence at bar {first_divergence + 1}")
        prod_bar = production_history[first_divergence]
        opt_bar = optimizer_history[first_divergence]
        print(f"Production: MA_short={prod_bar['ma_short']:.4f}, MA_long={prod_bar['ma_long']:.4f}")
        print(f"Optimizer:  MA_short={opt_bar['ma_short']:.4f}, MA_long={opt_bar['ma_long']:.4f}")
        print(f"Difference: MA_short={abs(prod_bar['ma_short'] - opt_bar['ma_short']):.4f}, MA_long={abs(prod_bar['ma_long'] - opt_bar['ma_long']):.4f}")
    else:
        print("No MA divergence found in first 20 bars")

def main():
    print("MA Initialization Analysis")
    
    # Extract MA calculation history from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nExtracting MA history from production: {production_file}")
    production_history = extract_ma_calculation_history(production_file, "PRODUCTION", max_bars=30)
    
    print(f"Extracting MA history from optimizer: {optimizer_file}")
    optimizer_history = extract_ma_calculation_history(optimizer_file, "OPTIMIZER", max_bars=30)
    
    # Find data start points
    print(f"\nFinding data start points...")
    production_starts = find_data_start_points(production_file, "PRODUCTION")
    optimizer_starts = find_data_start_points(optimizer_file, "OPTIMIZER")
    
    print(f"\nProduction data start events:")
    for event in production_starts[:3]:
        print(f"  Line {event['line_num']}: {event['event']}")
    
    print(f"\nOptimizer data start events:")
    for event in optimizer_starts[:3]:
        print(f"  Line {event['line_num']}: {event['event']}")
    
    # Compare MA initialization
    compare_ma_initialization(production_history, optimizer_history)

if __name__ == "__main__":
    main()