#!/usr/bin/env python3
"""
Enhanced indicator state analysis with absolute bar indexing and detailed warm-up tracking
"""
import re
from datetime import datetime

def extract_enhanced_bar_data(log_file_path, source_name, max_bars=100):
    """Extract detailed bar data with enhanced indicator state information"""
    bars = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        # Look for bar indicators and enhanced debug info
        for line_num, line in enumerate(lines):
            # Standard bar indicators
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
                    
                    # Convert values
                    try:
                        ma_short_val = float(ma_short) if ma_short not in ["N/A", "None"] else None
                        ma_long_val = float(ma_long) if ma_long not in ["N/A", "None"] else None
                        rsi_val = float(rsi) if rsi not in ["N/A", "None"] else None
                    except:
                        ma_short_val = None
                        ma_long_val = None
                        rsi_val = None
                    
                    bars.append({
                        'source': source_name,
                        'bar_num': bar_num,
                        'timestamp': bar_timestamp,
                        'price': price,
                        'ma_short': ma_short_val,
                        'ma_long': ma_long_val,
                        'rsi': rsi_val,
                        'rsi_raw': rsi,
                        'regime': regime,
                        'ma_weight': ma_weight,
                        'rsi_weight': rsi_weight,
                        'line_num': line_num,
                        'full_line': line.strip()
                    })
                    
                    if len(bars) >= max_bars:
                        break
            
            # Look for indicator initialization/validity messages
            elif 'First valid' in line or 'becomes valid' in line or 'indicator.*valid' in line.lower():
                bars.append({
                    'source': source_name,
                    'bar_num': 'INDICATOR_INIT',
                    'timestamp': 'INIT',
                    'init_message': line.strip(),
                    'line_num': line_num
                })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return bars

def extract_signal_timing_context(log_file_path, source_name, max_signals=20):
    """Extract signals with detailed timing and bar context"""
    signals = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            if '🚨 SIGNAL GENERATED' in line:
                signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+)', line)
                if signal_match:
                    signal_num = signal_match.group(1)
                    signal_type = int(signal_match.group(2))
                    price = float(signal_match.group(3))
                    regime = signal_match.group(4)
                    
                    # Look for the corresponding bar data around this signal
                    bar_context = []
                    for ctx_line_num in range(max(0, line_num - 10), min(len(lines), line_num + 5)):
                        ctx_line = lines[ctx_line_num].strip()
                        if '📊 BAR_' in ctx_line and 'INDICATORS:' in ctx_line:
                            bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+)', ctx_line)
                            if bar_match:
                                bar_context.append({
                                    'bar_num': int(bar_match.group(1)),
                                    'timestamp': bar_match.group(2),
                                    'price': float(bar_match.group(3)),
                                    'ma_short': bar_match.group(4),
                                    'ma_long': bar_match.group(5),
                                    'rsi': bar_match.group(6)
                                })
                    
                    signals.append({
                        'source': source_name,
                        'signal_num': signal_num,
                        'signal_type': signal_type,
                        'price': price,
                        'regime': regime,
                        'line_num': line_num,
                        'bar_context': bar_context
                    })
                    
                    if len(signals) >= max_signals:
                        break
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return signals

def analyze_indicator_warm_up_differences(production_bars, optimizer_bars):
    """Analyze indicator warm-up and validity differences"""
    
    print("=" * 140)
    print("INDICATOR WARM-UP AND STATE ANALYSIS")
    print("=" * 140)
    
    # Find when indicators become valid in each system
    print(f"\n🔍 INDICATOR VALIDITY TIMELINE:")
    print("-" * 80)
    
    # Extract non-N/A indicator first appearances
    prod_first_valid = {}
    opt_first_valid = {}
    
    for bar in production_bars:
        if bar.get('bar_num') != 'INDICATOR_INIT':
            if bar.get('ma_short') is not None and 'ma_short' not in prod_first_valid:
                prod_first_valid['ma_short'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['ma_short']}
            if bar.get('ma_long') is not None and 'ma_long' not in prod_first_valid:
                prod_first_valid['ma_long'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['ma_long']}
            if bar.get('rsi') is not None and 'rsi' not in prod_first_valid:
                prod_first_valid['rsi'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['rsi']}
    
    for bar in optimizer_bars:
        if bar.get('bar_num') != 'INDICATOR_INIT':
            if bar.get('ma_short') is not None and 'ma_short' not in opt_first_valid:
                opt_first_valid['ma_short'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['ma_short']}
            if bar.get('ma_long') is not None and 'ma_long' not in opt_first_valid:
                opt_first_valid['ma_long'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['ma_long']}
            if bar.get('rsi') is not None and 'rsi' not in opt_first_valid:
                opt_first_valid['rsi'] = {'bar': bar['bar_num'], 'timestamp': bar['timestamp'], 'value': bar['rsi']}
    
    print(f"Production First Valid Indicators:")
    for indicator, info in prod_first_valid.items():
        ts = info['timestamp'].split(' ')[1][:8] if ' ' in info['timestamp'] else info['timestamp'][:8]
        print(f"  {indicator.upper()}: Bar {info['bar']} at {ts} = {info['value']}")
    
    print(f"\nOptimizer First Valid Indicators:")
    for indicator, info in opt_first_valid.items():
        ts = info['timestamp'].split(' ')[1][:8] if ' ' in info['timestamp'] else info['timestamp'][:8]
        print(f"  {indicator.upper()}: Bar {info['bar']} at {ts} = {info['value']}")
    
    # Compare side-by-side for first 25 bars
    print(f"\n📊 SIDE-BY-SIDE BAR COMPARISON (First 25 bars):")
    print("-" * 140)
    print(f"{'Bar':<4} {'Production':<65} {'Optimizer':<65} {'Match':<6}")
    print(f"{'#':<4} {'Time | Price | MA_s | MA_l | RSI | Regime':<65} {'Time | Price | MA_s | MA_l | RSI | Regime':<65} {'Status':<6}")
    print("-" * 140)
    
    max_compare = min(25, len(production_bars), len(optimizer_bars))
    
    # Filter out INDICATOR_INIT entries
    prod_bars_filtered = [b for b in production_bars if b.get('bar_num') != 'INDICATOR_INIT']
    opt_bars_filtered = [b for b in optimizer_bars if b.get('bar_num') != 'INDICATOR_INIT']
    
    for i in range(max_compare):
        prod_bar = prod_bars_filtered[i] if i < len(prod_bars_filtered) else None
        opt_bar = opt_bars_filtered[i] if i < len(opt_bars_filtered) else None
        
        if prod_bar and opt_bar:
            # Format production info
            prod_ts = prod_bar['timestamp'].split(' ')[1][:8] if ' ' in prod_bar['timestamp'] else prod_bar['timestamp'][:8]
            prod_ma_s = f"{prod_bar['ma_short']:.2f}" if prod_bar['ma_short'] is not None else "N/A"
            prod_ma_l = f"{prod_bar['ma_long']:.2f}" if prod_bar['ma_long'] is not None else "N/A"
            prod_rsi = f"{prod_bar['rsi']:.1f}" if prod_bar['rsi'] is not None else "N/A"
            prod_info = f"{prod_ts} | {prod_bar['price']:.3f} | {prod_ma_s} | {prod_ma_l} | {prod_rsi} | {prod_bar['regime']}"
            
            # Format optimizer info
            opt_ts = opt_bar['timestamp'].split(' ')[1][:8] if ' ' in opt_bar['timestamp'] else opt_bar['timestamp'][:8]
            opt_ma_s = f"{opt_bar['ma_short']:.2f}" if opt_bar['ma_short'] is not None else "N/A"
            opt_ma_l = f"{opt_bar['ma_long']:.2f}" if opt_bar['ma_long'] is not None else "N/A"
            opt_rsi = f"{opt_bar['rsi']:.1f}" if opt_bar['rsi'] is not None else "N/A"
            opt_info = f"{opt_ts} | {opt_bar['price']:.3f} | {opt_ma_s} | {opt_ma_l} | {opt_rsi} | {opt_bar['regime']}"
            
            # Check for matches
            price_match = abs(prod_bar['price'] - opt_bar['price']) < 0.01
            ma_s_match = (prod_bar['ma_short'] is None and opt_bar['ma_short'] is None) or \
                        (prod_bar['ma_short'] is not None and opt_bar['ma_short'] is not None and abs(prod_bar['ma_short'] - opt_bar['ma_short']) < 0.01)
            ma_l_match = (prod_bar['ma_long'] is None and opt_bar['ma_long'] is None) or \
                        (prod_bar['ma_long'] is not None and opt_bar['ma_long'] is not None and abs(prod_bar['ma_long'] - opt_bar['ma_long']) < 0.01)
            rsi_match = (prod_bar['rsi'] is None and opt_bar['rsi'] is None) or \
                       (prod_bar['rsi'] is not None and opt_bar['rsi'] is not None and abs(prod_bar['rsi'] - opt_bar['rsi']) < 0.1)
            regime_match = prod_bar['regime'] == opt_bar['regime']
            
            if price_match and ma_s_match and ma_l_match and rsi_match and regime_match:
                match_status = "✅"
            else:
                match_status = "❌"
            
            print(f"{prod_bar['bar_num']:<4} {prod_info:<65} {opt_info:<65} {match_status:<6}")
        
        elif prod_bar:
            prod_ts = prod_bar['timestamp'].split(' ')[1][:8] if ' ' in prod_bar['timestamp'] else prod_bar['timestamp'][:8]
            prod_info = f"{prod_ts} | {prod_bar['price']:.3f} | ... | {prod_bar['regime']}"
            print(f"{prod_bar['bar_num']:<4} {prod_info:<65} {'[MISSING]':<65} {'❌':<6}")
        
        elif opt_bar:
            opt_ts = opt_bar['timestamp'].split(' ')[1][:8] if ' ' in opt_bar['timestamp'] else opt_bar['timestamp'][:8]
            opt_info = f"{opt_ts} | {opt_bar['price']:.3f} | ... | {opt_bar['regime']}"
            print(f"{opt_bar['bar_num']:<4} {'[MISSING]':<65} {opt_info:<65} {'❌':<6}")

def analyze_signal_timing_mismatch(production_signals, optimizer_signals):
    """Analyze timing mismatch in signal generation"""
    
    print(f"\n🚨 SIGNAL GENERATION TIMING ANALYSIS:")
    print("-" * 100)
    
    print(f"Production Signals ({len(production_signals)}):")
    for i, signal in enumerate(production_signals[:10]):
        if signal['bar_context']:
            bar = signal['bar_context'][0]  # Most recent bar context
            ts = bar['timestamp'].split(' ')[1][:8] if ' ' in bar['timestamp'] else bar['timestamp'][:8]
            print(f"  Signal #{signal['signal_num']}: Type={signal['signal_type']}, Price={signal['price']}, Regime={signal['regime']}")
            print(f"    Bar context: {ts}, MA_short={bar['ma_short']}, MA_long={bar['ma_long']}, RSI={bar['rsi']}")
    
    print(f"\nOptimizer Signals ({len(optimizer_signals)}):")
    for i, signal in enumerate(optimizer_signals[:10]):
        if signal['bar_context']:
            bar = signal['bar_context'][0]  # Most recent bar context
            ts = bar['timestamp'].split(' ')[1][:8] if ' ' in bar['timestamp'] else bar['timestamp'][:8]
            print(f"  Signal #{signal['signal_num']}: Type={signal['signal_type']}, Price={signal['price']}, Regime={signal['regime']}")
            print(f"    Bar context: {ts}, MA_short={bar['ma_short']}, MA_long={bar['ma_long']}, RSI={bar['rsi']}")

def main():
    print("Enhanced Indicator State Analysis")
    
    # Extract enhanced bar data from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nExtracting enhanced bar data from production: {production_file}")
    production_bars = extract_enhanced_bar_data(production_file, "PRODUCTION", max_bars=50)
    
    print(f"Extracting enhanced bar data from optimizer: {optimizer_file}")
    optimizer_bars = extract_enhanced_bar_data(optimizer_file, "OPTIMIZER", max_bars=50)
    
    print(f"Extracting signal timing from production: {production_file}")
    production_signals = extract_signal_timing_context(production_file, "PRODUCTION", max_signals=15)
    
    print(f"Extracting signal timing from optimizer: {optimizer_file}")
    optimizer_signals = extract_signal_timing_context(optimizer_file, "OPTIMIZER", max_signals=15)
    
    # Analyze indicator warm-up differences
    analyze_indicator_warm_up_differences(production_bars, optimizer_bars)
    
    # Analyze signal timing mismatch
    analyze_signal_timing_mismatch(production_signals, optimizer_signals)

if __name__ == "__main__":
    main()