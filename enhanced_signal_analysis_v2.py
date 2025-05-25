#!/usr/bin/env python3
"""
Enhanced signal analysis v2 with detailed indicator tracking and side-by-side comparison
"""
import re
from datetime import datetime
from collections import defaultdict, OrderedDict
import json

def extract_comprehensive_data(log_file_path, source_name):
    """Extract comprehensive data including regime detector inputs, indicator states, and parameters"""
    data = {
        'test_start': None,
        'first_bar': None,
        'signals': [],
        'bar_states': OrderedDict(),  # timestamp -> full bar state
        'regime_changes': [],
        'regime_params': {},
        'indicator_warmups': {},
        'regime_detector_inputs': OrderedDict(),  # timestamp -> detector inputs
        'param_sources': {}  # regime -> source (JSON/FALLBACK)
    }
    
    current_regime = 'default'
    current_params = {}
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Extract test dataset range
            if 'Test dataset range:' in line:
                test_match = re.search(r'Test dataset range: ([0-9-]+\s[0-9:]+\+[0-9:]+) to ([0-9-]+\s[0-9:]+\+[0-9:]+)', line)
                if test_match:
                    data['test_start'] = test_match.group(1)
            
            # Track first bar
            if '📊 BAR_' in line and data['first_bar'] is None:
                bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', line)
                if bar_match:
                    data['first_bar'] = bar_match.group(1)
            
            # Track parameter loading source
            if 'Loading optimized parameters from' in line:
                if 'regime_optimized_parameters.json' in line:
                    # Next few lines will show which regimes loaded from JSON
                    for i in range(1, 10):
                        if line_num + i < len(lines):
                            next_line = lines[line_num + i]
                            if "Loaded parameters for regime" in next_line:
                                regime_match = re.search(r"Loaded parameters for regime '([^']+)'", next_line)
                                if regime_match:
                                    data['param_sources'][regime_match.group(1)] = 'JSON'
            
            # Track fallback parameters
            if "No optimized parameters found for regime" in line:
                regime_match = re.search(r"No optimized parameters found for regime '([^']+)'", line)
                if regime_match:
                    data['param_sources'][regime_match.group(1)] = 'FALLBACK'
            
            # Extract regime detector classification details
            if 'Regime classification:' in line:
                detector_match = re.search(r'Regime classification: trend_strength=([^,]+), volatility=([^,]+), rsi_level=([^,]+) → regime=(\w+)', line)
                if detector_match:
                    # Find the timestamp for this classification
                    timestamp = None
                    for prev_line_num in range(line_num-5, line_num):
                        if prev_line_num >= 0 and '📊 BAR_' in lines[prev_line_num]:
                            bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[prev_line_num])
                            if bar_match:
                                timestamp = bar_match.group(1)
                                break
                    
                    if timestamp:
                        data['regime_detector_inputs'][timestamp] = {
                            'trend_strength': detector_match.group(1),
                            'volatility': detector_match.group(2),
                            'rsi_level': detector_match.group(3),
                            'regime': detector_match.group(4)
                        }
            
            # Extract regime changes
            if "REGIME CHANGED:" in line:
                change_match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", line)
                if change_match:
                    data['regime_changes'].append({
                        'from': change_match.group(1),
                        'to': change_match.group(2),
                        'timestamp': change_match.group(3)
                    })
                    current_regime = change_match.group(2)
            
            # Extract regime parameters
            if "REGIME PARAMETER UPDATE:" in line:
                param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", line)
                if param_match:
                    regime = param_match.group(1)
                    params = eval(param_match.group(2))
                    data['regime_params'][regime] = params
                    if regime == current_regime:
                        current_params = params
            
            # Extract detailed bar state with indicator values and processing info
            if '📊 BAR_' in line and 'INDICATORS:' in line:
                bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=(\w+)', line)
                if bar_match:
                    timestamp = bar_match.group(2)
                    bar_num = int(bar_match.group(1))
                    
                    # Try to find bars processed info from nearby debug lines
                    bars_processed = {'ma_short': bar_num, 'ma_long': bar_num, 'rsi': bar_num}
                    
                    # Look for RSI calculation details nearby
                    for offset in range(-5, 5):
                        if 0 <= line_num + offset < len(lines):
                            nearby_line = lines[line_num + offset]
                            if 'RSI calculation:' in nearby_line:
                                rsi_calc_match = re.search(r'RSI calculation: period=(\d+), values_count=(\d+)', nearby_line)
                                if rsi_calc_match:
                                    bars_processed['rsi'] = int(rsi_calc_match.group(2))
                    
                    data['bar_states'][timestamp] = {
                        'bar_num': bar_num,
                        'price': float(bar_match.group(3)),
                        'ma_short': bar_match.group(4),
                        'ma_long': bar_match.group(5),
                        'rsi': bar_match.group(6),
                        'rsi_thresholds': bar_match.group(7),
                        'regime': bar_match.group(8),
                        'params': current_params.copy() if current_params else {},
                        'param_source': data['param_sources'].get(bar_match.group(8), 'UNKNOWN'),
                        'bars_processed': bars_processed
                    }
            
            # Extract signals
            if '🚨 SIGNAL GENERATED' in line:
                signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\), Combined_strength=([+-]?[0-9.]+), Final_multiplier=([0-9.]+)', line)
                if signal_match:
                    # Find timestamp for this signal
                    signal_timestamp = None
                    for prev_line_num in range(line_num-10, line_num):
                        if prev_line_num >= 0 and '📊 BAR_' in lines[prev_line_num]:
                            bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[prev_line_num])
                            if bar_match:
                                signal_timestamp = bar_match.group(1)
                                break
                    
                    data['signals'].append({
                        'timestamp': signal_timestamp,
                        'signal_num': signal_match.group(1),
                        'type': int(signal_match.group(2)),
                        'price': float(signal_match.group(3)),
                        'regime': signal_match.group(4),
                        'ma_signal': int(signal_match.group(5)),
                        'ma_weight': float(signal_match.group(6)),
                        'rsi_signal': int(signal_match.group(7)),
                        'rsi_weight': float(signal_match.group(8)),
                        'combined_strength': float(signal_match.group(9)),
                        'final_multiplier': float(signal_match.group(10))
                    })
    
    except Exception as e:
        print(f"Error processing {log_file_path}: {e}")
    
    return data

def format_indicator_value(value, period, bars_processed):
    """Format indicator value with period and bars processed info"""
    if value == 'N/A':
        return f"N/A(P:{period} B:{bars_processed})"
    else:
        try:
            float_val = float(value)
            return f"{float_val:.2f}(P:{period} B:{bars_processed})"
        except:
            return f"{value}(P:{period} B:{bars_processed})"

def compare_at_timestamps(prod_data, opt_data, timestamps_to_compare):
    """Compare state at specific timestamps"""
    print("\n" + "=" * 140)
    print("SIDE-BY-SIDE COMPARISON AT KEY TIMESTAMPS")
    print("=" * 140)
    
    for ts in timestamps_to_compare:
        print(f"\n--- TIMESTAMP: {ts} ---")
        
        prod_state = prod_data['bar_states'].get(ts)
        opt_state = opt_data['bar_states'].get(ts)
        
        if prod_state:
            print("\nPRODUCTION:")
            params = prod_state['params']
            print(f"  Regime: {prod_state['regime']} (Params from: {prod_state['param_source']})")
            print(f"  Price: {prod_state['price']:.4f}")
            
            # Extract periods from params
            ma_short_period = params.get('short_window', 'UNK')
            ma_long_period = params.get('long_window', 'UNK')
            rsi_period = params.get('rsi_indicator.period', 'UNK')
            
            print(f"  MA_S: {format_indicator_value(prod_state['ma_short'], ma_short_period, prod_state['bars_processed']['ma_short'])}")
            print(f"  MA_L: {format_indicator_value(prod_state['ma_long'], ma_long_period, prod_state['bars_processed']['ma_long'])}")
            print(f"  RSI:  {format_indicator_value(prod_state['rsi'], rsi_period, prod_state['bars_processed']['rsi'])}")
            
            # Check if signal generated at this timestamp
            prod_signal = next((s for s in prod_data['signals'] if s['timestamp'] == ts), None)
            if prod_signal:
                print(f"  SIGNAL: Type={prod_signal['type']}, MA={prod_signal['ma_signal']}, RSI={prod_signal['rsi_signal']}, Combined={prod_signal['combined_strength']:.2f}")
        else:
            print("\nPRODUCTION: No data at this timestamp")
        
        if opt_state:
            print("\nOPTIMIZER:")
            params = opt_state['params']
            print(f"  Regime: {opt_state['regime']} (Params from: {opt_state['param_source']})")
            print(f"  Price: {opt_state['price']:.4f}")
            
            # Extract periods from params
            ma_short_period = params.get('short_window', 'UNK')
            ma_long_period = params.get('long_window', 'UNK')
            rsi_period = params.get('rsi_indicator.period', 'UNK')
            
            print(f"  MA_S: {format_indicator_value(opt_state['ma_short'], ma_short_period, opt_state['bars_processed']['ma_short'])}")
            print(f"  MA_L: {format_indicator_value(opt_state['ma_long'], ma_long_period, opt_state['bars_processed']['ma_long'])}")
            print(f"  RSI:  {format_indicator_value(opt_state['rsi'], rsi_period, opt_state['bars_processed']['rsi'])}")
            
            # Check if signal generated at this timestamp
            opt_signal = next((s for s in opt_data['signals'] if s['timestamp'] == ts), None)
            if opt_signal:
                print(f"  SIGNAL: Type={opt_signal['type']}, MA={opt_signal['ma_signal']}, RSI={opt_signal['rsi_signal']}, Combined={opt_signal['combined_strength']:.2f}")
        else:
            print("\nOPTIMIZER: No data at this timestamp")
        
        # Show regime detector inputs if available
        prod_detector = prod_data['regime_detector_inputs'].get(ts)
        opt_detector = opt_data['regime_detector_inputs'].get(ts)
        
        if prod_detector or opt_detector:
            print("\nREGIME DETECTOR INPUTS:")
            if prod_detector:
                print(f"  Production: trend={prod_detector['trend_strength']}, vol={prod_detector['volatility']}, rsi={prod_detector['rsi_level']} → {prod_detector['regime']}")
            if opt_detector:
                print(f"  Optimizer:  trend={opt_detector['trend_strength']}, vol={opt_detector['volatility']}, rsi={opt_detector['rsi_level']} → {opt_detector['regime']}")
        
        # Highlight key differences
        if prod_state and opt_state:
            print("\nKEY DIFFERENCES:")
            if prod_state['regime'] != opt_state['regime']:
                print(f"  ❌ Regime: {prod_state['regime']} vs {opt_state['regime']}")
            if prod_state['param_source'] != opt_state['param_source']:
                print(f"  ❌ Param Source: {prod_state['param_source']} vs {opt_state['param_source']}")
            if (prod_state['rsi'] == 'N/A') != (opt_state['rsi'] == 'N/A'):
                print(f"  ❌ RSI Availability: Prod={prod_state['rsi']} vs Opt={opt_state['rsi']}")

def analyze_early_divergence(prod_data, opt_data):
    """Analyze early divergence in regime detection"""
    print("\n" + "=" * 140)
    print("EARLY DIVERGENCE ANALYSIS (First 30 bars)")
    print("=" * 140)
    
    # Get first 30 common timestamps
    common_timestamps = sorted(set(prod_data['bar_states'].keys()) & set(opt_data['bar_states'].keys()))[:30]
    
    print(f"\nREGIME DETECTOR COMPARISON FOR FIRST {len(common_timestamps)} BARS:")
    print(f"{'IDX':>3} {'TIMESTAMP':^20} | {'PROD REGIME':^15} {'DETECTOR INPUTS':^40} | {'OPT REGIME':^15} {'DETECTOR INPUTS':^40}")
    print("-" * 140)
    
    for idx, ts in enumerate(common_timestamps):
        prod_state = prod_data['bar_states'].get(ts, {})
        opt_state = opt_data['bar_states'].get(ts, {})
        
        prod_detector = prod_data['regime_detector_inputs'].get(ts, {})
        opt_detector = opt_data['regime_detector_inputs'].get(ts, {})
        
        prod_regime = prod_state.get('regime', 'N/A')
        opt_regime = opt_state.get('regime', 'N/A')
        
        prod_inputs = f"t:{prod_detector.get('trend_strength', 'N/A')}, v:{prod_detector.get('volatility', 'N/A')}, r:{prod_detector.get('rsi_level', 'N/A')}" if prod_detector else "N/A"
        opt_inputs = f"t:{opt_detector.get('trend_strength', 'N/A')}, v:{opt_detector.get('volatility', 'N/A')}, r:{opt_detector.get('rsi_level', 'N/A')}" if opt_detector else "N/A"
        
        match = "✓" if prod_regime == opt_regime else "✗"
        print(f"{idx:>3} {ts[:19]:^20} | {prod_regime:^15} {prod_inputs:^40} | {opt_regime:^15} {opt_inputs:^40} {match}")

def main():
    print("Enhanced Signal Analysis v2 - Detailed Indicator and Regime Tracking")
    
    # Extract comprehensive data
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print(f"\nAnalyzing production: {production_file}")
    prod_data = extract_comprehensive_data(production_file, "PRODUCTION")
    
    print(f"Analyzing optimizer: {optimizer_file}")
    opt_data = extract_comprehensive_data(optimizer_file, "OPTIMIZER")
    
    # Summary
    print("\n" + "=" * 140)
    print("ANALYSIS SUMMARY")
    print("=" * 140)
    
    print(f"\n📅 TEST INFO:")
    print(f"  Production - First Bar: {prod_data['first_bar']}, Signals: {len(prod_data['signals'])}")
    print(f"  Optimizer  - First Bar: {opt_data['first_bar']}, Signals: {len(opt_data['signals'])}")
    
    print(f"\n📊 PARAMETER SOURCES:")
    print(f"  Production:")
    for regime, source in prod_data['param_sources'].items():
        print(f"    {regime}: {source}")
    print(f"  Optimizer:")
    for regime, source in opt_data['param_sources'].items():
        print(f"    {regime}: {source}")
    
    # Analyze early divergence
    analyze_early_divergence(prod_data, opt_data)
    
    # Compare at key timestamps
    key_timestamps = [
        "2024-03-28 13:46:00+00:00",  # First bar
        "2024-03-28 14:06:00+00:00",  # First production regime change
        "2024-03-28 14:11:00+00:00",  # Regime divergence point
        "2024-03-28 14:12:00+00:00",  # First production signal
        "2024-03-28 14:30:00+00:00",  # Signal 2 comparison point
    ]
    
    compare_at_timestamps(prod_data, opt_data, key_timestamps)

if __name__ == "__main__":
    main()