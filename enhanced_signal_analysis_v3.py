#!/usr/bin/env python3
"""
Enhanced signal analysis v3 - Complete warm-up tracking with detector inputs
"""
import re
from datetime import datetime
from collections import defaultdict, OrderedDict
import json

def extract_comprehensive_data_v3(log_file_path, source_name):
    """Extract all data including detector inputs, RSI lifecycle, and parameter sources"""
    data = {
        'test_start': None,
        'first_bar': None,
        'signals': [],
        'bar_states': OrderedDict(),
        'regime_changes': [],
        'regime_params': {},
        'indicator_warmups': defaultdict(dict),
        'regime_detector_states': OrderedDict(),  # timestamp -> detector state
        'param_sources': {},
        'rsi_lifecycle': [],  # Track RSI state changes
        'indicator_instances': defaultdict(lambda: {'created': None, 'reset_count': 0, 'values': []})
    }
    
    current_regime = 'default'
    current_params = {}
    current_param_source = 'UNKNOWN'
    
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
            
            # Track parameter source when loading
            if 'Loading parameters for regime' in line:
                source_match = re.search(r"Loading parameters for regime '([^']+)' from ([^:]+)", line)
                if source_match:
                    regime = source_match.group(1)
                    source = 'JSON' if 'json' in source_match.group(2).lower() else 'CONFIG'
                    data['param_sources'][regime] = source
            
            # Track when using fallback parameters
            if "using fallback parameters" in line or "No optimized parameters found" in line:
                regime_match = re.search(r"regime '([^']+)'", line)
                if regime_match:
                    data['param_sources'][regime_match.group(1)] = 'FALLBACK'
            
            # Track loaded parameters confirmation
            if "Loaded parameters for regime" in line:
                regime_match = re.search(r"Loaded parameters for regime '([^']+)'", line)
                if regime_match:
                    regime = regime_match.group(1)
                    if regime not in data['param_sources']:
                        data['param_sources'][regime] = 'JSON'
            
            # Extract NEW regime classification format with detector values
            if 'Regime classification:' in line:
                class_match = re.search(r'Regime classification: trend_strength=([^,]+), volatility=([^,]+), rsi_level=([^,]+) → regime=(\w+)', line)
                if class_match:
                    # Find timestamp
                    timestamp = None
                    for prev_line_num in range(line_num-10, line_num):
                        if prev_line_num >= 0 and '📊 BAR_' in lines[prev_line_num]:
                            bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[prev_line_num])
                            if bar_match:
                                timestamp = bar_match.group(1)
                                break
                    
                    if timestamp:
                        data['regime_detector_states'][timestamp] = {
                            'trend_strength': class_match.group(1),
                            'volatility': class_match.group(2),
                            'rsi_level': class_match.group(3),
                            'classified_regime': class_match.group(4)
                        }
            
            # Extract regime detector raw indicator values
            if "RegimeDet" in line and "Indicator values:" in line:
                detector_match = re.search(r"at ([0-9-]+\s[0-9:]+\+[0-9:]+): .*Indicator values: ({[^}]+})", line)
                if detector_match:
                    timestamp = detector_match.group(1)
                    try:
                        indicator_vals = eval(detector_match.group(2))
                        if timestamp in data['regime_detector_states']:
                            data['regime_detector_states'][timestamp]['raw_indicators'] = indicator_vals
                        else:
                            data['regime_detector_states'][timestamp] = {'raw_indicators': indicator_vals}
                    except:
                        pass
            
            # Track RSI lifecycle events
            if "RSI" in line and ("first valid value" in line or "reset" in line or "state reset" in line):
                data['rsi_lifecycle'].append({
                    'line_num': line_num,
                    'content': line.strip(),
                    'timestamp': 'TBD'  # Will be filled from context
                })
            
            # Track RSI calculation details
            if 'RSI calculation:' in line:
                rsi_calc_match = re.search(r'RSI calculation: period=(\d+), values_count=(\d+), value=([0-9.]+|N/A)', line)
                if rsi_calc_match:
                    # Find associated timestamp
                    for offset in range(-5, 5):
                        if 0 <= line_num + offset < len(lines) and '📊 BAR_' in lines[line_num + offset]:
                            bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[line_num + offset])
                            if bar_match:
                                ts = bar_match.group(1)
                                if 'rsi_calc_details' not in data:
                                    data['rsi_calc_details'] = OrderedDict()
                                data['rsi_calc_details'][ts] = {
                                    'period': int(rsi_calc_match.group(1)),
                                    'values_count': int(rsi_calc_match.group(2)),
                                    'value': rsi_calc_match.group(3)
                                }
                                break
            
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
                    # Look for parameter source in nearby lines
                    for offset in range(1, 10):
                        if line_num + offset < len(lines):
                            next_line = lines[line_num + offset]
                            if "Loading parameters" in next_line or "using fallback" in next_line:
                                if current_regime not in data['param_sources']:
                                    data['param_sources'][current_regime] = 'CHECKING'
                                break
            
            # Extract regime parameters
            if "REGIME PARAMETER UPDATE:" in line:
                param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", line)
                if param_match:
                    regime = param_match.group(1)
                    params = eval(param_match.group(2))
                    data['regime_params'][regime] = params
                    if regime == current_regime:
                        current_params = params
                        current_param_source = data['param_sources'].get(regime, 'UNKNOWN')
            
            # Extract detailed bar state
            if '📊 BAR_' in line and 'INDICATORS:' in line:
                bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=(\w+)', line)
                if bar_match:
                    timestamp = bar_match.group(2)
                    bar_num = int(bar_match.group(1))
                    
                    # Get RSI calculation details if available
                    rsi_details = data.get('rsi_calc_details', {}).get(timestamp, {})
                    
                    data['bar_states'][timestamp] = {
                        'bar_num': bar_num,
                        'price': float(bar_match.group(3)),
                        'ma_short': bar_match.group(4),
                        'ma_long': bar_match.group(5),
                        'rsi': bar_match.group(6),
                        'rsi_thresholds': bar_match.group(7),
                        'regime': bar_match.group(8),
                        'params': current_params.copy() if current_params else {},
                        'param_source': current_param_source,
                        'rsi_calc': rsi_details,
                        'bars_processed': {
                            'ma_short': bar_num,
                            'ma_long': bar_num,
                            'rsi': rsi_details.get('values_count', bar_num)
                        }
                    }
            
            # Extract signals
            if '🚨 SIGNAL GENERATED' in line:
                signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\), Combined_strength=([+-]?[0-9.]+), Final_multiplier=([0-9.]+)', line)
                if signal_match:
                    # Find timestamp
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

def format_indicator_value_v3(value, period, bars_processed, calc_details=None):
    """Format indicator value with detailed info"""
    if value == 'N/A':
        if calc_details and 'values_count' in calc_details:
            return f"N/A(P:{period} B:{calc_details['values_count']})"
        return f"N/A(P:{period} B:{bars_processed})"
    else:
        try:
            float_val = float(value)
            if calc_details and 'values_count' in calc_details:
                return f"{float_val:.2f}(P:{period} B:{calc_details['values_count']})"
            return f"{float_val:.2f}(P:{period} B:{bars_processed})"
        except:
            return f"{value}(P:{period} B:{bars_processed})"

def analyze_detector_warmup(prod_data, opt_data):
    """Analyze regime detector warm-up states"""
    print("\n" + "=" * 140)
    print("REGIME DETECTOR WARM-UP ANALYSIS")
    print("=" * 140)
    
    # Get first 30 timestamps from both
    all_timestamps = sorted(set(list(prod_data['regime_detector_states'].keys())[:30] + 
                               list(opt_data['regime_detector_states'].keys())[:30]))[:30]
    
    print(f"\nDETECTOR INDICATOR VALUES FOR FIRST {len(all_timestamps)} CLASSIFICATIONS:")
    print(f"{'IDX':>3} {'TIMESTAMP':^20} | {'PRODUCTION':^50} | {'OPTIMIZER':^50}")
    print(f"{'':>3} {'':^20} | {'Trend / Vol / RSI → Regime':^50} | {'Trend / Vol / RSI → Regime':^50}")
    print("-" * 140)
    
    for idx, ts in enumerate(all_timestamps):
        prod_state = prod_data['regime_detector_states'].get(ts, {})
        opt_state = opt_data['regime_detector_states'].get(ts, {})
        
        # Format production detector state
        if prod_state:
            prod_str = f"{prod_state.get('trend_strength', 'N/A'):>6} / {prod_state.get('volatility', 'N/A'):>6} / {prod_state.get('rsi_level', 'N/A'):>6} → {prod_state.get('classified_regime', 'N/A'):>15}"
        else:
            prod_str = "No classification data"
        
        # Format optimizer detector state
        if opt_state:
            opt_str = f"{opt_state.get('trend_strength', 'N/A'):>6} / {opt_state.get('volatility', 'N/A'):>6} / {opt_state.get('rsi_level', 'N/A'):>6} → {opt_state.get('classified_regime', 'N/A'):>15}"
        else:
            opt_str = "No classification data"
        
        match = "✓" if (prod_state.get('classified_regime') == opt_state.get('classified_regime') and prod_state and opt_state) else "✗"
        print(f"{idx:>3} {ts[:19]:^20} | {prod_str:^50} | {opt_str:^50} {match}")

def analyze_rsi_lifecycle(prod_data, opt_data):
    """Analyze RSI lifecycle events"""
    print("\n" + "=" * 140)
    print("RSI LIFECYCLE ANALYSIS")
    print("=" * 140)
    
    print("\nPRODUCTION RSI EVENTS:")
    for event in prod_data['rsi_lifecycle'][:10]:
        print(f"  Line {event['line_num']}: {event['content']}")
    
    print("\nOPTIMIZER RSI EVENTS:")
    for event in opt_data['rsi_lifecycle'][:10]:
        print(f"  Line {event['line_num']}: {event['content']}")

def compare_at_timestamps_v3(prod_data, opt_data, timestamps_to_compare):
    """Enhanced side-by-side comparison"""
    print("\n" + "=" * 140)
    print("ENHANCED SIDE-BY-SIDE COMPARISON AT KEY TIMESTAMPS")
    print("=" * 140)
    
    for ts in timestamps_to_compare:
        print(f"\n{'='*140}")
        print(f"TIMESTAMP: {ts}")
        print(f"{'='*140}")
        
        prod_state = prod_data['bar_states'].get(ts)
        opt_state = opt_data['bar_states'].get(ts)
        
        # Show bar states
        if prod_state:
            print("\n🏭 PRODUCTION STATE:")
            params = prod_state['params']
            print(f"  Regime: {prod_state['regime']} (Params: {prod_state['param_source']})")
            print(f"  Parameters: short={params.get('short_window', 'N/A')}, long={params.get('long_window', 'N/A')}, rsi_period={params.get('rsi_indicator.period', 'N/A')}")
            print(f"  Price: {prod_state['price']:.4f}")
            
            ma_short_period = params.get('short_window', 'UNK')
            ma_long_period = params.get('long_window', 'UNK')
            rsi_period = params.get('rsi_indicator.period', 'UNK')
            
            print(f"  MA_S: {format_indicator_value_v3(prod_state['ma_short'], ma_short_period, prod_state['bars_processed']['ma_short'])}")
            print(f"  MA_L: {format_indicator_value_v3(prod_state['ma_long'], ma_long_period, prod_state['bars_processed']['ma_long'])}")
            print(f"  RSI:  {format_indicator_value_v3(prod_state['rsi'], rsi_period, prod_state['bars_processed']['rsi'], prod_state.get('rsi_calc'))}")
            
            # Check for signal
            prod_signal = next((s for s in prod_data['signals'] if s['timestamp'] == ts), None)
            if prod_signal:
                print(f"  📢 SIGNAL: Type={prod_signal['type']}, MA={prod_signal['ma_signal']}, RSI={prod_signal['rsi_signal']}, Combined={prod_signal['combined_strength']:.2f}")
        else:
            print("\n🏭 PRODUCTION STATE: No data")
        
        if opt_state:
            print("\n🔧 OPTIMIZER STATE:")
            params = opt_state['params']
            print(f"  Regime: {opt_state['regime']} (Params: {opt_state['param_source']})")
            print(f"  Parameters: short={params.get('short_window', 'N/A')}, long={params.get('long_window', 'N/A')}, rsi_period={params.get('rsi_indicator.period', 'N/A')}")
            print(f"  Price: {opt_state['price']:.4f}")
            
            ma_short_period = params.get('short_window', 'UNK')
            ma_long_period = params.get('long_window', 'UNK')
            rsi_period = params.get('rsi_indicator.period', 'UNK')
            
            print(f"  MA_S: {format_indicator_value_v3(opt_state['ma_short'], ma_short_period, opt_state['bars_processed']['ma_short'])}")
            print(f"  MA_L: {format_indicator_value_v3(opt_state['ma_long'], ma_long_period, opt_state['bars_processed']['ma_long'])}")
            print(f"  RSI:  {format_indicator_value_v3(opt_state['rsi'], rsi_period, opt_state['bars_processed']['rsi'], opt_state.get('rsi_calc'))}")
            
            # Check for signal
            opt_signal = next((s for s in opt_data['signals'] if s['timestamp'] == ts), None)
            if opt_signal:
                print(f"  📢 SIGNAL: Type={opt_signal['type']}, MA={opt_signal['ma_signal']}, RSI={opt_signal['rsi_signal']}, Combined={opt_signal['combined_strength']:.2f}")
        else:
            print("\n🔧 OPTIMIZER STATE: No data")
        
        # Show regime detector states
        prod_detector = prod_data['regime_detector_states'].get(ts)
        opt_detector = opt_data['regime_detector_states'].get(ts)
        
        if prod_detector or opt_detector:
            print("\n🔍 REGIME DETECTOR STATES:")
            if prod_detector:
                print(f"  Production Detector: trend={prod_detector.get('trend_strength', 'N/A')}, vol={prod_detector.get('volatility', 'N/A')}, rsi={prod_detector.get('rsi_level', 'N/A')} → {prod_detector.get('classified_regime', 'N/A')}")
                if 'raw_indicators' in prod_detector:
                    print(f"    Raw indicators: {prod_detector['raw_indicators']}")
            if opt_detector:
                print(f"  Optimizer Detector:  trend={opt_detector.get('trend_strength', 'N/A')}, vol={opt_detector.get('volatility', 'N/A')}, rsi={opt_detector.get('rsi_level', 'N/A')} → {opt_detector.get('classified_regime', 'N/A')}")
                if 'raw_indicators' in opt_detector:
                    print(f"    Raw indicators: {opt_detector['raw_indicators']}")
        
        # Highlight differences
        if prod_state and opt_state:
            print("\n⚠️  KEY DIFFERENCES:")
            diffs = []
            
            if prod_state['regime'] != opt_state['regime']:
                diffs.append(f"Regime: {prod_state['regime']} vs {opt_state['regime']}")
            
            if prod_state['param_source'] != opt_state['param_source']:
                diffs.append(f"Param Source: {prod_state['param_source']} vs {opt_state['param_source']}")
            
            # Check MA availability
            prod_ma_valid = prod_state['ma_short'] != 'N/A' and prod_state['ma_long'] != 'N/A'
            opt_ma_valid = opt_state['ma_short'] != 'N/A' and opt_state['ma_long'] != 'N/A'
            if prod_ma_valid != opt_ma_valid:
                diffs.append(f"MA Availability: Prod={'Valid' if prod_ma_valid else 'N/A'} vs Opt={'Valid' if opt_ma_valid else 'N/A'}")
            
            # Check RSI availability
            if (prod_state['rsi'] == 'N/A') != (opt_state['rsi'] == 'N/A'):
                diffs.append(f"RSI Availability: Prod={prod_state['rsi']} vs Opt={opt_state['rsi']}")
            
            for diff in diffs:
                print(f"  ❌ {diff}")

def main():
    print("Enhanced Signal Analysis v3 - Complete Warm-up and Lifecycle Tracking")
    
    # Extract comprehensive data
    production_file = "/Users/daws/ADMF/logs/admf_20250524_002523.log"
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"
    
    print(f"\nAnalyzing production: {production_file}")
    prod_data = extract_comprehensive_data_v3(production_file, "PRODUCTION")
    
    print(f"Analyzing optimizer: {optimizer_file}")
    opt_data = extract_comprehensive_data_v3(optimizer_file, "OPTIMIZER")
    
    # Summary
    print("\n" + "=" * 140)
    print("ANALYSIS SUMMARY")
    print("=" * 140)
    
    print(f"\n📅 TEST INFO:")
    print(f"  Production - First Bar: {prod_data['first_bar']}, Signals: {len(prod_data['signals'])}")
    print(f"  Optimizer  - First Bar: {opt_data['first_bar']}, Signals: {len(opt_data['signals'])}")
    
    print(f"\n📊 PARAMETER SOURCES:")
    print(f"  Production: {dict(prod_data['param_sources'])}")
    print(f"  Optimizer:  {dict(opt_data['param_sources'])}")
    
    # Analyze detector warm-up
    analyze_detector_warmup(prod_data, opt_data)
    
    # Analyze RSI lifecycle
    analyze_rsi_lifecycle(prod_data, opt_data)
    
    # Compare at key timestamps
    key_timestamps = [
        "2024-03-28 13:46:00+00:00",  # First bar
        "2024-03-28 14:06:00+00:00",  # First regime divergence
        "2024-03-28 14:11:00+00:00",  # Regime divergence
        "2024-03-28 14:12:00+00:00",  # First production signal
        "2024-03-28 14:27:00+00:00",  # Before RSI anomaly
        "2024-03-28 14:28:00+00:00",  # RSI valid
        "2024-03-28 14:29:00+00:00",  # RSI becomes N/A?
        "2024-03-28 14:30:00+00:00",  # Signal with N/A RSI
    ]
    
    compare_at_timestamps_v3(prod_data, opt_data, key_timestamps)

if __name__ == "__main__":
    main()