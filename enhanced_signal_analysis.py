#!/usr/bin/env python3
"""
Enhanced signal analysis with absolute bar indexing and warm-up tracking
"""
import re
from datetime import datetime
from collections import defaultdict

def extract_test_start_and_signals(log_file_path, source_name, max_signals=10):
    """Extract test start timestamp and signals with absolute indexing"""
    signals = []
    test_start_time = None
    first_bar_time = None
    regime_params = {}
    indicator_warm_ups = defaultdict(lambda: {'first_valid': None, 'bars_processed': 0})
    bar_index_map = {}  # Map timestamp to absolute index
    current_abs_index = 0
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Find test dataset start
            if 'Test dataset range:' in line:
                test_match = re.search(r'Test dataset range: ([0-9-]+\s[0-9:]+\+[0-9:]+) to ([0-9-]+\s[0-9:]+\+[0-9:]+)', line)
                if test_match:
                    test_start_time = test_match.group(1)
            
            # Track first bar processed
            if '📊 BAR_' in line and first_bar_time is None:
                bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', line)
                if bar_match:
                    first_bar_time = bar_match.group(1)
            
            # Map bar timestamps to absolute indices
            if '📊 BAR_' in line and 'INDICATORS:' in line:
                bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\]', line)
                if bar_match:
                    bar_timestamp = bar_match.group(2)
                    if bar_timestamp not in bar_index_map:
                        bar_index_map[bar_timestamp] = current_abs_index
                        current_abs_index += 1
            
            # Track regime parameter updates
            if 'REGIME PARAMETER UPDATE:' in line:
                param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", line)
                if param_match:
                    regime_params[param_match.group(1)] = eval(param_match.group(2))
            
            # Track indicator warm-ups (when RSI becomes valid)
            if 'RSI calculation:' in line and 'value=' in line:
                rsi_match = re.search(r'RSI calculation: period=(\d+), values_count=(\d+), value=([0-9.]+|N/A)', line)
                if rsi_match:
                    period = int(rsi_match.group(1))
                    values_count = int(rsi_match.group(2))
                    value = rsi_match.group(3)
                    
                    if value != 'N/A' and indicator_warm_ups['RSI']['first_valid'] is None:
                        # Find the timestamp for this RSI calculation
                        for prev_line_num in range(line_num-5, line_num):
                            if prev_line_num >= 0 and '📊 BAR_' in lines[prev_line_num]:
                                bar_match = re.search(r'📊 BAR_\d+ \[([^\]]+)\]', lines[prev_line_num])
                                if bar_match:
                                    indicator_warm_ups['RSI']['first_valid'] = bar_match.group(1)
                                    indicator_warm_ups['RSI']['bars_processed'] = values_count
                                    indicator_warm_ups['RSI']['period'] = period
                                    break
            
            # Find signal generation with enhanced context
            if '🚨 SIGNAL GENERATED' in line:
                signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\), Combined_strength=([+-]?[0-9.]+), Final_multiplier=([0-9.]+)', line)
                if signal_match:
                    signal_data = {
                        'source': source_name,
                        'line_num': line_num,
                        'signal_num': signal_match.group(1),
                        'signal_type': int(signal_match.group(2)),
                        'price': float(signal_match.group(3)),
                        'regime': signal_match.group(4),
                        'ma_signal': int(signal_match.group(5)),
                        'ma_weight': float(signal_match.group(6)),
                        'rsi_signal': int(signal_match.group(7)),
                        'rsi_weight': float(signal_match.group(8)),
                        'combined_strength': float(signal_match.group(9)),
                        'final_multiplier': float(signal_match.group(10)),
                        'context_window': [],
                        'full_line': line.strip()
                    }
                    
                    # Get current regime parameters
                    if signal_data['regime'] in regime_params:
                        signal_data['regime_params'] = regime_params[signal_data['regime']]
                    
                    # Collect extended context window (10 bars before, 2 after)
                    start_line = max(0, line_num - 50)  # Look further back
                    end_line = min(len(lines), line_num + 10)
                    
                    bar_contexts = []
                    for ctx_line_num in range(start_line, end_line):
                        ctx_line = lines[ctx_line_num].strip()
                        
                        if '📊 BAR_' in ctx_line and 'INDICATORS:' in ctx_line:
                            bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=([^,]+)', ctx_line)
                            if bar_match:
                                bar_timestamp = bar_match.group(2)
                                abs_index = bar_index_map.get(bar_timestamp, -1)
                                
                                bar_contexts.append({
                                    'relative_bar': int(bar_match.group(1)),
                                    'abs_index': abs_index,
                                    'timestamp': bar_timestamp,
                                    'price': float(bar_match.group(3)),
                                    'ma_short': bar_match.group(4),
                                    'ma_long': bar_match.group(5),
                                    'rsi': bar_match.group(6),
                                    'rsi_thresholds': bar_match.group(7),
                                    'regime': bar_match.group(8),
                                    'line_offset': ctx_line_num - line_num
                                })
                    
                    # Get the 10 most recent bars before signal
                    signal_data['context_window'] = [bc for bc in bar_contexts if bc['line_offset'] < 0][-10:]
                    
                    signals.append(signal_data)
                    
                    if len(signals) >= max_signals:
                        break
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None, None, [], {}
    
    return test_start_time, first_bar_time, signals, indicator_warm_ups

def compare_enhanced_signals(prod_data, opt_data):
    """Compare signals with enhanced context"""
    test_start_prod, first_bar_prod, prod_signals, prod_warm_ups = prod_data
    test_start_opt, first_bar_opt, opt_signals, opt_warm_ups = opt_data
    
    print("=" * 140)
    print("ENHANCED SIGNAL GENERATION ANALYSIS WITH ABSOLUTE INDEXING")
    print("=" * 140)
    
    print(f"\n📅 TEST DATASET INFO:")
    print(f"  Production Test Start: {test_start_prod}")
    print(f"  Production First Bar:  {first_bar_prod}")
    print(f"  Optimizer Test Start:  {test_start_opt}")
    print(f"  Optimizer First Bar:   {first_bar_opt}")
    
    print(f"\n🔥 INDICATOR WARM-UP TRACKING:")
    print(f"  Production RSI First Valid: {prod_warm_ups['RSI']['first_valid']} (after {prod_warm_ups['RSI']['bars_processed']} bars, period={prod_warm_ups['RSI'].get('period', 'N/A')})")
    print(f"  Optimizer RSI First Valid:  {opt_warm_ups['RSI']['first_valid']} (after {opt_warm_ups['RSI']['bars_processed']} bars, period={opt_warm_ups['RSI'].get('period', 'N/A')})")
    
    print(f"\n📊 SIGNAL COUNTS:")
    print(f"  Production: {len(prod_signals)} signals")
    print(f"  Optimizer:  {len(opt_signals)} signals")
    
    # Compare first few signals in detail
    for i in range(min(3, len(prod_signals), len(opt_signals))):
        print(f"\n{'='*140}")
        print(f"SIGNAL {i+1} DETAILED COMPARISON WITH ABSOLUTE INDEXING")
        print(f"{'='*140}")
        
        prod_signal = prod_signals[i] if i < len(prod_signals) else None
        opt_signal = opt_signals[i] if i < len(opt_signals) else None
        
        if prod_signal and opt_signal:
            # Show side-by-side comparison of context windows
            print(f"\n📊 CONTEXT WINDOW COMPARISON (10 bars before signal):")
            print(f"{'─'*70} PRODUCTION {'─'*58} OPTIMIZER {'─'*50}")
            print(f"ABS  REL  TIMESTAMP            PRICE     MA_S      MA_L      RSI      │ ABS  REL  TIMESTAMP            PRICE     MA_S      MA_L      RSI")
            print(f"{'─'*140}")
            
            # Find common timestamps
            prod_ctx_by_time = {ctx['timestamp']: ctx for ctx in prod_signal['context_window']}
            opt_ctx_by_time = {ctx['timestamp']: ctx for ctx in opt_signal['context_window']}
            
            all_timestamps = sorted(set(prod_ctx_by_time.keys()) | set(opt_ctx_by_time.keys()))
            
            for ts in all_timestamps[-10:]:  # Last 10 bars
                prod_ctx = prod_ctx_by_time.get(ts)
                opt_ctx = opt_ctx_by_time.get(ts)
                
                # Production side
                if prod_ctx:
                    print(f"{prod_ctx['abs_index']:>3}  {prod_ctx['relative_bar']:>3}  {ts[:19]}  {prod_ctx['price']:>8.2f}  {prod_ctx['ma_short']:>8}  {prod_ctx['ma_long']:>8}  {prod_ctx['rsi']:>7}", end='')
                else:
                    print(f"{'---':>3}  {'---':>3}  {'---':^19}  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>7}", end='')
                
                print(" │", end='')
                
                # Optimizer side
                if opt_ctx:
                    print(f" {opt_ctx['abs_index']:>3}  {opt_ctx['relative_bar']:>3}  {ts[:19]}  {opt_ctx['price']:>8.2f}  {opt_ctx['ma_short']:>8}  {opt_ctx['ma_long']:>8}  {opt_ctx['rsi']:>7}")
                else:
                    print(f" {'---':>3}  {'---':>3}  {'---':^19}  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>7}")
            
            # Show signal details
            print(f"\n🚨 SIGNAL DETAILS:")
            print(f"  Production: Type={prod_signal['signal_type']}, Price={prod_signal['price']:.2f}, Regime={prod_signal['regime']}")
            if 'regime_params' in prod_signal:
                params = prod_signal['regime_params']
                rsi_period = params.get('rsi_indicator.period', 'N/A')
                print(f"              RSI Period={rsi_period}, MA Windows=({params.get('short_window', 'N/A')}, {params.get('long_window', 'N/A')})")
            print(f"              MA={prod_signal['ma_signal']}(w={prod_signal['ma_weight']}), RSI={prod_signal['rsi_signal']}(w={prod_signal['rsi_weight']})")
            
            print(f"\n  Optimizer:  Type={opt_signal['signal_type']}, Price={opt_signal['price']:.2f}, Regime={opt_signal['regime']}")
            if 'regime_params' in opt_signal:
                params = opt_signal['regime_params']
                rsi_period = params.get('rsi_indicator.period', 'N/A')
                print(f"              RSI Period={rsi_period}, MA Windows=({params.get('short_window', 'N/A')}, {params.get('long_window', 'N/A')})")
            print(f"              MA={opt_signal['ma_signal']}(w={opt_signal['ma_weight']}), RSI={opt_signal['rsi_signal']}(w={opt_signal['rsi_weight']})")
            
            # Highlight key differences
            print(f"\n  🔍 KEY DIFFERENCES:")
            differences = []
            
            if prod_signal['signal_type'] != opt_signal['signal_type']:
                differences.append(f"Signal Type: {prod_signal['signal_type']} vs {opt_signal['signal_type']}")
            
            if prod_signal['regime'] != opt_signal['regime']:
                differences.append(f"Regime: {prod_signal['regime']} vs {opt_signal['regime']}")
            
            # Check if RSI was N/A in one but not the other
            prod_has_valid_rsi = any(ctx['rsi'] != 'N/A' for ctx in prod_signal['context_window'])
            opt_has_valid_rsi = any(ctx['rsi'] != 'N/A' for ctx in opt_signal['context_window'])
            
            if prod_has_valid_rsi != opt_has_valid_rsi:
                differences.append(f"RSI Availability: Production={'Valid' if prod_has_valid_rsi else 'N/A'} vs Optimizer={'Valid' if opt_has_valid_rsi else 'N/A'}")
            
            for diff in differences:
                print(f"     ❌ {diff}")

def main():
    print("Enhanced Signal Analysis with Absolute Bar Indexing and Warm-up Tracking")
    
    # Extract data from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nAnalyzing production: {production_file}")
    prod_data = extract_test_start_and_signals(production_file, "PRODUCTION", max_signals=10)
    
    print(f"Analyzing optimizer: {optimizer_file}")
    opt_data = extract_test_start_and_signals(optimizer_file, "OPTIMIZER", max_signals=10)
    
    # Compare the signals
    compare_enhanced_signals(prod_data, opt_data)

if __name__ == "__main__":
    main()