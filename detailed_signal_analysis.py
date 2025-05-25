#!/usr/bin/env python3
"""
Detailed signal analysis showing exact context around signal generation
"""
import re
from datetime import datetime

def extract_signal_context(log_file_path, source_name, max_signals=10):
    """Extract signals and their immediate context"""
    signals = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Find signal generation
            if '🚨 SIGNAL GENERATED' in line:
                signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\), Combined_strength=([+-]?[0-9.]+), Final_multiplier=([0-9.]+)', line)
                if signal_match:
                    signal_num = signal_match.group(1)
                    signal_type = int(signal_match.group(2))
                    price = float(signal_match.group(3))
                    regime = signal_match.group(4)
                    ma_signal = int(signal_match.group(5))
                    ma_weight = float(signal_match.group(6))
                    rsi_signal = int(signal_match.group(7))
                    rsi_weight = float(signal_match.group(8))
                    combined_strength = float(signal_match.group(9))
                    final_multiplier = float(signal_match.group(10))
                    
                    # Collect context around this signal
                    context_before = []
                    context_after = []
                    
                    # Look 20 lines before and 10 after for relevant events
                    start_line = max(0, line_num - 20)
                    end_line = min(len(lines), line_num + 10)
                    
                    for ctx_line_num in range(start_line, end_line):
                        ctx_line = lines[ctx_line_num].strip()
                        
                        # Extract relevant events
                        event_info = None
                        
                        if '📊 BAR_' in ctx_line and 'INDICATORS:' in ctx_line:
                            # Parse bar indicators
                            bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=([^,]+), Weights=\(MA:([^,]+),RSI:([^)]+)\)', ctx_line)
                            if bar_match:
                                event_info = {
                                    'type': 'BAR_INDICATORS',
                                    'bar_num': bar_match.group(1),
                                    'bar_timestamp': bar_match.group(2),
                                    'price': bar_match.group(3),
                                    'ma_short': bar_match.group(4),
                                    'ma_long': bar_match.group(5),
                                    'rsi': bar_match.group(6),
                                    'rsi_thresholds': bar_match.group(7),
                                    'regime': bar_match.group(8),
                                    'ma_weight': bar_match.group(9),
                                    'rsi_weight': bar_match.group(10)
                                }
                        
                        elif 'REGIME CHANGED:' in ctx_line:
                            regime_match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", ctx_line)
                            if regime_match:
                                event_info = {
                                    'type': 'REGIME_CHANGE',
                                    'from_regime': regime_match.group(1),
                                    'to_regime': regime_match.group(2),
                                    'timestamp': regime_match.group(3)
                                }
                        
                        elif 'REGIME PARAMETER UPDATE:' in ctx_line:
                            param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", ctx_line)
                            if param_match:
                                event_info = {
                                    'type': 'PARAM_UPDATE',
                                    'regime': param_match.group(1),
                                    'params': param_match.group(2)
                                }
                        
                        elif 'Current weights for voting:' in ctx_line:
                            weight_match = re.search(r'Current weights for voting: MA=([0-9.]+), RSI=([0-9.]+)', ctx_line)
                            if weight_match:
                                event_info = {
                                    'type': 'VOTING_WEIGHTS',
                                    'ma_weight': weight_match.group(1),
                                    'rsi_weight': weight_match.group(2)
                                }
                        
                        elif 'Signal inputs' in ctx_line:
                            input_match = re.search(r'Signal inputs - MA: ([+-]?\d+) \(weighted: ([0-9.]+)\), RSI: ([+-]?\d+) \(weighted: ([0-9.]+)\), Combined strength: ([+-]?[0-9.]+), Final multiplier: ([0-9.]+)', ctx_line)
                            if input_match:
                                event_info = {
                                    'type': 'SIGNAL_INPUTS',
                                    'ma_signal': input_match.group(1),
                                    'ma_weighted': input_match.group(2),
                                    'rsi_signal': input_match.group(3),
                                    'rsi_weighted': input_match.group(4),
                                    'combined_strength': input_match.group(5),
                                    'final_multiplier': input_match.group(6)
                                }
                        
                        if event_info:
                            event_info['line_num'] = ctx_line_num
                            event_info['full_line'] = ctx_line
                            
                            if ctx_line_num < line_num:
                                context_before.append(event_info)
                            elif ctx_line_num > line_num:
                                context_after.append(event_info)
                    
                    signals.append({
                        'source': source_name,
                        'line_num': line_num,
                        'signal_num': signal_num,
                        'signal_type': signal_type,
                        'price': price,
                        'regime': regime,
                        'ma_signal': ma_signal,
                        'ma_weight': ma_weight,
                        'rsi_signal': rsi_signal,
                        'rsi_weight': rsi_weight,
                        'combined_strength': combined_strength,
                        'final_multiplier': final_multiplier,
                        'context_before': context_before,
                        'context_after': context_after,
                        'full_line': line.strip()
                    })
                
                # Limit signals to prevent overflow
                if len(signals) >= max_signals:
                    break
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return signals

def compare_signal_generation(production_signals, optimizer_signals, max_compare=5):
    """Compare detailed signal generation between production and optimizer"""
    
    print("=" * 140)
    print("DETAILED SIGNAL GENERATION ANALYSIS")
    print("=" * 140)
    
    print(f"\nSignal Counts:")
    print(f"  Production: {len(production_signals)} signals")
    print(f"  Optimizer:  {len(optimizer_signals)} signals")
    
    # Compare first few signals in detail
    for i in range(min(max_compare, len(production_signals), len(optimizer_signals))):
        print(f"\n{'='*140}")
        print(f"SIGNAL {i+1} DETAILED COMPARISON")
        print(f"{'='*140}")
        
        prod_signal = production_signals[i] if i < len(production_signals) else None
        opt_signal = optimizer_signals[i] if i < len(optimizer_signals) else None
        
        if prod_signal:
            print(f"\n🏭 PRODUCTION SIGNAL #{prod_signal['signal_num']}:")
            print(f"   Type={prod_signal['signal_type']}, Price={prod_signal['price']}, Regime={prod_signal['regime']}")
            print(f"   MA_signal={prod_signal['ma_signal']}(w={prod_signal['ma_weight']}), RSI_signal={prod_signal['rsi_signal']}(w={prod_signal['rsi_weight']})")
            print(f"   Combined_strength={prod_signal['combined_strength']}, Final_multiplier={prod_signal['final_multiplier']}")
            
            print(f"\n   Context BEFORE signal:")
            for event in prod_signal['context_before'][-5:]:  # Last 5 events before
                if event['type'] == 'BAR_INDICATORS':
                    print(f"     📊 BAR {event['bar_num']} [{event['bar_timestamp']}]: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'REGIME_CHANGE':
                    print(f"     🔄 REGIME: {event['from_regime']} → {event['to_regime']} at {event['timestamp']}")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"     ⚙️  PARAMS: {event['regime']} - {event['params']}")
                elif event['type'] == 'VOTING_WEIGHTS':
                    print(f"     ⚖️  WEIGHTS: MA={event['ma_weight']}, RSI={event['rsi_weight']}")
                elif event['type'] == 'SIGNAL_INPUTS':
                    print(f"     🎯 INPUTS: MA={event['ma_signal']}(w={event['ma_weighted']}), RSI={event['rsi_signal']}(w={event['rsi_weighted']}), strength={event['combined_strength']}, multiplier={event['final_multiplier']}")
        
        if opt_signal:
            print(f"\n🔧 OPTIMIZER SIGNAL #{opt_signal['signal_num']}:")
            print(f"   Type={opt_signal['signal_type']}, Price={opt_signal['price']}, Regime={opt_signal['regime']}")
            print(f"   MA_signal={opt_signal['ma_signal']}(w={opt_signal['ma_weight']}), RSI_signal={opt_signal['rsi_signal']}(w={opt_signal['rsi_weight']})")
            print(f"   Combined_strength={opt_signal['combined_strength']}, Final_multiplier={opt_signal['final_multiplier']}")
            
            print(f"\n   Context BEFORE signal:")
            for event in opt_signal['context_before'][-5:]:  # Last 5 events before
                if event['type'] == 'BAR_INDICATORS':
                    print(f"     📊 BAR {event['bar_num']} [{event['bar_timestamp']}]: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'REGIME_CHANGE':
                    print(f"     🔄 REGIME: {event['from_regime']} → {event['to_regime']} at {event['timestamp']}")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"     ⚙️  PARAMS: {event['regime']} - {event['params']}")
                elif event['type'] == 'VOTING_WEIGHTS':
                    print(f"     ⚖️  WEIGHTS: MA={event['ma_weight']}, RSI={event['rsi_weight']}")
                elif event['type'] == 'SIGNAL_INPUTS':
                    print(f"     🎯 INPUTS: MA={event['ma_signal']}(w={event['ma_weighted']}), RSI={event['rsi_signal']}(w={event['rsi_weighted']}), strength={event['combined_strength']}, multiplier={event['final_multiplier']}")
        
        # Compare key differences
        if prod_signal and opt_signal:
            print(f"\n   🔍 COMPARISON:")
            if prod_signal['signal_type'] != opt_signal['signal_type']:
                print(f"     ❌ Signal Type: {prod_signal['signal_type']} vs {opt_signal['signal_type']}")
            if abs(prod_signal['price'] - opt_signal['price']) > 0.01:
                print(f"     ❌ Price: {prod_signal['price']} vs {opt_signal['price']}")
            if prod_signal['regime'] != opt_signal['regime']:
                print(f"     ❌ Regime: {prod_signal['regime']} vs {opt_signal['regime']}")
            if prod_signal['ma_signal'] != opt_signal['ma_signal']:
                print(f"     ❌ MA Signal: {prod_signal['ma_signal']} vs {opt_signal['ma_signal']}")
            if prod_signal['rsi_signal'] != opt_signal['rsi_signal']:
                print(f"     ❌ RSI Signal: {prod_signal['rsi_signal']} vs {opt_signal['rsi_signal']}")
            if abs(prod_signal['combined_strength'] - opt_signal['combined_strength']) > 0.001:
                print(f"     ❌ Combined Strength: {prod_signal['combined_strength']} vs {opt_signal['combined_strength']}")

def main():
    print("Detailed Signal Generation Analysis")
    
    # Extract signal context from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nAnalyzing production signals: {production_file}")
    production_signals = extract_signal_context(production_file, "PRODUCTION", max_signals=10)
    
    print(f"Analyzing optimizer signals: {optimizer_file}")
    optimizer_signals = extract_signal_context(optimizer_file, "OPTIMIZER", max_signals=10)
    
    # Compare the signal generation
    compare_signal_generation(production_signals, optimizer_signals, max_compare=3)

if __name__ == "__main__":
    main()