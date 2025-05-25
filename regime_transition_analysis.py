#!/usr/bin/env python3
"""
Detailed analysis of indicators and bars around regime transitions
"""
import re
from datetime import datetime

def extract_regime_transition_context(log_file_path, source_name):
    """Extract regime changes and surrounding bar/indicator data"""
    events = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            # Find regime changes
            if 'REGIME CHANGED:' in line:
                regime_match = re.search(r"REGIME CHANGED: '([^']+)' → '([^']+)' at ([0-9-]+\s[0-9:]+\+[0-9:]+)", line)
                if regime_match:
                    from_regime = regime_match.group(1)
                    to_regime = regime_match.group(2)
                    timestamp = regime_match.group(3)
                    
                    # Collect context around this regime change
                    context_before = []
                    context_after = []
                    
                    # Look 10 lines before and after for relevant events
                    start_line = max(0, line_num - 10)
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
                                    'timestamp': bar_match.group(2),
                                    'price': bar_match.group(3),
                                    'ma_short': bar_match.group(4),
                                    'ma_long': bar_match.group(5),
                                    'rsi': bar_match.group(6),
                                    'rsi_thresholds': bar_match.group(7),
                                    'regime': bar_match.group(8),
                                    'ma_weight': bar_match.group(9),
                                    'rsi_weight': bar_match.group(10)
                                }
                        
                        elif '🚨 SIGNAL GENERATED' in ctx_line:
                            signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+), MA_signal=([+-]?\d+)\(w=([0-9.]+)\), RSI_signal=([+-]?\d+)\(w=([0-9.]+)\)', ctx_line)
                            if signal_match:
                                event_info = {
                                    'type': 'SIGNAL',
                                    'signal_num': signal_match.group(1),
                                    'signal_type': signal_match.group(2),
                                    'price': signal_match.group(3),
                                    'regime': signal_match.group(4),
                                    'ma_signal': signal_match.group(5),
                                    'ma_weight': signal_match.group(6),
                                    'rsi_signal': signal_match.group(7),
                                    'rsi_weight': signal_match.group(8)
                                }
                        
                        elif 'REGIME PARAMETER UPDATE:' in ctx_line:
                            param_match = re.search(r"REGIME PARAMETER UPDATE: '([^']+)' applying: ({[^}]+})", ctx_line)
                            if param_match:
                                event_info = {
                                    'type': 'PARAM_UPDATE',
                                    'regime': param_match.group(1),
                                    'params': param_match.group(2)
                                }
                        
                        if event_info:
                            event_info['line_num'] = ctx_line_num
                            event_info['full_line'] = ctx_line
                            
                            if ctx_line_num < line_num:
                                context_before.append(event_info)
                            elif ctx_line_num > line_num:
                                context_after.append(event_info)
                    
                    events.append({
                        'source': source_name,
                        'line_num': line_num,
                        'from_regime': from_regime,
                        'to_regime': to_regime,
                        'timestamp': timestamp,
                        'context_before': context_before,
                        'context_after': context_after
                    })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return events

def analyze_regime_transitions(production_transitions, optimizer_transitions):
    """Analyze regime transitions and their context"""
    
    print("=" * 120)
    print("REGIME TRANSITION CONTEXT ANALYSIS")
    print("=" * 120)
    
    # Focus on the first few transitions where differences occur
    print(f"\nProduction Transitions: {len(production_transitions)}")
    print(f"Optimizer Transitions:  {len(optimizer_transitions)}")
    
    # Create timestamp-based mapping
    prod_by_timestamp = {trans['timestamp']: trans for trans in production_transitions}
    opt_by_timestamp = {trans['timestamp']: trans for trans in optimizer_transitions}
    
    # Get all unique timestamps and sort them
    all_timestamps = sorted(set(prod_by_timestamp.keys()) | set(opt_by_timestamp.keys()))
    
    # Analyze first 5 transitions by timestamp alignment
    print(f"\n🔍 DETAILED ANALYSIS BY TIMESTAMP ALIGNMENT")
    print("=" * 120)
    
    for i, timestamp in enumerate(all_timestamps[:5]):
        print(f"\n--- TIMESTAMP: {timestamp} ---")
        
        prod_trans = prod_by_timestamp.get(timestamp)
        opt_trans = opt_by_timestamp.get(timestamp)
        
        if prod_trans:
            print(f"\n✅ PRODUCTION: {prod_trans['from_regime']} → {prod_trans['to_regime']}")
            print("Context BEFORE transition:")
            for event in prod_trans['context_before'][-3:]:  # Last 3 events before
                if event['type'] == 'BAR_INDICATORS':
                    print(f"  📊 BAR {event['bar_num']}: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'SIGNAL':
                    print(f"  🚨 SIGNAL #{event['signal_num']}: Type={event['signal_type']}, Price={event['price']}, Regime={event['regime']}, MA={event['ma_signal']}(w={event['ma_weight']}), RSI={event['rsi_signal']}(w={event['rsi_weight']})")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"  🔄 PARAM UPDATE: {event['regime']} - {event['params']}")
            
            print("Context AFTER transition:")
            for event in prod_trans['context_after'][:3]:  # First 3 events after
                if event['type'] == 'BAR_INDICATORS':
                    print(f"  📊 BAR {event['bar_num']}: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'SIGNAL':
                    print(f"  🚨 SIGNAL #{event['signal_num']}: Type={event['signal_type']}, Price={event['price']}, Regime={event['regime']}, MA={event['ma_signal']}(w={event['ma_weight']}), RSI={event['rsi_signal']}(w={event['rsi_weight']})")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"  🔄 PARAM UPDATE: {event['regime']} - {event['params']}")
        else:
            print(f"\n❌ PRODUCTION: No transition at this timestamp")
        
        if opt_trans:
            print(f"\n✅ OPTIMIZER: {opt_trans['from_regime']} → {opt_trans['to_regime']}")
            print("Context BEFORE transition:")
            for event in opt_trans['context_before'][-3:]:  # Last 3 events before
                if event['type'] == 'BAR_INDICATORS':
                    print(f"  📊 BAR {event['bar_num']}: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'SIGNAL':
                    print(f"  🚨 SIGNAL #{event['signal_num']}: Type={event['signal_type']}, Price={event['price']}, Regime={event['regime']}, MA={event['ma_signal']}(w={event['ma_weight']}), RSI={event['rsi_signal']}(w={event['rsi_weight']})")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"  🔄 PARAM UPDATE: {event['regime']} - {event['params']}")
            
            print("Context AFTER transition:")
            for event in opt_trans['context_after'][:3]:  # First 3 events after
                if event['type'] == 'BAR_INDICATORS':
                    print(f"  📊 BAR {event['bar_num']}: Price={event['price']}, MA_short={event['ma_short']}, MA_long={event['ma_long']}, RSI={event['rsi']}, Regime={event['regime']}")
                elif event['type'] == 'SIGNAL':
                    print(f"  🚨 SIGNAL #{event['signal_num']}: Type={event['signal_type']}, Price={event['price']}, Regime={event['regime']}, MA={event['ma_signal']}(w={event['ma_weight']}), RSI={event['rsi_signal']}(w={event['rsi_weight']})")
                elif event['type'] == 'PARAM_UPDATE':
                    print(f"  🔄 PARAM UPDATE: {event['regime']} - {event['params']}")
        else:
            print(f"\n❌ OPTIMIZER: No transition at this timestamp")
        
        print("-" * 120)

def main():
    print("Regime Transition Context Analysis")
    
    # Extract regime transition context from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nAnalyzing production transitions: {production_file}")
    production_transitions = extract_regime_transition_context(production_file, "PRODUCTION")
    
    print(f"Analyzing optimizer transitions: {optimizer_file}")
    optimizer_transitions = extract_regime_transition_context(optimizer_file, "OPTIMIZER")
    
    # Analyze the transitions
    analyze_regime_transitions(production_transitions, optimizer_transitions)

if __name__ == "__main__":
    main()