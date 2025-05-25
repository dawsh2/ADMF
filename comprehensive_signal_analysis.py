#!/usr/bin/env python3
"""
Comprehensive analysis of signal discrepancies between production and optimizer runs.
"""

import re
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

class BarData:
    def __init__(self):
        self.timestamp = None
        self.bar_num = None
        self.price = None
        self.ma_short = None
        self.ma_long = None
        self.rsi = None
        self.regime = None
        self.weights = {}
        self.signal = None
        self.signal_type = None
        
    def __repr__(self):
        return f"Bar({self.timestamp}, #{self.bar_num}, P={self.price}, MA_S={self.ma_short}, MA_L={self.ma_long}, Regime={self.regime}, Signal={self.signal})"

def parse_log_file(filepath):
    """Parse log file and extract comprehensive data."""
    bars_by_time = OrderedDict()
    signals = []
    regime_changes = []
    current_bar = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Parse BAR indicators
        bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([\d.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\([^)]+\), Regime=(\w+), Weights=\(MA:([\d.]+),RSI:([\d.]+)\)', line)
        if bar_match:
            bar = BarData()
            bar.bar_num = int(bar_match.group(1))
            
            timestamp_str = bar_match.group(2)
            if '+' in timestamp_str:
                timestamp_str = timestamp_str.split('+')[0]
            bar.timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            bar.price = float(bar_match.group(3))
            
            # Parse MA values
            ma_short_str = bar_match.group(4)
            ma_long_str = bar_match.group(5)
            bar.ma_short = None if ma_short_str == 'N/A' else float(ma_short_str)
            bar.ma_long = None if ma_long_str == 'N/A' else float(ma_long_str)
            
            bar.rsi = bar_match.group(6)
            bar.regime = bar_match.group(7)
            bar.weights['MA'] = float(bar_match.group(8))
            bar.weights['RSI'] = float(bar_match.group(9))
            
            current_bar = bar
            bars_by_time[bar.timestamp] = bar
        
        # Parse signals
        signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=(-?\d+), Price=([\d.]+), Regime=(\w+)', line)
        if signal_match and current_bar and current_bar.timestamp.year == 2024:
            signal_data = {
                'timestamp': current_bar.timestamp,
                'bar_num': int(signal_match.group(1)),
                'signal_type': int(signal_match.group(2)),
                'price': float(signal_match.group(3)),
                'regime': signal_match.group(4),
                'bar_data': current_bar
            }
            signals.append(signal_data)
            current_bar.signal = signal_data['signal_type']
            current_bar.signal_type = 'BUY' if signal_data['signal_type'] == 1 else 'SELL'
        
        # Parse regime changes
        regime_match = re.search(r"REGIME CHANGED: '(\w+)' → '(\w+)' at ([^']+)", line)
        if regime_match:
            regime_changes.append({
                'from_regime': regime_match.group(1),
                'to_regime': regime_match.group(2),
                'timestamp_str': regime_match.group(3)
            })
    
    return bars_by_time, signals, regime_changes

def get_context_bars(bars_by_time, target_time, before=5, after=2):
    """Get bars before and after a target timestamp."""
    all_times = list(bars_by_time.keys())
    context_bars = []
    
    # Find nearby bars
    for t in all_times:
        time_diff = (t - target_time).total_seconds() / 60  # Convert to minutes
        if -before * 60 <= time_diff <= after * 60:  # Assuming 1-minute bars
            bar = bars_by_time[t]
            bar.relative_time = int(time_diff)
            context_bars.append(bar)
    
    return sorted(context_bars, key=lambda x: x.timestamp)

def analyze_mismatched_signals():
    """Analyze the specific mismatched signals."""
    # Parse both logs
    print("Parsing production log...")
    prod_bars, prod_signals, prod_regimes = parse_log_file('logs/admf_20250524_210903.log')
    
    print("Parsing optimizer log...")
    opt_bars, opt_signals, opt_regimes = parse_log_file('logs/admf_20250523_230532.log')
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SIGNAL DISCREPANCY ANALYSIS")
    print("="*80)
    
    # Extract signal timestamps
    prod_signal_times = {s['timestamp'] for s in prod_signals}
    opt_signal_times = {s['timestamp'] for s in opt_signals}
    
    # Find mismatched timestamps
    prod_only = prod_signal_times - opt_signal_times
    opt_only = opt_signal_times - prod_signal_times
    matched = prod_signal_times & opt_signal_times
    
    print(f"\nSignal Summary:")
    print(f"  Production signals: {len(prod_signals)}")
    print(f"  Optimizer signals: {len(opt_signals)}")
    print(f"  Matched signals: {len(matched)}")
    print(f"  Production-only: {len(prod_only)}")
    print(f"  Optimizer-only: {len(opt_only)}")
    
    # Analyze each mismatch
    print("\n" + "="*80)
    print("DETAILED MISMATCH ANALYSIS")
    print("="*80)
    
    # 1. Production-only signal at 13:59
    print("\n1. PRODUCTION-ONLY SIGNAL at 2024-03-28 13:59:00")
    print("-"*60)
    target_time = datetime(2024, 3, 28, 13, 59, 0)
    
    print("\nProduction context:")
    prod_context = get_context_bars(prod_bars, target_time)
    for bar in prod_context:
        prefix = ">>>" if bar.timestamp == target_time else "   "
        ma_signal = "SELL" if bar.ma_short and bar.ma_long and bar.ma_short < bar.ma_long else "BUY" if bar.ma_short and bar.ma_long and bar.ma_short > bar.ma_long else "N/A"
        ma_s_str = f"{bar.ma_short:.2f}" if bar.ma_short else "N/A"
        ma_l_str = f"{bar.ma_long:.2f}" if bar.ma_long else "N/A"
        print(f"{prefix} {bar.timestamp} | Bar#{bar.bar_num:3d} | P={bar.price:.2f} | MA_S={ma_s_str:>7s} | MA_L={ma_l_str:>7s} | MA_Sig={ma_signal:4s} | Regime={bar.regime:20s} | Signal={bar.signal_type or 'None'}")
    
    print("\nOptimizer context:")
    opt_context = get_context_bars(opt_bars, target_time)
    for bar in opt_context:
        prefix = ">>>" if bar.timestamp == target_time else "   "
        ma_signal = "SELL" if bar.ma_short and bar.ma_long and bar.ma_short < bar.ma_long else "BUY" if bar.ma_short and bar.ma_long and bar.ma_short > bar.ma_long else "N/A"
        ma_s_str = f"{bar.ma_short:.2f}" if bar.ma_short else "N/A"
        ma_l_str = f"{bar.ma_long:.2f}" if bar.ma_long else "N/A"
        print(f"{prefix} {bar.timestamp} | Bar#{bar.bar_num:3d} | P={bar.price:.2f} | MA_S={ma_s_str:>7s} | MA_L={ma_l_str:>7s} | MA_Sig={ma_signal:4s} | Regime={bar.regime:20s} | Signal={bar.signal_type or 'None'}")
    
    # 2. Optimizer-only signals
    for opt_time in sorted(opt_only):
        print(f"\n2. OPTIMIZER-ONLY SIGNAL at {opt_time}")
        print("-"*60)
        
        print("\nProduction context:")
        prod_context = get_context_bars(prod_bars, opt_time)
        for bar in prod_context:
            prefix = ">>>" if bar.timestamp == opt_time else "   "
            ma_signal = "SELL" if bar.ma_short and bar.ma_long and bar.ma_short < bar.ma_long else "BUY" if bar.ma_short and bar.ma_long and bar.ma_short > bar.ma_long else "N/A"
            ma_s_str = f"{bar.ma_short:.2f}" if bar.ma_short else "N/A"
            ma_l_str = f"{bar.ma_long:.2f}" if bar.ma_long else "N/A"
            print(f"{prefix} {bar.timestamp} | Bar#{bar.bar_num:3d} | P={bar.price:.2f} | MA_S={ma_s_str:>7s} | MA_L={ma_l_str:>7s} | MA_Sig={ma_signal:4s} | Regime={bar.regime:20s} | Signal={bar.signal_type or 'None'}")
        
        print("\nOptimizer context:")
        opt_context = get_context_bars(opt_bars, opt_time)
        for bar in opt_context:
            prefix = ">>>" if bar.timestamp == opt_time else "   "
            ma_signal = "SELL" if bar.ma_short and bar.ma_long and bar.ma_short < bar.ma_long else "BUY" if bar.ma_short and bar.ma_long and bar.ma_short > bar.ma_long else "N/A"
            ma_s_str = f"{bar.ma_short:.2f}" if bar.ma_short else "N/A"
            ma_l_str = f"{bar.ma_long:.2f}" if bar.ma_long else "N/A"
            print(f"{prefix} {bar.timestamp} | Bar#{bar.bar_num:3d} | P={bar.price:.2f} | MA_S={ma_s_str:>7s} | MA_L={ma_l_str:>7s} | MA_Sig={ma_signal:4s} | Regime={bar.regime:20s} | Signal={bar.signal_type or 'None'}")
    
    # Analyze patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Check if production starts later
    prod_first_bar = min(prod_bars.keys()) if prod_bars else None
    opt_first_bar = min(opt_bars.keys()) if opt_bars else None
    
    print(f"\nData alignment:")
    print(f"  Production first bar: {prod_first_bar}")
    print(f"  Optimizer first bar: {opt_first_bar}")
    if prod_first_bar and opt_first_bar:
        print(f"  Time difference: {prod_first_bar - opt_first_bar}")
    
    # Check regime alignment
    print("\nRegime alignment at mismatched signals:")
    for t in sorted(prod_only | opt_only):
        prod_regime = prod_bars[t].regime if t in prod_bars else "N/A"
        opt_regime = opt_bars[t].regime if t in opt_bars else "N/A"
        match = "✓" if prod_regime == opt_regime else "✗"
        print(f"  {t}: Prod={prod_regime:20s}, Opt={opt_regime:20s} {match}")
    
    # Check MA values at mismatched points
    print("\nMA indicator comparison at mismatched signals:")
    for t in sorted(prod_only | opt_only):
        if t in prod_bars and t in opt_bars:
            p_bar = prod_bars[t]
            o_bar = opt_bars[t]
            ma_short_diff = abs(p_bar.ma_short - o_bar.ma_short) if p_bar.ma_short and o_bar.ma_short else "N/A"
            ma_long_diff = abs(p_bar.ma_long - o_bar.ma_long) if p_bar.ma_long and o_bar.ma_long else "N/A"
            print(f"  {t}:")
            print(f"    MA_short: Prod={p_bar.ma_short:.4f if p_bar.ma_short else 'N/A'}, Opt={o_bar.ma_short:.4f if o_bar.ma_short else 'N/A'}, Diff={ma_short_diff}")
            print(f"    MA_long:  Prod={p_bar.ma_long:.4f if p_bar.ma_long else 'N/A'}, Opt={o_bar.ma_long:.4f if o_bar.ma_long else 'N/A'}, Diff={ma_long_diff}")

def main():
    analyze_mismatched_signals()

if __name__ == "__main__":
    main()