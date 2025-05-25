#!/usr/bin/env python3
"""
Analyze and compare the first few bars with detailed indicator values
"""
import re
from datetime import datetime

def extract_bar_indicators(log_file_path, source_name, start_bar=50, max_bars=20):
    """Extract detailed indicator values for the first few bars"""
    bars = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Look for bar indicator logging
                if '📊 BAR_' in line and 'INDICATORS:' in line:
                    # Parse the bar details
                    bar_match = re.search(r'📊 BAR_(\d+) \[([^\]]+)\] INDICATORS: Price=([0-9.]+), MA_short=([^,]+), MA_long=([^,]+), RSI=([^,]+), RSI_thresholds=\(([^)]+)\), Regime=([^,]+), Weights=\(MA:([^,]+),RSI:([^)]+)\)', line)
                    if bar_match:
                        bar_num = int(bar_match.group(1))
                        timestamp = bar_match.group(2)
                        price = float(bar_match.group(3))
                        ma_short = bar_match.group(4).strip()
                        ma_long = bar_match.group(5).strip()
                        rsi = bar_match.group(6).strip()
                        rsi_thresholds = bar_match.group(7).strip()
                        regime = bar_match.group(8).strip()
                        ma_weight = bar_match.group(9).strip()
                        rsi_weight = bar_match.group(10).strip()
                        
                        # Convert numeric values
                        try:
                            ma_short_val = float(ma_short) if ma_short != "N/A" else None
                            ma_long_val = float(ma_long) if ma_long != "N/A" else None
                            rsi_val = float(rsi) if rsi != "N/A" else None
                            ma_weight_val = float(ma_weight)
                            rsi_weight_val = float(rsi_weight)
                        except:
                            ma_short_val = None
                            ma_long_val = None
                            rsi_val = None
                            ma_weight_val = None
                            rsi_weight_val = None
                        
                        # Only collect bars from start_bar onwards
                        if bar_num >= start_bar:
                            bars.append({
                            'source': source_name,
                            'bar_num': bar_num,
                            'timestamp': timestamp,
                            'price': price,
                            'ma_short': ma_short_val,
                            'ma_long': ma_long_val,
                            'rsi': rsi_val,
                            'rsi_thresholds': rsi_thresholds,
                            'regime': regime,
                            'ma_weight': ma_weight_val,
                            'rsi_weight': rsi_weight_val,
                            'line_num': line_num,
                            'full_line': line.strip()
                        })
                        
                            if len(bars) >= max_bars:
                                break
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return bars

def compare_bars(production_bars, optimizer_bars, max_compare=10):
    """Compare bars between production and optimizer runs"""
    
    print("=" * 100)
    print("BAR-BY-BAR INDICATOR COMPARISON")
    print("=" * 100)
    
    print(f"\nBar Counts Found:")
    print(f"  Production: {len(production_bars)} bars")
    print(f"  Optimizer:  {len(optimizer_bars)} bars")
    
    if len(production_bars) == 0 or len(optimizer_bars) == 0:
        print("❌ No bar data found in one or both logs!")
        return
    
    max_bars = min(max_compare, len(production_bars), len(optimizer_bars))
    
    print(f"\nComparing first {max_bars} bars:")
    print("-" * 100)
    
    for i in range(max_bars):
        prod_bar = production_bars[i] if i < len(production_bars) else None
        opt_bar = optimizer_bars[i] if i < len(optimizer_bars) else None
        
        if prod_bar and opt_bar:
            print(f"\n🔍 BAR {i+1:2d}:")
            
            # Compare basic info
            print(f"  Price:      Prod={prod_bar['price']:.4f}  |  Opt={opt_bar['price']:.4f}")
            print(f"  Regime:     Prod={prod_bar['regime']:<20}  |  Opt={opt_bar['regime']:<20}")
            print(f"  Weights:    Prod=(MA:{prod_bar['ma_weight']:.3f},RSI:{prod_bar['rsi_weight']:.3f})  |  Opt=(MA:{opt_bar['ma_weight']:.3f},RSI:{opt_bar['rsi_weight']:.3f})")
            
            # Compare MA values
            if prod_bar['ma_short'] is not None and opt_bar['ma_short'] is not None:
                ma_short_diff = abs(prod_bar['ma_short'] - opt_bar['ma_short'])
                ma_short_status = "✅" if ma_short_diff < 0.001 else f"❌({ma_short_diff:.4f})"
                print(f"  MA_short:   Prod={prod_bar['ma_short']:.4f}  |  Opt={opt_bar['ma_short']:.4f}  {ma_short_status}")
            else:
                print(f"  MA_short:   Prod={prod_bar['ma_short']}  |  Opt={opt_bar['ma_short']} ❌(N/A mismatch)")
                
            if prod_bar['ma_long'] is not None and opt_bar['ma_long'] is not None:
                ma_long_diff = abs(prod_bar['ma_long'] - opt_bar['ma_long'])
                ma_long_status = "✅" if ma_long_diff < 0.001 else f"❌({ma_long_diff:.4f})"
                print(f"  MA_long:    Prod={prod_bar['ma_long']:.4f}  |  Opt={opt_bar['ma_long']:.4f}  {ma_long_status}")
            else:
                print(f"  MA_long:    Prod={prod_bar['ma_long']}  |  Opt={opt_bar['ma_long']} ❌(N/A mismatch)")
            
            # Compare RSI values
            if prod_bar['rsi'] is not None and opt_bar['rsi'] is not None:
                rsi_diff = abs(prod_bar['rsi'] - opt_bar['rsi'])
                rsi_status = "✅" if rsi_diff < 0.01 else f"❌({rsi_diff:.2f})"
                print(f"  RSI:        Prod={prod_bar['rsi']:.2f}  |  Opt={opt_bar['rsi']:.2f}  {rsi_status}")
            else:
                print(f"  RSI:        Prod={prod_bar['rsi']}  |  Opt={opt_bar['rsi']} ❌(N/A mismatch)")
            
            print(f"  RSI_thresh: Prod={prod_bar['rsi_thresholds']}  |  Opt={opt_bar['rsi_thresholds']}")
            
            # Check for major differences
            differences = []
            if prod_bar['regime'] != opt_bar['regime']:
                differences.append("REGIME")
            if abs(prod_bar['ma_weight'] - opt_bar['ma_weight']) > 0.001:
                differences.append("MA_WEIGHT")
            if abs(prod_bar['rsi_weight'] - opt_bar['rsi_weight']) > 0.001:
                differences.append("RSI_WEIGHT")
            if prod_bar['ma_short'] and opt_bar['ma_short'] and abs(prod_bar['ma_short'] - opt_bar['ma_short']) > 0.001:
                differences.append("MA_SHORT")
            if prod_bar['ma_long'] and opt_bar['ma_long'] and abs(prod_bar['ma_long'] - opt_bar['ma_long']) > 0.001:
                differences.append("MA_LONG")
            if prod_bar['rsi'] and opt_bar['rsi'] and abs(prod_bar['rsi'] - opt_bar['rsi']) > 0.01:
                differences.append("RSI")
                
            if differences:
                print(f"  ❌ DIFFERENCES: {', '.join(differences)}")
            else:
                print(f"  ✅ All indicators match!")
                
        elif prod_bar and not opt_bar:
            print(f"\n🔍 BAR {i+1:2d}: ❌ Production has bar, Optimizer missing")
        elif opt_bar and not prod_bar:
            print(f"\n🔍 BAR {i+1:2d}: ❌ Optimizer has bar, Production missing")
    
    print(f"\n" + "=" * 100)

def main():
    print("Bar Analysis: Comparing Production vs Optimizer Indicator Values")
    
    # Extract bar data from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log
    
    print(f"\nExtracting bar data from production log: {production_file} (bars 10-25)")
    production_bars = extract_bar_indicators(production_file, "PRODUCTION", start_bar=10, max_bars=15)
    
    print(f"Extracting bar data from optimizer log: {optimizer_file} (bars 10-25)")
    optimizer_bars = extract_bar_indicators(optimizer_file, "OPTIMIZER", start_bar=10, max_bars=15)
    
    # Compare the bars
    compare_bars(production_bars, optimizer_bars, max_compare=10)

if __name__ == "__main__":
    main()