#!/usr/bin/env python3
"""
Analyze and compare signals from production and optimizer log files
"""
import re
from datetime import datetime

def extract_signals_from_log(log_file_path, source_name):
    """Extract all signal generations from a log file"""
    signals = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Look for signal generation messages
                if '🚨 SIGNAL GENERATED' in line:
                    # Parse the signal details
                    signal_match = re.search(r'🚨 SIGNAL GENERATED #(\d+): Type=([+-]?\d+), Price=([0-9.]+), Regime=(\w+)', line)
                    if signal_match:
                        bar_num = signal_match.group(1)
                        signal_type = int(signal_match.group(2))
                        price = float(signal_match.group(3))
                        regime = signal_match.group(4)
                        
                        # Extract timestamp if available
                        timestamp_match = re.search(r'\[([0-9-]+\s[0-9:]+\+[0-9:]+)\]', line)
                        timestamp = timestamp_match.group(1) if timestamp_match else None
                        
                        signals.append({
                            'source': source_name,
                            'line_num': line_num,
                            'bar_num': bar_num,
                            'signal_type': signal_type,
                            'price': price,
                            'regime': regime,
                            'timestamp': timestamp,
                            'full_line': line.strip()
                        })
                
                # Also look for any other signal-related messages
                elif 'Signal:' in line and ('at' in line or 'signal' in line.lower()):
                    # Alternative signal format
                    alt_signal_match = re.search(r'Signal:\s*([+-]?\d+)\s*at\s*([0-9.]+)', line)
                    if alt_signal_match:
                        signal_type = int(alt_signal_match.group(1))
                        price = float(alt_signal_match.group(2))
                        
                        signals.append({
                            'source': source_name,
                            'line_num': line_num,
                            'bar_num': 'unknown',
                            'signal_type': signal_type,
                            'price': price,
                            'regime': 'unknown',
                            'timestamp': None,
                            'full_line': line.strip()
                        })
    
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return []
    
    return signals

def compare_signals(production_signals, optimizer_signals):
    """Compare signals between production and optimizer runs"""
    
    print("=" * 80)
    print("SIGNAL COMPARISON ANALYSIS")
    print("=" * 80)
    
    print(f"\nSignal Counts:")
    print(f"  Production: {len(production_signals)} signals")
    print(f"  Optimizer:  {len(optimizer_signals)} signals")
    
    if len(production_signals) != len(optimizer_signals):
        print(f"  ❌ DIFFERENT SIGNAL COUNTS! Difference: {abs(len(production_signals) - len(optimizer_signals))}")
    else:
        print(f"  ✅ Same signal count")
    
    # Compare signals by timestamp/order
    print(f"\nSignal-by-Signal Comparison:")
    print("-" * 50)
    
    max_signals = max(len(production_signals), len(optimizer_signals))
    differences = []
    
    for i in range(max_signals):
        prod_signal = production_signals[i] if i < len(production_signals) else None
        opt_signal = optimizer_signals[i] if i < len(optimizer_signals) else None
        
        if prod_signal and opt_signal:
            # Compare key attributes
            type_match = prod_signal['signal_type'] == opt_signal['signal_type']
            price_match = abs(prod_signal['price'] - opt_signal['price']) < 0.01  # Allow small price differences
            regime_match = prod_signal['regime'] == opt_signal['regime']
            
            if type_match and price_match and regime_match:
                status = "✅"
            else:
                status = "❌"
                differences.append(i + 1)
            
            print(f"  Signal {i+1:2d}: {status} Prod({prod_signal['signal_type']}, {prod_signal['price']:.4f}, {prod_signal['regime']}) vs Opt({opt_signal['signal_type']}, {opt_signal['price']:.4f}, {opt_signal['regime']})")
            
            if not type_match:
                print(f"           └─ Signal type differs: {prod_signal['signal_type']} vs {opt_signal['signal_type']}")
            if not price_match:
                print(f"           └─ Price differs: {prod_signal['price']:.4f} vs {opt_signal['price']:.4f}")
            if not regime_match:
                print(f"           └─ Regime differs: {prod_signal['regime']} vs {opt_signal['regime']}")
                
        elif prod_signal and not opt_signal:
            print(f"  Signal {i+1:2d}: ❌ Production has signal, Optimizer missing: ({prod_signal['signal_type']}, {prod_signal['price']:.4f}, {prod_signal['regime']})")
            differences.append(i + 1)
        elif opt_signal and not prod_signal:
            print(f"  Signal {i+1:2d}: ❌ Optimizer has signal, Production missing: ({opt_signal['signal_type']}, {opt_signal['price']:.4f}, {opt_signal['regime']})")
            differences.append(i + 1)
    
    print(f"\nSummary:")
    if differences:
        print(f"  ❌ Found {len(differences)} signal differences at positions: {differences}")
        print(f"  Signal generation is NOT identical")
    else:
        print(f"  ✅ All signals match! Signal generation is identical")
    
    # Show first few signals from each for detailed inspection
    print(f"\nFirst 5 Production Signals:")
    for i, signal in enumerate(production_signals[:5]):
        print(f"  {i+1}. Type:{signal['signal_type']}, Price:{signal['price']:.4f}, Regime:{signal['regime']}, Timestamp:{signal['timestamp']}")
    
    print(f"\nFirst 5 Optimizer Signals:")
    for i, signal in enumerate(optimizer_signals[:5]):
        print(f"  {i+1}. Type:{signal['signal_type']}, Price:{signal['price']:.4f}, Regime:{signal['regime']}, Timestamp:{signal['timestamp']}")
    
    return len(differences) == 0

def main():
    print("Signal Analysis: Comparing Production vs Optimizer Signal Generation")
    
    # Extract signals from both log files
    production_file = "/Users/daws/ADMF/logs/admf_20250523_231324.log"  # Latest production log with JSON weights
    optimizer_file = "/Users/daws/ADMF/logs/admf_20250523_230532.log"   # Latest optimizer log with full logging
    
    print(f"\nExtracting signals from production log: {production_file}")
    production_signals = extract_signals_from_log(production_file, "PRODUCTION")
    
    print(f"Extracting signals from optimizer log: {optimizer_file}")
    optimizer_signals = extract_signals_from_log(optimizer_file, "OPTIMIZER")
    
    # Compare the signals
    signals_match = compare_signals(production_signals, optimizer_signals)
    
    if signals_match:
        print(f"\n🎯 CONCLUSION: Signal generation is IDENTICAL between production and optimizer!")
        print(f"   The weight difference (0.5,0.5) vs (0.2,0.8) is not affecting signal output.")
        print(f"   Different trade counts must be due to other factors.")
    else:
        print(f"\n⚠️  CONCLUSION: Signal generation DIFFERS between production and optimizer!")
        print(f"   The weight difference (0.5,0.5) vs (0.2,0.8) IS affecting signal output.")
        print(f"   This explains the different trade counts and performance.")

if __name__ == "__main__":
    main()