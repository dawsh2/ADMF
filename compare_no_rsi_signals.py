#!/usr/bin/env python3
"""
Compare signals between RSI-disabled production run and optimizer run.
Analyzes whether disabling RSI makes production match optimizer behavior.
"""

import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def parse_signal_from_line(line: str) -> Tuple[str, str, float, str]:
    """Extract timestamp, symbol, value, and direction from a signal line."""
    # Match patterns like: "Signal: -1 at 523.24 (StrongSignal(strength=-1.000))"
    signal_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Signal:\s*([-\d.]+)\s*at\s*([\d.]+)', line)
    if signal_match:
        timestamp = signal_match.group(1)
        value = float(signal_match.group(2))
        price = float(signal_match.group(3))
        direction = "BUY" if value > 0 else "SELL"
        # Extract symbol from component name if present
        symbol = "SPY"  # Default to SPY
        if "SPY_Ensemble_Strategy" in line:
            symbol = "SPY"
        return timestamp, symbol, value, direction
    
    # Also match "MA BUY signal" or "MA SELL signal" patterns
    ma_signal_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*MA\s+(BUY|SELL)\s+signal.*price=([\d.]+)', line)
    if ma_signal_match:
        timestamp = ma_signal_match.group(1)
        direction = ma_signal_match.group(2)
        price = float(ma_signal_match.group(3))
        value = 1.0 if direction == "BUY" else -1.0
        symbol = "SPY"
        return timestamp, symbol, value, direction
    
    return None

def extract_signals(log_file: str) -> Dict[str, List[Tuple[str, float, str]]]:
    """Extract all signals from a log file."""
    signals = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for "Signal:" pattern or "MA BUY/SELL signal" pattern
                if ("Signal:" in line and " at " in line) or ("MA " in line and " signal" in line):
                    result = parse_signal_from_line(line)
                    if result:
                        timestamp, symbol, value, direction = result
                        signals[symbol].append((timestamp, value, direction))
    except FileNotFoundError:
        print(f"Error: File {log_file} not found")
        return {}
    
    # Sort signals by timestamp
    for symbol in signals:
        signals[symbol].sort(key=lambda x: x[0])
    
    return dict(signals)

def find_regime_info(log_file: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract regime information from log file."""
    regimes = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for regime detection patterns
                if "Detected regime:" in line or "Current regime:" in line:
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    regime_match = re.search(r'regime:\s*(\w+)', line, re.IGNORECASE)
                    if timestamp_match and regime_match:
                        timestamp = timestamp_match.group(1)
                        regime = regime_match.group(1)
                        regimes['SPY'].append((timestamp, regime))
    except FileNotFoundError:
        pass
    
    return dict(regimes)

def find_rsi_mentions(log_file: str) -> List[str]:
    """Find any mentions of RSI in the log file."""
    rsi_lines = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'RSI' in line.upper():
                    rsi_lines.append(line.strip())
    except FileNotFoundError:
        pass
    
    return rsi_lines

def compare_signals(prod_signals: Dict, opt_signals: Dict) -> None:
    """Compare signals between production and optimizer runs."""
    print("=" * 80)
    print("SIGNAL COMPARISON: RSI-DISABLED PRODUCTION vs OPTIMIZER")
    print("=" * 80)
    
    # Get all unique symbols
    all_symbols = set(prod_signals.keys()) | set(opt_signals.keys())
    
    for symbol in sorted(all_symbols):
        print(f"\n{symbol} Signals:")
        print("-" * 60)
        
        prod_sigs = prod_signals.get(symbol, [])
        opt_sigs = opt_signals.get(symbol, [])
        
        print(f"Production signals: {len(prod_sigs)}")
        print(f"Optimizer signals: {len(opt_sigs)}")
        
        if len(prod_sigs) != len(opt_sigs):
            print(f"⚠️  MISMATCH: Different number of signals!")
        
        # Create time windows for matching (within 1 minute)
        matched_signals = []
        unmatched_prod = list(prod_sigs)
        unmatched_opt = list(opt_sigs)
        
        for prod_sig in prod_sigs:
            prod_time = datetime.strptime(prod_sig[0], "%Y-%m-%d %H:%M:%S")
            
            for opt_sig in opt_sigs:
                opt_time = datetime.strptime(opt_sig[0], "%Y-%m-%d %H:%M:%S")
                
                # Check if timestamps are within 1 minute
                time_diff = abs((prod_time - opt_time).total_seconds())
                if time_diff <= 60:
                    matched_signals.append((prod_sig, opt_sig))
                    if prod_sig in unmatched_prod:
                        unmatched_prod.remove(prod_sig)
                    if opt_sig in unmatched_opt:
                        unmatched_opt.remove(opt_sig)
                    break
        
        print(f"\nMatched signals: {len(matched_signals)}")
        
        # Show all signals side by side for comparison
        print("\nSignal Details (chronological order):")
        print("\nProduction signals:")
        print("Time | Value | Direction")
        print("-" * 40)
        for sig in prod_sigs[:10]:  # Show first 10
            print(f"{sig[0]} | {sig[1]:>7.4f} | {sig[2]}")
        if len(prod_sigs) > 10:
            print(f"... and {len(prod_sigs) - 10} more")
        
        print("\nOptimizer signals:")
        print("Time | Value | Direction")
        print("-" * 40)
        for sig in opt_sigs[:10]:  # Show first 10
            print(f"{sig[0]} | {sig[1]:>7.4f} | {sig[2]}")
        if len(opt_sigs) > 10:
            print(f"... and {len(opt_sigs) - 10} more")
        
        # Show matched signals with value comparison
        if matched_signals:
            print("\nMatched Signal Details:")
            print("Time (Prod) | Time (Opt) | Value (Prod) | Value (Opt) | Diff | Match")
            print("-" * 80)
            
            exact_matches = 0
            close_matches = 0
            
            for prod_sig, opt_sig in matched_signals:
                prod_time = prod_sig[0].split()[1]
                opt_time = opt_sig[0].split()[1]
                prod_val = prod_sig[1]
                opt_val = opt_sig[1]
                diff = abs(prod_val - opt_val)
                
                # Check if values match
                if diff < 0.0001:  # Exact match (accounting for float precision)
                    match_status = "✓ EXACT"
                    exact_matches += 1
                elif diff < 0.01:  # Close match
                    match_status = "~ CLOSE"
                    close_matches += 1
                else:
                    match_status = "✗ DIFF"
                
                print(f"{prod_time} | {opt_time} | {prod_val:>12.4f} | {opt_val:>11.4f} | {diff:>6.4f} | {match_status}")
            
            print(f"\nMatch Summary:")
            print(f"  Exact matches: {exact_matches}/{len(matched_signals)} ({exact_matches/len(matched_signals)*100:.1f}%)")
            print(f"  Close matches: {close_matches}/{len(matched_signals)} ({close_matches/len(matched_signals)*100:.1f}%)")
        
        # Show unmatched signals
        if unmatched_prod:
            print(f"\nUnmatched Production Signals ({len(unmatched_prod)}):")
            for sig in unmatched_prod[:5]:  # Show first 5
                print(f"  {sig[0]} | Value: {sig[1]:.4f} | {sig[2]}")
            if len(unmatched_prod) > 5:
                print(f"  ... and {len(unmatched_prod) - 5} more")
        
        if unmatched_opt:
            print(f"\nUnmatched Optimizer Signals ({len(unmatched_opt)}):")
            for sig in unmatched_opt[:5]:  # Show first 5
                print(f"  {sig[0]} | Value: {sig[1]:.4f} | {sig[2]}")
            if len(unmatched_opt) > 5:
                print(f"  ... and {len(unmatched_opt) - 5} more")

def analyze_rsi_presence(prod_file: str, opt_file: str) -> None:
    """Check for RSI mentions in both log files."""
    print("\n" + "=" * 80)
    print("RSI PRESENCE ANALYSIS")
    print("=" * 80)
    
    # Check for RSI weight configuration
    prod_rsi_weight = None
    opt_rsi_weight = None
    
    try:
        with open(prod_file, 'r') as f:
            for line in f:
                if "RSI weight:" in line or "RSI=" in line:
                    weight_match = re.search(r'RSI\s*(?:weight)?[:\s=]\s*([\d.]+)', line)
                    if weight_match:
                        prod_rsi_weight = float(weight_match.group(1))
                        break
    except:
        pass
    
    try:
        with open(opt_file, 'r') as f:
            for line in f:
                if "RSI weight:" in line or "RSI=" in line:
                    weight_match = re.search(r'RSI\s*(?:weight)?[:\s=]\s*([\d.]+)', line)
                    if weight_match:
                        opt_rsi_weight = float(weight_match.group(1))
                        break
    except:
        pass
    
    print(f"\nRSI Weight Configuration:")
    print(f"  Production RSI weight: {prod_rsi_weight}")
    print(f"  Optimizer RSI weight: {opt_rsi_weight}")
    
    if prod_rsi_weight == 0.0:
        print("  ✓ RSI is disabled in production (weight = 0.0)")
    elif prod_rsi_weight is not None:
        print(f"  ⚠️  RSI is ENABLED in production with weight {prod_rsi_weight}")
    
    prod_rsi = find_rsi_mentions(prod_file)
    opt_rsi = find_rsi_mentions(opt_file)
    
    print(f"\nProduction RSI mentions: {len(prod_rsi)}")
    if prod_rsi:
        print("Sample production RSI lines:")
        for line in prod_rsi[:3]:
            print(f"  {line[:100]}...")
    
    print(f"\nOptimizer RSI mentions: {len(opt_rsi)}")
    if opt_rsi:
        print("Sample optimizer RSI lines:")
        for line in opt_rsi[:3]:
            print(f"  {line[:100]}...")

def analyze_regime_consistency(prod_file: str, opt_file: str) -> None:
    """Compare regime detections between runs."""
    print("\n" + "=" * 80)
    print("REGIME CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    prod_regimes = find_regime_info(prod_file)
    opt_regimes = find_regime_info(opt_file)
    
    for symbol in set(prod_regimes.keys()) | set(opt_regimes.keys()):
        print(f"\n{symbol} Regimes:")
        prod_reg = prod_regimes.get(symbol, [])
        opt_reg = opt_regimes.get(symbol, [])
        
        print(f"  Production regime changes: {len(prod_reg)}")
        print(f"  Optimizer regime changes: {len(opt_reg)}")
        
        if prod_reg and opt_reg:
            print("\n  First few regime detections:")
            print("  Production:")
            for i, (timestamp, regime) in enumerate(prod_reg[:3]):
                print(f"    {timestamp}: {regime}")
            
            print("  Optimizer:")
            for i, (timestamp, regime) in enumerate(opt_reg[:3]):
                print(f"    {timestamp}: {regime}")

def main():
    """Main analysis function."""
    prod_file = "logs/admf_20250524_210903.log"
    opt_file = "logs/admf_20250523_230532.log"
    
    print("Analyzing RSI-disabled production run vs optimizer run...")
    print(f"Production log: {prod_file}")
    print(f"Optimizer log: {opt_file}")
    
    # Extract signals
    prod_signals = extract_signals(prod_file)
    opt_signals = extract_signals(opt_file)
    
    if not prod_signals and not opt_signals:
        print("\nNo signals found in either log file!")
        return
    
    # Compare signals
    compare_signals(prod_signals, opt_signals)
    
    # Analyze RSI presence
    analyze_rsi_presence(prod_file, opt_file)
    
    # Analyze regime consistency
    analyze_regime_consistency(prod_file, opt_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_prod_signals = sum(len(sigs) for sigs in prod_signals.values())
    total_opt_signals = sum(len(sigs) for sigs in opt_signals.values())
    
    print(f"\nTotal signals:")
    print(f"  Production: {total_prod_signals}")
    print(f"  Optimizer: {total_opt_signals}")
    
    if total_prod_signals == total_opt_signals:
        print("\n✓ Signal counts match!")
    else:
        diff = abs(total_prod_signals - total_opt_signals)
        print(f"\n✗ Signal count mismatch: {diff} difference")
    
    print("\nKey observations:")
    print("- Check if signal values are now more consistent")
    print("- Verify RSI is truly disabled in production")
    print("- Look for any remaining discrepancies in timing or values")

if __name__ == "__main__":
    main()