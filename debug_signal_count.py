#!/usr/bin/env python3
"""
Debug script to count signals in log files
"""

import sys
import re

def analyze_log_file(filename):
    """Analyze a log file for signals and trades"""
    
    signals = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
    trades = 0
    regime_changes = 0
    
    with open(filename, 'r') as f:
        for line in f:
            # Count SIGNAL events
            if "Publishing event: Event(type=SIGNAL" in line or "Publishing SIGNAL event" in line:
                if "signal_type': 1" in line or "BUY" in line:
                    signals['BUY'] += 1
                elif "signal_type': -1" in line or "SELL" in line:
                    signals['SELL'] += 1
                elif "signal_type': 0" in line or "NEUTRAL" in line:
                    signals['NEUTRAL'] += 1
                    
            # Count trades/fills
            if "FILL event" in line or "Trade logged" in line:
                trades += 1
                
            # Count regime changes
            if "REGIME CHANGE DETECTED" in line or "Regime changed from" in line:
                regime_changes += 1
    
    print(f"\nAnalysis of {filename}:")
    print(f"  BUY signals: {signals['BUY']}")
    print(f"  SELL signals: {signals['SELL']}")
    print(f"  NEUTRAL signals: {signals['NEUTRAL']}")
    print(f"  Total non-neutral signals: {signals['BUY'] + signals['SELL']}")
    print(f"  Trades/Fills: {trades}")
    print(f"  Regime changes: {regime_changes}")
    
    return signals, trades

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_signal_count.py <log_file1> [log_file2] ...")
        sys.exit(1)
        
    for logfile in sys.argv[1:]:
        try:
            analyze_log_file(logfile)
        except Exception as e:
            print(f"Error analyzing {logfile}: {e}")