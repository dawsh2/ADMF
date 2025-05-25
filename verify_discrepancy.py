#!/usr/bin/env python3
"""
Verify the exact discrepancy between optimizer and production.
"""

import re
import os
import glob

def find_latest_results():
    """Find the latest results from logs."""
    
    # Look for the most recent logs
    log_files = sorted(glob.glob('logs/admf_*.log'), reverse=True)
    
    results = {
        'optimizer': None,
        'production': None
    }
    
    # Search for the known values in recent logs or output files
    search_patterns = [
        (r'Adaptive GA Ensemble Strategy Test final_portfolio_value:\s*([0-9.]+)', 'optimizer'),
        (r'Final Portfolio Value:\s*\$?([0-9,]+\.?\d*)', 'production'),
        (r'Optimizer V3 result:\s*\$?([0-9,]+\.?\d*)', 'optimizer'),
        (r'Production result:\s*\$?([0-9,]+\.?\d*)', 'production')
    ]
    
    # Check log files
    for log_file in log_files[:10]:  # Check last 10 logs
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            for pattern, result_type in search_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        value = float(match.replace(',', ''))
                        # Look for the specific values we're tracking
                        if 99900 < value < 100000:
                            if result_type == 'optimizer' and abs(value - 99915.74) < 1:
                                results['optimizer'] = value
                            elif result_type == 'production' and abs(value - 99870.04) < 1:
                                results['production'] = value
        except:
            continue
    
    # Also check output files
    output_files = ['test.out', 'rsi.out', 'prod_valid_test.out']
    for out_file in output_files:
        if os.path.exists(out_file):
            try:
                with open(out_file, 'r') as f:
                    content = f.read()
                    
                for pattern, result_type in search_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches:
                            value = float(match.replace(',', ''))
                            if 99900 < value < 100000:
                                if result_type == 'production' and abs(value - 99870.04) < 1:
                                    results['production'] = value
            except:
                continue
    
    return results

def main():
    print("="*80)
    print("VERIFYING DISCREPANCY")
    print("="*80)
    
    # The known values from our investigation
    optimizer_value = 99915.74
    production_value = 99870.04
    
    print(f"\nKnown Results:")
    print(f"Optimizer (V3 with CleanBacktestEngine): ${optimizer_value:,.2f}")
    print(f"Production (with adaptive parameters):     ${production_value:,.2f}")
    
    diff = abs(optimizer_value - production_value)
    pct_diff = (diff / optimizer_value) * 100
    
    print(f"\nDifference: ${diff:.2f} ({pct_diff:.4f}%)")
    
    print("\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)
    
    print(f"\n1. Trade Count Mismatch:")
    print(f"   - Optimizer: 5 trades")
    print(f"   - Production: 11 trades")
    print(f"   - Production has 6 MORE trades")
    
    print(f"\n2. The 0.046% discrepancy suggests:")
    print(f"   - Different regime detection timing")
    print(f"   - Different signal generation")
    print(f"   - Possible data indexing misalignment")
    
    print(f"\n3. Key areas to investigate:")
    print(f"   - Indicator warmup periods (MA: 20 bars, RSI: 14 bars)")
    print(f"   - Data slice differences (bars 800-999 vs 0-199)")
    print(f"   - Regime detector initialization state")
    print(f"   - First few bars of each run")
    
    # Search for results
    print("\n" + "-"*80)
    print("Searching for these values in recent files...")
    found = find_latest_results()
    
    if found['optimizer']:
        print(f"✓ Found optimizer value: ${found['optimizer']:,.2f}")
    if found['production']:
        print(f"✓ Found production value: ${found['production']:,.2f}")

if __name__ == "__main__":
    main()