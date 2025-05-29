#!/usr/bin/env python3
"""
Debug script to investigate the trade counting discrepancy.
"""

import subprocess
import re
import json

def run_test_and_extract_info():
    """Run a test backtest and extract trade information."""
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--bars", "5000",  # Enough bars to generate some trades
        "--dataset", "test",
        "--log-level", "INFO"
    ]
    
    print("Running test backtest...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Write full output to file for inspection
    with open("debug_trade_count_output.log", "w") as f:
        f.write(output)
    
    # Extract trade execution logs
    trade_logs = []
    for match in re.finditer(r"TRADE: (BUY|SELL) ([\d.]+) (\w+) @ \$([\d.]+)", output):
        trade_logs.append({
            'side': match.group(1),
            'quantity': float(match.group(2)),
            'symbol': match.group(3),
            'price': float(match.group(4))
        })
    
    # Extract trade completion logs
    trade_completions = []
    for match in re.finditer(r"Trade #(\d+) completed", output):
        trade_completions.append(int(match.group(1)))
    
    # Extract closed position logs
    closed_positions = []
    for match in re.finditer(r"(Closed LONG|Covered SHORT) .* ([\d.]+) (\w+) at ([\d.]+).*PnL: ([-\d.]+)", output):
        closed_positions.append({
            'type': match.group(1),
            'quantity': float(match.group(2)),
            'symbol': match.group(3),
            'price': float(match.group(4)),
            'pnl': float(match.group(5))
        })
    
    # Extract final performance
    total_return_match = re.search(r"Total Return: ([-\d.]+)%", output)
    total_return = float(total_return_match.group(1)) if total_return_match else None
    
    num_trades_match = re.search(r"Number of Trades: (\d+)", output)
    num_trades = int(num_trades_match.group(1)) if num_trades_match else None
    
    # Extract regime performance
    regime_trades = {}
    for match in re.finditer(r"(\w+): PnL=\$[-\d.,]+, Trades=(\d+)", output):
        regime = match.group(1)
        trades = int(match.group(2))
        regime_trades[regime] = trades
    
    # Check for portfolio reset logs
    resets = []
    for match in re.finditer(r"Resetting portfolio '(\w+)'", output):
        resets.append(match.group(1))
    
    # Check for comprehensive reset
    comprehensive_reset = "COMPREHENSIVE RESET" in output
    
    return {
        'trade_logs': trade_logs,
        'trade_completions': trade_completions,
        'closed_positions': closed_positions,
        'total_return': total_return,
        'num_trades': num_trades,
        'regime_trades': regime_trades,
        'resets': resets,
        'comprehensive_reset': comprehensive_reset
    }

print("="*80)
print("TRADE COUNT DEBUG ANALYSIS")
print("="*80)

info = run_test_and_extract_info()

print(f"\nComprehensive Reset Found: {info['comprehensive_reset']}")
print(f"\nPortfolio Resets: {len(info['resets'])}")
for reset in info['resets']:
    print(f"  - {reset}")

print(f"\nTrade Execution Logs Found: {len(info['trade_logs'])}")
if info['trade_logs']:
    print("  First 5 trades:")
    for i, trade in enumerate(info['trade_logs'][:5]):
        print(f"    {i+1}. {trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']}")

print(f"\nClosed Positions Found: {len(info['closed_positions'])}")
if info['closed_positions']:
    print("  First 5 closed positions:")
    for i, pos in enumerate(info['closed_positions'][:5]):
        print(f"    {i+1}. {pos['type']} {pos['quantity']} {pos['symbol']} @ ${pos['price']}, PnL: ${pos['pnl']}")

print(f"\nTrade Completion Logs: {info['trade_completions']}")

print(f"\nFinal Performance:")
print(f"  Total Return: {info['total_return']}%")
print(f"  Number of Trades (from summary): {info['num_trades']}")

print(f"\nRegime Trade Counts:")
for regime, count in info['regime_trades'].items():
    print(f"  {regime}: {count} trades")
total_regime_trades = sum(info['regime_trades'].values())
print(f"  Total from regimes: {total_regime_trades}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Analyze discrepancies
if info['num_trades'] == 0 and total_regime_trades > 0:
    print("❌ ISSUE: Summary shows 0 trades but regime performance shows trades!")
    print("   This suggests _trade_log is being cleared or not properly populated.")
    
if len(info['closed_positions']) > 0 and info['num_trades'] == 0:
    print("❌ ISSUE: Positions were closed but trade count is 0!")
    print("   This confirms trades are being executed but not counted.")
    
if len(info['trade_logs']) > len(info['closed_positions']) * 2:
    print("⚠️  NOTE: More trade executions than closed positions.")
    print("   This is normal - not all trades close positions (some open new ones).")

if info['comprehensive_reset']:
    print("✅ Comprehensive reset is working.")
else:
    print("❌ Comprehensive reset not found!")

print("\nCheck debug_trade_count_output.log for full output.")