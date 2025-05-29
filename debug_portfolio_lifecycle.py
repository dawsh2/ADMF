#!/usr/bin/env python3
"""
Debug script to trace portfolio lifecycle and trade counting.
"""

import subprocess
import re

def analyze_portfolio_lifecycle():
    """Run test and analyze portfolio lifecycle."""
    cmd = [
        "python3", "main_ultimate.py",
        "--config", "config/test_ensemble_optimization.yaml",
        "--bars", "2000",
        "--dataset", "test",
        "--log-level", "DEBUG"  # Use DEBUG for more detail
    ]
    
    print("Running test with DEBUG logging...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Write to file
    with open("portfolio_lifecycle_debug.log", "w") as f:
        f.write(output)
    
    # Extract all portfolio-related events
    events = []
    
    # Portfolio resets
    for match in re.finditer(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Resetting portfolio.*had (\d+) trades", output):
        events.append({
            'time': match.group(1),
            'type': 'RESET',
            'trades_before': int(match.group(2))
        })
    
    # Trade completions
    for match in re.finditer(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Trade #(\d+) completed", output):
        events.append({
            'time': match.group(1),
            'type': 'TRADE',
            'trade_num': int(match.group(2))
        })
    
    # Performance calculations
    for match in re.finditer(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Number of Trades: (\d+)", output):
        events.append({
            'time': match.group(1),
            'type': 'PERFORMANCE',
            'num_trades': int(match.group(2))
        })
    
    # Portfolio creation/resolution
    for match in re.finditer(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*(Creating new instance|resolved.*portfolio)", output):
        events.append({
            'time': match.group(1),
            'type': 'LIFECYCLE',
            'message': match.group(2)
        })
    
    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    return events

print("="*80)
print("PORTFOLIO LIFECYCLE ANALYSIS")
print("="*80)

events = analyze_portfolio_lifecycle()

print(f"\nTotal events: {len(events)}\n")

# Group events by type
reset_events = [e for e in events if e['type'] == 'RESET']
trade_events = [e for e in events if e['type'] == 'TRADE']
perf_events = [e for e in events if e['type'] == 'PERFORMANCE']
lifecycle_events = [e for e in events if e['type'] == 'LIFECYCLE']

print(f"Reset events: {len(reset_events)}")
print(f"Trade events: {len(trade_events)}")
print(f"Performance events: {len(perf_events)}")
print(f"Lifecycle events: {len(lifecycle_events)}")

print("\n" + "-"*80)
print("EVENT TIMELINE:")
print("-"*80)

for event in events:
    if event['type'] == 'RESET':
        print(f"{event['time']} [RESET] Portfolio reset (had {event['trades_before']} trades)")
    elif event['type'] == 'TRADE':
        print(f"{event['time']} [TRADE] Trade #{event['trade_num']} completed")
    elif event['type'] == 'PERFORMANCE':
        print(f"{event['time']} [PERF] Number of Trades: {event['num_trades']}")
    elif event['type'] == 'LIFECYCLE':
        print(f"{event['time']} [LIFECYCLE] {event['message']}")

# Analyze the issue
print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

if reset_events and trade_events:
    last_reset = reset_events[-1]
    first_trade = trade_events[0] if trade_events else None
    last_trade = trade_events[-1] if trade_events else None
    
    if first_trade and last_reset['time'] < first_trade['time']:
        print(f"✅ Trades occur AFTER reset (reset at {last_reset['time']}, first trade at {first_trade['time']})")
    else:
        print(f"❌ Reset occurs AFTER trades!")
        
    if last_trade and perf_events:
        first_perf = perf_events[0]
        if last_trade['time'] < first_perf['time']:
            print(f"✅ Performance calculated AFTER trades complete")
            if first_perf['num_trades'] == 0:
                print(f"❌ BUT performance shows 0 trades despite {last_trade['trade_num']} trades executed!")
                print("   This suggests the portfolio instance used for performance is different")
                print("   from the one that executed trades, or was reset between trading and reporting.")
        else:
            print(f"❌ Performance calculated BEFORE trades complete!")

print("\nCheck portfolio_lifecycle_debug.log for full details.")