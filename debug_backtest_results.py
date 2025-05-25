#!/usr/bin/env python3
"""
Debug script to understand why we're getting -inf results.
"""

print("""
Potential issues causing -inf results:

1. BacktestEngine creates new strategy instance but container returns cached one
2. Parameters aren't being applied to the strategy
3. Data handler isn't emitting bars properly
4. Portfolio isn't recording trades

The issue is likely that BacktestEngine needs to:
- Use the existing strategy instance from container
- Apply parameters to it
- Ensure it's properly reset between runs

Let's check the logs for:
- "Backtest completed. Final portfolio value:"
- Any trade execution messages
- Strategy signal generation
""")