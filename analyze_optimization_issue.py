#!/usr/bin/env python3
"""
Analyze why optimization is returning -inf.
"""

print("""
Analysis of the -inf optimization results:

1. The ADAPTIVE TEST is working fine:
   - Shows 597 trades
   - Final portfolio value: 104639.13
   - Multiple regimes detected
   - Proper P&L calculations

2. The GRID SEARCH optimization is failing:
   - Returns -inf for training metric
   - No valid results found

The issue is likely:
- BacktestEngine reuses singleton components
- Original optimizer might create fresh instances
- Components need proper reset between parameter runs

Solution options:
1. Force components to reset between runs
2. Create fresh component instances (not singletons)
3. Ensure proper state cleanup

The original optimizer likely bypasses the singleton pattern
or has special handling for parameter application.
""")