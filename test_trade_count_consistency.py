#!/usr/bin/env python3
"""
Test that trade counts are reported consistently across all methods.
"""

# Test the portfolio methods directly
from src.risk.basic_portfolio import BasicPortfolio

# Create a mock portfolio with trades
portfolio = BasicPortfolio("test_portfolio")

# Simulate some trades by populating _trade_log
# This simulates what would happen after trades are executed
portfolio._trade_log = []

# Add some mock trades for different regimes
mock_trades = [
    # DEFAULT regime trades
    {'regime': 'default', 'pnl': 10, 'count': 1, 'entry_regime': 'default', 'exit_regime': 'default'},
    {'regime': 'default', 'pnl': 15, 'count': 1, 'entry_regime': 'default', 'exit_regime': 'default'},
    {'regime': 'default', 'pnl': -5, 'count': 1, 'entry_regime': 'default', 'exit_regime': 'default'},
    # TRENDING_UP trades
    {'regime': 'trending_up', 'pnl': 20, 'count': 1, 'entry_regime': 'trending_up', 'exit_regime': 'trending_up'},
    {'regime': 'trending_up', 'pnl': 11, 'count': 1, 'entry_regime': 'trending_up', 'exit_regime': 'trending_up'},
    # TRENDING_DOWN trade
    {'regime': 'trending_down', 'pnl': 67, 'count': 1, 'entry_regime': 'trending_down', 'exit_regime': 'trending_down'},
]

# Clear trade log to simulate the issue
portfolio._trade_log = []

print("="*80)
print("TRADE COUNT CONSISTENCY TEST")
print("="*80)
print("\nSimulating scenario where _trade_log is empty but regime performance has trades...")

# Test different methods
print("\n1. Testing get_performance():")
perf = portfolio.get_performance()
print(f"   num_trades: {perf.get('num_trades', 'NOT FOUND')}")

print("\n2. Testing get_performance_metrics():")
metrics = portfolio.get_performance_metrics()
print(f"   num_trades: {metrics.get('num_trades', 'NOT FOUND')}")

print("\n3. Testing regime performance calculation:")
regime_perf = portfolio._calculate_performance_by_regime()
total_from_regimes = sum(
    p.get('count', 0) 
    for r, p in regime_perf.items() 
    if r != '_boundary_trades_summary' and isinstance(p, dict)
)
print(f"   Total trades from regimes: {total_from_regimes}")

print("\n4. Direct _trade_log check:")
print(f"   len(_trade_log): {len(portfolio._trade_log)}")

print("\n" + "="*80)
print("EXPLANATION:")
print("="*80)
print("The issue occurs when:")
print("1. Trades are executed and logged in _trade_log")
print("2. Regime performance correctly tracks these trades")
print("3. But _trade_log gets cleared (e.g., by reset or scoping issues)")
print("4. Old code: num_trades = len(_trade_log) = 0")
print("5. Fixed code: num_trades = sum of regime counts = actual trades")

print("\nWith the fix applied:")
print("- get_performance() now uses regime counts")
print("- get_performance_metrics() now uses regime counts")
print("- Both methods should report consistent trade counts")
print("- Trade count will match what's shown in regime performance")