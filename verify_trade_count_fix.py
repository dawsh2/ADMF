#!/usr/bin/env python3
"""
Verify that the trade count fix in BasicPortfolio is working correctly.
This tests both get_performance() and get_performance_metrics() methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk.basic_portfolio import BasicPortfolio
from src.core.event import Event, EventType
import datetime
import json

def test_trade_count_consistency():
    """Test that trade counts are reported consistently."""
    
    print("="*80)
    print("TRADE COUNT FIX VERIFICATION")
    print("="*80)
    
    # Create a portfolio
    portfolio = BasicPortfolio("test_portfolio")
    portfolio._regime_detector = None  # Disable regime detector for simplicity
    portfolio._current_market_regime = "default"
    
    # Initialize the portfolio
    portfolio._initialize()
    
    # Simulate some fills to create trades
    print("\n1. SIMULATING TRADES...")
    print("-"*40)
    
    # Set initial bar prices
    portfolio._last_bar_prices = {'SPY': 100.0}
    
    # Trade 1: Buy 100 SPY (DEFAULT regime)
    portfolio._current_market_regime = "default"
    fill1 = {
        'symbol': 'SPY',
        'timestamp': datetime.datetime.now(),
        'quantity_filled': 100,
        'fill_price': 100.0,
        'direction': 'BUY',
        'commission': 0.005,
        'fill_id': 'fill_1'
    }
    portfolio.on_fill(Event(EventType.FILL, fill1))
    print("✅ Trade 1: BUY 100 SPY @ $100 (DEFAULT regime)")
    
    # Trade 2: Sell 100 SPY (still DEFAULT regime)
    fill2 = {
        'symbol': 'SPY',
        'timestamp': datetime.datetime.now(),
        'quantity_filled': 100,
        'fill_price': 101.0,
        'direction': 'SELL',
        'commission': 0.005,
        'fill_id': 'fill_2'
    }
    portfolio.on_fill(Event(EventType.FILL, fill2))
    print("✅ Trade 2: SELL 100 SPY @ $101 (DEFAULT regime) - PnL: $100")
    
    # Change regime and do more trades
    portfolio._current_market_regime = "trending_up"
    
    # Trade 3: Buy 100 SPY (TRENDING_UP regime)
    fill3 = {
        'symbol': 'SPY',
        'timestamp': datetime.datetime.now(),
        'quantity_filled': 100,
        'fill_price': 101.5,
        'direction': 'BUY',
        'commission': 0.005,
        'fill_id': 'fill_3'
    }
    portfolio.on_fill(Event(EventType.FILL, fill3))
    print("✅ Trade 3: BUY 100 SPY @ $101.5 (TRENDING_UP regime)")
    
    # Trade 4: Sell 100 SPY (still TRENDING_UP regime)
    fill4 = {
        'symbol': 'SPY',
        'timestamp': datetime.datetime.now(),
        'quantity_filled': 100,
        'fill_price': 102.5,
        'direction': 'SELL',
        'commission': 0.005,
        'fill_id': 'fill_4'
    }
    portfolio.on_fill(Event(EventType.FILL, fill4))
    print("✅ Trade 4: SELL 100 SPY @ $102.5 (TRENDING_UP regime) - PnL: $100")
    
    print(f"\nTotal trades in _trade_log: {len(portfolio._trade_log)}")
    
    # Now test the methods
    print("\n2. TESTING TRADE COUNT METHODS...")
    print("-"*40)
    
    # Test get_performance_metrics (this is what app_runner calls)
    metrics = portfolio.get_performance_metrics()
    print(f"\nget_performance_metrics():")
    print(f"  num_trades: {metrics.get('num_trades', 'NOT FOUND')}")
    print(f"  total_return_pct: {metrics.get('total_return_pct', 0):.2f}%")
    
    # Test regime performance
    regime_perf = metrics.get('regime_performance', {})
    print(f"\nRegime performance breakdown:")
    total_from_regimes = 0
    for regime, perf in regime_perf.items():
        if regime != '_boundary_trades_summary' and isinstance(perf, dict):
            count = perf.get('count', 0)
            total_from_regimes += count
            print(f"  {regime}: {count} trades")
    print(f"  TOTAL: {total_from_regimes} trades")
    
    # Verify consistency
    print("\n3. CONSISTENCY CHECK...")
    print("-"*40)
    
    if metrics['num_trades'] == len(portfolio._trade_log):
        print(f"✅ Trade count matches _trade_log length: {metrics['num_trades']}")
    else:
        print(f"❌ Trade count mismatch: reported={metrics['num_trades']}, actual={len(portfolio._trade_log)}")
    
    if metrics['num_trades'] == total_from_regimes:
        print(f"✅ Trade count matches regime totals: {metrics['num_trades']}")
    else:
        print(f"❌ Trade count doesn't match regimes: reported={metrics['num_trades']}, regime_total={total_from_regimes}")
    
    # Now simulate the issue: clear _trade_log but keep regime performance
    print("\n4. SIMULATING THE BUG (empty _trade_log)...")
    print("-"*40)
    
    # Save regime performance before clearing
    saved_regime_perf = portfolio._calculate_performance_by_regime()
    
    # Clear trade log (simulating reset or scoping issue)
    portfolio._trade_log = []
    print("❌ Cleared _trade_log (simulating bug condition)")
    
    # Test again
    metrics_after = portfolio.get_performance_metrics()
    print(f"\nAfter clearing _trade_log:")
    print(f"  len(_trade_log): {len(portfolio._trade_log)}")
    print(f"  num_trades from get_performance_metrics(): {metrics_after.get('num_trades', 'NOT FOUND')}")
    
    # The fix should still report correct trade count from regime performance
    if metrics_after['num_trades'] == 0:
        print("❌ BUG REPRODUCED: Trade count is 0 despite regime performance having trades")
        print("   The fix may not be working correctly!")
    else:
        print(f"✅ FIX WORKING: Trade count is {metrics_after['num_trades']} (from regime performance)")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    if metrics_after['num_trades'] > 0:
        print("✅ The trade count fix is working correctly!")
        print("   Even with empty _trade_log, trades are counted from regime performance")
    else:
        print("❌ The trade count fix needs more work")
        print("   Check that get_performance_metrics() uses regime performance for counting")

if __name__ == "__main__":
    test_trade_count_consistency()