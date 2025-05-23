#!/usr/bin/env python3
"""Test that portfolio cash accounting is fixed"""

from src.risk.basic_portfolio import BasicPortfolio
from src.core.event_bus import EventBus
from src.core.event import Event, EventType
from src.core.config import SimpleConfigLoader
import datetime

# Create a minimal portfolio
event_bus = EventBus()
config = SimpleConfigLoader({'components': {'basic_portfolio': {'initial_cash': 100000}}})
portfolio = BasicPortfolio("TestPortfolio", config, event_bus, "components.basic_portfolio")
portfolio.setup()
portfolio.start()

# Initial state
print(f"Initial cash: ${portfolio.current_cash:,.2f}")
print(f"Initial portfolio value: ${portfolio.current_total_value:,.2f}")

# Simulate buying 100 shares at $500
buy_fill = {
    'symbol': 'TEST',
    'timestamp': datetime.datetime.now(),
    'quantity_filled': 100,
    'fill_price': 500.0,
    'direction': 'BUY',
    'commission': 0.5
}
event_bus.publish(Event(EventType.FILL, buy_fill))

print(f"\nAfter BUY 100 @ $500:")
print(f"Cash: ${portfolio.current_cash:,.2f} (should be ~$49,999.50)")
print(f"Expected: $100,000 - $50,000 - $0.50 = $49,999.50")

# Simulate selling 100 shares at $510 (closing the position)
sell_fill = {
    'symbol': 'TEST',
    'timestamp': datetime.datetime.now(),
    'quantity_filled': 100,
    'fill_price': 510.0,
    'direction': 'SELL',
    'commission': 0.5
}
event_bus.publish(Event(EventType.FILL, sell_fill))

print(f"\nAfter SELL 100 @ $510 (closing position):")
print(f"Cash: ${portfolio.current_cash:,.2f} (should be ~$100,999)")
print(f"Expected: $49,999.50 + $51,000 - $0.50 = $100,999")
print(f"Realized P&L: ${portfolio.realized_pnl:,.2f} (should be $1,000)")
print(f"Portfolio value: ${portfolio.current_total_value:,.2f}")

# Check the math
net_profit = portfolio.current_cash - 100000
print(f"\nNet profit from cash: ${net_profit:,.2f}")
print(f"Commission paid: $1.00")
print(f"Gross profit: ${net_profit + 1:,.2f} (should match realized P&L)")

if abs(portfolio.realized_pnl - (net_profit + 1)) < 0.01:
    print("\n✓ Portfolio math is CORRECT!")
else:
    print("\n✗ Portfolio math is BROKEN!")
    print(f"  Realized P&L: ${portfolio.realized_pnl:,.2f}")
    print(f"  Cash-based profit: ${net_profit + 1:,.2f}")