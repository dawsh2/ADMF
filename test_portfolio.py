#!/usr/bin/env python3
# test_portfolio.py
import os
import sys
import datetime
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.core.event import Event, EventType
from src.risk.basic_portfolio import BasicPortfolio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("test_portfolio")

# Setup test environment
config_loader = SimpleConfigLoader(config_file_path="test_config.yaml")

event_bus = EventBus()
container = Container()
container.register_instance("config_loader", config_loader)
container.register_instance("event_bus", event_bus)
container.register_instance("container", container)

# Create a dummy regime detector that can be resolved by the portfolio
class DummyRegimeDetector:
    def __init__(self):
        self.name = "test_regime_detector"
        self.current_classification = "default"
    
    def get_current_classification(self):
        return self.current_classification
    
    def set_classification(self, new_classification):
        self.current_classification = new_classification
        return self.current_classification

# Register the regime detector
dummy_regime_detector = DummyRegimeDetector()
container.register_instance("test_regime_detector", dummy_regime_detector)

# Create the portfolio
portfolio = BasicPortfolio(
    instance_name="TestPortfolio",
    config_loader=config_loader,
    event_bus=event_bus,
    container=container,
    component_config_key="components.basic_portfolio"
)

# Setup and start the portfolio
portfolio.setup()
portfolio.start()

def test_simple_trade():
    """Test a simple buy and sell sequence for accuracy in equity calculation"""
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    symbol = "SPY"
    
    # First, send a bar to establish a price
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": 100.0,
            "high": 101.0, 
            "low": 99.0,
            "close": 100.0,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    initial_value = portfolio.current_total_value
    logger.info(f"Initial portfolio value: {initial_value}")
    
    # Buy 100 shares @ 100.0
    buy_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "quantity_filled": 100,
            "fill_price": 100.0,
            "direction": "BUY",
            "commission": 5.0
        }
    )
    portfolio.on_fill(buy_fill_event)
    
    after_buy_value = portfolio.current_total_value
    logger.info(f"After buy: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, Total={after_buy_value}")
    
    # Issue a new bar to change prices
    new_timestamp = timestamp + datetime.timedelta(minutes=5)
    new_price = 105.0
    
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "open": 100.0,
            "high": 106.0, 
            "low": 99.0,
            "close": new_price,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    after_price_change_value = portfolio.current_total_value
    expected_value = portfolio.current_cash + (100 * new_price)
    
    logger.info(f"After price change: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_price_change_value}, Unrealized PnL={portfolio.unrealized_pnl}")
    logger.info(f"Expected value: {expected_value}, Difference: {after_price_change_value - expected_value}")
    
    # Sell half the position
    sell_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "quantity_filled": 50,
            "fill_price": new_price,
            "direction": "SELL",
            "commission": 2.5
        }
    )
    portfolio.on_fill(sell_fill_event)
    
    after_partial_sell_value = portfolio.current_total_value
    logger.info(f"After partial sell: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_partial_sell_value}, Unrealized PnL={portfolio.unrealized_pnl}, Realized PnL={portfolio.realized_pnl}")
    
    # Sell the rest
    sell_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "quantity_filled": 50,
            "fill_price": new_price,
            "direction": "SELL",
            "commission": 2.5
        }
    )
    portfolio.on_fill(sell_fill_event)
    
    after_full_sell_value = portfolio.current_total_value
    logger.info(f"After full sell: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_full_sell_value}, Unrealized PnL={portfolio.unrealized_pnl}, Realized PnL={portfolio.realized_pnl}")
    
    # Check expected final value
    expected_final_value = 100000.0 + (100 * (new_price - 100.0)) - 10.0  # Initial + profit - commissions
    logger.info(f"Expected final value: {expected_final_value}, Actual: {after_full_sell_value}, " +
                f"Difference: {after_full_sell_value - expected_final_value}")

def test_short_position():
    """Test a short position to check for sign errors in PnL calculations"""
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    symbol = "AAPL"
    
    # Reset portfolio
    portfolio.current_cash = portfolio.initial_cash
    portfolio.realized_pnl = 0.0
    portfolio.unrealized_pnl = 0.0
    portfolio.current_holdings_value = 0.0
    portfolio.current_total_value = portfolio.initial_cash
    portfolio.open_positions = {}
    portfolio._last_bar_prices = {}
    
    # First, send a bar to establish a price
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": 150.0,
            "high": 155.0, 
            "low": 149.0,
            "close": 150.0,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    # Short 50 shares @ 150.0
    short_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "quantity_filled": 50,
            "fill_price": 150.0,
            "direction": "SELL",
            "commission": 5.0
        }
    )
    portfolio.on_fill(short_fill_event)
    
    after_short_value = portfolio.current_total_value
    logger.info(f"After short: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_short_value}, Unrealized PnL={portfolio.unrealized_pnl}")
    
    # Issue a new bar to change prices (price drops, good for short)
    new_timestamp = timestamp + datetime.timedelta(minutes=5)
    new_price = 140.0
    
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "open": 150.0,
            "high": 151.0, 
            "low": 139.0,
            "close": new_price,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    after_price_drop_value = portfolio.current_total_value
    logger.info(f"After price drop: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_price_drop_value}, Unrealized PnL={portfolio.unrealized_pnl}")
    
    # Cover the short position
    cover_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "quantity_filled": 50,
            "fill_price": new_price,
            "direction": "BUY",
            "commission": 5.0
        }
    )
    portfolio.on_fill(cover_fill_event)
    
    after_cover_value = portfolio.current_total_value
    logger.info(f"After covering short: Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Total={after_cover_value}, Unrealized PnL={portfolio.unrealized_pnl}, Realized PnL={portfolio.realized_pnl}")
    
    # Check expected final value
    expected_short_profit = 50 * (150.0 - 140.0)  # Quantity * (short price - cover price)
    expected_final_value = 100000.0 + expected_short_profit - 10.0  # Initial + profit - commissions
    logger.info(f"Expected final value: {expected_final_value}, Actual: {after_cover_value}, " +
                f"Difference: {after_cover_value - expected_final_value}")

def test_regime_change():
    """Test portfolio value updates during regime changes"""
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    symbol = "TSLA"
    
    # Reset portfolio
    portfolio.current_cash = portfolio.initial_cash
    portfolio.realized_pnl = 0.0
    portfolio.unrealized_pnl = 0.0
    portfolio.current_holdings_value = 0.0
    portfolio.current_total_value = portfolio.initial_cash
    portfolio.open_positions = {}
    portfolio._last_bar_prices = {}
    
    # First, send a bar to establish a price
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": 200.0,
            "high": 205.0, 
            "low": 198.0,
            "close": 200.0,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    # Buy 50 shares in default regime
    buy_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "quantity_filled": 50,
            "fill_price": 200.0,
            "direction": "BUY",
            "commission": 5.0
        }
    )
    portfolio.on_fill(buy_fill_event)
    
    logger.info(f"After buy in default regime: Equity={portfolio.current_total_value}, " +
                f"Regime={portfolio._current_market_regime}")
    
    # Change regime to "volatile"
    dummy_regime_detector.set_classification("volatile")
    
    classification_event = Event(
        EventType.CLASSIFICATION,
        {
            "classification": "volatile",
            "timestamp": timestamp + datetime.timedelta(minutes=5)
        }
    )
    portfolio.on_classification_change(classification_event)
    
    # Update with a new price in the new regime
    new_timestamp = timestamp + datetime.timedelta(minutes=10)
    
    bar_event = Event(
        EventType.BAR,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "open": 200.0,
            "high": 220.0, 
            "low": 198.0,
            "close": 210.0,
            "volume": 1000
        }
    )
    portfolio.on_bar(bar_event)
    
    logger.info(f"After regime change to 'volatile': Equity={portfolio.current_total_value}, " +
                f"Position's current segment regime={portfolio.open_positions[symbol]['current_segment_regime']}")
    
    # Buy more in the new regime
    buy_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "quantity_filled": 50,
            "fill_price": 210.0,
            "direction": "BUY",
            "commission": 5.0
        }
    )
    portfolio.on_fill(buy_fill_event)
    
    position_data = portfolio.open_positions.get(symbol, {})
    logger.info(f"After buying more in 'volatile' regime: Equity={portfolio.current_total_value}, " +
                f"Total quantity={position_data.get('quantity')}, " +
                f"Avg entry price={position_data.get('avg_entry_price')}, " +
                f"Current segment entry price={position_data.get('current_segment_entry_price')}")
    
    # Sell everything
    sell_fill_event = Event(
        EventType.FILL,
        {
            "symbol": symbol,
            "timestamp": new_timestamp,
            "quantity_filled": 100, # Total position
            "fill_price": 210.0,
            "direction": "SELL",
            "commission": 5.0
        }
    )
    portfolio.on_fill(sell_fill_event)
    
    logger.info(f"After selling all: Equity={portfolio.current_total_value}, " +
                f"Cash={portfolio.current_cash}, Holdings={portfolio.current_holdings_value}, " +
                f"Realized PnL={portfolio.realized_pnl}")

if __name__ == "__main__":
    logger.info("=== SIMPLE TRADE TEST ===")
    test_simple_trade()
    
    logger.info("\n=== SHORT POSITION TEST ===")
    test_short_position()
    
    logger.info("\n=== REGIME CHANGE TEST ===")
    test_regime_change()

    # Print final summary
    portfolio._log_final_performance_summary()