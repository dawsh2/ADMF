#!/usr/bin/env python3
# test_portfolio_reset_simple.py
"""
A simplified test script to verify that the portfolio reset functionality works correctly.
This script directly tests the BasicPortfolio class without needing the full component stack.
"""

import logging
import datetime
import sys
from typing import Dict, Any, List, Optional

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.risk.basic_portfolio import BasicPortfolio

def setup_logging():
    """Configure logging for the test."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("test_portfolio_reset_simple.log")
        ]
    )
    return logging.getLogger("PortfolioResetSimpleTest")

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("Starting simplified portfolio reset test")
    
    # Initialize core components
    config_loader = SimpleConfigLoader("config/config.yaml")
    container = Container()
    event_bus = EventBus()
    
    # Register core components
    container.register_instance("config", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create portfolio with test configuration
    test_config = {
        "components": {
            "test_portfolio": {
                "initial_cash": 100000.0
            }
        }
    }
    
    # Create portfolio manager
    logger.info("Creating BasicPortfolio instance")
    portfolio = BasicPortfolio(
        "TestPortfolio", config_loader, event_bus, container, "basic_portfolio"
    )
    
    # Test initial state
    logger.info("=== Initial Portfolio State ===")
    logger.info(f"Initial cash: {portfolio.initial_cash:.2f}")
    logger.info(f"Current cash: {portfolio.current_cash:.2f}")
    logger.info(f"Current total value: {portfolio.current_total_value:.2f}")
    logger.info(f"Open positions: {portfolio.open_positions}")
    logger.info(f"Realized P&L: {portfolio.realized_pnl:.2f}")
    
    # Manually modify portfolio state to simulate trading
    logger.info("=== Modifying Portfolio State ===")
    portfolio.current_cash = 50000.0
    portfolio.realized_pnl = 10000.0
    portfolio.current_total_value = 110000.0
    
    # Add some fake positions
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    portfolio.open_positions["SPY"] = {
        'quantity': 100.0,
        'cost_basis': 50000.0,
        'avg_entry_price': 500.0,
        'entry_timestamp': timestamp,
        'trade_id': "test_trade_123",
        'current_segment_entry_price': 500.0,
        'current_segment_regime': "trending_up",
        'initial_entry_regime_for_trade': "trending_up"
    }
    
    # Log modified state
    logger.info("Portfolio after simulated trading:")
    logger.info(f"Current cash: {portfolio.current_cash:.2f}")
    logger.info(f"Current total value: {portfolio.current_total_value:.2f}")
    logger.info(f"Open positions: {len(portfolio.open_positions)} positions")
    logger.info(f"Realized P&L: {portfolio.realized_pnl:.2f}")
    
    # Reset portfolio
    logger.info("=== Resetting Portfolio ===")
    portfolio.reset()
    
    # Check state after reset
    logger.info("=== Portfolio State After Reset ===")
    logger.info(f"Initial cash: {portfolio.initial_cash:.2f}")
    logger.info(f"Current cash: {portfolio.current_cash:.2f}")
    logger.info(f"Current total value: {portfolio.current_total_value:.2f}")
    logger.info(f"Open positions: {portfolio.open_positions}")
    logger.info(f"Realized P&L: {portfolio.realized_pnl:.2f}")
    
    # Verify reset was successful
    if portfolio.current_cash == portfolio.initial_cash and not portfolio.open_positions:
        logger.info("TEST RESULT: PASSED - Portfolio was successfully reset to initial state")
    else:
        logger.info("TEST RESULT: FAILED - Portfolio was not properly reset")
    
    logger.info("Simplified portfolio reset test completed")

if __name__ == "__main__":
    main()