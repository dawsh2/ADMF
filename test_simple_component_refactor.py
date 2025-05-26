#!/usr/bin/env python3
"""Simple test of refactored components without full Bootstrap."""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.component_base import ComponentBase
from src.core.event_bus import EventBus
from src.core.event import Event, EventType
import datetime

def test_refactored_components():
    """Test the refactored components directly."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("test_components")
    
    try:
        # Create mock dependencies
        event_bus = EventBus()
        container = MagicMock()
        config_loader = MagicMock()
        
        # Mock config loader responses - return config regardless of key
        config_loader.get_component_config = MagicMock(return_value={
            "symbol": "SPY",
            "csv_file_path": "data/SPY_1min.csv",
            "timestamp_column": "timestamp",
            "initial_cash": 100000.0,
            "short_window_default": 10,
            "long_window_default": 20
        })
        
        # Mock full config object
        config = {
            "components": {
                "csv_data_handler": {
                    "symbol": "SPY",
                    "csv_file_path": "data/SPY_1min.csv",
                    "timestamp_column": "timestamp"
                },
                "portfolio": {
                    "initial_cash": 100000.0
                },
                "ma_strategy": {
                    "symbol": "SPY",
                    "short_window_default": 10,
                    "long_window_default": 20
                }
            }
        }
        
        # Create context object with config
        context = {
            'event_bus': event_bus,
            'container': container,
            'config_loader': config_loader,
            'config': config,
            'logger': logger
        }
        
        logger.info("Testing ComponentBase lifecycle...")
        
        # Test CSVDataHandler
        logger.info("\n=== Testing CSVDataHandler ===")
        from src.data.csv_data_handler import CSVDataHandler
        
        csv_handler = CSVDataHandler("csv_data_handler", config_key="csv_data_handler")
        logger.info(f"Created: {csv_handler}")
        
        csv_handler.initialize(context)
        logger.info(f"Initialized: {csv_handler.initialized}")
        
        # Test BasicPortfolio
        logger.info("\n=== Testing BasicPortfolio ===")
        from src.risk.basic_portfolio import BasicPortfolio
        
        portfolio = BasicPortfolio("portfolio", config_key="portfolio")
        logger.info(f"Created: {portfolio}")
        
        portfolio.initialize(context)
        logger.info(f"Initialized: {portfolio.initialized}")
        
        # Test MAStrategy
        logger.info("\n=== Testing MAStrategy ===")
        from src.strategy.ma_strategy import MAStrategy
        
        strategy = MAStrategy("ma_strategy", config_key="ma_strategy")
        logger.info(f"Created: {strategy}")
        
        strategy.initialize(context)
        logger.info(f"Initialized: {strategy.initialized}")
        
        # Test lifecycle methods
        logger.info("\n=== Testing Lifecycle Methods ===")
        
        # Test setup
        logger.info("Setup phase...")
        # Note: We can't test csv_handler.setup() without actual CSV file
        portfolio.setup()
        logger.info(f"Portfolio after setup: initialized={portfolio.initialized}")
        
        strategy.setup()
        logger.info(f"Strategy after setup: initialized={strategy.initialized}")
        
        # Test start
        logger.info("\nStart phase...")
        portfolio.start()
        logger.info(f"Portfolio after start: running={portfolio.running}")
        
        strategy.start()
        logger.info(f"Strategy after start: running={strategy.running}")
        
        # Test event handling
        logger.info("\n=== Testing Event Handling ===")
        test_bar = Event(EventType.BAR, {
            "symbol": "SPY",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000
        })
        
        logger.info("Publishing test BAR event...")
        event_bus.publish(test_bar)
        
        # Test stop
        logger.info("\nStop phase...")
        strategy.stop()
        logger.info(f"Strategy after stop: running={strategy.running}")
        
        portfolio.stop()
        logger.info(f"Portfolio after stop: running={portfolio.running}")
        
        # Test teardown
        logger.info("\nTeardown phase...")
        strategy.teardown()
        logger.info(f"Strategy after teardown: initialized={strategy.initialized}")
        
        portfolio.teardown()
        logger.info(f"Portfolio after teardown: initialized={portfolio.initialized}")
        
        logger.info("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_refactored_components()
    sys.exit(0 if success else 1)