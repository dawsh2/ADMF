#!/usr/bin/env python3
"""Test script to verify refactored components work with Bootstrap system."""

import sys
import logging
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import Bootstrap
from src.core.config import SimpleConfigLoader
from src.core.event import Event, EventType
import datetime

def test_bootstrap_with_refactored_components():
    """Test the Bootstrap system with our refactored components."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("test_bootstrap")
    
    try:
        # Create test configuration
        test_config = {
            "components": {
                "csv_data_handler": {
                    "class": "src.data.csv_data_handler.CSVDataHandler",
                    "config": {
                        "symbol": "SPY",
                        "csv_file_path": "data/SPY_1min.csv",
                        "timestamp_column": "timestamp",
                        "train_test_split_ratio": 0.8
                    }
                },
                "portfolio": {
                    "class": "src.risk.basic_portfolio.BasicPortfolio",
                    "config": {
                        "initial_cash": 100000.0,
                        "regime_detector_service_name": "regime_detector"
                    },
                    "dependencies": ["csv_data_handler"]
                },
                "ma_strategy": {
                    "class": "src.strategy.ma_strategy.MAStrategy",
                    "config": {
                        "symbol": "SPY",
                        "short_window_default": 10,
                        "long_window_default": 20
                    },
                    "dependencies": ["csv_data_handler", "portfolio"]
                }
            }
        }
        
        # Create a temporary config file
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        logger.info(f"Created test config at: {config_path}")
        
        # Load configuration
        config_loader = SimpleConfigLoader(config_path)
        
        # Initialize Bootstrap
        logger.info("Initializing Bootstrap system...")
        bootstrap = Bootstrap()
        
        # Initialize the system
        logger.info("Initializing components...")
        context = bootstrap.initialize(config_loader)
        
        logger.info("Bootstrap initialization successful!")
        logger.info(f"Event bus: {context.event_bus}")
        logger.info(f"Container: {context.container}")
        logger.info(f"Logger: {context.logger}")
        
        # Verify components were created
        csv_handler = context.container.resolve("csv_data_handler")
        portfolio = context.container.resolve("portfolio")
        strategy = context.container.resolve("ma_strategy")
        
        logger.info(f"CSV Handler: {csv_handler}")
        logger.info(f"Portfolio: {portfolio}")
        logger.info(f"Strategy: {strategy}")
        
        # Test component lifecycle
        logger.info("\nTesting component lifecycle...")
        
        # Setup components
        logger.info("Setting up components...")
        csv_handler.setup()
        portfolio.setup()
        strategy.setup()
        
        # Check states
        logger.info(f"CSV Handler state: {csv_handler._state}")
        logger.info(f"Portfolio state: {portfolio._state}")
        logger.info(f"Strategy state: {strategy._state}")
        
        # Start components
        logger.info("\nStarting components...")
        csv_handler.set_active_dataset("train")
        csv_handler.start()
        portfolio.start()
        strategy.start()
        
        # Test event publishing
        logger.info("\nTesting event publishing...")
        test_bar_event = Event(EventType.BAR, {
            "symbol": "SPY",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000
        })
        
        context.event_bus.publish(test_bar_event)
        logger.info("Published test BAR event")
        
        # Stop components
        logger.info("\nStopping components...")
        strategy.stop()
        portfolio.stop()
        csv_handler.stop()
        
        # Dispose components
        logger.info("\nDisposing components...")
        strategy.dispose()
        portfolio.dispose()
        csv_handler.dispose()
        
        logger.info("\nTest completed successfully!")
        
        # Clean up
        import os
        os.unlink(config_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_bootstrap_with_refactored_components()
    sys.exit(0 if success else 1)