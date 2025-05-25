#!/usr/bin/env python3
"""
Modified main.py that pre-warms indicators to match optimizer behavior.
"""

import sys
import logging
import argparse
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Import the original main components
from src.core.logging_setup import setup_logging
from src.core.config import SimpleConfigLoader, SharedConfig
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.exceptions import (
    ComponentError, DataError, ValidationError, ConfigurationError, ErrorContext, ErrorSeverity
)

from src.core.event import EventType
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.dummy_component import DummyComponent

logger = logging.getLogger(__name__)

def prewarm_strategy_indicators(strategy, data_file, split_ratio):
    """Pre-warm strategy indicators using training data."""
    # Load data
    df = pd.read_csv(data_file)
    split_idx = int(len(df) * split_ratio)
    train_data = df[:split_idx]
    
    if len(train_data) < 20:  # Need at least long_window bars
        logger.warning("Not enough training data to pre-warm indicators")
        return
    
    # Extract prices for warming up
    prices = train_data['Close'].values
    
    # Pre-warm the MA indicators by setting their internal buffers
    # This simulates what happens in the optimizer
    if hasattr(strategy, '_ma_short_buffer') and hasattr(strategy, '_ma_long_buffer'):
        # Set the buffers to the last N prices from training
        strategy._ma_short_buffer = list(prices[-strategy.short_window:])
        strategy._ma_long_buffer = list(prices[-strategy.long_window:])
        
        # Calculate and set the current MA values
        strategy._ma_short = np.mean(strategy._ma_short_buffer)
        strategy._ma_long = np.mean(strategy._ma_long_buffer)
        
        logger.info(f"Pre-warmed MA indicators: MA_short={strategy._ma_short:.4f}, MA_long={strategy._ma_long:.4f}")
        logger.info(f"Buffer sizes: short={len(strategy._ma_short_buffer)}, long={len(strategy._ma_long_buffer)}")
    else:
        logger.warning("Strategy doesn't have expected MA buffer attributes")

def main(args):
    """Enhanced main function with indicator pre-warming."""
    try:
        # Load configuration
        config_path = args.config if args.config else "config/config.yaml"
        logger.info(f"Attempting to load configuration from: {config_path}")
        config_loader = SimpleConfigLoader(config_file_path=config_path)
        config = config_loader.load()
        
        # Setup logging
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_dir = 'logs'
        additional_setup_logging(log_level, log_dir, args.verbose, args.debug)
        
        # Initialize core components
        container = Container()
        event_bus = EventBus()
        container.register('EventBus', event_bus)
        container.register('Config', config_loader)
        
        # Register strategies and components (same as original main.py)
        # ... (copy the registration logic from original main.py) ...
        
        # After strategy is created and set up, pre-warm its indicators
        strategy = container.resolve('strategy')
        data_config = config.get('components', {}).get('data_handler_csv', {})
        csv_file = data_config.get('csv_file_path', 'data/SPY_1min.csv')
        split_ratio = data_config.get('train_test_split_ratio', 0.8)
        
        logger.info("Pre-warming strategy indicators from training data...")
        prewarm_strategy_indicators(strategy, csv_file, split_ratio)
        
        # Continue with normal execution
        # ... (rest of main.py logic) ...
        
    except Exception as e:
        logger.critical(f"CRITICAL: Unhandled exception in main - {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ADMF Trading System with Pre-warming')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--bars', type=int, help='Limit number of bars to process')
    
    args = parser.parse_args()
    sys.exit(main(args))