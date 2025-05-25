#!/usr/bin/env python3
"""
Modified main.py to pre-warm indicators like the optimizer does.
This processes the last 20 bars of training data before running test.
"""

import sys
import argparse
from src.core.event import Event
from src.data.csv_data_handler import CSVDataHandler

# Import the existing main.py functionality
from main import (
    parse_arguments, setup_components, setup_logging,
    DI_Container, EventBus, Config
)

def pre_warm_indicators(container, logger, warmup_bars=20):
    """Pre-warm indicators by processing the last bars from training data."""
    try:
        # Get components
        data_handler = container.resolve("data_handler_service")
        strategy = container.resolve("strategy")
        event_bus = container.resolve("event_bus")
        regime_detector = container.resolve("MyPrimaryRegimeDetector")
        
        logger.info(f"Starting indicator pre-warming with {warmup_bars} bars from training data...")
        
        # Get train/test data
        if not hasattr(data_handler, '_train_df') or data_handler._train_df is None:
            logger.warning("No training data available for warmup")
            return
            
        train_df = data_handler._train_df
        
        # Take last warmup_bars from training data
        warmup_data = train_df.iloc[-warmup_bars:].copy()
        logger.info(f"Using bars from {warmup_data.index[0]} to {warmup_data.index[-1]} for warmup")
        
        # Process warmup bars
        bar_count = 0
        for idx, row in warmup_data.iterrows():
            bar_count += 1
            
            # Create BAR event
            bar_event = Event(
                event_type="BAR",
                payload={
                    "symbol": data_handler.symbol,
                    "timestamp": idx,
                    "open": row[data_handler.open_column],
                    "high": row[data_handler.high_column],
                    "low": row[data_handler.low_column],
                    "close": row[data_handler.close_column],
                    "volume": row[data_handler.volume_column] if data_handler.volume_column else 0
                }
            )
            
            # Publish event to warm up indicators
            event_bus.publish(bar_event)
            
            if bar_count % 5 == 0:
                logger.debug(f"Processed {bar_count}/{warmup_bars} warmup bars")
        
        logger.info(f"✓ Pre-warming complete. Processed {bar_count} bars.")
        logger.info("Indicators are now warmed up and ready for test data.")
        
        # Log current indicator states
        if hasattr(strategy, '_short_ma_series') and strategy._short_ma_series:
            logger.info(f"Short MA has {len(strategy._short_ma_series)} values")
        if hasattr(strategy, '_long_ma_series') and strategy._long_ma_series:
            logger.info(f"Long MA has {len(strategy._long_ma_series)} values")
            
    except Exception as e:
        logger.error(f"Error during pre-warming: {e}", exc_info=True)

def main():
    """Modified main function with indicator pre-warming."""
    args = parse_arguments()
    
    # Setup logging
    logger, debug_logger = setup_logging(args.log_level, args.debug_log)
    
    # Load configuration
    config = Config(args.config)
    logger.info(f"Attempting to load configuration from: {args.config}")
    
    # Initialize DI container and event bus
    container = DI_Container()
    event_bus = EventBus()
    container.register_singleton("event_bus", event_bus)
    container.register_singleton("config", config)
    
    # Setup all components
    components = setup_components(container, config, args, logger)
    
    # Extract key components
    data_handler = components['data_handler']
    strategy = components['strategy']
    portfolio_manager = components['portfolio_manager']
    
    # Start components in proper order
    # (Same as original main.py startup sequence)
    regime_detector = container.resolve("MyPrimaryRegimeDetector")
    execution_handler = container.resolve("execution_handler_service")
    risk_manager = container.resolve("risk_manager_service")
    signal_logger_comp = container.resolve("signal_logger_service")
    order_logger_comp = container.resolve("order_logger_service")
    
    # Setup and start components
    for comp in [regime_detector, execution_handler, risk_manager, strategy, 
                 portfolio_manager, data_handler, signal_logger_comp, order_logger_comp]:
        if hasattr(comp, 'setup'):
            comp.setup()
        if hasattr(comp, 'start'):
            comp.start()
    
    # CRITICAL: Pre-warm indicators before processing test data
    logger.info("\n" + "="*60)
    logger.info("PRE-WARMING INDICATORS TO MATCH OPTIMIZER")
    logger.info("="*60)
    pre_warm_indicators(container, logger, warmup_bars=20)
    logger.info("="*60 + "\n")
    
    # Now switch to test data and run normally
    if hasattr(data_handler, 'set_active_dataset'):
        logger.info("Switching to test dataset for evaluation...")
        data_handler.set_active_dataset("test")
    
    # Continue with normal data processing
    logger.info("Starting test data processing with pre-warmed indicators...")
    # The data_handler.start() was already called above, so test data will be processed
    
    # Rest of shutdown sequence (same as original)
    # ... (shutdown code remains the same)

if __name__ == "__main__":
    main()