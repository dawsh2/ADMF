#!/usr/bin/env python3
"""
Run production with manual warmup by processing training data bars before test data
"""
import sys
import logging
import pandas as pd
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import ServiceContainer
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.core.event import EventType, Event

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/production_warmup_test.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting production run with manual warmup...")
    
    # Load configuration
    config_loader = SimpleConfigLoader('config/config_adaptive_production.yaml')
    
    # Create core components
    event_bus = EventBus()
    container = ServiceContainer(config_loader, event_bus)
    
    # Register components
    container.bootstrap_core_components()
    container.start_all()
    
    # Get data handler and load data
    data_handler = container.resolve('data_handler')
    
    # Load the full dataset first
    logger.info("Loading full dataset...")
    df = pd.read_csv("data/1000_1min.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Split data manually
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    logger.info(f"Dataset split - Training: {len(train_df)} bars, Test: {len(test_df)} bars")
    logger.info(f"Test data range: {test_df.iloc[0]['timestamp']} to {test_df.iloc[-1]['timestamp']}")
    
    # WARMUP PHASE: Process last 30 training bars
    warmup_bars = 30
    warmup_df = train_df.iloc[-warmup_bars:]
    
    logger.info(f"=== WARMUP PHASE: Processing {len(warmup_df)} training bars ===")
    logger.info(f"Warmup range: {warmup_df.iloc[0]['timestamp']} to {warmup_df.iloc[-1]['timestamp']}")
    
    # Get strategy
    strategy = container.resolve('strategy')
    
    # Process warmup bars (no trading, just indicator updates)
    for idx, row in warmup_df.iterrows():
        data = {
            'timestamp': row['timestamp'],
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
        # Create market data event
        event = Event(EventType.MARKET_DATA, data)
        event_bus.publish(event)
    
    logger.info("=== WARMUP COMPLETE - Starting test phase ===")
    
    # Get portfolio for tracking
    portfolio = container.resolve('portfolio')
    
    # TEST PHASE: Process test data with trading enabled
    total_signals = 0
    bar_count = 0
    
    for idx, row in test_df.iterrows():
        bar_count += 1
        data = {
            'timestamp': row['timestamp'],
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        }
        
        # Create and publish market data event
        event = Event(EventType.MARKET_DATA, data)
        event_bus.publish(event)
        
        # Log progress periodically
        if bar_count % 50 == 0:
            equity = portfolio.get_equity(row['Close'])
            logger.info(f"Progress: {bar_count}/{len(test_df)} bars, Equity: ${equity:,.2f}")
    
    # Final results
    final_price = test_df.iloc[-1]['Close']
    final_equity = portfolio.get_equity(final_price)
    total_return = ((final_equity - 100000) / 100000) * 100
    
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Total bars processed: {bar_count}")
    logger.info(f"Final equity: ${final_equity:,.2f}")
    logger.info(f"Total return: {total_return:.2f}%")
    logger.info(f"Total trades: {portfolio.get_trade_count()}")
    
    # Cleanup
    container.stop_all()

if __name__ == "__main__":
    main()