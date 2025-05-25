#!/usr/bin/env python3
"""
Run exact validation matching optimizer behavior.
This ensures we process all data with proper warmup.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.event import Event, EventType
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler

logger = logging.getLogger(__name__)

def main():
    """Run exact validation."""
    # Load config
    config = SimpleConfigLoader('config/config.yaml')
    
    # Setup logging
    setup_logging(config, cmd_log_level='INFO', optimization_mode=False)
    
    # Create container and event bus
    container = Container()
    event_bus = EventBus()
    
    # Register core services
    container.register_instance('EventBus', event_bus)
    container.register_instance('Config', config)
    
    logger.info("=== EXACT VALIDATION RUN ===")
    logger.info("Processing ALL data (training + test) with warmup")
    
    # CRITICAL: Disable adaptive mode BEFORE creating strategy
    # We do this by temporarily removing the adaptive parameters file
    adaptive_file = 'adaptive_regime_parameters.json'
    adaptive_backup = None
    if os.path.exists(adaptive_file):
        adaptive_backup = adaptive_file + '.backup'
        os.rename(adaptive_file, adaptive_backup)
        logger.info("Temporarily disabled adaptive parameters file")
    
    try:
        # Create components
        data_handler = CSVDataHandler(
            instance_name="SPY_CSV_Loader",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.data_handler_csv",
            max_bars=1000
        )
        
        strategy = EnsembleSignalStrategy(
            instance_name="SPY_Ensemble_Strategy",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.ensemble_strategy",
            container=container
        )
        
        portfolio = BasicPortfolio(
            instance_name="BasicPortfolio",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.basic_portfolio",
            container=container
        )
        
        risk_manager = BasicRiskManager(
            instance_name="BasicRiskMan1",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.basic_risk_manager",
            container=container
        )
        
        execution_handler = SimulatedExecutionHandler(
            instance_name="SimExec_1",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.simulated_execution_handler"
        )
        
        # Register all components
        container.register_instance('data_handler', data_handler)
        container.register_instance('strategy', strategy)
        container.register_instance('portfolio', portfolio)
        container.register_instance('portfolio_manager', portfolio)  # Risk manager needs this
        container.register_instance('risk_manager', risk_manager)
        container.register_instance('execution_handler', execution_handler)
        
        # Setup components
        data_handler.setup()
        strategy.setup()
        portfolio.setup()
        risk_manager.setup()
        execution_handler.setup()
        
        # Apply optimizer settings
        logger.info("=== APPLYING OPTIMIZER SETTINGS ===")
        
        # Set rule isolation mode for MA-only
        strategy.set_rule_isolation_mode('ma')
        logger.info(f"Rule isolation mode set to: MA")
        logger.info(f"Weights after isolation - MA: {strategy._ma_weight}, RSI: {strategy._rsi_weight}")
        
        # Force disable adaptive mode after setup
        if hasattr(strategy, '_adaptive_mode_enabled'):
            strategy._adaptive_mode_enabled = False
            strategy._regime_parameters = {}
            logger.info("Forced adaptive mode disabled after setup")
            
        # Verify final state
        logger.info(f"Adaptive mode enabled: {getattr(strategy, '_adaptive_mode_enabled', False)}")
        logger.info(f"MA weight: {strategy._ma_weight}, RSI weight: {strategy._rsi_weight}")
        
        # Start components
        portfolio.start()
        risk_manager.start()
        execution_handler.start()
        strategy.start()
        
        # Track signals
        signal_log = []
        def track_signal(event):
            if event.type == EventType.SIGNAL:
                signal_data = event.data_dict
                signal_log.append({
                    'timestamp': signal_data['timestamp'],
                    'type': signal_data['signal_type'],
                    'price': signal_data['price_at_signal'],
                    'reason': signal_data.get('reason', '')
                })
                logger.info(f"SIGNAL #{len(signal_log)}: {signal_data['timestamp']}, "
                           f"Type={signal_data['signal_type']}, Price={signal_data['price_at_signal']:.2f}")
        
        event_bus.subscribe(EventType.SIGNAL, track_signal)
        
        # Process ALL data (no train/test split for validation)
        logger.info("=== PROCESSING FULL DATASET ===")
        
        # Get the full dataset directly
        import pandas as pd
        full_data = pd.read_csv('data/1000_1min.csv')
        full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])
        
        # Limit to 998 bars to match optimizer
        if len(full_data) > 998:
            full_data = full_data.iloc[:998]
            
        total_bars = len(full_data)
        test_start_bar = 798
        
        logger.info(f"Total bars to process: {total_bars}")
        logger.info(f"Warmup phase: bars 0-{test_start_bar-1}")
        logger.info(f"Test phase: bars {test_start_bar}-{total_bars-1}")
        
        # Process each bar
        for bar_num, row in full_data.iterrows():
            # Log phase transition
            if bar_num == test_start_bar:
                logger.info(f"=== WARMUP COMPLETE at bar {bar_num} ===")
                logger.info(f"Starting test phase at {row['timestamp']}")
                
                # Log indicator states
                logger.info("=== INDICATOR STATES ===")
                if hasattr(strategy, '_prices'):
                    logger.info(f"Price buffer length: {len(strategy._prices)}")
                if hasattr(strategy, '_prev_short_ma') and strategy._prev_short_ma:
                    logger.info(f"MA short: {strategy._prev_short_ma:.4f}")
                if hasattr(strategy, '_prev_long_ma') and strategy._prev_long_ma:
                    logger.info(f"MA long: {strategy._prev_long_ma:.4f}")
            
            # Create bar event
            bar_data = {
                'symbol': 'SPY',
                'timestamp': row['timestamp'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            }
            
            bar_event = Event(EventType.BAR, bar_data)
            event_bus.publish(bar_event)
        
        # Log results
        logger.info("=== VALIDATION COMPLETE ===")
        logger.info(f"Total signals generated: {len(signal_log)}")
        logger.info(f"Final portfolio value: ${portfolio.get_portfolio_value():.2f}")
        
        # Only count test period signals for comparison
        test_signals = [s for i, s in enumerate(signal_log) if i >= 0]  # All signals for now
        logger.info(f"Test period signals: {len(test_signals)}")
        
        if test_signals:
            logger.info(f"First signal: {test_signals[0]['timestamp']}")
            logger.info(f"Last signal: {test_signals[-1]['timestamp']}")
            
            # Show first few signals
            logger.info("\nFirst 5 signals:")
            for i, sig in enumerate(test_signals[:5]):
                logger.info(f"  {i+1}. {sig['timestamp']}, {sig['type']}, ${sig['price']:.2f}")
        
        # Stop components
        strategy.stop()
        execution_handler.stop()
        risk_manager.stop()
        portfolio.stop()
        data_handler.stop()
        
    finally:
        # Restore adaptive file if it was moved
        if adaptive_backup and os.path.exists(adaptive_backup):
            os.rename(adaptive_backup, adaptive_file)
            logger.info("Restored adaptive parameters file")
    
    logger.info("=== VALIDATION RUN COMPLETE ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())