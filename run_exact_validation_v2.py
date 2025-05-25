#!/usr/bin/env python3
"""
Run exact validation matching optimizer behavior - Version 2.
Process ALL data like optimizer does.
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
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler

logger = logging.getLogger(__name__)

def main():
    """Run exact validation matching optimizer behavior."""
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
    
    logger.info("=== EXACT VALIDATION RUN V2 ===")
    logger.info("Processing ALL 998 bars like optimizer")
    
    # Keep adaptive parameters file active to match optimizer
    logger.info("Keeping adaptive_regime_parameters.json active for regime-specific parameters")
    
    try:
        # Create regime detector for adaptive mode
        from src.strategy.regime_detector import RegimeDetector
        regime_detector = RegimeDetector(
            instance_name="MyPrimaryRegimeDetector",
            config_loader=config,
            event_bus=event_bus,
            component_config_key="components.MyPrimaryRegimeDetector"
        )
        
        # Create components (NO data handler - we'll feed data manually)
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
        container.register_instance('MyPrimaryRegimeDetector', regime_detector)
        container.register_instance('strategy', strategy)
        container.register_instance('portfolio', portfolio)
        container.register_instance('portfolio_manager', portfolio)  # Risk manager needs this
        container.register_instance('risk_manager', risk_manager)
        container.register_instance('execution_handler', execution_handler)
        
        # Setup components (NO data handler)
        regime_detector.setup()
        strategy.setup()
        portfolio.setup()
        risk_manager.setup()
        execution_handler.setup()
        
        # Apply optimizer settings
        logger.info("=== APPLYING OPTIMIZER SETTINGS ===")
        
        # Set rule isolation mode for MA-only
        strategy.set_rule_isolation_mode('ma')
        logger.info(f"Rule isolation mode set to: MA")
        
        # CRITICAL: Set weights to match optimizer behavior
        # When optimize_ma is used, weights are set to MA=0.8, RSI=0.2
        # But with RSI disabled, effective weight is MA=1.0
        strategy._ma_weight = 0.8
        strategy._rsi_weight = 0.2
        logger.info(f"Set weights to match optimizer: MA={strategy._ma_weight}, RSI={strategy._rsi_weight}")
        
        # Keep adaptive mode ENABLED to match optimizer
        if hasattr(strategy, '_adaptive_mode_enabled'):
            logger.info(f"Adaptive mode status: {strategy._adaptive_mode_enabled}")
            if strategy._adaptive_mode_enabled:
                logger.info("Keeping adaptive mode ENABLED to match optimizer")
                if hasattr(strategy, '_regime_best_parameters'):
                    logger.info(f"Loaded parameters for regimes: {list(strategy._regime_best_parameters.keys())}")
                else:
                    logger.info("Regime parameters loaded")
            else:
                logger.info("WARNING: Adaptive mode is disabled, but optimizer uses it!")
            
        # Verify final state
        logger.info(f"Adaptive mode: {getattr(strategy, '_adaptive_mode_enabled', False)}")
        logger.info(f"Final weights - MA: {strategy._ma_weight}, RSI: {strategy._rsi_weight}")
        
        # Start components
        regime_detector.start()
        portfolio.start()
        risk_manager.start()
        execution_handler.start()
        strategy.start()
        
        # Track signals
        signal_log = []
        test_signal_log = []
        current_bar = 0  # Initialize here
        test_start_timestamp = None  # Will be set when we reach bar 798
        
        def track_signal(event):
            nonlocal current_bar, test_start_timestamp  # Access the outer scope variables
            if event.type == EventType.SIGNAL:
                signal_data = event.data_dict
                signal_info = {
                    'bar': current_bar,
                    'timestamp': signal_data['timestamp'],
                    'type': signal_data['signal_type'],
                    'price': signal_data['price_at_signal'],
                    'reason': signal_data.get('reason', '')
                }
                signal_log.append(signal_info)
                
                # Track test signals by timestamp
                if test_start_timestamp and pd.to_datetime(signal_data['timestamp']) >= test_start_timestamp:
                    test_signal_log.append(signal_info)
                    logger.info(f"TEST SIGNAL #{len(test_signal_log)}: Bar {current_bar}, {signal_data['timestamp']}, "
                               f"Type={signal_data['signal_type']}, Price={signal_data['price_at_signal']:.2f}")
        
        event_bus.subscribe(EventType.SIGNAL, track_signal)
        
        # Load and process ALL data
        logger.info("=== PROCESSING FULL DATASET ===")
        
        import pandas as pd
        full_data = pd.read_csv('data/1000_1min.csv')
        full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])
        
        # Limit to 998 bars to match optimizer
        if len(full_data) > 998:
            full_data = full_data.iloc[:998]
            
        total_bars = len(full_data)
        test_start_bar = 798
        
        logger.info(f"Total bars: {total_bars}")
        logger.info(f"Training bars: 0-{test_start_bar-1} ({test_start_bar} bars)")
        logger.info(f"Test bars: {test_start_bar}-{total_bars-1} ({total_bars-test_start_bar} bars)")
        
        # Process each bar
        for bar_index, row in full_data.iterrows():
            current_bar = bar_index
            
            # Log phase transition
            if bar_index == test_start_bar:
                test_start_timestamp = row['timestamp']  # Set the test start timestamp
                logger.info(f"\n=== ENTERING TEST PHASE at bar {bar_index} ===")
                logger.info(f"Test start timestamp: {row['timestamp']}")
                logger.info(f"Training signals so far: {len(signal_log)}")
                
                # Log indicator states
                logger.info("=== INDICATOR STATES AT TEST START ===")
                if hasattr(strategy, '_prices'):
                    logger.info(f"Price buffer length: {len(strategy._prices)}")
                    if len(strategy._prices) >= 5:
                        logger.info(f"Last 5 prices: {[f'{p:.2f}' for p in list(strategy._prices)[-5:]]}")
                if hasattr(strategy, '_prev_short_ma') and strategy._prev_short_ma:
                    logger.info(f"MA short (10): {strategy._prev_short_ma:.4f}")
                if hasattr(strategy, '_prev_long_ma') and strategy._prev_long_ma:
                    logger.info(f"MA long (20): {strategy._prev_long_ma:.4f}")
            
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
        logger.info("\n=== VALIDATION COMPLETE ===")
        logger.info(f"Total bars processed: {total_bars}")
        logger.info(f"Total signals: {len(signal_log)}")
        logger.info(f"Training signals: {len(signal_log) - len(test_signal_log)}")
        logger.info(f"Test signals: {len(test_signal_log)}")
        logger.info(f"Final portfolio value: ${portfolio.get_portfolio_value():.2f}")
        
        # Compare with expected
        logger.info("\n=== COMPARISON WITH OPTIMIZER ===")
        logger.info(f"Expected test signals: 17")
        logger.info(f"Actual test signals: {len(test_signal_log)}")
        logger.info(f"Match: {'YES ✓' if len(test_signal_log) == 17 else 'NO ✗'}")
        
        if test_signal_log:
            logger.info(f"\nFirst test signal: Bar {test_signal_log[0]['bar']}, {test_signal_log[0]['timestamp']}")
            logger.info(f"Last test signal: Bar {test_signal_log[-1]['bar']}, {test_signal_log[-1]['timestamp']}")
            
            # Show all test signals for debugging
            logger.info("\nAll test signals:")
            for i, sig in enumerate(test_signal_log):
                logger.info(f"  {i+1}. Bar {sig['bar']}, {sig['timestamp']}, "
                           f"Type={'BUY' if sig['type'] == 1 else 'SELL'}, Price=${sig['price']:.2f}")
        
        # Stop components
        strategy.stop()
        execution_handler.stop()
        risk_manager.stop()
        portfolio.stop()
        
    finally:
        pass  # No need to restore since we didn't move the file
    
    logger.info("\n=== VALIDATION RUN COMPLETE ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())