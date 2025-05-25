#!/usr/bin/env python3
"""
Exact validation script that reproduces optimizer behavior.
This script ensures we can exactly match optimizer results.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Optional

# Import required components
from src.core.logging_setup import setup_logging
from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.event import Event, EventType
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.dummy_component import DummyComponent

logger = logging.getLogger(__name__)

class ValidationOrchestrator:
    """Orchestrates exact reproduction of optimizer behavior."""
    
    def __init__(self, config_path: str, optimization_mode: str = 'ma'):
        self.config_path = config_path
        self.optimization_mode = optimization_mode
        self.config_loader = None
        self.container = None
        self.event_bus = None
        self.components = {}
        self.signal_log = []
        self.warmup_complete = False
        self.test_start_bar = 798  # Bar where test period starts
        self.current_bar = 0
        
    def setup(self):
        """Setup all components matching optimizer initialization."""
        logger.info("=== VALIDATION SETUP ===")
        
        # Load configuration first
        self.config_loader = SimpleConfigLoader(config_file_path=self.config_path)
        
        # Setup logging with config loader
        setup_logging(self.config_loader, cmd_log_level='INFO', optimization_mode=False)
        
        # Initialize core components
        self.container = Container()
        self.event_bus = EventBus()
        
        # Register core services
        self.container.register_instance('EventBus', self.event_bus)
        self.container.register_instance('Config', self.config_loader)
        
        # Create components in optimizer order
        self._create_components()
        
        # Setup components
        self._setup_components()
        
        # Subscribe to signals for tracking
        self.event_bus.subscribe(EventType.SIGNAL, self._track_signal)
        
        logger.info("Setup complete")
        
    def _create_components(self):
        """Create components matching optimizer initialization order."""
        # Data handler
        self.components['data_handler'] = CSVDataHandler(
            instance_name="SPY_CSV_Loader",
            config_loader=self.config_loader,
            event_bus=self.event_bus,
            component_config_key="components.data_handler_csv",
            max_bars=1000  # Match optimizer
        )
        
        # Strategy
        self.components['strategy'] = EnsembleSignalStrategy(
            instance_name="SPY_Ensemble_Strategy",
            config_loader=self.config_loader,
            event_bus=self.event_bus,
            component_config_key="components.ensemble_strategy",
            container=self.container
        )
        
        # Portfolio
        self.components['portfolio'] = BasicPortfolio(
            instance_name="BasicPortfolio",
            config_loader=self.config_loader,
            event_bus=self.event_bus,
            component_config_key="components.basic_portfolio",
            container=self.container
        )
        
        # Risk manager
        self.components['risk_manager'] = BasicRiskManager(
            instance_name="BasicRiskMan1",
            config_loader=self.config_loader,
            event_bus=self.event_bus,
            component_config_key="components.basic_risk_manager",
            container=self.container
        )
        
        # Execution handler
        self.components['execution_handler'] = SimulatedExecutionHandler(
            instance_name="SimExec_1",
            config_loader=self.config_loader,
            event_bus=self.event_bus,
            component_config_key="components.simulated_execution_handler"
        )
        
        # Register in container
        for name, component in self.components.items():
            self.container.register_instance(name, component)
            
        # Risk manager expects portfolio_manager key
        self.container.register_instance('portfolio_manager', self.components['portfolio'])
            
    def _setup_components(self):
        """Setup components in correct order."""
        # Setup order matches optimizer
        setup_order = ['data_handler', 'strategy', 'portfolio', 'risk_manager', 'execution_handler']
        
        for name in setup_order:
            component = self.components[name]
            component.setup()
            logger.info(f"Setup complete: {name}")
            
        # Apply optimizer-specific settings
        self._apply_optimizer_settings()
        
    def _apply_optimizer_settings(self):
        """Apply settings to match optimizer behavior."""
        strategy = self.components['strategy']
        
        # 1. Disable adaptive mode FIRST (before it loads parameters)
        if hasattr(strategy, '_adaptive_mode_enabled'):
            strategy._adaptive_mode_enabled = False
            strategy._regime_parameters = {}  # Clear any loaded parameters
            logger.info("Disabled adaptive mode and cleared regime parameters")
        
        # 2. Set rule isolation mode to match optimizer
        if hasattr(strategy, 'set_rule_isolation_mode'):
            strategy.set_rule_isolation_mode(self.optimization_mode)
            logger.info(f"Set rule isolation mode: {self.optimization_mode}")
            
        # 3. Log initial weights after rule isolation
        logger.info(f"Weights after configuration - MA: {strategy._ma_weight}, RSI: {strategy._rsi_weight}")
        
    def run_validation(self):
        """Run validation matching optimizer execution."""
        logger.info("=== STARTING VALIDATION RUN ===")
        
        # Start components
        self._start_components()
        
        # Phase 1: Process ALL data (training + test)
        self._process_all_data()
        
        # Phase 2: Analyze results
        results = self._analyze_results()
        
        # Stop components
        self._stop_components()
        
        return results
        
    def _start_components(self):
        """Start components in correct order."""
        start_order = ['portfolio', 'risk_manager', 'execution_handler', 'strategy']
        
        for name in start_order:
            if name in self.components:
                self.components[name].start()
                
    def _process_all_data(self):
        """Process all data with warmup phase tracking."""
        data_handler = self.components['data_handler']
        
        # Setup data handler
        data_handler.setup()
        
        # Get full dataset (before train/test split)
        # We need to access the raw data, not the split data
        if hasattr(data_handler, '_data'):
            full_data = data_handler._data
        else:
            # Fallback - load data directly
            import pandas as pd
            full_data = pd.read_csv(data_handler._csv_file_path)
            full_data[data_handler._timestamp_column] = pd.to_datetime(full_data[data_handler._timestamp_column])
            full_data.set_index(data_handler._timestamp_column, inplace=True)
            
        # Limit to first 1000 bars to match optimizer
        if len(full_data) > 1000:
            full_data = full_data.iloc[:1000]
            
        total_bars = len(full_data)
        
        logger.info(f"Processing {total_bars} total bars")
        logger.info(f"Warmup phase: bars 0-{self.test_start_bar-1}")
        logger.info(f"Test phase: bars {self.test_start_bar}-{total_bars-1}")
        
        # Process each bar
        for bar_num, (timestamp, row) in enumerate(full_data.iterrows()):
            # Check phase transition
            if bar_num == self.test_start_bar and not self.warmup_complete:
                self.warmup_complete = True
                logger.info(f"=== WARMUP COMPLETE at bar {bar_num} ===")
                logger.info(f"Starting test phase at {timestamp}")
                self._log_indicator_states()
            
            # Prepare bar data
            bar_data = {
                'symbol': 'SPY',
                'timestamp': timestamp,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            }
            
            # Store bar number for signal tracking
            self.current_bar = bar_num
            
            # Publish bar event
            bar_event = Event(EventType.BAR, bar_data)
            self.event_bus.publish(bar_event)
            
        logger.info(f"Completed processing {total_bars} bars")
        
    def _track_signal(self, event: Event):
        """Track signals for analysis."""
        if event.event_type == EventType.SIGNAL:
            signal_data = event.data
            bar_num = self.current_bar
            
            # Only track test period signals
            if bar_num >= self.test_start_bar:
                self.signal_log.append({
                    'bar': bar_num,
                    'timestamp': signal_data['timestamp'],
                    'type': signal_data['signal_type'],
                    'price': signal_data['price_at_signal'],
                    'reason': signal_data.get('reason', ''),
                    'in_test': True
                })
                
                logger.info(f"SIGNAL #{len(self.signal_log)}: "
                           f"Bar {bar_num}, "
                           f"{signal_data['timestamp']}, "
                           f"Type={signal_data['signal_type']}, "
                           f"Price={signal_data['price_at_signal']:.2f}")
                
    def _log_indicator_states(self):
        """Log indicator states at warmup/test transition."""
        strategy = self.components['strategy']
        
        logger.info("=== INDICATOR STATES AT TEST START ===")
        
        # MA buffers
        if hasattr(strategy, '_prices'):
            logger.info(f"Price buffer length: {len(strategy._prices)}")
            if len(strategy._prices) >= 20:
                logger.info(f"Last 5 prices: {list(strategy._prices)[-5:]}")
                
        # MA values
        if hasattr(strategy, '_prev_short_ma') and strategy._prev_short_ma:
            logger.info(f"MA short: {strategy._prev_short_ma:.4f}")
        if hasattr(strategy, '_prev_long_ma') and strategy._prev_long_ma:
            logger.info(f"MA long: {strategy._prev_long_ma:.4f}")
            
        # RSI state
        if hasattr(strategy, 'rsi_indicator'):
            rsi = strategy.rsi_indicator
            logger.info(f"RSI enabled: {strategy._rsi_enabled}")
            logger.info(f"RSI value: {rsi._current_value if hasattr(rsi, '_current_value') else 'N/A'}")
            
    def _analyze_results(self):
        """Analyze validation results."""
        portfolio = self.components['portfolio']
        
        results = {
            'signal_count': len(self.signal_log),
            'signals': self.signal_log,
            'final_value': portfolio.get_portfolio_value(),
            'total_trades': len(portfolio._trade_history) if hasattr(portfolio, '_trade_history') else 0,
            'first_signal': self.signal_log[0] if self.signal_log else None,
            'last_signal': self.signal_log[-1] if self.signal_log else None
        }
        
        logger.info("=== VALIDATION RESULTS ===")
        logger.info(f"Signals generated: {results['signal_count']}")
        logger.info(f"Final portfolio value: ${results['final_value']:.2f}")
        logger.info(f"Total trades: {results['total_trades']}")
        
        if results['first_signal']:
            logger.info(f"First signal: {results['first_signal']['timestamp']} "
                       f"(bar {results['first_signal']['bar']})")
            
        return results
        
    def _stop_components(self):
        """Stop components in reverse order."""
        stop_order = ['strategy', 'execution_handler', 'risk_manager', 'portfolio', 'data_handler']
        
        for name in stop_order:
            if name in self.components:
                self.components[name].stop()

def main():
    """Run exact validation."""
    parser = argparse.ArgumentParser(description='Exact validation of optimizer results')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--mode', default='ma', choices=['ma', 'rsi', 'all'], 
                       help='Optimization mode to validate')
    parser.add_argument('--compare', help='Path to optimizer results for comparison')
    
    args = parser.parse_args()
    
    # Run validation
    validator = ValidationOrchestrator(args.config, args.mode)
    validator.setup()
    results = validator.run_validation()
    
    # Save results
    output_path = f"validation_results_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Compare with optimizer if provided
    if args.compare:
        logger.info("\n=== COMPARISON WITH OPTIMIZER ===")
        # Load optimizer results and compare
        # TODO: Implement comparison logic
        
    return 0

if __name__ == "__main__":
    sys.exit(main())