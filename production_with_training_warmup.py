#!/usr/bin/env python3
"""
Run production with training data warmup to exactly match optimizer behavior.

This approach:
1. Loads the full dataset
2. Processes training data (0-797) to warm up indicators
3. Then processes test data (798-997) for signal generation
4. Only counts signals from test period
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys

# Import required components
from src.core.logging_setup import setup_logging
from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.event import Event, EventType
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.dummy_component import DummyComponent

import logging
logger = logging.getLogger(__name__)

class WarmupDataHandler(CSVDataHandler):
    """Modified data handler that can process training data for warmup."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_mode = False
        self._test_start_idx = None
        self._signal_count = 0
        self._test_signals = []
        
    def set_warmup_mode(self, warmup=True):
        """Enable/disable warmup mode."""
        self._warmup_mode = warmup
        logger.info(f"Warmup mode: {'ENABLED' if warmup else 'DISABLED'}")
        
    def load_and_prepare_data(self):
        """Override to handle both training and test data."""
        super().load_and_prepare_data()
        
        # Calculate where test data starts
        total_rows = len(self._data)
        self._test_start_idx = int(total_rows * self._train_test_split_ratio)
        
        logger.info(f"Data loaded: {total_rows} total bars")
        logger.info(f"Training: bars 0-{self._test_start_idx-1}")
        logger.info(f"Test: bars {self._test_start_idx}-{total_rows-1}")
        
        # For warmup, we'll process ALL data but track which is test
        self._active_data = self._data.copy()
        
    def _publish_bar_events(self):
        """Override to process all data but track test period."""
        if self._active_data is None or self._active_data.empty:
            self.logger.warning(f"No active data to publish for '{self.name}'.")
            return
            
        # Process ALL bars for warmup
        for idx, (timestamp, row) in enumerate(self._active_data.iterrows()):
            bar_data = self._prepare_bar_data(timestamp, row)
            
            # Mark whether this is training or test data
            is_test = idx >= self._test_start_idx
            bar_data['is_test'] = is_test
            bar_data['bar_index'] = idx
            
            if is_test and self._warmup_mode:
                # Skip test bars during warmup
                continue
            elif not is_test and not self._warmup_mode:
                # Skip training bars during test mode
                continue
                
            bar_event = Event(
                event_type=EventType.BAR,
                data=bar_data
            )
            self._event_bus.publish(bar_event)
            
        mode = "warmup" if self._warmup_mode else "test"
        self.logger.info(f"Finished publishing BAR events in {mode} mode")

def run_with_warmup():
    """Run production with proper warmup sequence."""
    
    print("PRODUCTION RUN WITH TRAINING WARMUP")
    print("=" * 60)
    
    # Setup logging
    setup_logging('INFO', 'logs', False, True)
    
    # Load config
    config_path = "config/config_match_optimizer.yaml"
    config_loader = SimpleConfigLoader(config_file_path=config_path)
    config = config_loader.load()
    
    # Initialize core components
    container = Container()
    event_bus = EventBus()
    container.register('EventBus', event_bus)
    container.register('Config', config_loader)
    
    # Create data handler with warmup capability
    data_handler = WarmupDataHandler(
        instance_name="SPY_CSV_Loader",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.data_handler_csv"
    )
    
    # Create strategy
    strategy = EnsembleSignalStrategy(
        instance_name="SPY_Ensemble_Strategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    # Create other components
    portfolio = BasicPortfolio(
        instance_name="BasicPortfolio",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.basic_portfolio"
    )
    
    risk_manager = BasicRiskManager(
        instance_name="BasicRiskMan1",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.basic_risk_manager",
        container=container
    )
    
    execution_handler = SimulatedExecutionHandler(
        instance_name="SimExec_1",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.simulated_execution_handler"
    )
    
    # Register components
    container.register('data_handler', data_handler)
    container.register('strategy', strategy)
    container.register('portfolio_manager', portfolio)
    container.register('risk_manager', risk_manager)
    container.register('execution_handler', execution_handler)
    
    # Setup components
    data_handler.setup()
    strategy.setup()
    portfolio.setup()
    risk_manager.setup()
    execution_handler.setup()
    
    # Load adaptive parameters if they exist
    if os.path.exists('adaptive_regime_parameters.json'):
        with open('adaptive_regime_parameters.json', 'r') as f:
            regime_params = json.load(f)
        strategy._regime_best_parameters = regime_params
        strategy._adaptive_mode_enabled = True
        logger.info("Loaded adaptive parameters")
    
    print("\nPhase 1: Warming up indicators on training data...")
    print("-" * 40)
    
    # Start components
    portfolio.start()
    risk_manager.start()
    execution_handler.start()
    strategy.start()
    
    # Track signals during warmup (should be none in test period)
    warmup_signals = []
    
    def track_warmup_signal(event):
        if event.event_type == EventType.SIGNAL:
            warmup_signals.append(event.data)
    
    event_bus.subscribe(EventType.SIGNAL, track_warmup_signal)
    
    # Process training data for warmup
    data_handler.set_warmup_mode(True)
    data_handler.start()
    
    print(f"Warmup complete. Signals during warmup: {len(warmup_signals)}")
    
    # Now run on test data with warmed indicators
    print("\nPhase 2: Processing test data with warmed indicators...")
    print("-" * 40)
    
    # Clear warmup signal tracking
    event_bus.unsubscribe(EventType.SIGNAL, track_warmup_signal)
    
    # Track test signals
    test_signals = []
    
    def track_test_signal(event):
        if event.event_type == EventType.SIGNAL:
            sig = event.data
            test_signals.append({
                'timestamp': sig['timestamp'],
                'type': sig['signal_type'],
                'price': sig['price_at_signal'],
                'reason': sig['reason']
            })
            print(f"Signal #{len(test_signals)}: {sig['signal_type']} at {sig['timestamp']} price={sig['price_at_signal']:.2f}")
    
    event_bus.subscribe(EventType.SIGNAL, track_test_signal)
    
    # Process test data
    data_handler.set_warmup_mode(False)
    data_handler._active_data = data_handler._data.copy()  # Reset to process test data
    data_handler.start()
    
    print(f"\nTest complete. Total test signals: {len(test_signals)}")
    
    # Compare with expected optimizer signals
    print("\nExpected first two optimizer signals:")
    print("1. BUY at 2024-03-28 13:46:00 (first test bar)")
    print("2. SELL at 2024-03-28 14:00:00")
    
    if len(test_signals) >= 2:
        print("\nActual first two production signals:")
        for i in range(min(2, len(test_signals))):
            sig = test_signals[i]
            sig_type = "BUY" if sig['type'] == 1 else "SELL"
            print(f"{i+1}. {sig_type} at {sig['timestamp']} price={sig['price']:.2f}")
    
    # Cleanup
    strategy.stop()
    risk_manager.stop()
    portfolio.stop()
    execution_handler.stop()
    data_handler.stop()
    
    return test_signals

if __name__ == "__main__":
    signals = run_with_warmup()