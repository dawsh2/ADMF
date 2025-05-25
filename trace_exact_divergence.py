#!/usr/bin/env python3
"""
Trace the exact point where optimizer and production diverge.
Focus on the trade count difference: 5 vs 11 trades.
"""

import sys
sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.core.logging_setup import setup_logging
from src.strategy.optimization.engines.clean_backtest_engine import CleanBacktestEngine
import json
import logging

class DivergenceTracer:
    """Trace divergence between optimizer and production runs."""
    
    def __init__(self):
        self.optimizer_events = []
        self.production_events = []
        
    def trace_optimizer_run(self):
        """Run optimizer test and capture all events."""
        print("Running Optimizer Test with Event Tracing...")
        
        # Create clean engine like EnhancedOptimizerV3 does
        config_loader = SimpleConfigLoader("config/config_optimization_exact.yaml")
        config = config_loader.load_config()
        
        # Override to test dataset only
        if 'data_handler_csv' in config['components']:
            config['components']['data_handler_csv']['train_test_split_ratio'] = 0.8
        
        engine = CleanBacktestEngine(config_loader)
        
        # Load adaptive parameters
        with open('regime_optimized_parameters.json', 'r') as f:
            regime_params = json.load(f)
        
        # Get test data parameters
        test_params = regime_params.get('ranging_low_vol', {})
        
        # Create strategy params
        strategy_params = {
            'ma_weight': test_params.get('ma_weight', 0.6),
            'rsi_weight': test_params.get('rsi_weight', 0.1),
            'bb_weight': test_params.get('bb_weight', 0.2),
            'volume_weight': test_params.get('volume_weight', 0.1),
            'ma_short_period': test_params.get('ma_short_period', 10),
            'ma_long_period': test_params.get('ma_long_period', 20),
            'rsi_period': test_params.get('rsi_period', 21),
            'rsi_oversold': test_params.get('rsi_oversold', 30),
            'rsi_overbought': test_params.get('rsi_overbought', 70)
        }
        
        # Hook into event bus to capture events
        captured_events = []
        
        def capture_event(event):
            if event.type in ['SIGNAL', 'ORDER', 'FILL', 'REGIME_CHANGE']:
                captured_events.append({
                    'type': event.type,
                    'timestamp': str(event.payload.get('timestamp', '')),
                    'details': event.payload
                })
        
        # Run backtest with event capture
        result = engine.run_backtest(
            dataset='test',
            strategy_params=strategy_params,
            adaptive_params=regime_params,
            event_callback=capture_event
        )
        
        self.optimizer_events = captured_events
        
        print(f"Optimizer Result: ${result.get('final_portfolio_value', 0):,.2f}")
        print(f"Trades: {result.get('total_trades', 0)}")
        print(f"Events captured: {len(captured_events)}")
        
        return result
        
    def trace_production_run(self):
        """Run production test and capture events."""
        print("\nRunning Production Test with Event Tracing...")
        
        # Import production components
        from src.data.csv_data_handler import CsvDataHandler
        from src.risk.basic_portfolio import BasicPortfolio
        from src.risk.basic_risk_manager import BasicRiskManager
        from src.execution.simulated_execution_handler import SimulatedExecutionHandler
        from src.strategy.regime_detector import RegimeDetector
        from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
        
        # Setup
        config_loader = SimpleConfigLoader("config/config_production.yaml")
        event_bus = EventBus()
        container = Container()
        
        # Create components
        data_handler = CsvDataHandler(
            instance_name="DataHandler",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="data_handler_csv",
            container=container
        )
        
        portfolio = BasicPortfolio(
            instance_name="Portfolio",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="basic_portfolio",
            container=container
        )
        
        risk_manager = BasicRiskManager(
            instance_name="RiskManager",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="basic_risk_manager",
            container=container
        )
        
        execution_handler = SimulatedExecutionHandler(
            instance_name="ExecutionHandler",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="simulated_execution_handler",
            container=container
        )
        
        regime_detector = RegimeDetector(
            instance_name="RegimeDetector",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="MyPrimaryRegimeDetector",
            container=container
        )
        
        # Load adaptive parameters
        with open('regime_optimized_parameters.json', 'r') as f:
            regime_params = json.load(f)
        
        # Create adaptive strategy
        strategy = RegimeAdaptiveStrategy(
            instance_name="Strategy",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key="ensemble_strategy",
            container=container
        )
        
        # Configure adaptive mode
        strategy._adaptive_mode = True
        strategy._regime_parameters = regime_params
        
        # Capture events
        captured_events = []
        
        def capture_event(event):
            if event.type in ['SIGNAL', 'ORDER', 'FILL', 'REGIME_CHANGE']:
                captured_events.append({
                    'type': event.type,
                    'timestamp': str(event.payload.get('timestamp', '')),
                    'details': event.payload
                })
        
        # Subscribe to events
        event_bus.subscribe('SIGNAL', capture_event)
        event_bus.subscribe('ORDER', capture_event)
        event_bus.subscribe('FILL', capture_event)
        event_bus.subscribe('REGIME_CHANGE', capture_event)
        
        # Setup and start
        portfolio.setup()
        risk_manager.setup()
        execution_handler.setup()
        regime_detector.setup()
        strategy.setup()
        
        portfolio.start()
        risk_manager.start()
        execution_handler.start()
        regime_detector.start()
        strategy.start()
        
        # Run on test data
        test_bars = data_handler.get_test_data()
        for bar in test_bars:
            event_bus.publish('BAR', bar)
        
        # Get results
        final_value = portfolio.get_total_portfolio_value()
        trades = portfolio.get_trade_history()
        
        self.production_events = captured_events
        
        print(f"Production Result: ${final_value:,.2f}")
        print(f"Trades: {len(trades)}")
        print(f"Events captured: {len(captured_events)}")
        
        return {
            'final_portfolio_value': final_value,
            'total_trades': len(trades)
        }
        
    def compare_events(self):
        """Compare events to find divergence."""
        print("\n" + "="*80)
        print("EVENT COMPARISON")
        print("="*80)
        
        # Find first divergence
        min_len = min(len(self.optimizer_events), len(self.production_events))
        
        first_divergence = None
        for i in range(min_len):
            opt_event = self.optimizer_events[i]
            prod_event = self.production_events[i]
            
            if opt_event['type'] != prod_event['type'] or opt_event['timestamp'] != prod_event['timestamp']:
                first_divergence = i
                break
        
        if first_divergence is not None:
            print(f"\nFirst divergence at event #{first_divergence}:")
            print(f"Optimizer: {self.optimizer_events[first_divergence]}")
            print(f"Production: {self.production_events[first_divergence]}")
        else:
            print("\nNo divergence found in matching events")
            print(f"Optimizer has {len(self.optimizer_events)} events")
            print(f"Production has {len(self.production_events)} events")
        
        # Count event types
        opt_signals = sum(1 for e in self.optimizer_events if e['type'] == 'SIGNAL')
        prod_signals = sum(1 for e in self.production_events if e['type'] == 'SIGNAL')
        
        opt_orders = sum(1 for e in self.optimizer_events if e['type'] == 'ORDER')
        prod_orders = sum(1 for e in self.production_events if e['type'] == 'ORDER')
        
        print(f"\nSignal Count:")
        print(f"  Optimizer: {opt_signals}")
        print(f"  Production: {prod_signals}")
        print(f"  Difference: {prod_signals - opt_signals}")
        
        print(f"\nOrder Count:")
        print(f"  Optimizer: {opt_orders}")
        print(f"  Production: {prod_orders}")
        print(f"  Difference: {prod_orders - opt_orders}")

def main():
    tracer = DivergenceTracer()
    
    # Run both tests
    opt_result = tracer.trace_optimizer_run()
    prod_result = tracer.trace_production_run()
    
    # Compare
    tracer.compare_events()
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    opt_value = opt_result.get('final_portfolio_value', 0)
    prod_value = prod_result.get('final_portfolio_value', 0)
    
    diff = abs(opt_value - prod_value)
    pct_diff = (diff / opt_value) * 100 if opt_value else 0
    
    print(f"Optimizer:  ${opt_value:,.2f} ({opt_result.get('total_trades', 0)} trades)")
    print(f"Production: ${prod_value:,.2f} ({prod_result.get('total_trades', 0)} trades)")
    print(f"Difference: ${diff:.2f} ({pct_diff:.4f}%)")

if __name__ == "__main__":
    main()