#!/usr/bin/env python3

"""
Test script to verify that different MA/RSI weights actually produce different results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.execution.simulated_execution_handler import SimulatedExecutionHandler

def test_weight_impact():
    """Test if different weights actually produce different portfolio results."""
    
    print("Testing if MA/RSI weights actually impact results...")
    
    # Set up basic components
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Register data handler
    data_args = {"instance_name": "DataHandler", "config_loader": config_loader, 
                 "event_bus": event_bus, "component_config_key": "components.data_handler",
                 "container": container}
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=data_args)
    
    # Register portfolio
    portfolio_args = {"instance_name": "Portfolio", "config_loader": config_loader,
                     "event_bus": event_bus, "component_config_key": "components.portfolio",
                     "container": container}
    container.register_type("portfolio", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    
    # Register execution handler
    exec_args = {"instance_name": "ExecutionHandler", "config_loader": config_loader,
                "event_bus": event_bus, "component_config_key": "components.execution_handler",
                "container": container}
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=exec_args)
    
    # Register strategy
    strategy_args = {"instance_name": "Strategy", "config_loader": config_loader,
                    "event_bus": event_bus, "component_config_key": "components.strategy",
                    "container": container}
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=strategy_args)
    
    # Test different weight combinations
    weight_tests = [
        {"ma_rule.weight": 0.1, "rsi_rule.weight": 0.9},
        {"ma_rule.weight": 0.5, "rsi_rule.weight": 0.5}, 
        {"ma_rule.weight": 0.9, "rsi_rule.weight": 0.1},
    ]
    
    results = []
    
    for i, weights in enumerate(weight_tests):
        print(f"\n--- Test {i+1}: MA={weights['ma_rule.weight']}, RSI={weights['rsi_rule.weight']} ---")
        
        # Get fresh instances
        data_handler = container.resolve("data_handler")
        portfolio = container.resolve("portfolio")
        execution_handler = container.resolve("execution_handler")
        strategy = container.resolve("strategy")
        
        # Setup and start components
        for component in [data_handler, portfolio, execution_handler, strategy]:
            component.setup()
            if component.get_state() == component.STATE_INITIALIZED:
                component.start()
        
        # Set strategy weights
        strategy.set_parameters(weights)
        
        # Verify weights were set
        current_params = strategy.get_parameters()
        actual_ma = current_params.get('ma_rule.weight', 'NOT_SET')
        actual_rsi = current_params.get('rsi_rule.weight', 'NOT_SET')
        print(f"Strategy weights after setting: MA={actual_ma}, RSI={actual_rsi}")
        
        # Run limited backtest
        data_handler.limit_data_to_bars(500)  # Use small dataset for speed
        
        # Process data
        for timestamp, row in data_handler.get_iterator():
            strategy.on_data(timestamp, row)
        
        # Get final portfolio value
        final_value = portfolio.get_final_portfolio_value()
        results.append((weights, final_value))
        print(f"Final portfolio value: {final_value:.4f}")
        
        # Stop components
        for component in [strategy, execution_handler, portfolio, data_handler]:
            if component.get_state() == component.STATE_STARTED:
                component.stop()
    
    # Analyze results
    print(f"\n=== WEIGHT IMPACT ANALYSIS ===")
    for weights, value in results:
        print(f"MA={weights['ma_rule.weight']:.1f}, RSI={weights['rsi_rule.weight']:.1f} -> {value:.4f}")
    
    # Check if results are different
    values = [result[1] for result in results]
    unique_values = len(set(round(v, 2) for v in values))
    
    if unique_values == 1:
        print("❌ PROBLEM: All weight combinations produced identical results!")
        print("   This suggests weights are not actually being applied to the strategy.")
    elif unique_values < len(values):
        print("⚠️  WARNING: Some weight combinations produced identical results.")
    else:
        print("✅ SUCCESS: Different weights produced different results.")
    
    return results

if __name__ == "__main__":
    test_weight_impact()