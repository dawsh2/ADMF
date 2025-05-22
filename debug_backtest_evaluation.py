#!/usr/bin/env python3

"""
Debug script to trace exactly what happens during genetic optimization backtest evaluation.
This will help us understand why different weight combinations produce identical fitness values.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.optimization.genetic_optimizer import GeneticOptimizer

def debug_backtest_evaluation():
    """Debug backtest evaluation to understand parameter propagation"""
    
    print("=== Debugging Backtest Evaluation ===")
    
    # Setup infrastructure
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Register components (full setup for genetic optimizer)
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create and register all required components
    from src.data.csv_data_handler import CSVDataHandler
    from src.risk.basic_portfolio import BasicPortfolio
    from src.risk.basic_risk_manager import BasicRiskManager
    from src.execution.simulated_execution_handler import SimulatedExecutionHandler
    from src.core.dummy_component import DummyComponent
    
    # Data handler
    csv_args = {"instance_name": "TestDataHandler", "config_loader": config_loader,
                "event_bus": event_bus, "component_config_key": "components.data_handler_csv",
                "max_bars": 50}  # Small dataset for debugging
    data_handler = CSVDataHandler(**csv_args)
    container.register_instance("data_handler", data_handler)
    
    # Strategy
    strategy = EnsembleSignalStrategy(
        instance_name="TestStrategy",
        config_loader=config_loader,
        event_bus=event_bus,  
        component_config_key="components.ensemble_strategy",
        container=container
    )
    container.register_instance("strategy", strategy)
    
    # Portfolio
    portfolio_args = {"instance_name": "TestPortfolio", "config_loader": config_loader,
                      "event_bus": event_bus, "container": container,
                      "component_config_key": "components.basic_portfolio"}
    portfolio = BasicPortfolio(**portfolio_args)
    container.register_instance("portfolio_manager", portfolio)
    
    # Risk manager
    risk_args = {"instance_name": "TestRisk", "config_loader": config_loader,
                 "event_bus": event_bus, "component_config_key": "components.basic_risk_manager",
                 "container": container, "portfolio_manager_key": "portfolio_manager"}
    risk_manager = BasicRiskManager(**risk_args)
    container.register_instance("risk_manager", risk_manager)
    
    # Execution handler
    exec_args = {"instance_name": "TestExec", "config_loader": config_loader,
                 "event_bus": event_bus, "component_config_key": "components.simulated_execution_handler"}
    execution_handler = SimulatedExecutionHandler(**exec_args)
    container.register_instance("execution_handler", execution_handler)
    
    # Create genetic optimizer
    genetic_optimizer = GeneticOptimizer(
        instance_name="TestGeneticOptimizer",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.genetic_optimizer",
        container=container
    )
    
    print("\n1. Testing parameter propagation with different weights")
    
    # Test two very different weight combinations
    test_cases = [
        {"ma_rule.weight": 0.9, "rsi_rule.weight": 0.1, "name": "MA-Heavy"},
        {"ma_rule.weight": 0.1, "rsi_rule.weight": 0.9, "name": "RSI-Heavy"}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        
        # Extract weights
        weights = {k: v for k, v in test_case.items() if k != 'name'}
        print(f"Testing weights: {weights}")
        
        # Call the genetic optimizer's fitness evaluation method directly
        print("Calling genetic optimizer._evaluate_fitness()...")
        
        try:
            fitness = genetic_optimizer._evaluate_fitness(weights, "train")
            print(f"Fitness result: {fitness}")
            
            # Check strategy state after evaluation
            strategy_params = strategy.get_parameters()
            print(f"Strategy parameters after evaluation: {strategy_params}")
            
            # Check internal weights
            ma_weight = getattr(strategy, '_ma_weight', 'NOT_FOUND')
            rsi_weight = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
            print(f"Internal strategy weights: MA={ma_weight}, RSI={rsi_weight}")
            
            results.append({
                'name': test_case['name'],
                'weights': weights,
                'fitness': fitness,
                'final_ma_weight': ma_weight,
                'final_rsi_weight': rsi_weight
            })
            
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n2. Summary of Results:")
    print(f"{'Test Case':<12} {'MA Weight':<10} {'RSI Weight':<11} {'Fitness':<12} {'Different?'}")
    print("=" * 60)
    
    prev_fitness = None
    for result in results:
        fitness_diff = "N/A" if prev_fitness is None else ("YES" if abs(result['fitness'] - prev_fitness) > 0.01 else "NO")
        print(f"{result['name']:<12} {result['final_ma_weight']:<10.3f} {result['final_rsi_weight']:<11.3f} {result['fitness']:<12.2f} {fitness_diff}")
        prev_fitness = result['fitness']
    
    print("\n3. Analysis:")
    if len(results) >= 2:
        fitness_diff = abs(results[0]['fitness'] - results[1]['fitness']) 
        weight_diff = abs(results[0]['final_ma_weight'] - results[1]['final_ma_weight'])
        
        print(f"Fitness difference: {fitness_diff:.6f}")
        print(f"Weight difference: {weight_diff:.6f}")
        
        if fitness_diff < 0.01:
            print("❌ PROBLEM: Different weights produce identical fitness values")
            print("This suggests weights are not affecting strategy behavior during backtest")
        else:
            print("✅ SUCCESS: Different weights produce different fitness values")
            
        if weight_diff < 0.01:
            print("❌ PROBLEM: Weights are not being applied to strategy")
        else:
            print("✅ SUCCESS: Weights are being applied to strategy")

if __name__ == "__main__":
    debug_backtest_evaluation()