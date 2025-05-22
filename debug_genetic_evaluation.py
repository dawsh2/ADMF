#!/usr/bin/env python3

"""
Debug script to check what parameters are actually being tested during genetic optimization.
"""

import sys
import os
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.optimization.genetic_optimizer import GeneticOptimizer
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio

def debug_genetic_evaluation():
    """Debug what parameters are actually being evaluated"""
    
    print("=== Debugging Genetic Evaluation ===")
    
    # Setup infrastructure
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Register components (minimal setup)
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create strategy
    strategy = EnsembleSignalStrategy(
        instance_name="TestStrategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.ensemble_strategy",
        container=container
    )
    
    container.register_instance("strategy", strategy)
    
    # Create a minimal genetic optimizer
    genetic_optimizer = GeneticOptimizer(
        instance_name="TestGeneticOptimizer",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.genetic_optimizer",
        container=container
    )
    
    print("\n1. Testing population initialization")
    population = genetic_optimizer._initialize_population()
    
    print(f"Population size: {len(population)}")
    print("Sample individuals:")
    for i, individual in enumerate(population[:5]):
        print(f"  Individual {i+1}: {individual}")
    
    print("\n2. Testing parameter merging with strategy")
    strategy.setup()
    
    # Get base parameters from strategy
    base_params = strategy.get_parameters()
    print(f"Base strategy parameters: {base_params}")
    
    # Test parameter merging for a few individuals
    print("\n3. Testing parameter combinations")
    for i, individual in enumerate(population[:3]):
        combined_params = base_params.copy()
        combined_params.update(individual)
        
        print(f"\nIndividual {i+1}:")
        print(f"  Weights from GA: {individual}")
        print(f"  Combined params: {combined_params}")
        
        # Test if setting parameters works
        success = strategy.set_parameters(combined_params)
        print(f"  Set parameters success: {success}")
        
        # Check what parameters are actually set
        actual_params = strategy.get_parameters()
        print(f"  Actual params after set: {actual_params}")
        
        # Check internal weights
        ma_weight = getattr(strategy, '_ma_weight', 'NOT_FOUND')
        rsi_weight = getattr(strategy, '_rsi_weight', 'NOT_FOUND')
        print(f"  Internal weights: MA={ma_weight}, RSI={rsi_weight}")
        
        # Reset strategy state
        if hasattr(strategy, 'reset'):
            strategy.reset()
            print(f"  Strategy reset")
    
    print("\n4. Summary")
    print("This test shows whether:")
    print("- Population initialization creates diverse individuals")
    print("- Parameter merging works correctly") 
    print("- Strategy accepts and stores the parameters")
    print("- Reset clears state between evaluations")

if __name__ == "__main__":
    debug_genetic_evaluation()