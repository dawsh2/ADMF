#!/usr/bin/env python3

"""
Simple test to see if genetic algorithm individuals are actually different.
"""

import sys
import os
import random

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.strategy.optimization.genetic_optimizer import GeneticOptimizer
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container

def test_simple_genetic():
    """Test if GA individuals are actually different"""
    
    print("=== Simple Genetic Algorithm Test ===")
    
    # Setup
    config_loader = SimpleConfigLoader("config/config.yaml")
    event_bus = EventBus()
    container = Container()
    
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Create genetic optimizer
    genetic_optimizer = GeneticOptimizer(
        instance_name="TestGA",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.genetic_optimizer",
        container=container
    )
    
    print("\n1. Testing population generation")
    population = genetic_optimizer._initialize_population()
    
    print(f"Population size: {len(population)}")
    for i, individual in enumerate(population):
        ma_w = individual.get("ma_rule.weight", 0.5)
        rsi_w = individual.get("rsi_rule.weight", 0.5)
        print(f"Individual {i+1}: MA={ma_w:.4f}, RSI={rsi_w:.4f}")
    
    print("\n2. Testing mutation")
    original = population[0].copy()
    print(f"Original: MA={original['ma_rule.weight']:.4f}, RSI={original['rsi_rule.weight']:.4f}")
    
    mutated = genetic_optimizer._mutate(original.copy())
    print(f"Mutated:  MA={mutated['ma_rule.weight']:.4f}, RSI={mutated['rsi_rule.weight']:.4f}")
    
    print("\n3. Testing crossover")
    parent1 = {"ma_rule.weight": 0.8, "rsi_rule.weight": 0.2}
    parent2 = {"ma_rule.weight": 0.2, "rsi_rule.weight": 0.8}
    
    child1, child2 = genetic_optimizer._crossover(parent1, parent2)
    print(f"Parent1: MA={parent1['ma_rule.weight']:.4f}, RSI={parent1['rsi_rule.weight']:.4f}")
    print(f"Parent2: MA={parent2['ma_rule.weight']:.4f}, RSI={parent2['rsi_rule.weight']:.4f}")
    print(f"Child1:  MA={child1['ma_rule.weight']:.4f}, RSI={child1['rsi_rule.weight']:.4f}")
    print(f"Child2:  MA={child2['ma_rule.weight']:.4f}, RSI={child2['rsi_rule.weight']:.4f}")
    
    print("\n4. Summary")
    print("This test verifies that:")
    print("- Population initialization creates diverse individuals")
    print("- Mutation changes individual parameters")
    print("- Crossover creates new combinations")
    print("If these work, the problem is likely in fitness evaluation, not GA operations")

if __name__ == "__main__":
    test_simple_genetic()