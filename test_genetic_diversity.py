#!/usr/bin/env python3

"""
Simple test script to verify genetic algorithm diversity and evolution.
"""

import random
import copy
from typing import List, Dict

def test_genetic_operations():
    """Test genetic algorithm components for diversity."""
    
    # Simulate GA parameters
    mutation_rate = 0.6
    crossover_rate = 0.5
    min_weight = 0.1
    max_weight = 0.9
    population_size = 10
    
    # Initialize test population
    population = []
    for _ in range(population_size):
        ma_weight = random.uniform(min_weight, max_weight)
        individual = {
            "ma_rule.weight": ma_weight,
            "rsi_rule.weight": 1.0 - ma_weight
        }
        population.append(individual)
    
    print("Initial population:")
    for i, ind in enumerate(population):
        print(f"  Ind{i+1}: MA={ind['ma_rule.weight']:.3f}, RSI={ind['rsi_rule.weight']:.3f}")
    
    # Test mutation
    print(f"\nTesting mutation (rate={mutation_rate}):")
    mutation_count = 0
    for i, individual in enumerate(population):
        if random.random() < mutation_rate:
            old_ma = individual["ma_rule.weight"]
            mutation_amount = random.gauss(0, 0.15)
            new_ma_weight = old_ma + mutation_amount
            new_ma_weight = max(min_weight, min(max_weight, new_ma_weight))
            
            individual["ma_rule.weight"] = new_ma_weight
            individual["rsi_rule.weight"] = 1.0 - new_ma_weight
            
            print(f"  Ind{i+1}: MA {old_ma:.3f} -> {new_ma_weight:.3f} (change: {new_ma_weight-old_ma:+.3f})")
            mutation_count += 1
    
    print(f"Total mutations: {mutation_count}/{population_size}")
    
    # Test crossover
    print(f"\nTesting crossover (rate={crossover_rate}):")
    crossover_count = 0
    for i in range(0, len(population)-1, 2):
        if random.random() < crossover_rate:
            parent1 = population[i]
            parent2 = population[i+1]
            
            print(f"  Parents: MA1={parent1['ma_rule.weight']:.3f}, MA2={parent2['ma_rule.weight']:.3f}")
            
            # Arithmetic crossover
            alpha = random.random()
            child1_ma = alpha * parent1["ma_rule.weight"] + (1-alpha) * parent2["ma_rule.weight"]
            child2_ma = (1-alpha) * parent1["ma_rule.weight"] + alpha * parent2["ma_rule.weight"]
            
            print(f"  Children: MA1={child1_ma:.3f}, MA2={child2_ma:.3f}")
            crossover_count += 1
    
    print(f"Total crossovers: {crossover_count}")
    
    print("\nFinal population:")
    for i, ind in enumerate(population):
        print(f"  Ind{i+1}: MA={ind['ma_rule.weight']:.3f}, RSI={ind['rsi_rule.weight']:.3f}")
        
    # Check diversity
    ma_weights = [ind['ma_rule.weight'] for ind in population]
    ma_range = max(ma_weights) - min(ma_weights)
    unique_weights = len(set(round(w, 3) for w in ma_weights))
    
    print(f"\nDiversity metrics:")
    print(f"  MA weight range: {ma_range:.3f}")
    print(f"  Unique weights (rounded to 3 decimals): {unique_weights}/{population_size}")
    print(f"  Min MA weight: {min(ma_weights):.3f}")
    print(f"  Max MA weight: {max(ma_weights):.3f}")

if __name__ == "__main__":
    print("Testing Genetic Algorithm Components")
    print("=" * 50)
    test_genetic_operations()