"""
Genetic algorithm optimization mixin for components.

This module provides the GeneticOptimizationMixin that implements
genetic algorithm optimization for components.
"""

from typing import Dict, Any, Optional, List, Tuple
import random
import numpy as np
from .base import OptimizationMixin


class GeneticOptimizationMixin(OptimizationMixin):
    """
    Mixin that adds genetic algorithm optimization capabilities to components.
    
    This mixin implements genetic algorithm optimization with:
    - Population-based search
    - Crossover and mutation operators
    - Fitness-based selection
    - Elitism to preserve best solutions
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize genetic algorithm state."""
        super().__init__(*args, **kwargs)
        
        # GA specific parameters
        self._population_size: int = 50
        self._mutation_rate: float = 0.1
        self._crossover_rate: float = 0.8
        self._elitism_count: int = 2
        self._tournament_size: int = 3
        
        # GA state
        self._population: List[Dict[str, Any]] = []
        self._fitness_scores: Dict[str, float] = {}
        self._generation: int = 0
        self._parameter_space = None
        
    def initialize_genetic_search(self,
                                parameter_space: Optional['ParameterSpace'] = None,
                                population_size: int = 50,
                                mutation_rate: float = 0.1,
                                crossover_rate: float = 0.8,
                                elitism_count: int = 2) -> None:
        """
        Initialize genetic algorithm with parameters.
        
        Args:
            parameter_space: Parameter space to search
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_count: Number of best individuals to preserve
        """
        # Get parameter space
        if parameter_space is None:
            if hasattr(self, 'get_parameter_space'):
                parameter_space = self.get_parameter_space()
            else:
                raise ValueError("No parameter space provided")
                
        self._parameter_space = parameter_space
        
        # Set GA parameters
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elitism_count = min(elitism_count, population_size // 2)
        
        # Initialize population
        self._initialize_population()
        self._generation = 0
        
    def _initialize_population(self) -> None:
        """Initialize random population."""
        self._population = []
        
        # Generate random individuals
        for _ in range(self._population_size):
            individual = self._generate_random_individual()
            self._population.append(individual)
            
    def _generate_random_individual(self) -> Dict[str, Any]:
        """Generate a random individual from parameter space."""
        if not self._parameter_space:
            raise ValueError("Parameter space not initialized")
            
        individual = {}
        
        for name, param in self._parameter_space._parameters.items():
            if param.param_type == 'discrete' and param.values:
                individual[name] = random.choice(param.values)
            elif param.param_type == 'continuous':
                min_val = param.min_value or 0
                max_val = param.max_value or 1
                individual[name] = random.uniform(min_val, max_val)
                
        return individual
        
    def suggest_next_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Suggest next parameters using genetic algorithm.
        
        Returns:
            Next parameter set to evaluate or None
        """
        # If we haven't evaluated the whole population yet, return next individual
        unevaluated = []
        for individual in self._population:
            param_id = self._generate_parameter_id(individual)
            if param_id not in self._fitness_scores:
                unevaluated.append(individual)
                
        if unevaluated:
            return unevaluated[0]
            
        # All current population evaluated - evolve to next generation
        self._evolve_population()
        self._generation += 1
        
        # Return first individual from new population
        if self._population:
            return self._population[0]
            
        return None
        
    def record_fitness(self, parameters: Dict[str, Any], fitness: float) -> None:
        """
        Record fitness score for a parameter set.
        
        Args:
            parameters: The evaluated parameters
            fitness: Fitness score (higher is better)
        """
        param_id = self._generate_parameter_id(parameters)
        self._fitness_scores[param_id] = fitness
        
    def _evolve_population(self) -> None:
        """Evolve population to next generation."""
        if not self._population:
            return
            
        # Sort population by fitness
        sorted_pop = self._sort_by_fitness(self._population)
        
        new_population = []
        
        # Elitism - keep best individuals
        new_population.extend(sorted_pop[:self._elitism_count])
        
        # Generate rest of population
        while len(new_population) < self._population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self._crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            # Mutation
            if random.random() < self._mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self._mutation_rate:
                child2 = self._mutate(child2)
                
            new_population.append(child1)
            if len(new_population) < self._population_size:
                new_population.append(child2)
                
        self._population = new_population[:self._population_size]
        
    def _sort_by_fitness(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort population by fitness (descending)."""
        def get_fitness(individual):
            param_id = self._generate_parameter_id(individual)
            return self._fitness_scores.get(param_id, float('-inf'))
            
        return sorted(population, key=get_fitness, reverse=True)
        
    def _tournament_selection(self) -> Dict[str, Any]:
        """Select individual using tournament selection."""
        tournament = random.sample(self._population, min(self._tournament_size, len(self._population)))
        return max(tournament, key=lambda x: self._fitness_scores.get(self._generate_parameter_id(x), float('-inf')))
        
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
                
        return child1, child2
        
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        
        # Select random parameter to mutate
        if not self._parameter_space:
            return mutated
            
        param_names = list(self._parameter_space._parameters.keys())
        if not param_names:
            return mutated
            
        param_to_mutate = random.choice(param_names)
        param = self._parameter_space._parameters[param_to_mutate]
        
        if param.param_type == 'discrete' and param.values:
            # For discrete, pick a different value
            current = mutated.get(param_to_mutate)
            choices = [v for v in param.values if v != current]
            if choices:
                mutated[param_to_mutate] = random.choice(choices)
        elif param.param_type == 'continuous':
            # For continuous, add gaussian noise
            min_val = param.min_value or 0
            max_val = param.max_value or 1
            current = mutated.get(param_to_mutate, (min_val + max_val) / 2)
            
            # Add noise proportional to range
            noise_scale = (max_val - min_val) * 0.1
            new_val = current + random.gauss(0, noise_scale)
            
            # Clip to bounds
            mutated[param_to_mutate] = max(min_val, min(max_val, new_val))
            
        return mutated
        
    def get_genetic_progress(self) -> Dict[str, Any]:
        """Get current progress of genetic algorithm."""
        evaluated = len(self._fitness_scores)
        best_fitness = max(self._fitness_scores.values()) if self._fitness_scores else None
        
        return {
            'method': 'genetic_algorithm',
            'generation': self._generation,
            'population_size': self._population_size,
            'individuals_evaluated': evaluated,
            'best_fitness': best_fitness,
            'mutation_rate': self._mutation_rate,
            'crossover_rate': self._crossover_rate,
            'elitism_count': self._elitism_count
        }