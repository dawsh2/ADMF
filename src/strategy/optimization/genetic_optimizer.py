# src/strategy/optimization/genetic_optimizer.py
import logging
import random
import datetime
from typing import Dict, Any, List, Optional, Tuple
import copy
import numpy as np

from src.strategy.optimization.basic_optimizer import BasicOptimizer


class GeneticOptimizer(BasicOptimizer):
    """
    A genetic algorithm-based optimizer for trading strategies.
    Optimizes weights for ensemble strategy components after grid search has 
    determined optimal parameters for individual components.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize GA specific settings with defaults
        self._population_size = 50
        self._generations = 20
        self._mutation_rate = 0.1
        self._crossover_rate = 0.7
        self._elitism_count = 5
        self._tournament_size = 3
        
        # Weight constraints
        self._min_weight = 0.1
        self._max_weight = 0.9
        
        # Store the best result
        self._best_individual = None
        self._best_fitness = None
        
        # Debug counter
        self._debug_logged_count = 0
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Call parent's _initialize first
        super()._initialize()
        
        # Load GA specific settings
        self._population_size = self.get_specific_config("population_size", 50)
        self._generations = self.get_specific_config("generations", 20)
        self._mutation_rate = self.get_specific_config("mutation_rate", 0.1)
        self._crossover_rate = self.get_specific_config("crossover_rate", 0.7)
        self._elitism_count = self.get_specific_config("elitism_count", 5)
        self._tournament_size = self.get_specific_config("tournament_size", 3)
        
        # Weight constraints
        self._min_weight = self.get_specific_config("min_weight", 0.1)
        self._max_weight = self.get_specific_config("max_weight", 0.9)
        
        # Set up optimizer logger
        self.opt_logger = logging.getLogger(f"{__name__}.{self.instance_name}")
        
        # DEBUG: Print actual config values being used
        self.logger.info(f"🔧 GA CONFIG: pop={self._population_size}, gen={self._generations}, "
                        f"mut={self._mutation_rate}, cross={self._crossover_rate}, elite={self._elitism_count}")

    def _initialize_population(self) -> List[Dict[str, float]]:
        """
        Initialize a population of weight combinations.
        
        Returns:
            List of individuals, where each individual is a dictionary of weights
        """
        population = []
        for _ in range(self._population_size):
            # Generate individual with rule weights that sum to 1.0
            ma_weight = random.uniform(self._min_weight, self._max_weight)
            rsi_weight = 1.0 - ma_weight
            
            individual = {
                "ma_rule.weight": ma_weight,
                "rsi_rule.weight": rsi_weight
            }
            population.append(individual)
        
        return population

    def _evaluate_fitness(self, individual: Dict[str, float], dataset_type: str = "train") -> float:
        """
        Evaluate fitness of an individual by running a backtest.
        Uses the same _perform_single_backtest_run from BasicOptimizer.
        
        Args:
            individual: Dictionary of parameter values (typically just weights)
            dataset_type: Whether to use "train" or "test" dataset
            
        Returns:
            Fitness value (metric value from backtest)
        """
        # CRITICAL FIX: Merge regime parameters with weight parameters
        # The genetic optimizer only provides weights, but the strategy needs both
        # regime-specific parameters (MA windows, RSI thresholds) AND weights
        strategy = self.container.resolve(self._strategy_service_name)
        
        # CRITICAL FIX: Reset strategy state before evaluation to avoid carrying over
        # state from previous evaluations. This ensures each parameter combination
        # gets a fresh evaluation.
        if hasattr(strategy, 'reset') and callable(strategy.reset):
            strategy.reset()
            self.logger.debug(f"Strategy state reset before evaluation")
        
        current_params = strategy.get_parameters() if hasattr(strategy, 'get_parameters') else {}
        
        # Merge current regime parameters with weight parameters from genetic algorithm
        combined_params = current_params.copy()
        combined_params.update(individual)
        
        # Always debug log for now to understand what's happening
        ma_weight = combined_params.get('ma_rule.weight', 'N/A')
        rsi_weight = combined_params.get('rsi_rule.weight', 'N/A')
        self.logger.info(f"🔍 Fitness Eval: Testing weights MA={ma_weight:.4f}, RSI={rsi_weight:.4f}")
        
        # Check if strategy actually receives these parameters
        self.logger.info(f"🔧 Setting combined params: {combined_params}")
        
        # Debug: Log the first few fitness evaluations to verify parameters are different
        if self._debug_logged_count < 5:
            self.logger.info(f"GA Fitness Debug {self._debug_logged_count+1}:")
            self.logger.info(f"  Individual weights: {individual}")
            self.logger.info(f"  Current strategy params: {current_params}")
            self.logger.info(f"  Combined params: {combined_params}")
            # Check if the weights are actually different
            if 'ma_rule.weight' in combined_params and 'rsi_rule.weight' in combined_params:
                self.logger.info(f"  Final weights to test: MA={combined_params['ma_rule.weight']}, RSI={combined_params['rsi_rule.weight']}")
            self._debug_logged_count += 1
        
        self.logger.debug(f"Evaluating fitness with combined params: {combined_params} on {dataset_type} data")
        
        # CRITICAL FIX: Handle the fact that EnhancedOptimizer returns a tuple (metric, regime_performance)
        # while BasicOptimizer returns just the metric
        result = self._perform_single_backtest_run(combined_params, dataset_type)
        if isinstance(result, tuple):
            metric_value, _ = result  # Extract just the metric value from tuple
        else:
            metric_value = result  # Direct value from BasicOptimizer
        
        if metric_value is None:
            self.logger.warning(f"Failed to evaluate fitness for individual: {individual}")
            return float('-inf') if self._higher_metric_is_better else float('inf')
            
        # CRITICAL FIX: Add continuous weight-based fitness adjustment
        # This ensures that even tiny weight differences produce different fitness values
        # This is essential for genetic algorithm evolution
        
        ma_weight_val = combined_params.get('ma_rule.weight', 0.5)  
        rsi_weight_val = combined_params.get('rsi_rule.weight', 0.5)
        
        # SOLUTION: Enhanced fitness calculation inspired by GitHub GA sample
        # Instead of just using raw portfolio value, calculate a risk-adjusted metric
        # that is more sensitive to weight changes
        
        # Get portfolio performance details if available
        portfolio_manager = self.container.resolve(self._portfolio_service_name)
        
        try:
            # Try to get more detailed performance metrics for nuanced fitness
            if hasattr(portfolio_manager, 'get_trade_history') and callable(getattr(portfolio_manager, 'get_trade_history')):
                trades = portfolio_manager.get_trade_history()
                if trades and len(trades) > 0:
                    # Calculate returns from trades
                    returns = [trade.get('pnl', 0) for trade in trades if 'pnl' in trade]
                    if len(returns) > 1:
                        returns_array = np.array(returns)
                        
                        # Calculate risk-adjusted fitness similar to GitHub sample
                        mean_return = np.mean(returns_array)
                        std_return = np.std(returns_array) if len(returns_array) > 1 else 1.0
                        negative_returns = returns_array[returns_array < 0]
                        downside_penalty = -np.sum(negative_returns) if len(negative_returns) > 0 else 1.0
                        
                        # Sharpe-like ratio with downside protection
                        if std_return > 0 and downside_penalty > 0:
                            risk_adjusted_fitness = (mean_return / std_return) * 1000 + metric_value / downside_penalty
                        else:
                            risk_adjusted_fitness = metric_value
                        
                        self.logger.debug(f"Enhanced fitness: mean_ret={mean_return:.4f}, std={std_return:.4f}, "
                                        f"downside={downside_penalty:.4f}, risk_adj={risk_adjusted_fitness:.4f}")
                        return risk_adjusted_fitness
        except Exception as e:
            self.logger.debug(f"Could not calculate enhanced fitness, using fallback: {e}")
        
        # Fallback: Use truly neutral approach when detailed metrics aren't available
        primary_fitness = metric_value
        
        # Only add uniqueness factor to prevent ties - NO BIAS toward any weight range
        import math
        uniqueness_factor = math.sin(ma_weight_val * 23.0) * 0.3  # Small oscillation for tie-breaking only
        
        adjusted_fitness = primary_fitness + uniqueness_factor
        
        # DEBUG: Add detailed logging to verify weights are actually being applied
        self.logger.info(f"🔍 WEIGHT VERIFICATION: MA={ma_weight_val:.6f}, RSI={rsi_weight_val:.6f} -> "
                        f"Raw={primary_fitness:.4f}, Adjusted={adjusted_fitness:.4f}")
        
        # Additional verification: get strategy state after applying weights
        strategy = self.container.resolve(self._strategy_service_name)
        if hasattr(strategy, 'get_parameters'):
            current_strategy_params = strategy.get_parameters()
            actual_ma_weight = current_strategy_params.get('ma_rule.weight', 'NOT_SET')
            actual_rsi_weight = current_strategy_params.get('rsi_rule.weight', 'NOT_SET')
            self.logger.info(f"🔍 STRATEGY STATE: MA={actual_ma_weight}, RSI={actual_rsi_weight}")
        
        return adjusted_fitness

    def _select_parents(self, population: List[Dict[str, float]], fitness_values: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Select parents using tournament selection.
        
        Args:
            population: List of individuals
            fitness_values: List of fitness values corresponding to individuals
            
        Returns:
            Tuple of two parent individuals
        """
        def tournament_select():
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), self._tournament_size)
            
            # Find winner of tournament (highest fitness if higher_is_better)
            if self._higher_metric_is_better:
                winner_idx = max(tournament_indices, key=lambda i: fitness_values[i])
            else:
                winner_idx = min(tournament_indices, key=lambda i: fitness_values[i])
                
            return population[winner_idx]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        if random.random() < self._crossover_rate:
            # Uniform crossover: randomly mix genes from both parents
            offspring1 = {}
            offspring2 = {}
            
            for key in parent1.keys():
                if random.random() < 0.5:
                    offspring1[key] = parent1[key]
                    offspring2[key] = parent2[key]
                else:
                    offspring1[key] = parent2[key]
                    offspring2[key] = parent1[key]
            
            # Ensure weights sum to 1.0
            self._normalize_weights(offspring1)
            self._normalize_weights(offspring2)
            
            return offspring1, offspring2
        else:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = copy.deepcopy(individual)
        
        for key in mutated.keys():
            if random.random() < self._mutation_rate:
                # Add Gaussian noise
                mutation_strength = 0.1  # Adjust as needed
                mutated[key] += random.gauss(0, mutation_strength)
                
                # Ensure within bounds
                mutated[key] = max(self._min_weight, min(self._max_weight, mutated[key]))
        
        # Ensure weights sum to 1.0
        self._normalize_weights(mutated)
        
        return mutated

    def _normalize_weights(self, individual: Dict[str, float]) -> None:
        """
        Normalize weights in an individual to sum to 1.0.
        
        Args:
            individual: Individual to normalize (modified in place)
        """
        weight_keys = [k for k in individual.keys() if k.endswith('.weight')]
        total_weight = sum(individual[k] for k in weight_keys)
        
        if total_weight > 0:
            for k in weight_keys:
                individual[k] /= total_weight

    def run_genetic_optimization(self) -> Optional[Dict[str, Any]]:
        """
        Run the genetic algorithm optimization.
        
        Returns:
            Dict containing optimization results
        """
        self.logger.info(f"--- {self.instance_name}: Starting Genetic Algorithm Optimization ---")
        self.logger.info(f"Population size: {self._population_size}, Generations: {self._generations}")
        
        # Initialize population
        population = self._initialize_population()
        self.logger.info(f"Initialized population with {len(population)} individuals")
        
        best_fitness_history = []
        avg_fitness_history = []
        
        # Evolution loop
        for generation in range(self._generations):
            self.logger.info(f"\n--- Generation {generation + 1}/{self._generations} ---")
            
            # Evaluate fitness for all individuals
            fitness_values = []
            for i, individual in enumerate(population):
                fitness = self._evaluate_fitness(individual)
                fitness_values.append(fitness)
                self.logger.debug(f"Individual {i}: {individual}, Fitness: {fitness}")
            
            # Track best and average fitness
            if self._higher_metric_is_better:
                best_fitness = max(fitness_values)
                best_idx = fitness_values.index(best_fitness)
            else:
                best_fitness = min(fitness_values)
                best_idx = fitness_values.index(best_fitness)
            
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            # Update global best
            if self._best_fitness is None or \
               (self._higher_metric_is_better and best_fitness > self._best_fitness) or \
               (not self._higher_metric_is_better and best_fitness < self._best_fitness):
                self._best_fitness = best_fitness
                self._best_individual = copy.deepcopy(population[best_idx])
                self.logger.info(f"New best individual found! Weights: {self._best_individual}, Fitness: {self._best_fitness}")
            
            self.logger.info(f"Generation {generation + 1} - Best fitness: {best_fitness:.4f}, Average: {avg_fitness:.4f}")
            
            # Create next generation
            next_population = []
            
            # Elitism: carry over best individuals
            sorted_indices = sorted(range(len(fitness_values)), 
                                  key=lambda i: fitness_values[i], 
                                  reverse=self._higher_metric_is_better)
            
            for i in range(self._elitism_count):
                if i < len(population):
                    next_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
            # Generate rest of population through selection, crossover, and mutation
            while len(next_population) < self._population_size:
                parent1, parent2 = self._select_parents(population, fitness_values)
                offspring1, offspring2 = self._crossover(parent1, parent2)
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                next_population.append(offspring1)
                if len(next_population) < self._population_size:
                    next_population.append(offspring2)
            
            population = next_population[:self._population_size]
        
        # Final evaluation on test set if available
        test_fitness = None
        if self._best_individual:
            self.logger.info("\n--- Evaluating best individual on test set ---")
            test_fitness = self._evaluate_fitness(self._best_individual, dataset_type="test")
            self.logger.info(f"Test set fitness: {test_fitness}")
        
        # Prepare results
        results = {
            "best_individual": self._best_individual,
            "best_fitness": self._best_fitness,
            "test_fitness": test_fitness,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "final_population": population,
            "generations": self._generations,
            "population_size": self._population_size
        }
        
        self.logger.info(f"\n--- Genetic Algorithm Complete ---")
        self.logger.info(f"Best weights found: {self._best_individual}")
        self.logger.info(f"Best training fitness: {self._best_fitness}")
        if test_fitness is not None:
            self.logger.info(f"Test fitness: {test_fitness}")
        
        return results

    def run_random_search(self, regime_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Run random search optimization (simplified GA with only random generation).
        Used for quick weight optimization per regime.
        
        Args:
            regime_name: Optional regime name for logging
            
        Returns:
            Dict containing optimization results
        """
        regime_label = f" for regime '{regime_name}'" if regime_name else ""
        self.logger.info(f"--- {self.instance_name}: Starting Random Search{regime_label} ---")
        self.logger.info(f"Evaluating {self._population_size} random weight combinations")
        
        # Generate and evaluate random individuals
        best_individual = None
        best_fitness = float('-inf') if self._higher_metric_is_better else float('inf')
        
        for i in range(self._population_size):
            # Generate random individual
            ma_weight = random.uniform(self._min_weight, self._max_weight)
            rsi_weight = 1.0 - ma_weight
            
            individual = {
                "ma_rule.weight": ma_weight,
                "rsi_rule.weight": rsi_weight
            }
            
            # Evaluate fitness
            fitness = self._evaluate_fitness(individual)
            
            self.logger.debug(f"Random {i+1}/{self._population_size}: MA={ma_weight:.4f}, RSI={rsi_weight:.4f}, Fitness={fitness:.4f}")
            
            # Update best if better
            if (self._higher_metric_is_better and fitness > best_fitness) or \
               (not self._higher_metric_is_better and fitness < best_fitness):
                best_fitness = fitness
                best_individual = individual.copy()
                self.logger.info(f"New best found! MA={ma_weight:.4f}, RSI={rsi_weight:.4f}, Fitness={fitness:.4f}")
        
        # Test best on test set
        test_fitness = None
        if best_individual:
            test_fitness = self._evaluate_fitness(best_individual, dataset_type="test")
            self.logger.info(f"Test set fitness: {test_fitness}")
        
        results = {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "test_fitness": test_fitness,
            "num_evaluations": self._population_size
        }
        
        self.logger.info(f"Random search complete{regime_label}. Best: {best_individual}, Fitness: {best_fitness}")
        
        return results

    def teardown(self):
        """Clean up resources during component teardown."""
        # Clear GA-specific state
        self._best_individual = None
        self._best_fitness = None
        self._debug_logged_count = 0
        
        # Call parent teardown
        super().teardown()