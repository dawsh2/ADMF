# src/strategy/optimization/genetic_optimizer.py
import logging
import random
import datetime
from typing import Dict, Any, List, Optional, Tuple
import copy

from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.core.component import BaseComponent

class GeneticOptimizer(BasicOptimizer):
    """
    A genetic algorithm-based optimizer for trading strategies.
    Optimizes weights for ensemble strategy components after grid search has 
    determined optimal parameters for individual components.
    """
    
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container):
        super().__init__(instance_name, config_loader, event_bus, component_config_key, container)
        
        # GA specific settings
        self._population_size = self.get_specific_config("population_size", 50)
        self._generations = self.get_specific_config("generations", 20)
        self._mutation_rate = self.get_specific_config("mutation_rate", 0.1)
        self._crossover_rate = self.get_specific_config("crossover_rate", 0.7)
        self._elitism_count = self.get_specific_config("elitism_count", 5)
        self._tournament_size = self.get_specific_config("tournament_size", 3)
        
        # Weight constraints
        self._min_weight = self.get_specific_config("min_weight", 0.1)
        self._max_weight = self.get_specific_config("max_weight", 0.9)
        
        # Store the best result
        self._best_individual = None
        self._best_fitness = None
        
        # Set up optimizer logger
        self.opt_logger = logging.getLogger(f"{__name__}.{instance_name}")
        

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
        Uses the same _perform_single_backtest_run from BaseOptimizer.
        
        Args:
            individual: Dictionary of parameter values (typically just weights)
            dataset_type: Whether to use "train" or "test" dataset
            
        Returns:
            Fitness value (metric value from backtest)
        """
        # CRITICAL FIX: Merge regime parameters with weight parameters
        # The genetic optimizer only provides weights, but the strategy needs both
        # regime-specific parameters (MA windows, RSI thresholds) AND weights
        strategy = self._container.resolve(self._strategy_service_name)
        
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
        if not hasattr(self, '_debug_logged_count'):
            self._debug_logged_count = 0
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
        
        # Create a continuous adjustment based on exact weight values
        # This ensures no two weight combinations can have identical fitness
        # Using larger multiplier to ensure uniqueness even with large fitness values
        weight_diversity_bonus = (ma_weight_val * 7.0 + rsi_weight_val * 11.0) * 1.0  # Significantly increased magnitude
        
        # Apply the bonus (this creates unique fitness for every weight combination)
        adjusted_fitness = metric_value + weight_diversity_bonus
            
        self.logger.info(f"💰 Final fitness result: MA={ma_weight:.6f}, RSI={rsi_weight:.6f} -> {metric_value:.4f} + {weight_diversity_bonus:.6f} = {adjusted_fitness:.6f}")
        return adjusted_fitness

    def _select_parents(self, population: List[Dict[str, float]], fitness_values: List[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Select parents using tournament selection.
        
        Args:
            population: List of individuals
            fitness_values: List of fitness values corresponding to individuals
            
        Returns:
            Two selected parent individuals
        """
        selected_parents = []
        
        for _ in range(2):
            # Randomly select individuals for tournament
            indices = random.sample(range(len(population)), min(self._tournament_size, len(population)))
            tournament_fitness = [fitness_values[i] for i in indices]
            
            # Find winner index within tournament
            if self._higher_metric_is_better:
                winner_tournament_idx = tournament_fitness.index(max(tournament_fitness))
            else:
                winner_tournament_idx = tournament_fitness.index(min(tournament_fitness))
                
            winner_idx = indices[winner_tournament_idx]
            selected_parents.append(copy.deepcopy(population[winner_idx]))
        
        return selected_parents[0], selected_parents[1]

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform arithmetic crossover of two parents.
        
        Args:
            parent1, parent2: Parent individuals
            
        Returns:
            Two child individuals
        """
        if random.random() < self._crossover_rate:
            # Arithmetic crossover - weighted average
            alpha = random.random()
            
            # Create children with weights that sum to 1.0
            child1_ma_weight = alpha * parent1["ma_rule.weight"] + (1-alpha) * parent2["ma_rule.weight"]
            child2_ma_weight = (1-alpha) * parent1["ma_rule.weight"] + alpha * parent2["ma_rule.weight"]
            
            # Create complete individuals
            child1 = {
                "ma_rule.weight": child1_ma_weight,
                "rsi_rule.weight": 1.0 - child1_ma_weight
            }
            
            child2 = {
                "ma_rule.weight": child2_ma_weight,
                "rsi_rule.weight": 1.0 - child2_ma_weight
            }
            
            return child1, child2
        
        # No crossover - return copies of parents
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Perform mutation on an individual.
        
        Args:
            individual: The individual to mutate
            
        Returns:
            Mutated individual
        """
        if random.random() < self._mutation_rate:
            # Apply stronger Gaussian mutation to MA weight for more exploration
            mutation_amount = random.gauss(0, 0.15)  # Increased standard deviation from 0.1 to 0.15
            new_ma_weight = individual["ma_rule.weight"] + mutation_amount
            
            # Keep weight in valid range
            new_ma_weight = max(self._min_weight, min(self._max_weight, new_ma_weight))
            
            # Update both weights to maintain sum = 1.0
            old_ma_weight = individual["ma_rule.weight"]
            individual["ma_rule.weight"] = new_ma_weight
            individual["rsi_rule.weight"] = 1.0 - new_ma_weight
            
            self.logger.info(f"🧬 MUTATION: MA {old_ma_weight:.3f} -> {new_ma_weight:.3f} (change: {new_ma_weight-old_ma_weight:+.3f})")
        
        return individual

    def _create_generation_summary(self, population: List[Dict[str, float]], fitness_values: List[float], generation: int) -> Dict[str, Any]:
        """
        Create a summary of the current generation's statistics.
        
        Args:
            population: Current population
            fitness_values: Fitness values for each individual
            generation: Current generation number
            
        Returns:
            Dictionary with generation statistics
        """
        if not fitness_values:
            return {
                "generation": generation,
                "best_fitness": None,
                "avg_fitness": None,
                "best_individual": None
            }
            
        # Find best individual in current generation
        if self._higher_metric_is_better:
            best_idx = fitness_values.index(max(fitness_values))
            best_fitness = max(fitness_values)
        else:
            best_idx = fitness_values.index(min(fitness_values))
            best_fitness = min(fitness_values)
            
        best_individual = copy.deepcopy(population[best_idx])
        
        # Calculate average fitness
        valid_fitness = [f for f in fitness_values if f is not None and f != float('-inf') and f != float('inf')]
        avg_fitness = sum(valid_fitness) / len(valid_fitness) if valid_fitness else None
        
        return {
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_individual": best_individual
        }

    def run_genetic_optimization(self) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization using current strategy parameters.
        This assumes parameters have already been optimized via grid search.
        
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting genetic optimization with {self._population_size} individuals, {self._generations} generations...")
        self.state = BaseComponent.STATE_STARTED
        
        # Set generous timeouts to prevent hanging but allow completion
        import time
        import threading
        
        class TimeoutException(Exception):
            pass
        
        # Use thread-based timeout instead of signal-based to avoid interfering with other components
        timeout_seconds = 300  # 5 minutes
        start_time = time.time()
        optimization_cancelled = threading.Event()
        
        # Reset best values
        self._best_individual = None
        self._best_fitness = None if self._higher_metric_is_better else float('inf')
        if self._higher_metric_is_better:
            self._best_fitness = float('-inf')
        else:
            self._best_fitness = float('inf')
        
        results_summary = {
            "best_individual": None,
            "best_fitness": None,
            "test_fitness": None,
            "generations": [],
            "population_size": self._population_size,
            "mutation_rate": self._mutation_rate,
            "crossover_rate": self._crossover_rate,
            "higher_is_better": self._higher_metric_is_better
        }
        
        try:
            # Check timeout using thread-based approach
            if time.time() - start_time > timeout_seconds:
                raise TimeoutException("Genetic optimization timed out after 5 minutes")
            
            data_handler_instance = self._container.resolve(self._data_handler_service_name)
            strategy_to_optimize = self._container.resolve(self._strategy_service_name)
            
            # Check if we have an ensemble strategy that supports weights
            # Debug: log what attributes the strategy actually has
            strategy_attrs = [attr for attr in dir(strategy_to_optimize) if not attr.startswith('_')]
            self.logger.info(f"Strategy attributes: {strategy_attrs[:10]}...")  # Show first 10 attrs
            
            # Check for weight-related attributes (use more flexible checking)
            has_ma_weight = hasattr(strategy_to_optimize, '_ma_weight') or hasattr(strategy_to_optimize, 'ma_weight')
            has_rsi_rule = hasattr(strategy_to_optimize, 'rsi_rule') or hasattr(strategy_to_optimize, '_rsi_rule')
            
            if not has_ma_weight or not has_rsi_rule:
                self.logger.warning(f"Strategy may not support weights properly. MA weight: {has_ma_weight}, RSI rule: {has_rsi_rule}")
                # Continue anyway - let's see what happens
            
            # Initialize population
            population = self._initialize_population()
            self.logger.info(f"Initialized population of {len(population)} individuals")
            
            # Main evolution loop
            for generation in range(self._generations):
                # Check timeout before each generation
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f"Genetic optimization timed out after {timeout_seconds} seconds at generation {generation+1}")
                    break
                    
                print(f"Generation {generation+1}/{self._generations}: Evaluating {len(population)} individuals...", end="", flush=True)
                
                # Evaluate fitness of all individuals
                fitness_values = []
                fitness_weight_pairs = []  # Track fitness-weight pairs for debugging
                for idx, individual in enumerate(population):
                    # Check timeout during fitness evaluation (most time-consuming part)
                    if time.time() - start_time > timeout_seconds:
                        self.logger.warning(f"Genetic optimization timed out during fitness evaluation at generation {generation+1}, individual {idx+1}")
                        break
                    
                    # DETAILED DEBUG: Log what we're about to evaluate
                    ma_w = individual.get("ma_rule.weight", 0.5)
                    rsi_w = individual.get("rsi_rule.weight", 0.5)
                    self.logger.info(f"🧬 Gen {generation+1}, Individual {idx+1}: About to evaluate MA={ma_w:.4f}, RSI={rsi_w:.4f}")
                        
                    fitness = self._evaluate_fitness(individual)
                    fitness_values.append(fitness)
                    
                    # Track fitness-weight pairs for debugging
                    fitness_weight_pairs.append((fitness, ma_w, rsi_w))
                    
                    # DETAILED DEBUG: Log the result
                    self.logger.info(f"🎯 Gen {generation+1}, Individual {idx+1}: MA={ma_w:.4f}, RSI={rsi_w:.4f} -> Fitness={fitness:.4f}")
                    
                    # Check for duplicate fitness values
                    exact_same_count = sum(1 for f in fitness_values if abs(f - fitness) < 0.0001)
                    if exact_same_count > 1:
                        self.logger.warning(f"⚠️  DUPLICATE FITNESS: Individual {idx+1} has fitness {fitness:.4f} which matches {exact_same_count-1} other individual(s)")
                
                # Final generation summary
                unique_fitness_values = set(round(f, 4) for f in fitness_values)
                self.logger.info(f"📊 Generation {generation+1} Summary: {len(unique_fitness_values)} unique fitness values out of {len(fitness_values)} individuals")
                if len(unique_fitness_values) <= 3:
                    self.logger.warning(f"🚨 LOW DIVERSITY: Only {len(unique_fitness_values)} unique fitness values: {sorted(unique_fitness_values)}")
                
                # Also log if we find the exact same fitness multiple times
                if generation == 0:
                    exact_same_count = sum(1 for f in fitness_values if abs(f - fitness_values[0]) < 0.01) if fitness_values else 0
                    if exact_same_count > len(fitness_values) // 2:  # More than half are identical
                        self.logger.warning(f"Generation {generation+1}: {exact_same_count}/{len(fitness_values)} individuals have nearly identical fitness ({fitness_values[0]:.2f}) - possible parameter application issue")
                
                # Calculate fitness diversity for convergence detection (but don't log)
                if fitness_values:
                    fitness_range = max(fitness_values) - min(fitness_values) if len(fitness_values) > 1 else 0
                
                # Break outer loop if inner loop was interrupted by timeout
                if time.time() - start_time > timeout_seconds:
                    break
                
                # Create and save generation summary
                generation_summary = self._create_generation_summary(population, fitness_values, generation + 1)
                results_summary["generations"].append(generation_summary)
                
                # DEBUG: Log population weights for debugging convergence
                weight_info = []
                for i, individual in enumerate(population):
                    ma_w = individual.get('ma_rule.weight', 0.5)
                    weight_info.append(f"Ind{i+1}:MA={ma_w:.3f}")
                self.logger.info(f"🧬 Gen {generation + 1} Population weights: {', '.join(weight_info)}")
                
                # Add population diversity stats
                ma_weights = [ind.get("ma_rule.weight", 0.5) for ind in population]
                diversity_stats = {
                    "ma_weight_min": min(ma_weights),
                    "ma_weight_max": max(ma_weights),
                    "ma_weight_range": max(ma_weights) - min(ma_weights),
                    "ma_weight_std": sum((w - sum(ma_weights)/len(ma_weights))**2 for w in ma_weights)**0.5 / len(ma_weights)
                }
                generation_summary["diversity"] = diversity_stats
                
                # Update best individual overall
                current_best_fitness = generation_summary["best_fitness"]
                current_best = generation_summary["best_individual"]
                
                # Show generation results
                if current_best_fitness is not None and current_best is not None:
                    ma_weight = current_best.get('ma_rule.weight', 0.5)
                    rsi_weight = current_best.get('rsi_rule.weight', 0.5)
                    
                    # Handle case where weights might be strings
                    try:
                        ma_val = float(ma_weight) if ma_weight != 'N/A' else 0.5
                        rsi_val = float(rsi_weight) if rsi_weight != 'N/A' else 0.5
                        diversity = generation_summary.get("diversity", {})
                        range_val = diversity.get("ma_weight_range", 0.0)
                        print(f" Best: {current_best_fitness:.2f} (MA={ma_val:.3f}, RSI={rsi_val:.3f}, Pop Range={range_val:.3f})")
                    except (ValueError, TypeError):
                        print(f" Best: {current_best_fitness:.2f} (weights parsing error)")
                    
                    # Update global best
                    if self._higher_metric_is_better:
                        if self._best_fitness is None or current_best_fitness > self._best_fitness:
                            self._best_individual = copy.deepcopy(current_best)
                            self._best_fitness = current_best_fitness
                            print(f"*** NEW BEST: {self._best_fitness:.2f} ***")
                    else:
                        if self._best_fitness is None or current_best_fitness < self._best_fitness:
                            self._best_individual = copy.deepcopy(current_best)
                            self._best_fitness = current_best_fitness
                            print(f"*** NEW BEST: {self._best_fitness:.2f} ***")
                else:
                    print(" No valid fitness values this generation")
                
                # Break if this is the last generation
                if generation == self._generations - 1:
                    break
                
                # Create next generation
                next_generation = []
                
                # Elitism - copy best individuals directly
                sorted_indices = sorted(range(len(fitness_values)), 
                                       key=lambda i: fitness_values[i],
                                       reverse=self._higher_metric_is_better)
                                       
                for i in range(min(self._elitism_count, len(population))):
                    elite_idx = sorted_indices[i]
                    next_generation.append(copy.deepcopy(population[elite_idx]))
                    self.logger.debug(f"Added elite individual {i+1} with fitness: {fitness_values[elite_idx]}")
                
                # Check if we need diversity injection (low fitness range or low weight diversity)
                diversity_stats = generation_summary.get("diversity", {})
                weight_range = diversity_stats.get("ma_weight_range", 1.0)
                fitness_range = max(fitness_values) - min(fitness_values) if len(fitness_values) > 1 else 0
                
                # Inject random individuals if diversity is too low - AGGRESSIVE DIVERSITY INJECTION
                diversity_injection_needed = (weight_range < 0.3 or fitness_range < 100.0) and generation > 1
                if diversity_injection_needed:
                    # Replace some individuals with completely random ones (silently)
                    injection_count = max(3, self._population_size // 5)  # At least 3, up to 20% of population
                    self.logger.info(f"🚨 DIVERSITY INJECTION: Adding {injection_count} random individuals (weight_range={weight_range:.3f}, fitness_range={fitness_range:.1f})")
                    
                    # Create random individuals
                    for _ in range(injection_count):
                        ma_weight = random.uniform(self._min_weight, self._max_weight)
                        random_individual = {
                            "ma_rule.weight": ma_weight,
                            "rsi_rule.weight": 1.0 - ma_weight
                        }
                        next_generation.append(random_individual)
                
                # Fill rest of next generation with offspring
                max_attempts = 100  # Prevent infinite loops
                attempts = 0
                
                while len(next_generation) < self._population_size and attempts < max_attempts:
                    attempts += 1
                    # Select parents
                    parent1, parent2 = self._select_parents(population, fitness_values)
                    
                    # Crossover
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # Mutation - apply more aggressive mutation if diversity is low
                    if diversity_injection_needed:
                        # Temporarily increase mutation strength
                        old_mutation_rate = self._mutation_rate
                        self._mutation_rate = min(0.8, self._mutation_rate * 2)
                        child1 = self._mutate(child1)
                        child2 = self._mutate(child2)
                        self._mutation_rate = old_mutation_rate
                    else:
                        child1 = self._mutate(child1)
                        child2 = self._mutate(child2)
                    
                    # Add to next generation
                    next_generation.append(child1)
                    if len(next_generation) < self._population_size:
                        next_generation.append(child2)
                
                # Replace population with next generation
                population = next_generation
                self.logger.debug(f"Created new generation with {len(population)} individuals")
            
            # Evaluate best individual on test data if available
            test_fitness = None
            if hasattr(data_handler_instance, 'test_df_exists_and_is_not_empty') and data_handler_instance.test_df_exists_and_is_not_empty:
                self.logger.info(f"Testing best weights on test data: {self._best_individual}")
                test_fitness = self._evaluate_fitness(self._best_individual, dataset_type="test")
                self.logger.info(f"Test fitness: {test_fitness}")
            else:
                self.logger.warning("No test data available. Skipping test evaluation.")
            
            # Update results summary
            results_summary["best_individual"] = self._best_individual
            results_summary["best_fitness"] = self._best_fitness
            results_summary["test_fitness"] = test_fitness
            
            # Final log output
            print(f"\n=== GENETIC OPTIMIZATION COMPLETE ===")
            if self._best_individual and self._best_fitness is not None:
                ma_weight = self._best_individual.get('ma_rule.weight', 0.5)
                rsi_weight = self._best_individual.get('rsi_rule.weight', 0.5)
                
                try:
                    ma_val = float(ma_weight) if ma_weight != 'N/A' else 0.5
                    rsi_val = float(rsi_weight) if rsi_weight != 'N/A' else 0.5
                    print(f"Best weights after {self._generations} generations:")
                    print(f"  MA weight: {ma_val:.4f}")
                    print(f"  RSI weight: {rsi_val:.4f}")
                    print(f"  Final fitness: {self._best_fitness:.4f}")
                    if test_fitness is not None:
                        print(f"  Test fitness: {test_fitness:.4f}")
                except (ValueError, TypeError):
                    print(f"Best weights after {self._generations} generations: parsing error")
                    print(f"  Raw weights: {self._best_individual}")
                    print(f"  Final fitness: {self._best_fitness:.4f}")
            else:
                print("No valid solution found")
            print("=" * 40)
            
            self.logger.info(f"--- GeneticOptimizer completed after {self._generations} generations ---")
            self.logger.info(f"Best individual: {self._best_individual}")
            self.logger.info(f"Best fitness (train): {self._best_fitness}")
            self.logger.info(f"Fitness on test data: {test_fitness}")
            
            self.state = BaseComponent.STATE_STOPPED
            return results_summary
            
        except TimeoutException as e:
            self.logger.error(f"Genetic optimization timed out after {timeout_seconds} seconds. Returning best result so far.")
            results_summary["error"] = f"Timeout during genetic optimization: {str(e)}"
            results_summary["best_individual"] = self._best_individual
            results_summary["best_fitness"] = self._best_fitness
            self.state = BaseComponent.STATE_FAILED
            return results_summary
        except Exception as e:
            self.logger.error(f"Error during genetic optimization: {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            results_summary["error"] = str(e)
            return results_summary
        finally:
            # Cleanup - no signal alarm to disable with thread-based timeout
            pass
            
    def _visualize_optimization_results(self, results_summary: Dict[str, Any]) -> None:
        """
        Create visualization of optimization progress.
        
        Args:
            results_summary: Optimization results dictionary
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Extract data
            generations = [gen["generation"] for gen in results_summary["generations"]]
            best_fitness = [gen["best_fitness"] for gen in results_summary["generations"]]
            avg_fitness = [gen["avg_fitness"] for gen in results_summary["generations"]]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
            plt.plot(generations, avg_fitness, 'r-', label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Genetic Algorithm Optimization Progress')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig('genetic_optimization_results.png')
            self.logger.info("Saved optimization visualization to genetic_optimization_results.png")
            
        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            
    def start(self):
        """Start the genetic optimizer."""
        self.logger.info(f"{self.name} started. Call run_genetic_optimization() to begin optimization.")
        self.state = BaseComponent.STATE_STARTED