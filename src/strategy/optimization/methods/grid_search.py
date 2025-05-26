"""
Grid search optimization implementation.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime

from ..base import OptimizationMethod, OptimizationResult
from ...base.parameter import ParameterSpace


class GridSearchOptimizer(OptimizationMethod):
    """
    Grid search optimization method.
    
    Exhaustively searches all parameter combinations in the space.
    """
    
    def __init__(self, name: str = "GridSearch"):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def optimize(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        parameter_space: ParameterSpace,
        n_iterations: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run grid search optimization.
        
        Args:
            objective_func: Function that takes parameters and returns score
            parameter_space: ParameterSpace defining search space
            n_iterations: Not used for grid search (searches all combinations)
            **kwargs: Additional arguments
                - early_stopping_rounds: Stop if no improvement for N rounds
                - min_improvement: Minimum improvement to continue
                
        Returns:
            OptimizationResult with best parameters and score
        """
        # Reset state
        self.reset()
        
        # Get all parameter combinations
        param_combinations = parameter_space.sample(method='grid')
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        # Early stopping parameters
        early_stopping_rounds = kwargs.get('early_stopping_rounds', None)
        min_improvement = kwargs.get('min_improvement', 1e-6)
        rounds_without_improvement = 0
        
        # Search all combinations
        for i, params in enumerate(param_combinations):
            self._iteration = i + 1
            
            # Log progress
            if (i + 1) % 10 == 0 or i == 0:
                self.logger.info(f"Evaluating combination {i+1}/{total_combinations}")
                
            try:
                # Log parameters being tested
                self.logger.info(f"Testing parameters: {params}")
                
                # Evaluate parameters
                score = objective_func(params)
                
                # Record result
                previous_best = self._best_score
                self._record_result(params, score)
                
                # Check for improvement
                if previous_best is not None and self._best_score is not None:
                    improvement = self._best_score - previous_best
                    if improvement > min_improvement:
                        rounds_without_improvement = 0
                        self.logger.info(f"New best score: {self._best_score:.6f} "
                                       f"(improvement: {improvement:.6f})")
                    else:
                        rounds_without_improvement += 1
                        
                # Early stopping check
                if (early_stopping_rounds is not None and 
                    rounds_without_improvement >= early_stopping_rounds):
                    self.logger.info(f"Early stopping: No improvement for "
                                   f"{early_stopping_rounds} rounds")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {e}")
                # Record failed evaluation
                self._record_result(params, float('-inf'), {'error': str(e)})
                
        # Create result
        metadata = {
            'total_combinations': total_combinations,
            'evaluated_combinations': self._iteration,
            'early_stopped': self._iteration < total_combinations
        }
        
        return self._create_result(metadata)
        
    def get_search_progress(self) -> Dict[str, Any]:
        """Get current search progress."""
        return {
            'iterations_completed': self._iteration,
            'best_score': self._best_score,
            'best_params': self._best_params
        }