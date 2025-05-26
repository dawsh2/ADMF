"""
Random search optimization implementation.
"""

import random
from typing import Dict, Any, List, Optional, Callable
import logging

from ..base import OptimizationMethod, OptimizationResult
from ...base.parameter import ParameterSpace, Parameter


class RandomSearchOptimizer(OptimizationMethod):
    """
    Random search optimization method.
    
    Randomly samples parameter combinations from the space.
    Often more efficient than grid search for high-dimensional spaces.
    """
    
    def __init__(self, name: str = "RandomSearch", seed: Optional[int] = None):
        super().__init__(name)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            
    def optimize(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        parameter_space: ParameterSpace,
        n_iterations: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run random search optimization.
        
        Args:
            objective_func: Function that takes parameters and returns score
            parameter_space: ParameterSpace defining search space
            n_iterations: Number of random samples to evaluate (default: 100)
            **kwargs: Additional arguments
                
        Returns:
            OptimizationResult with best parameters and score
        """
        # Reset state
        self.reset()
        
        # Default iterations if not specified
        if n_iterations is None:
            # Use 10% of grid search space or 100, whichever is smaller
            grid_size = len(parameter_space.sample(method='grid'))
            n_iterations = min(100, max(10, grid_size // 10))
            
        self.logger.info(f"Starting random search with {n_iterations} iterations")
        
        # Generate random samples
        for i in range(n_iterations):
            self._iteration = i + 1
            
            # Log progress
            if (i + 1) % 10 == 0 or i == 0:
                self.logger.info(f"Evaluating sample {i+1}/{n_iterations}")
                
            try:
                # Generate random parameters
                params = self._sample_parameters(parameter_space)
                
                # Evaluate parameters
                score = objective_func(params)
                
                # Record result
                previous_best = self._best_score
                self._record_result(params, score)
                
                # Log improvement
                if previous_best is not None and self._best_score is not None:
                    if self._best_score > previous_best:
                        improvement = self._best_score - previous_best
                        self.logger.info(f"New best score: {self._best_score:.6f} "
                                       f"(improvement: {improvement:.6f})")
                        
            except Exception as e:
                self.logger.error(f"Error evaluating parameters: {e}")
                # Continue with next sample
                
        # Create result
        metadata = {
            'iterations': n_iterations,
            'seed': self.seed
        }
        
        return self._create_result(metadata)
        
    def _sample_parameters(self, parameter_space: ParameterSpace) -> Dict[str, Any]:
        """Generate random parameter sample."""
        params = {}
        
        # Sample local parameters
        for name, param in parameter_space._parameters.items():
            params[name] = self._sample_parameter(param)
            
        # Sample from subspaces
        for subspace_name, subspace in parameter_space._subspaces.items():
            subspace_params = self._sample_parameters(subspace)
            for key, value in subspace_params.items():
                params[f"{subspace_name}.{key}"] = value
                
        return params
        
    def _sample_parameter(self, param: Parameter) -> Any:
        """Sample a single parameter value."""
        if param.param_type == 'discrete':
            if param.values:
                return random.choice(param.values)
            else:
                return param.default
                
        elif param.param_type == 'continuous':
            min_val = param.min_value or 0
            max_val = param.max_value or 1
            
            if param.step:
                # Sample with step size
                steps = int((max_val - min_val) / param.step)
                step_idx = random.randint(0, steps)
                return min_val + step_idx * param.step
            else:
                # Continuous sampling
                return random.uniform(min_val, max_val)
                
        elif param.param_type == 'categorical':
            if param.values:
                return random.choice(param.values)
                
        return param.default