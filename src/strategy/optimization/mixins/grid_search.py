"""
Grid search optimization mixin for components.

This module provides the GridSearchMixin that implements grid search
optimization logic for components.
"""

from typing import Dict, Any, Optional, List, Set
import itertools
from .base import OptimizationMixin


class GridSearchMixin(OptimizationMixin):
    """
    Mixin that adds grid search optimization capabilities to components.
    
    This mixin implements exhaustive grid search over the parameter space,
    trying all possible combinations of discrete parameter values.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize grid search state."""
        super().__init__(*args, **kwargs)
        
        # Grid search specific state
        self._grid_combinations: List[Dict[str, Any]] = []
        self._tried_parameters: Set[str] = set()
        self._current_grid_index: int = 0
        
    def initialize_grid_search(self, parameter_space: Optional['ParameterSpace'] = None) -> int:
        """
        Initialize grid search with parameter combinations.
        
        Args:
            parameter_space: Optional parameter space to use. If None,
                           uses component's get_parameter_space()
                           
        Returns:
            Number of combinations to try
        """
        # Get parameter space
        if parameter_space is None:
            if hasattr(self, 'get_parameter_space'):
                parameter_space = self.get_parameter_space()
            else:
                raise ValueError("No parameter space provided and component has no get_parameter_space method")
                
        if parameter_space is None:
            raise ValueError("Parameter space is None")
            
        # Generate all combinations
        self._grid_combinations = parameter_space.sample(method='grid')
        self._current_grid_index = 0
        self._tried_parameters.clear()
        
        return len(self._grid_combinations)
        
    def suggest_next_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Suggest next parameters in the grid search.
        
        Returns:
            Next parameter set or None if all combinations tried
        """
        # Skip already tried combinations
        while self._current_grid_index < len(self._grid_combinations):
            params = self._grid_combinations[self._current_grid_index]
            param_id = self._generate_parameter_id(params)
            
            if param_id not in self._tried_parameters:
                self._tried_parameters.add(param_id)
                return params
                
            self._current_grid_index += 1
            
        return None  # All combinations tried
        
    def get_grid_progress(self) -> Dict[str, Any]:
        """
        Get current progress of grid search.
        
        Returns:
            Dictionary with progress information
        """
        total = len(self._grid_combinations)
        completed = len(self._tried_parameters)
        
        return {
            'method': 'grid_search',
            'total_combinations': total,
            'completed': completed,
            'remaining': max(0, total - completed),
            'progress_percent': (completed / total * 100) if total > 0 else 100.0,
            'current_index': self._current_grid_index
        }
        
    def get_search_space_info(self) -> Dict[str, Any]:
        """
        Get information about the search space.
        
        Returns:
            Dictionary with search space details
        """
        if not hasattr(self, 'get_parameter_space'):
            return {'error': 'Component has no parameter space'}
            
        space = self.get_parameter_space()
        if space is None:
            return {'error': 'Parameter space is None'}
            
        info = {
            'parameters': {},
            'total_combinations': len(self._grid_combinations) if self._grid_combinations else 0
        }
        
        # Get parameter details
        for name, param in space._parameters.items():
            param_info = {
                'type': param.param_type,
                'default': param.default
            }
            
            if param.param_type == 'discrete' and param.values:
                param_info['values'] = param.values
                param_info['num_values'] = len(param.values)
            elif param.param_type == 'continuous':
                param_info['min'] = param.min_value
                param_info['max'] = param.max_value
                param_info['step'] = param.step
                
            info['parameters'][name] = param_info
            
        return info
        
    def reset_grid_search(self) -> None:
        """Reset grid search state."""
        self._grid_combinations.clear()
        self._tried_parameters.clear()
        self._current_grid_index = 0