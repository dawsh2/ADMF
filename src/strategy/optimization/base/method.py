"""
Base class for optimization methods/algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class OptimizationResult:
    """Result from an optimization run."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Add timestamp if not present."""
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.all_results,
            'metadata': self.metadata
        }
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create from dictionary representation."""
        return cls(
            best_params=data['best_params'],
            best_score=data['best_score'],
            all_results=data.get('all_results', []),
            metadata=data.get('metadata', {})
        )


class OptimizationMethod(ABC):
    """
    Base class for optimization algorithms.
    
    Subclasses implement specific search strategies like
    grid search, genetic algorithms, Bayesian optimization, etc.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None
        self._all_results: List[Dict[str, Any]] = []
        self._iteration = 0
        
    @abstractmethod
    def optimize(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        parameter_space: Any,  # ParameterSpace
        n_iterations: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Run optimization.
        
        Args:
            objective_func: Function that takes parameters and returns score
            parameter_space: ParameterSpace defining search space
            n_iterations: Maximum iterations (method-specific meaning)
            **kwargs: Method-specific arguments
            
        Returns:
            OptimizationResult with best parameters and score
        """
        pass
        
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        return self._best_params
        
    def get_best_score(self) -> Optional[float]:
        """Get best score found so far."""
        return self._best_score
        
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results."""
        return self._all_results
        
    def _record_result(
        self,
        params: Dict[str, Any],
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an evaluation result."""
        result = {
            'iteration': self._iteration,
            'params': params.copy(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            result['metadata'] = metadata
            
        self._all_results.append(result)
        
        # Update best if needed
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._best_params = params.copy()
            
    def _create_result(self, metadata: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Create optimization result."""
        result_metadata = {
            'method': self.name,
            'iterations': self._iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            result_metadata.update(metadata)
            
        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score or float('-inf'),
            all_results=self._all_results,
            metadata=result_metadata
        )
        
    def reset(self) -> None:
        """Reset optimizer state."""
        self._best_params = None
        self._best_score = None
        self._all_results = []
        self._iteration = 0