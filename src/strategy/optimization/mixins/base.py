"""
Base optimization mixin for components.

This module provides the OptimizationMixin class that adds optimization
tracking and metadata capabilities to components.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import json
import hashlib


class OptimizationMixin(ABC):
    """
    Base mixin that adds optimization capabilities to components.
    
    This mixin provides:
    - Parameter history tracking
    - Performance metric storage
    - Optimization metadata management
    - Parameter versioning
    
    Components that support optimization should inherit from both
    ComponentBase and this mixin.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize optimization tracking."""
        super().__init__(*args, **kwargs)
        
        # Optimization state
        self._parameter_history: List[Dict[str, Any]] = []
        self._performance_history: List[Dict[str, Any]] = []
        self._current_optimization_id: Optional[str] = None
        self._optimization_metadata: Dict[str, Any] = {}
        
    def record_parameter_set(self, 
                           parameters: Dict[str, Any], 
                           performance_metrics: Dict[str, float],
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a parameter set and its performance.
        
        Args:
            parameters: The parameter values used
            performance_metrics: Performance metrics achieved
            metadata: Optional additional metadata
            
        Returns:
            Unique ID for this parameter set
        """
        # Generate unique ID for this parameter set
        param_id = self._generate_parameter_id(parameters)
        
        # Create parameter record
        param_record = {
            'id': param_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters.copy(),
            'component': self.instance_name if hasattr(self, 'instance_name') else self.__class__.__name__,
            'optimization_id': self._current_optimization_id
        }
        
        if metadata:
            param_record['metadata'] = metadata
            
        # Create performance record
        perf_record = {
            'parameter_id': param_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_metrics.copy(),
            'component': self.instance_name if hasattr(self, 'instance_name') else self.__class__.__name__
        }
        
        # Store records
        self._parameter_history.append(param_record)
        self._performance_history.append(perf_record)
        
        return param_id
        
    def _generate_parameter_id(self, parameters: Dict[str, Any]) -> str:
        """Generate a unique ID for a parameter set."""
        # Create a deterministic string representation
        param_string = json.dumps(parameters, sort_keys=True)
        param_hash = hashlib.md5(param_string.encode()).hexdigest()
        return param_hash[:12]
        
    def get_best_parameters(self, 
                          metric: str = 'sharpe_ratio',
                          minimize: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best performing parameter set.
        
        Args:
            metric: The metric to optimize
            minimize: Whether to minimize (True) or maximize (False) the metric
            
        Returns:
            Best parameter set or None if no history
        """
        if not self._performance_history:
            return None
            
        # Find best performance
        best_perf = None
        best_idx = -1
        
        for i, perf in enumerate(self._performance_history):
            if metric not in perf['metrics']:
                continue
                
            value = perf['metrics'][metric]
            
            if best_perf is None:
                best_perf = value
                best_idx = i
            elif minimize and value < best_perf:
                best_perf = value
                best_idx = i
            elif not minimize and value > best_perf:
                best_perf = value
                best_idx = i
                
        if best_idx < 0:
            return None
            
        # Get corresponding parameters
        param_id = self._performance_history[best_idx]['parameter_id']
        
        for param_record in self._parameter_history:
            if param_record['id'] == param_id:
                return param_record['parameters']
                
        return None
        
    def get_parameter_history(self, 
                            optimization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get parameter history, optionally filtered by optimization run.
        
        Args:
            optimization_id: Optional optimization run ID to filter by
            
        Returns:
            List of parameter records
        """
        if optimization_id is None:
            return self._parameter_history.copy()
            
        return [
            record for record in self._parameter_history
            if record.get('optimization_id') == optimization_id
        ]
        
    def get_performance_history(self,
                               metric: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get performance history, optionally filtered by metric.
        
        Args:
            metric: Optional metric name to extract
            
        Returns:
            List of performance records
        """
        if metric is None:
            return self._performance_history.copy()
            
        # Extract specific metric
        history = []
        for record in self._performance_history:
            if metric in record['metrics']:
                history.append({
                    'parameter_id': record['parameter_id'],
                    'timestamp': record['timestamp'],
                    'value': record['metrics'][metric]
                })
                
        return history
        
    def start_optimization_run(self, optimization_id: str, metadata: Dict[str, Any]) -> None:
        """
        Mark the start of an optimization run.
        
        Args:
            optimization_id: Unique ID for this optimization run
            metadata: Metadata about the optimization (method, dataset, etc.)
        """
        self._current_optimization_id = optimization_id
        self._optimization_metadata[optimization_id] = {
            'start_time': datetime.now().isoformat(),
            'metadata': metadata,
            'component': self.instance_name if hasattr(self, 'instance_name') else self.__class__.__name__
        }
        
    def end_optimization_run(self) -> None:
        """Mark the end of the current optimization run."""
        if self._current_optimization_id:
            if self._current_optimization_id in self._optimization_metadata:
                self._optimization_metadata[self._current_optimization_id]['end_time'] = datetime.now().isoformat()
            self._current_optimization_id = None
            
    def get_optimization_summary(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of a specific optimization run.
        
        Args:
            optimization_id: The optimization run ID
            
        Returns:
            Summary dict or None if not found
        """
        if optimization_id not in self._optimization_metadata:
            return None
            
        # Get metadata
        summary = self._optimization_metadata[optimization_id].copy()
        
        # Get parameter history for this run
        params = self.get_parameter_history(optimization_id)
        summary['num_iterations'] = len(params)
        
        # Get performance stats
        perf_ids = {p['id'] for p in params}
        performances = [
            p for p in self._performance_history
            if p['parameter_id'] in perf_ids
        ]
        
        if performances:
            # Calculate performance statistics
            metrics = {}
            for perf in performances:
                for metric, value in perf['metrics'].items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
                    
            summary['performance_stats'] = {}
            for metric, values in metrics.items():
                summary['performance_stats'][metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
                
        return summary
        
    def export_optimization_results(self, 
                                  optimization_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export optimization results for persistence or analysis.
        
        Args:
            optimization_id: Optional specific optimization run to export
            
        Returns:
            Dictionary containing all optimization data
        """
        if optimization_id:
            params = self.get_parameter_history(optimization_id)
            param_ids = {p['id'] for p in params}
            perfs = [
                p for p in self._performance_history
                if p['parameter_id'] in param_ids
            ]
            metadata = {optimization_id: self._optimization_metadata.get(optimization_id, {})}
        else:
            params = self._parameter_history
            perfs = self._performance_history
            metadata = self._optimization_metadata
            
        return {
            'component': self.instance_name if hasattr(self, 'instance_name') else self.__class__.__name__,
            'parameter_history': params,
            'performance_history': perfs,
            'optimization_metadata': metadata,
            'export_timestamp': datetime.now().isoformat()
        }
        
    @abstractmethod
    def suggest_next_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Suggest next parameters to try based on optimization history.
        
        This method should be implemented by specific optimization mixins
        (e.g., GridSearchMixin, GeneticOptimizationMixin) to provide
        their optimization logic.
        
        Returns:
            Suggested parameter set or None if optimization is complete
        """
        pass