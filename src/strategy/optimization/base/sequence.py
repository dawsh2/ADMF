"""
Base class for optimization sequences/workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging


class OptimizationSequence(ABC):
    """
    Base class for optimization workflows.
    
    Sequences orchestrate complex optimization processes involving
    multiple stages, components, or methods.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._stages: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the optimization sequence.
        
        Args:
            context: Execution context with components, data, etc.
            
        Returns:
            Dictionary of results from all stages
        """
        pass
        
    def add_stage(
        self,
        name: str,
        component: Any,
        method: Any,
        metric: Any,
        **kwargs
    ) -> None:
        """Add a stage to the sequence."""
        stage = {
            'name': name,
            'component': component,
            'method': method,
            'metric': metric,
            'kwargs': kwargs
        }
        self._stages.append(stage)
        
    def get_results(self) -> List[Dict[str, Any]]:
        """Get results from all stages."""
        return self._results
        
    def get_stage_result(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get result from a specific stage."""
        for result in self._results:
            if result.get('stage') == stage_name:
                return result
        return None
        
    def _run_stage(
        self,
        stage: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a single optimization stage."""
        stage_name = stage['name']
        self.logger.info(f"Running optimization stage: {stage_name}")
        
        # Get components
        component = stage['component']
        method = stage['method']
        metric = stage['metric']
        kwargs = stage.get('kwargs', {})
        
        # Create objective function
        def objective_func(params: Dict[str, Any]) -> float:
            # Set parameters on component
            component.set_parameters(params)
            
            # Run backtest (from context)
            backtest_func = context.get('backtest_func')
            if not backtest_func:
                raise ValueError("No backtest function in context")
                
            results = backtest_func(component)
            
            # Calculate metric
            score = metric.calculate(results)
            
            return score
            
        # Get parameter space
        parameter_space = component.get_parameter_space()
        
        # Run optimization
        opt_result = method.optimize(
            objective_func=objective_func,
            parameter_space=parameter_space,
            **kwargs
        )
        
        # Record stage result
        stage_result = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'component': type(component).__name__,
            'method': type(method).__name__,
            'metric': type(metric).__name__,
            'best_params': opt_result.best_params,
            'best_score': opt_result.best_score,
            'optimization_result': opt_result
        }
        
        self._results.append(stage_result)
        
        self.logger.info(f"Stage {stage_name} complete. Best score: {opt_result.best_score}")
        
        return stage_result
        
    def reset(self) -> None:
        """Reset sequence state."""
        self._results = []
        self._metadata = {}
        

class SingleStageSequence(OptimizationSequence):
    """Simple single-stage optimization."""
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run single-stage optimization."""
        if not self._stages:
            raise ValueError("No stages defined")
            
        if len(self._stages) > 1:
            self.logger.warning(f"SingleStageSequence has {len(self._stages)} stages, only running first")
            
        # Run the single stage
        result = self._run_stage(self._stages[0], context)
        
        return {
            'sequence': self.name,
            'type': 'single_stage',
            'results': [result]
        }
        

class SequentialOptimizationSequence(OptimizationSequence):
    """Multi-stage sequential optimization."""
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run stages sequentially, passing results forward."""
        results = []
        previous_result = None
        
        for i, stage in enumerate(self._stages):
            self.logger.info(f"Running stage {i+1}/{len(self._stages)}: {stage['name']}")
            
            # Pass previous results in context
            stage_context = context.copy()
            if previous_result:
                stage_context['previous_result'] = previous_result
                
            # Run stage
            result = self._run_stage(stage, stage_context, previous_result)
            results.append(result)
            
            # Apply best parameters before next stage
            component = stage['component']
            component.set_parameters(result['best_params'])
            
            previous_result = result
            
        return {
            'sequence': self.name,
            'type': 'sequential',
            'results': results
        }
        

class ParallelOptimizationSequence(OptimizationSequence):
    """Run multiple optimizations in parallel."""
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all stages in parallel (simulated for now)."""
        results = []
        
        # Note: This is a simplified implementation
        # Real parallel execution would use multiprocessing
        for stage in self._stages:
            result = self._run_stage(stage, context)
            results.append(result)
            
        return {
            'sequence': self.name,
            'type': 'parallel',
            'results': results
        }