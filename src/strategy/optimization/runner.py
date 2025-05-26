"""
Optimization runner using scoped contexts for isolation.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import os

from ...core.component_base import ComponentBase
from ...core.event import EventType
from ..base import Strategy, ParameterSet
from .base import (
    OptimizationTarget,
    OptimizationMethod,
    OptimizationMetric,
    OptimizationResult
)
from .methods import GridSearchOptimizer, RandomSearchOptimizer
from .base.metric import (
    SharpeRatioMetric,
    TotalReturnMetric,
    MaxDrawdownMetric,
    CompositeMetric
)


class OptimizationRunner(ComponentBase):
    """
    Runs optimization using scoped contexts for proper isolation.
    
    This is the main entry point for strategy optimization.
    """
    
    def __init__(self, instance_name: str = "optimization_runner", config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        self._bootstrap = None
        self._base_config = None
        self._results_dir = "optimization_results"
        
    def _initialize(self):
        """Initialize optimization runner."""
        config = self.component_config or {}
        self._results_dir = config.get('results_dir', 'optimization_results')
        
        # Create results directory
        os.makedirs(self._results_dir, exist_ok=True)
        
        self.logger.info(f"OptimizationRunner initialized. Results directory: {self._results_dir}")
        
    def set_bootstrap(self, bootstrap: Any) -> None:
        """Set bootstrap instance for creating scoped contexts."""
        self._bootstrap = bootstrap
        
    def set_base_config(self, config: Dict[str, Any]) -> None:
        """Set base configuration for optimization runs."""
        self._base_config = config
        
    def optimize_strategy(
        self,
        strategy_name: str,
        method: Optional[OptimizationMethod] = None,
        metric: Optional[OptimizationMetric] = None,
        n_iterations: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize a strategy using specified method and metric.
        
        Args:
            strategy_name: Name of strategy component to optimize
            method: Optimization method (default: GridSearch)
            metric: Optimization metric (default: SharpeRatio)
            n_iterations: Number of iterations
            **kwargs: Additional arguments for method
            
        Returns:
            OptimizationResult with best parameters
        """
        if not self._bootstrap:
            raise ValueError("Bootstrap not set. Call set_bootstrap() first.")
            
        if not self._base_config:
            raise ValueError("Base config not set. Call set_base_config() first.")
            
        # Default method and metric
        if method is None:
            method = GridSearchOptimizer()
        if metric is None:
            metric = SharpeRatioMetric()
            
        self.logger.info(f"Starting optimization of '{strategy_name}' "
                        f"using {method.name} optimizing {metric.name}")
                        
        # Create objective function
        def objective_func(params: Dict[str, Any]) -> float:
            return self._evaluate_parameters(strategy_name, params, metric)
            
        # Get strategy parameter space
        # First create a context to get the strategy
        temp_context = self._bootstrap.create_scoped_context("temp_param_space")
        
        # Get strategy from the container
        try:
            strategy = temp_context.container.resolve(strategy_name)
        except Exception:
            # Try from bootstrap components
            strategy = self._bootstrap.get_component(strategy_name)
            
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
            
        # Check if strategy has get_parameter_space method
        if not hasattr(strategy, 'get_parameter_space'):
            raise ValueError(f"Strategy '{strategy_name}' does not support optimization (no get_parameter_space method)")
            
        parameter_space = strategy.get_parameter_space()
        
        # Log parameter space info
        self.logger.info(f"Parameter space for {strategy_name}:")
        param_dict = parameter_space.to_dict()
        self.logger.info(f"Total parameters: {len(parameter_space._parameters)}")
        for name, param in parameter_space._parameters.items():
            self.logger.info(f"  {name}: type={param.param_type}, default={param.default}")
            if param.param_type == 'discrete':
                self.logger.info(f"    values: {param.values}")
            elif param.param_type == 'continuous':
                self.logger.info(f"    range: [{param.min_value}, {param.max_value}], step={param.step}")
        
        # Clean up temp context - SystemContext doesn't have teardown
        # temp_context.teardown()
        
        # Run optimization
        result = method.optimize(
            objective_func=objective_func,
            parameter_space=parameter_space,
            n_iterations=n_iterations,
            **kwargs
        )
        
        # Save results
        self._save_results(strategy_name, result, method.name, metric.name)
        
        return result
        
    def _evaluate_parameters(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        metric: OptimizationMetric
    ) -> float:
        """Evaluate parameters in isolated context."""
        iteration_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # Create scoped context for this iteration
            context = self._bootstrap.create_scoped_context(iteration_id)
            
            # Get components from container
            strategy = context.container.resolve(strategy_name)
            if not strategy:
                raise ValueError(f"Strategy '{strategy_name}' not found in context")
                
            # Set parameters
            strategy.set_parameters(params)
            
            # Run backtest
            backtest_runner = context.container.resolve('backtest_runner')
            if not backtest_runner:
                raise ValueError("BacktestRunner not found in context")
                
            # Execute backtest
            results = backtest_runner.execute()
            
            # Calculate metric
            score = metric.calculate(results)
            
            self.logger.debug(f"Parameters {params} scored {score:.6f}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return float('-inf')
            
        finally:
            # SystemContext doesn't have teardown - it will be cleaned up when optimization completes
            pass
                
    def _save_results(
        self,
        strategy_name: str,
        result: OptimizationResult,
        method_name: str,
        metric_name: str
    ) -> None:
        """Save optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{method_name}_{timestamp}.json"
        filepath = os.path.join(self._results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'strategy': strategy_name,
            'method': method_name,
            'metric': metric_name,
            'timestamp': timestamp,
            'best_params': result.best_params,
            'best_score': result.best_score,
            'metadata': result.metadata,
            'all_results': result.all_results[:10]  # Save top 10 results
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        self.logger.info(f"Saved optimization results to {filepath}")
        
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load optimization results from file."""
        filepath = os.path.join(self._results_dir, filename)
        
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def optimize_regime_specific(
        self,
        strategy_name: str,
        regimes: List[str],
        method: Optional[OptimizationMethod] = None,
        metric: Optional[OptimizationMetric] = None,
        **kwargs
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize strategy parameters for specific regimes.
        
        Returns dictionary of regime -> OptimizationResult
        """
        results = {}
        
        for regime in regimes:
            self.logger.info(f"Optimizing for regime: {regime}")
            
            # Modify base config to filter for regime
            regime_config = self._base_config.copy()
            # Add regime filtering configuration here
            
            # Run optimization
            result = self.optimize_strategy(
                strategy_name=strategy_name,
                method=method,
                metric=metric,
                **kwargs
            )
            
            results[regime] = result
            
        return results