#!/usr/bin/env python3
"""
OptimizationRunner - dedicated component for running optimizations.

This component handles optimization orchestration, including support
for scoped containers to ensure clean isolation between trials.
"""

from typing import Optional, Dict, Any, List
import time

from ...core.component_base import ComponentBase
from ...core.bootstrap import Bootstrap
from ...core.exceptions import ComponentError, DependencyNotFoundError


class OptimizationRunner(ComponentBase):
    """
    Dedicated component for running optimizations.
    
    This component:
    1. Orchestrates optimization flow
    2. Can use scoped containers for trial isolation
    3. Supports various optimization strategies (grid, genetic, random)
    4. Returns optimization results
    """
    
    def _initialize(self) -> None:
        """Initialize the optimization runner."""
        # Get CLI overrides
        cli_args = self.context.metadata.get('cli_args', {})
        self.max_bars = cli_args.get('bars')
        self.optimization_flags = {
            'genetic_optimize': cli_args.get('genetic_optimize', False),
            'random_search': cli_args.get('random_search', False),
        }
        
        # Get optimization configuration
        self.use_scoped_containers = self.component_config.get('use_scoped_containers', False)
        self.optimization_type = self.component_config.get('optimization_type', 'grid_search')
        
        self.logger.info(
            f"OptimizationRunner initialized. Type: {self.optimization_type}, "
            f"Scoped containers: {self.use_scoped_containers}"
        )
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute the optimization.
        
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting optimization execution")
        
        if not self.initialized or not self.running:
            raise ComponentError("OptimizationRunner not properly initialized")
            
        # Determine which optimizer to use
        if self.use_scoped_containers:
            return self._run_scoped_optimization()
        else:
            return self._run_standard_optimization()
            
    def _run_standard_optimization(self) -> Dict[str, Any]:
        """Run optimization using the traditional shared container approach."""
        self.logger.info("Running standard optimization (shared containers)")
        
        # Get optimizer component
        optimizer = self.container.get('optimizer')
        if not optimizer:
            raise DependencyNotFoundError("Optimizer component not found")
            
        # Configure max bars if specified
        if self.max_bars:
            data_handler = self.container.get('data_handler')
            if data_handler and hasattr(data_handler, 'set_max_bars'):
                data_handler.set_max_bars(self.max_bars)
                
        # Run base optimization
        self.logger.info("Starting grid search")
        results = optimizer.run_grid_search()
        
        if not results:
            self.logger.error("Grid search returned no results")
            return {'status': 'failed', 'reason': 'No optimization results'}
            
        # Run additional optimization if requested
        if self.optimization_flags['genetic_optimize']:
            self.logger.info("Running genetic optimization")
            if hasattr(optimizer, 'run_per_regime_genetic_optimization'):
                results = optimizer.run_per_regime_genetic_optimization(results)
                
        elif self.optimization_flags['random_search']:
            self.logger.info("Running random search")
            if hasattr(optimizer, 'run_per_regime_random_search_optimization'):
                results = optimizer.run_per_regime_random_search_optimization(results)
                
        # Run adaptive test
        if hasattr(optimizer, 'run_adaptive_test'):
            self.logger.info("Running adaptive test")
            optimizer.run_adaptive_test(results)
            
        return results
        
    def _run_scoped_optimization(self) -> Dict[str, Any]:
        """
        Run optimization using scoped containers for clean isolation.
        
        This is the preferred approach as it ensures no state pollution
        between trials.
        """
        self.logger.info("Running scoped optimization (isolated containers)")
        
        # Get the parent Bootstrap instance
        parent_bootstrap = self.context.metadata.get('bootstrap')
        if not parent_bootstrap:
            self.logger.warning("Bootstrap not in metadata, falling back to standard optimization")
            return self._run_standard_optimization()
            
        # Get parameter sets to test
        parameter_sets = self._get_parameter_sets()
        
        results = []
        best_result = None
        best_metric = float('-inf')
        
        # Run each trial in its own scope
        for i, params in enumerate(parameter_sets):
            trial_id = f"trial_{i:04d}"
            self.logger.info(f"Running {trial_id} with params: {params}")
            
            try:
                # Create scoped context for this trial
                scoped_context = parent_bootstrap.create_scoped_context(
                    scope_name=trial_id,
                    shared_services=['config', 'logger']
                )
                
                # Run trial in isolated scope
                trial_result = self._run_scoped_trial(
                    scoped_context=scoped_context,
                    trial_id=trial_id,
                    parameters=params
                )
                
                if trial_result:
                    results.append(trial_result)
                    
                    # Track best result
                    metric_value = trial_result.get('metric_value', float('-inf'))
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_result = trial_result
                        
            except Exception as e:
                self.logger.error(f"Trial {trial_id} failed: {e}")
                
        # Compile final results
        optimization_results = {
            'total_trials': len(parameter_sets),
            'successful_trials': len(results),
            'best_parameters': best_result.get('parameters') if best_result else None,
            'best_metric_value': best_metric if best_result else None,
            'all_results': results
        }
        
        self.logger.info(
            f"Scoped optimization complete. Best metric: {best_metric:.4f}"
        )
        
        return optimization_results
        
    def _run_scoped_trial(self, scoped_context, trial_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single optimization trial in a scoped context.
        
        This ensures complete isolation from other trials.
        """
        start_time = time.time()
        
        # Create fresh components in the scoped context
        # This is similar to what BacktestRunner does, but in isolation
        
        # 1. Create data handler
        from ...data.csv_data_handler import CSVDataHandler
        data_handler = CSVDataHandler(scoped_context.event_bus)
        scoped_context.container.register('data_handler', data_handler)
        
        # 2. Create portfolio
        from ...risk.basic_portfolio import BasicPortfolio
        portfolio = BasicPortfolio(scoped_context.event_bus)
        scoped_context.container.register('portfolio_manager', portfolio)
        
        # 3. Create strategy with test parameters
        strategy_class = self._get_strategy_class()
        strategy = strategy_class(scoped_context.event_bus)
        strategy.set_parameters(parameters)
        scoped_context.container.register('strategy', strategy)
        
        # 4. Create other required components...
        # (execution handler, risk manager, etc.)
        
        # Initialize all components
        for component in [data_handler, portfolio, strategy]:
            if hasattr(component, 'initialize'):
                component.initialize(scoped_context)
                
        # Start components
        for component in [data_handler, portfolio, strategy]:
            if hasattr(component, 'start'):
                component.start()
                
        # Run the backtest
        # ... event flow runs ...
        
        # Get results
        metric_value = self._calculate_metric(portfolio)
        
        elapsed_time = time.time() - start_time
        
        return {
            'trial_id': trial_id,
            'parameters': parameters,
            'metric_value': metric_value,
            'execution_time': elapsed_time
        }
        
    def _get_parameter_sets(self) -> List[Dict[str, Any]]:
        """Get parameter sets to test."""
        # This would normally come from optimizer configuration
        # For now, return a simple example
        return [
            {"ma_rule.weight": 0.7, "rsi_rule.weight": 0.3},
            {"ma_rule.weight": 0.6, "rsi_rule.weight": 0.4},
            {"ma_rule.weight": 0.5, "rsi_rule.weight": 0.5},
        ]
        
    def _get_strategy_class(self):
        """Get the strategy class to optimize."""
        # This would be configured
        from ...strategy.ma_strategy import MAStrategy
        return MAStrategy
        
    def _calculate_metric(self, portfolio) -> float:
        """Calculate the optimization metric."""
        metric_name = self.component_config.get('metric', 'total_return')
        
        if hasattr(portfolio, f'get_{metric_name}'):
            return getattr(portfolio, f'get_{metric_name}')()
        else:
            self.logger.warning(f"Metric {metric_name} not found, using total return")
            return portfolio.get_total_return() if hasattr(portfolio, 'get_total_return') else 0.0