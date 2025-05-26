#!/usr/bin/env python3
"""
Scoped Optimization Runner - Uses existing scoped context infrastructure.

This runner properly uses the documented optimization framework with:
- Scoped contexts for isolation
- Proper parameter injection
- Clean component lifecycle
"""

import datetime
import json
from typing import Dict, Any, List
from pathlib import Path

from ..core.component_base import ComponentBase
from ..core.exceptions import ComponentError


class ScopedOptimizationRunner(ComponentBase):
    """
    Optimization runner that uses scoped contexts for proper isolation.
    
    This implementation leverages the existing infrastructure documented in:
    - docs/scoped_containers_comparison.md
    - docs/modules/strategy/optimization/OPTIMIZATION_FRAMEWORK.md
    """
    
    def _initialize(self) -> None:
        """Initialize the optimization runner."""
        # Get configuration
        self.parameter_ranges = self.component_config.get('parameter_ranges', {})
        self.max_iterations = self.component_config.get('max_iterations', 10)
        self.optimization_metric = self.component_config.get('optimization_metric', 'total_return')
        
        # Get bootstrap reference from context
        self.bootstrap = self._context.get('bootstrap')
        if not self.bootstrap:
            raise ComponentError("Bootstrap not found in context")
        
        self.results = []
        self.best_result = None
        
        self.logger.info(
            f"ScopedOptimizationRunner initialized. "
            f"Max iterations: {self.max_iterations}, "
            f"Metric: {self.optimization_metric}"
        )
        
    def _start(self) -> None:
        """Start the optimization runner."""
        self.logger.info("ScopedOptimizationRunner started")
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute optimization using scoped contexts.
        
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting scoped optimization execution")
        
        try:
            # Import the basic optimizer for parameter generation
            from ..strategy.optimization.basic_optimizer import BasicParameterOptimizer
            
            # Create parameter generator
            param_generator = BasicParameterOptimizer()
            parameter_combinations = param_generator.generate_parameter_grid(self.parameter_ranges)
            
            # Limit iterations
            if len(parameter_combinations) > self.max_iterations:
                self.logger.warning(
                    f"Limiting {len(parameter_combinations)} combinations to {self.max_iterations}"
                )
                parameter_combinations = parameter_combinations[:self.max_iterations]
            
            self.logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
            
            # Run optimization iterations
            for i, params in enumerate(parameter_combinations):
                self.logger.info(f"\n=== Optimization Iteration {i+1}/{len(parameter_combinations)} ===")
                self.logger.info(f"Testing parameters: {params}")
                
                # Create scoped context for this iteration
                scope_name = f"opt_trial_{i+1}"
                scoped_context = self.bootstrap.create_scoped_context(
                    scope_name,
                    shared_services=['config', 'logger']
                )
                
                try:
                    # Run backtest in isolated scope
                    result = self._run_scoped_backtest(scoped_context, params)
                    
                    # Store result
                    self.results.append({
                        'iteration': i + 1,
                        'parameters': params,
                        'result': result
                    })
                    
                    # Check if best
                    metric_value = result.get(self.optimization_metric, float('-inf'))
                    if self.best_result is None or self._is_better(metric_value, self.best_result['result'].get(self.optimization_metric)):
                        self.best_result = {
                            'iteration': i + 1,
                            'parameters': params,
                            'result': result
                        }
                        self.logger.info(f"New best result! {self.optimization_metric}: {metric_value}")
                        
                except Exception as e:
                    self.logger.error(f"Iteration {i+1} failed: {e}")
                    
            # Save results
            self._save_results()
            
            return {
                'status': 'success',
                'total_iterations': len(parameter_combinations),
                'best_result': self.best_result,
                'summary': self._create_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _run_scoped_backtest(self, scoped_context: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a backtest in an isolated scope."""
        # Create components in the scoped context
        components = self._create_scoped_components(scoped_context, params)
        
        # Initialize all components
        for name, component in components.items():
            if hasattr(component, 'initialize'):
                component.initialize(scoped_context.__dict__)
            self.logger.debug(f"Initialized {name} in scope")
        
        # Start components
        for name, component in components.items():
            if hasattr(component, 'start'):
                component.start()
            self.logger.debug(f"Started {name} in scope")
        
        # Run the backtest
        data_handler = components['data_handler']
        data_handler.set_active_dataset('train')  # Use training data
        
        # Get CLI args from context for bars limit
        cli_args = self._context.get('metadata', {}).get('cli_args', {})
        max_bars = cli_args.get('bars')
        if max_bars and hasattr(data_handler, 'set_max_bars'):
            data_handler.set_max_bars(max_bars)
        
        # Stream the data (triggers the backtest)
        data_handler.start()  # This publishes all bar events
        
        # Get results from portfolio
        portfolio = components['portfolio_manager']
        return portfolio.get_performance()
        
    def _create_scoped_components(self, scoped_context: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create fresh components in the scoped context."""
        # Import component classes
        from ..data.csv_data_handler import CSVDataHandler
        from ..risk.basic_portfolio import BasicPortfolio
        from ..strategy.ma_strategy import MAStrategy
        from ..strategy.regime_detector import RegimeDetector
        
        # Create components
        components = {}
        
        # Data handler
        data_handler = CSVDataHandler('data_handler', 'components.data_handler_csv')
        components['data_handler'] = data_handler
        
        # Portfolio
        portfolio = BasicPortfolio('portfolio_manager', 'components.basic_portfolio')
        components['portfolio_manager'] = portfolio
        
        # Regime detector
        regime_detector = RegimeDetector('MyPrimaryRegimeDetector', 'components.MyPrimaryRegimeDetector')
        components['regime_detector'] = regime_detector
        
        # Strategy with parameters
        strategy = MAStrategy('strategy', 'components.ma_strategy')
        
        # Apply optimization parameters to strategy
        # MAStrategy uses these parameter names internally
        param_mapping = {
            'short_window': 'short_window',
            'long_window': 'long_window'
        }
        
        for opt_param, strategy_param in param_mapping.items():
            if opt_param in params:
                setattr(strategy, strategy_param, params[opt_param])
                
        components['strategy'] = strategy
        
        # Register all in scoped container
        for name, component in components.items():
            scoped_context.container.register(name, component)
            
        return components
        
    def _is_better(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2 based on metric."""
        if 'drawdown' in self.optimization_metric.lower():
            return value1 < value2  # Lower is better for drawdown
        return value1 > value2  # Higher is better for most metrics
        
    def _create_summary(self) -> Dict[str, Any]:
        """Create optimization summary."""
        if not self.results:
            return {}
            
        # Calculate statistics
        metric_values = [r['result'].get(self.optimization_metric, 0) for r in self.results]
        
        return {
            'total_iterations': len(self.results),
            'best_metric_value': max(metric_values) if metric_values else None,
            'worst_metric_value': min(metric_values) if metric_values else None,
            'average_metric_value': sum(metric_values) / len(metric_values) if metric_values else None
        }
        
    def _save_results(self) -> None:
        """Save optimization results."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'optimization_metric': self.optimization_metric,
                'results': self.results,
                'best_result': self.best_result,
                'summary': self._create_summary()
            }, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save best parameters for easy loading
        if self.best_result:
            with open('best_optimization_parameters.json', 'w') as f:
                json.dump(self.best_result['parameters'], f, indent=2)
            self.logger.info("Best parameters saved to best_optimization_parameters.json")