"""
Optimization entry point component for Bootstrap.
"""

from typing import Dict, Any, Optional
import logging

from ..core.component_base import ComponentBase
from ..core.exceptions import ComponentError
from ..strategy.optimization.runner import OptimizationRunner
from ..strategy.optimization.methods import GridSearchOptimizer, RandomSearchOptimizer
from ..strategy.optimization.base.metric import (
    SharpeRatioMetric,
    TotalReturnMetric,
    MaxDrawdownMetric,
    CompositeMetric
)


class OptimizationEntrypoint(ComponentBase):
    """
    Entry point component for optimization mode.
    
    This component is executed by Bootstrap when run_mode is 'optimization'.
    """
    
    def __init__(self, instance_name: str = "optimization_entrypoint", config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        self._runner = None
        
    def _initialize(self):
        """Initialize optimization entry point."""
        config = self.component_config or {}
        
        # Get optimization configuration
        self._strategy_name = config.get('strategy_name', 'strategy')
        self._method_type = config.get('method', 'grid')
        self._metric_type = config.get('metric', 'sharpe_ratio')
        self._n_iterations = config.get('n_iterations', None)
        self._regime_specific = config.get('regime_specific', False)
        self._regimes = config.get('regimes', [])
        
        self.logger.info(f"OptimizationEntrypoint initialized for strategy '{self._strategy_name}'")
        
    def _start(self):
        """Start the optimization entry point."""
        # Create optimization runner
        self._runner = OptimizationRunner("opt_runner")
        self._runner.initialize(self._context)
        
        self.logger.info("OptimizationEntrypoint started")
        
    def execute(self) -> Dict[str, Any]:
        """
        Execute optimization.
        
        This is the main entry point called by Bootstrap.
        """
        self.logger.info("Starting optimization execution")
        
        if not self._runner:
            raise ComponentError("OptimizationRunner not initialized")
            
        # Set bootstrap and config on runner
        bootstrap = self._context.get('bootstrap')
        if not bootstrap:
            raise ComponentError("Bootstrap not found in context")
            
        self._runner.set_bootstrap(bootstrap)
        
        # Get base configuration (minus optimization settings)
        base_config = self._get_base_config()
        self._runner.set_base_config(base_config)
        
        # Create method and metric
        method = self._create_method()
        metric = self._create_metric()
        
        # Run optimization
        if self._regime_specific and self._regimes:
            # Regime-specific optimization
            results = self._runner.optimize_regime_specific(
                strategy_name=self._strategy_name,
                regimes=self._regimes,
                method=method,
                metric=metric,
                n_iterations=self._n_iterations
            )
            
            # Format results
            return self._format_regime_results(results)
        else:
            # Standard optimization
            result = self._runner.optimize_strategy(
                strategy_name=self._strategy_name,
                method=method,
                metric=metric,
                n_iterations=self._n_iterations
            )
            
            # Format results
            return self._format_results(result)
            
    def _create_method(self):
        """Create optimization method based on configuration."""
        if self._method_type == 'grid':
            return GridSearchOptimizer()
        elif self._method_type == 'random':
            seed = self.component_config.get('random_seed', None)
            return RandomSearchOptimizer(seed=seed)
        else:
            raise ValueError(f"Unknown optimization method: {self._method_type}")
            
    def _create_metric(self):
        """Create optimization metric based on configuration."""
        if self._metric_type == 'sharpe_ratio':
            risk_free_rate = self.component_config.get('risk_free_rate', 0.0)
            return SharpeRatioMetric(risk_free_rate)
        elif self._metric_type == 'total_return':
            return TotalReturnMetric()
        elif self._metric_type == 'max_drawdown':
            return MaxDrawdownMetric()
        elif self._metric_type == 'composite':
            # Create composite metric
            composite = CompositeMetric()
            
            # Add configured metrics
            metrics_config = self.component_config.get('composite_metrics', {})
            for metric_name, weight in metrics_config.items():
                if metric_name == 'sharpe_ratio':
                    composite.add_metric(SharpeRatioMetric(), weight)
                elif metric_name == 'total_return':
                    composite.add_metric(TotalReturnMetric(), weight)
                elif metric_name == 'max_drawdown':
                    composite.add_metric(MaxDrawdownMetric(), weight)
                    
            return composite
        else:
            raise ValueError(f"Unknown optimization metric: {self._metric_type}")
            
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration without optimization settings."""
        # Get original config from context
        config = self._context.get('config', {})
        
        # Handle SimpleConfigLoader - convert to dict
        if hasattr(config, '_config_data'):
            # It's a SimpleConfigLoader, get the internal data
            base_config = dict(config._config_data)
        elif hasattr(config, 'copy'):
            # It's already a dict
            base_config = config.copy()
        else:
            # Fallback - create new dict
            base_config = {}
            
        # Remove optimization-specific settings
        if 'optimization' in base_config:
            del base_config['optimization']
            
        # Ensure backtest mode for optimization iterations
        base_config['run_mode'] = 'backtest'
        
        return base_config
        
    def _format_results(self, result) -> Dict[str, Any]:
        """Format optimization results for display."""
        return {
            'mode': 'optimization',
            'strategy': self._strategy_name,
            'method': self._method_type,
            'metric': self._metric_type,
            'best_parameters': result.best_params,
            'best_score': result.best_score,
            'total_iterations': len(result.all_results),
            'metadata': result.metadata
        }
        
    def _format_regime_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format regime-specific optimization results."""
        formatted = {
            'mode': 'regime_optimization',
            'strategy': self._strategy_name,
            'method': self._method_type,
            'metric': self._metric_type,
            'regimes': {}
        }
        
        for regime, result in results.items():
            formatted['regimes'][regime] = {
                'best_parameters': result.best_params,
                'best_score': result.best_score,
                'iterations': len(result.all_results)
            }
            
        return formatted