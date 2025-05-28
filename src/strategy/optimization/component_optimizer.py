"""
Component Optimizer for ADMF system.

This optimizer works with any ComponentBase-derived component that implements
the optimization interface (get_parameter_space, apply_parameters, etc).
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import json
from pathlib import Path

from src.core.component_base import ComponentBase
from .regime_analyzer import RegimePerformanceAnalyzer

# Create custom log level for optimization progress
PROGRESS_LEVEL = 35  # Just above WARNING (30) but below ERROR (40)
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")

def progress(self, message, *args, **kwargs):
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)

# Add the progress method to Logger
logging.Logger.progress = progress

logger = logging.getLogger(__name__)


class ComponentOptimizer(ComponentBase):
    """
    Optimizes individual components using their built-in optimization interface.
    
    This optimizer works with any component that inherits from ComponentBase
    and implements the optimization methods.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        self.results_cache = {}
        self.output_dir = Path("optimization_results")
        self._regime_analyzer = RegimePerformanceAnalyzer()  # Track regime performance
        self._component_factory = {}  # Store factory functions for component recreation
        
    def _initialize(self):
        """Initialize the component optimizer."""
        self.output_dir.mkdir(exist_ok=True)
        
    def _create_component_copy(self, original_component: ComponentBase) -> ComponentBase:
        """
        Create a fresh copy of a component with the same configuration.
        
        This ensures each optimization test has a clean component instance
        without accumulated state from previous tests.
        """
        # Get the component class
        component_class = original_component.__class__
        
        # Create new instance with same name and config
        instance_name = f"{original_component.instance_name}_copy"
        config_key = original_component.config_key
        
        # Handle different component types with dependencies
        if hasattr(original_component, 'rsi_indicator'):
            # RSI Rule - needs indicator dependency
            from src.strategy.components.indicators.oscillators import RSIIndicator
            # Create fresh indicator
            fresh_indicator = RSIIndicator(
                instance_name=f"{original_component.rsi_indicator.instance_name}_copy",
                config_key=original_component.rsi_indicator.config_key
            )
            # Initialize with context
            if hasattr(original_component.rsi_indicator, '_context'):
                fresh_indicator.initialize(original_component.rsi_indicator._context)
            
            # Create fresh rule with fresh indicator
            fresh_component = component_class(
                instance_name=instance_name,
                config_key=config_key,
                rsi_indicator=fresh_indicator
            )
        elif hasattr(original_component, 'fast_ma') and hasattr(original_component, 'slow_ma'):
            # MA Crossover Rule - needs two MA indicators
            # Create fresh indicators - MovingAverageIndicator expects 'name' not 'instance_name'
            fresh_fast_ma = original_component.fast_ma.__class__(
                name=f"{original_component.fast_ma.instance_name}_copy",
                lookback_period=original_component.fast_ma._lookback_period,
                config_key=original_component.fast_ma.config_key
            )
            fresh_slow_ma = original_component.slow_ma.__class__(
                name=f"{original_component.slow_ma.instance_name}_copy",
                lookback_period=original_component.slow_ma._lookback_period,
                config_key=original_component.slow_ma.config_key
            )
            
            # Initialize indicators
            if hasattr(original_component.fast_ma, '_context'):
                fresh_fast_ma.initialize(original_component.fast_ma._context)
                fresh_slow_ma.initialize(original_component.slow_ma._context)
            
            # Create fresh rule
            fresh_component = component_class(
                instance_name=instance_name,
                config_key=config_key,
                fast_ma=fresh_fast_ma,
                slow_ma=fresh_slow_ma
            )
        elif hasattr(original_component, 'bb_indicator'):
            # Bollinger Bands Rule
            # Create fresh BB indicator
            fresh_indicator = original_component.bb_indicator.__class__(
                instance_name=f"{original_component.bb_indicator.instance_name}_copy", 
                config_key=original_component.bb_indicator.config_key
            )
            if hasattr(original_component.bb_indicator, '_context'):
                fresh_indicator.initialize(original_component.bb_indicator._context)
                
            fresh_component = component_class(
                instance_name=instance_name,
                config_key=config_key,
                bb_indicator=fresh_indicator
            )
        elif hasattr(original_component, 'macd_indicator'):
            # MACD Rule
            # Create fresh MACD indicator
            fresh_indicator = original_component.macd_indicator.__class__(
                instance_name=f"{original_component.macd_indicator.instance_name}_copy",
                config_key=original_component.macd_indicator.config_key
            )
            if hasattr(original_component.macd_indicator, '_context'):
                fresh_indicator.initialize(original_component.macd_indicator._context)
                
            fresh_component = component_class(
                instance_name=instance_name,
                config_key=config_key,
                macd_indicator=fresh_indicator
            )
        else:
            # Simple component without dependencies or indicator
            fresh_component = component_class(
                instance_name=instance_name,
                config_key=config_key
            )
        
        # Initialize with same context
        if hasattr(original_component, '_context'):
            fresh_component.initialize(original_component._context)
            
        # DO NOT apply original parameters - let optimizer set them fresh
        # This ensures we start from default state
            
        return fresh_component
        
    def optimize_component(
        self,
        component: ComponentBase,
        evaluator: Callable[[ComponentBase], float],
        method: str = "grid_search",
        isolate: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize a single component.
        
        Args:
            component: The component to optimize (must have get_parameter_space, etc)
            evaluator: Function that evaluates component performance
            method: Optimization method ('grid_search', 'random_search', etc)
            isolate: Whether to evaluate component in isolation (for rules/indicators)
            **kwargs: Additional arguments for the optimization method
            
        Returns:
            Dictionary with optimization results including best parameters
        """
        isolation_mode = " in isolation" if isolate else ""
        self.logger.progress(f"Starting {method} optimization for component {component.instance_name}{isolation_mode}")
        
        # If isolate is requested, wrap the evaluator
        if isolate:
            # Import here to avoid circular dependencies
            from .isolated_evaluator import IsolatedComponentEvaluator
            
            # Check if we have the required components for isolation
            if not hasattr(self, '_isolated_evaluator'):
                self.logger.warning("Isolated evaluator not configured. Using standard evaluation.")
                isolate = False
            else:
                # Use the isolated evaluator
                evaluator = self._isolated_evaluator.create_evaluator_function(
                    metric=kwargs.get('metric', 'sharpe_ratio')
                )
        
        # Get parameter space from component
        param_space = component.get_parameter_space()
        if param_space is None:
            self.logger.warning(f"Component {component.instance_name} has no parameter space")
            return {
                "status": "no_parameters",
                "component": component.instance_name,
                "message": "Component has no optimizable parameters"
            }
        
        # For isolated optimization, exclude the weight parameter
        # since weight is only meaningful in ensemble context
        if isolate and hasattr(param_space, '_parameters') and 'weight' in param_space._parameters:
            self.logger.debug(f"Excluding 'weight' parameter for isolated optimization of {component.instance_name}")
            del param_space._parameters['weight']
        
        # Debug: Log parameter space details
        self.logger.debug(f"Parameter space for {component.instance_name}:")
        if hasattr(param_space, '_parameters'):
            for name, param in param_space._parameters.items():
                self.logger.debug(f"  - {name}: type={param.param_type}, values={param.values if param.param_type == 'discrete' else f'[{param.min_value}, {param.max_value}]'}")
        
        # Store original parameters
        original_params = component.get_optimizable_parameters()
        
        try:
            if method == "grid_search":
                # Extract grid search specific parameters
                maximize = kwargs.get('maximize', True)
                # Remove parameters that _grid_search doesn't accept
                grid_kwargs = {k: v for k, v in kwargs.items() if k in ['maximize']}
                results = self._grid_search(component, param_space, evaluator, **grid_kwargs)
            else:
                self.logger.warning(f"Optimization method {method} not yet implemented")
                results = {"status": "method_not_implemented", "method": method}
                
            # Restore original parameters after optimization
            component.apply_parameters(original_params)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            # Ensure we restore original parameters
            component.apply_parameters(original_params)
            raise
    
    def _grid_search(
        self,
        component: ComponentBase,
        param_space: Any,  # ParameterSpace
        evaluator: Callable[[ComponentBase], float],
        maximize: bool = True
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            component: Component to optimize
            param_space: Parameter space from component
            evaluator: Performance evaluation function
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            Optimization results
        """
        # Get all parameter combinations from space
        combinations = param_space.sample(method='grid') if hasattr(param_space, 'sample') else []
        if not combinations:
            self.logger.warning("No parameter combinations to test")
            return {
                "status": "no_combinations",
                "component": component.instance_name,
                "message": "Parameter space generated no combinations"
            }
        self.logger.progress(f"Found {len(combinations)} parameter combinations to test")
        
        # Debug: log first few combinations
        for i, combo in enumerate(combinations[:3]):
            self.logger.debug(f"  Combination {i+1}: {combo}")
        
        # Track results
        results = []
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        
        # Test each combination
        for i, params in enumerate(combinations):
            try:
                # Format parameters nicely
                param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())])
                # Don't log here - we'll log with result on same line
                
                # Create a fresh component copy for this test
                test_component = self._create_component_copy(component)
                
                # Validate parameters
                valid, error = test_component.validate_parameters(params)
                if not valid:
                    self.logger.warning(f"Invalid parameters {params}: {error}")
                    continue
                
                # Apply parameters to the fresh component
                test_component.apply_parameters(params)
                
                # Debug logging for parameter application
                self.logger.debug(f"Applied params to {test_component.instance_name}: {params}")
                if hasattr(test_component, 'get_optimizable_parameters'):
                    current_params = test_component.get_optimizable_parameters()
                    self.logger.debug(f"Component reports params: {current_params}")
                    
                    # Verify parameters were actually applied
                    for param, value in params.items():
                        if param in current_params:
                            if current_params[param] != value:
                                self.logger.warning(f"Parameter {param} not properly applied: expected {value}, got {current_params[param]}")
                
                # For indicators, log their current state
                if hasattr(test_component, 'rsi_indicator'):
                    ind_params = test_component.rsi_indicator.get_optimizable_parameters()
                    self.logger.debug(f"RSI indicator params: {ind_params}")
                elif hasattr(test_component, 'bb_indicator'):
                    ind_params = test_component.bb_indicator.get_optimizable_parameters()
                    self.logger.debug(f"BB indicator params: {ind_params}")
                elif hasattr(test_component, 'macd_indicator'):
                    ind_params = test_component.macd_indicator.get_optimizable_parameters()
                    self.logger.debug(f"MACD indicator params: {ind_params}")
                
                # Ensure data handler is reset if available
                if hasattr(self._context, 'container'):
                    try:
                        data_handler = self._context.container.resolve('data_handler')
                        if hasattr(data_handler, 'reset'):
                            data_handler.reset()
                    except:
                        pass
                
                # Evaluate performance with the fresh component
                try:
                    eval_result = evaluator(test_component)
                except Exception as eval_error:
                    self.logger.error(f"Error during evaluation of {test_component.instance_name} with params {params}: {eval_error}")
                    import traceback
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Handle different return types from evaluator
                if isinstance(eval_result, tuple):
                    # Evaluator returned (score, full_results)
                    score, backtest_results = eval_result
                else:
                    # Evaluator returned just score
                    score = eval_result
                    backtest_results = None
                
                # Handle None scores
                if score is None:
                    self.logger.warning(f"Evaluator returned None for parameters {params}")
                    score = float('-inf') if maximize else float('inf')
                
                # Record result
                result_entry = {
                    "parameters": params.copy(),
                    "score": score
                }
                
                # If we have backtest results, analyze regime performance
                if backtest_results and hasattr(self, '_regime_analyzer'):
                    # Get regime history from the data handler or strategy
                    regime_history = []
                    if hasattr(self._context, 'container'):
                        try:
                            # Try to get regime detector
                            regime_detector = self._context.container.resolve('regime_detector')
                            if hasattr(regime_detector, 'regime_history'):
                                regime_history = regime_detector.regime_history
                        except:
                            pass
                    
                    # Analyze regime performance
                    regime_performance = self._regime_analyzer.analyze_backtest_results(
                        backtest_results, params, regime_history
                    )
                    
                    # Add regime performance to result
                    result_entry['regime_performance'] = regime_performance
                    
                    # Log regime-specific performance
                    if regime_performance:
                        self.logger.info(f"Regime performance for {params}:")
                        for regime, perf in regime_performance.items():
                            self.logger.info(f"  {regime}: return={perf.get('total_return', 0):.2f}, "
                                           f"sharpe={perf.get('sharpe_ratio', 0):.2f}, "
                                           f"trades={perf.get('trade_count', 0)}")
                
                results.append(result_entry)
                
                # Track best
                is_new_best = False
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                    is_new_best = True
                    
                # Show the test and result on one line with global progress
                global_progress = ""
                if hasattr(self, '_global_progress'):
                    current = self._global_progress['current'] + i + 1
                    total = self._global_progress['total']
                    global_progress = f"[{current}/{total}] "
                    
                result_str = f"{global_progress}Testing {i+1}/{len(combinations)}: {param_str} → Score: {score:.4f}"
                if is_new_best:
                    result_str += " ⭐ NEW BEST!"
                self.logger.progress(result_str)
                    
            except Exception as e:
                global_progress = ""
                if hasattr(self, '_global_progress'):
                    current = self._global_progress['current'] + i + 1
                    total = self._global_progress['total']
                    global_progress = f"[{current}/{total}] "
                    
                error_str = f"{global_progress}Testing {i+1}/{len(combinations)}: {param_str} → Error: {str(e)}"
                self.logger.progress(error_str)
                continue
        
        # Grid search should NOT handle test evaluation - that's the workflow orchestrator's job
        test_score = None
        
        # Get regime-specific best parameters if we have regime analyzer
        regime_best_params = {}
        if hasattr(self, '_regime_analyzer'):
            regime_best_params = self._regime_analyzer.get_best_parameters_per_regime()
        
        # Prepare results
        optimization_results = {
            "method": "grid_search",
            "component": component.instance_name,
            "parameter_space": param_space.name if hasattr(param_space, 'name') else "unknown",
            "combinations_tested": len(combinations),
            "best_parameters": best_params,
            "best_score": best_score,
            "test_score": test_score,
            "all_results": results,
            "regime_best_parameters": regime_best_params,  # Add regime-specific best params
            "regime_statistics": self._regime_analyzer.get_regime_statistics() if hasattr(self, '_regime_analyzer') else {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Grid search complete. Best score: {best_score}")
        
        # Display regime-specific best parameters if available
        if regime_best_params:
            self.logger.warning("\n" + "="*80)
            self.logger.warning(f"REGIME-SPECIFIC BEST PARAMETERS for {component.instance_name}:")
            self.logger.warning("="*80)
            
            for regime, params in regime_best_params.items():
                # Format parameters nicely
                param_strs = []
                for param_name, param_value in sorted(params.items()):
                    if isinstance(param_value, float):
                        param_strs.append(f"{param_name}={param_value:.4f}")
                    else:
                        param_strs.append(f"{param_name}={param_value}")
                
                # Get regime performance if available
                perf_str = ""
                if hasattr(self, '_regime_analyzer') and self._regime_analyzer:
                    regime_stats = self._regime_analyzer.get_regime_statistics()
                    if regime in regime_stats:
                        stats = regime_stats[regime]
                        perf_str = f" | Sharpe: {stats.get('avg_sharpe', 'N/A'):.4f}, Trades: {stats.get('total_trades', 'N/A')}"
                
                self.logger.warning(f"  {regime.upper()}: {', '.join(param_strs)}{perf_str}")
            
            self.logger.warning("="*80 + "\n")
        
        # Save results
        self._save_results(optimization_results)
        
        # Save regime analysis separately if we have it
        if hasattr(self, '_regime_analyzer') and regime_best_params:
            regime_filename = f"regime_analysis_{component.instance_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._regime_analyzer.save_analysis(self.output_dir / regime_filename)
            self.logger.info(f"Saved regime analysis to {regime_filename}")
        
        return optimization_results
    
    def optimize_weights(
        self,
        strategy: ComponentBase,
        evaluator: Callable[[Dict[str, float]], float],
        weight_params: List[str],
        method: str = "grid_search",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize weights for ensemble strategies.
        
        Args:
            strategy: Strategy component with weight parameters
            evaluator: Function that evaluates weight performance
            weight_params: List of parameter names that are weights
            method: Optimization method
            **kwargs: Method-specific parameters
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Starting weight optimization for {len(weight_params)} weights")
        
        # Create a wrapper evaluator that works with the component
        def component_evaluator(component):
            # The evaluator expects just the weights
            current_params = component.get_optimizable_parameters()
            weights = {k: v for k, v in current_params.items() if k in weight_params}
            return evaluator(weights)
        
        # Use standard component optimization
        return self.optimize_component(strategy, component_evaluator, method, **kwargs)
    
    def set_isolated_evaluator(self, backtest_runner, data_handler, portfolio, 
                             risk_manager, execution_handler):
        """
        Configure the isolated evaluator for component isolation.
        
        Args:
            backtest_runner: BacktestRunner instance
            data_handler: Data handler for market data
            portfolio: Portfolio manager
            risk_manager: Risk management component
            execution_handler: Execution handler
        """
        from .isolated_evaluator import IsolatedComponentEvaluator
        
        self._isolated_evaluator = IsolatedComponentEvaluator(
            backtest_runner=backtest_runner,
            data_handler=data_handler,
            portfolio=portfolio,
            risk_manager=risk_manager,
            execution_handler=execution_handler
        )
        self.logger.info("Isolated evaluator configured for component optimization")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        component_name = results.get("component", "unknown")
        filename = f"component_opt_{component_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Saved optimization results to {filepath}")