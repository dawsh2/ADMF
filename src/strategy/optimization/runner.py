"""
Optimization runner using scoped contexts for isolation.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime
import json
import os

from ...core.component_base import ComponentBase
from ...core.event import EventType
from ..base import Strategy, ParameterSet
# Removed CleanBacktestEngine - using proper scoped contexts instead
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
from .regime_analyzer import RegimePerformanceAnalyzer


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
        self._regime_analyzer = RegimePerformanceAnalyzer()
        self._clean_engine = None  # Will be initialized when we have config
        
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
        
        # Store bootstrap reference for creating scoped contexts
        if self._bootstrap and hasattr(self._bootstrap, 'context'):
            self.logger.info("OptimizationRunner configured with bootstrap context")
        
    def _create_fresh_components_in_context(self, context, strategy_name: str) -> None:
        """
        Create fresh component instances in the scoped context.
        
        This ensures complete state isolation between optimization runs by creating
        new instances of all components rather than reusing parent instances.
        """
        # Get component configurations from base config
        components_config = self._base_config.get('components', {})
        
        # List of components that need fresh instances for each run
        components_to_create = [
            'data_handler',
            'regime_detector', 
            'portfolio_manager',
            'risk_manager',
            'execution_handler',
            'backtest_runner',
            strategy_name
        ]
        
        # Create fresh instances in the scoped container
        for comp_name in components_to_create:
            if comp_name in components_config:
                comp_config = components_config[comp_name]
                class_path = comp_config.get('class_path')
                
                if class_path:
                    try:
                        # Import the class
                        module_path, class_name = class_path.rsplit('.', 1)
                        module = __import__(module_path, fromlist=[class_name])
                        comp_class = getattr(module, class_name)
                        
                        # Create instance with correct parameters
                        # Most components just need instance_name and optionally config_key
                        instance = comp_class(
                            instance_name=comp_name,
                            config_key=comp_name  # Use component name as config key
                        )
                        
                        # Set the component-specific configuration as an override
                        # This ensures components can access their config even in scoped contexts
                        instance._bootstrap_override_config = comp_config.get('config', {})
                        
                        # Register in scoped container
                        context.container.register_instance(comp_name, instance)
                        
                        # Initialize the component with proper context
                        if hasattr(instance, 'initialize'):
                            # Create a dictionary context from SystemContext
                            init_context = {
                                'config': context.config,
                                'config_loader': context.config,  # Some components expect this
                                'container': context.container,
                                'event_bus': context.event_bus,
                                'logger': context.logger,
                                'metadata': context.metadata,
                                'bootstrap': self._bootstrap
                            }
                            instance.initialize(init_context)
                        elif hasattr(instance, 'set_context'):
                            # Some components might use set_context
                            instance.set_context(context)
                            
                        self.logger.debug(f"Created fresh {comp_name} instance in scope")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to create {comp_name}: {e}")
                        # For now, fall back to parent instance
                        parent_instance = self._bootstrap.get_component(comp_name)
                        if parent_instance:
                            context.container.register_instance(comp_name, parent_instance)
        
        # Start all components in the context
        for comp_name in components_to_create:
            comp = context.container.resolve(comp_name)
            if comp and hasattr(comp, 'start'):
                comp.start()
                self.logger.debug(f"Started {comp_name} in scoped context")
        
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
        
        This runs ONE backtest per parameter combination with regime detection active,
        then analyzes performance by regime to find optimal parameters for each regime.
        
        Returns dictionary of regime -> OptimizationResult
        """
        # Ensure we have a regime detector in the config
        if 'regime_detector' not in self._base_config.get('components', {}):
            self.logger.warning("No regime detector configured for regime-specific optimization")
            
        self.logger.info(f"Starting regime-specific optimization for regimes: {regimes}")
        
        # Get strategy parameter space
        temp_context = self._bootstrap.create_scoped_context("temp_param_space")
        
        try:
            strategy = temp_context.container.resolve(strategy_name)
        except Exception:
            strategy = self._bootstrap.get_component(strategy_name)
            
        if not strategy:
            raise ValueError(f"Strategy '{strategy_name}' not found")
            
        parameter_space = strategy.get_parameter_space()
        
        # Remove n_iterations from kwargs to avoid duplicate
        opt_kwargs = kwargs.copy()
        n_iterations = opt_kwargs.pop('n_iterations', None)
        
        # Create comprehensive objective function that collects all regime data
        all_backtest_results = []
        
        def comprehensive_objective_func(params: Dict[str, Any]) -> float:
            """Run backtest and collect performance data for ALL regimes."""
            
            try:
                # Create a scoped context for this parameter evaluation
                scoped_context = self._bootstrap.create_scoped_context(f"opt_eval_{len(all_backtest_results)}")
                
                # Create fresh components in the scoped context
                self._create_fresh_components_in_context(scoped_context, strategy_name)
                
                # Get the strategy and apply parameters
                strategy = scoped_context.container.resolve(strategy_name)
                if strategy and hasattr(strategy, 'set_parameters'):
                    strategy.set_parameters(params)
                
                # Get the backtest runner
                backtest_runner = scoped_context.container.resolve('backtest_runner')
                if not backtest_runner:
                    self.logger.error("Backtest runner not available in scoped context")
                    return float('-inf')
                
                # Run the backtest
                results = backtest_runner.execute()
                
                # Calculate metric value
                metric_value = metric.calculate(results)
                
                # Get regime performance from portfolio manager
                portfolio = scoped_context.container.resolve('portfolio_manager')
                regime_performance = {}
                if portfolio and hasattr(portfolio, 'get_performance_by_regime'):
                    regime_performance = portfolio.get_performance_by_regime()
                
                # Store results for later analysis
                all_backtest_results.append({
                    'params': params,
                    'metric_value': metric_value,
                    'regime_performance': regime_performance
                })
                
                # Log summary
                self.logger.info(f"\nParameter combination: {params}")
                self.logger.info(f"Overall metric value: {metric_value:.4f}")
                if regime_performance:
                    for regime, perf in regime_performance.items():
                        if regime.startswith('_'):  # Skip internal keys
                            continue
                        trade_count = perf.get('count', 0)
                        if trade_count > 0:
                            self.logger.info(f"  {regime}: trades={trade_count}, "
                                           f"net_pnl={perf.get('net_pnl', 0):.2f}, "
                                           f"sharpe={perf.get('sharpe_ratio', 0):.2f}")
                
                return metric_value
                    
            except Exception as e:
                self.logger.error(f"Error evaluating parameters: {e}")
                return float('-inf')
        
        # Run optimization to collect all data
        param_combinations = parameter_space.sample(method='grid')
        self.logger.info(f"Running {len(param_combinations)} parameter combinations...")
        
        # Use the optimization method to explore parameter space
        overall_result = method.optimize(
            objective_func=comprehensive_objective_func,
            parameter_space=parameter_space,
            n_iterations=n_iterations,
            **opt_kwargs
        )
        
        # Now analyze results to find best parameters per regime
        results = {}
        best_params_per_regime = {}
        
        # Aggregate results by regime
        for result in all_backtest_results:
            params = result['params']
            regime_perf = result.get('regime_performance', {})
            
            for regime, perf in regime_perf.items():
                if regime not in best_params_per_regime:
                    best_params_per_regime[regime] = {
                        'parameters': params,
                        'sharpe_ratio': perf.get('sharpe_ratio', float('-inf')),
                        'total_return': perf.get('net_pnl', 0),
                        'win_rate': perf.get('win_rate', 0),
                        'trade_count': perf.get('count', 0)
                    }
                else:
                    # Update if this is better
                    if perf.get('sharpe_ratio', float('-inf')) > best_params_per_regime[regime]['sharpe_ratio']:
                        best_params_per_regime[regime] = {
                            'parameters': params,
                            'sharpe_ratio': perf.get('sharpe_ratio', float('-inf')),
                            'total_return': perf.get('net_pnl', 0),
                            'win_rate': perf.get('win_rate', 0),
                            'trade_count': perf.get('count', 0)
                        }
        
        self.logger.info("\n=== REGIME-SPECIFIC OPTIMIZATION RESULTS ===")
        
        # Count total trades in training phase
        total_training_trades = 0
        for result in all_backtest_results:
            regime_perf = result.get('regime_performance', {})
            for regime, perf in regime_perf.items():
                total_training_trades += perf.get('count', 0)
        
        self.logger.info(f"\nTotal trades during training phase: {total_training_trades}")
        self.logger.info(f"Parameter combinations tested: {len(all_backtest_results)}")
        
        for regime in regimes:
            if regime in best_params_per_regime:
                regime_best = best_params_per_regime[regime]
                # Create OptimizationResult for this regime
                results[regime] = OptimizationResult(
                    best_params=regime_best['parameters'],
                    best_score=regime_best['sharpe_ratio'],
                    all_results=[],  # Could populate with regime-specific results
                    metadata={
                        'win_rate': regime_best['win_rate'],
                        'total_return': regime_best['total_return'],
                        'trade_count': regime_best['trade_count'],
                        'regime': regime
                    }
                )
                
                self.logger.info(f"\n{regime.upper()}:")
                self.logger.info(f"  Best params: {regime_best['parameters']}")
                self.logger.info(f"  Sharpe ratio: {regime_best['sharpe_ratio']:.2f}")
                self.logger.info(f"  Total return: {regime_best['total_return']:.2f}")
                self.logger.info(f"  Win rate: {regime_best['win_rate']:.2%}")
                self.logger.info(f"  Trade count: {regime_best['trade_count']}")
            else:
                self.logger.warning(f"\nNo trades found for regime: {regime}")
                results[regime] = OptimizationResult(
                    best_params={},
                    best_score=float('-inf'),
                    all_results=[],
                    metadata={'regime': regime, 'error': 'No trades in regime'}
                )
        
        # Save results
        for regime, result in results.items():
            self._save_regime_results(strategy_name, regime, result, method.name, metric.name)
            
        self._save_combined_regime_results(strategy_name, results, method.name, metric.name)
        
        # Save detailed analysis
        analyzer_file = os.path.join(self._results_dir, f"{strategy_name}_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self._regime_analyzer.save_analysis(analyzer_file)
        
        return results
        
    def optimize_regime_specific_with_split(
        self,
        strategy_name: str,
        regimes: List[str],
        train_ratio: float = 0.8,
        method: Optional[OptimizationMethod] = None,
        metric: Optional[OptimizationMetric] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for specific regimes using train/test split.
        
        This method:
        1. Runs optimization on training data to find best parameters per regime
        2. Runs a single backtest on test data using regime-adaptive strategy
        3. Returns performance metrics for both train and test periods
        
        Args:
            strategy_name: Name of strategy to optimize
            regimes: List of regimes to optimize for
            train_ratio: Proportion of data to use for training (default: 0.8)
            method: Optimization method (default: GridSearch)
            metric: Optimization metric (default: SharpeRatio)
            **kwargs: Additional arguments
            
        Returns:
            Dict containing:
                - 'train_results': Regime-specific optimization results from training
                - 'test_results': Performance on test set with adaptive parameters
                - 'regime_parameters': Best parameters per regime
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
            
        self.logger.info(f"Starting train/test optimization with {train_ratio:.0%} training split")
        
        # Ensure data handler is configured for train/test split
        if 'data_handler' not in self._base_config.get('components', {}):
            raise ConfigurationError("No data_handler configured")
            
        # Update config to enable train/test split
        original_data_config = self._base_config['components']['data_handler'].copy()
        # The train_test_split_ratio needs to be in the config section, not at the top level
        if 'config' not in self._base_config['components']['data_handler']:
            self._base_config['components']['data_handler']['config'] = {}
        self._base_config['components']['data_handler']['config']['train_test_split_ratio'] = train_ratio
        
        # IMPORTANT: Apply the train/test split to the existing data handler
        # The data handler was already created during bootstrap, so we need to update it
        if self._bootstrap:
            data_handler = self._bootstrap.get_component('data_handler')
            if data_handler:
                # First ensure max_bars is set from CLI args before split
                metadata = getattr(self._bootstrap.context, 'metadata', {})
                cli_args = metadata.get('cli_args', {})
                max_bars = cli_args.get('bars')
                
                if max_bars and hasattr(data_handler, 'set_max_bars'):
                    self.logger.info(f"Setting max_bars to {max_bars} before train/test split")
                    data_handler.set_max_bars(max_bars)
                
                # Now apply the split
                if hasattr(data_handler, 'apply_train_test_split'):
                    self.logger.info(f"Applying train/test split ratio {train_ratio} to data handler")
                    # This will split the already --bars limited data
                    data_handler.apply_train_test_split(train_ratio)
        
        # Step 1: Run optimization on training data
        self.logger.info("=== TRAINING PHASE: Finding optimal parameters per regime ===")
        
        # Stop the main regime detector to prevent it from counting optimization bars
        if self._bootstrap:
            main_regime_detector = self._bootstrap.get_component('regime_detector')
            if main_regime_detector and hasattr(main_regime_detector, 'stop'):
                main_regime_detector.stop()
                self.logger.info("Stopped main regime detector during optimization")
        
        # Modify backtest runner config to use training data
        original_backtest_config = self._base_config['components'].get('backtest_runner', {}).copy()
        self._base_config['components']['backtest_runner'] = {
            **original_backtest_config,
            'use_test_dataset': False  # Use training data
        }
        
        # Run regime-specific optimization on training data
        train_results = self.optimize_regime_specific(
            strategy_name=strategy_name,
            regimes=regimes,
            method=method,
            metric=metric,
            **kwargs
        )
        
        # Extract best parameters per regime
        regime_parameters = {}
        for regime, result in train_results.items():
            if result.best_score > float('-inf'):
                regime_parameters[regime] = result.best_params
                self.logger.info(f"Training - {regime}: best params = {result.best_params}")
        
        # Step 2: Test with regime-adaptive strategy on test data
        self.logger.info("\n=== TESTING PHASE: Evaluating adaptive strategy on test data ===")
        
        # Save regime parameters to a temporary file for the adaptive strategy
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write parameters in the format expected by RegimeAdaptiveStrategy
            json.dump(regime_parameters, f)
            regime_params_path = f.name
            
        self.logger.info(f"Saved regime parameters to {regime_params_path}")
        self.logger.info(f"Parameters content: {regime_parameters}")
            
        try:
            # Create a scoped context for test phase
            test_context = self._bootstrap.create_scoped_context("test_phase")
            
            # Update config to use regime adaptive strategy with parameters
            if 'components' not in self._base_config:
                self._base_config['components'] = {}
                
            # Configure regime adaptive strategy
            self._base_config['components']['regime_adaptive_strategy'] = {
                'class_path': 'src.strategy.regime_adaptive_strategy.RegimeAdaptiveStrategy',
                'config': {
                    'symbol': 'SPY',  # Add the symbol that data_handler expects
                    'regime_params_file_path': regime_params_path  # Fixed key name
                }
            }
            
            # Create fresh components with regime adaptive strategy
            self._create_fresh_components_in_context(test_context, 'regime_adaptive_strategy')
            
            # Copy train/test split data from main data handler to test context data handler
            main_data_handler = self._bootstrap.get_component('data_handler')
            test_data_handler = test_context.container.resolve('data_handler')
            
            if main_data_handler and test_data_handler:
                # Copy the train/test split data
                if hasattr(main_data_handler, '_train_df') and hasattr(test_data_handler, '_train_df'):
                    test_data_handler._train_df = main_data_handler._train_df
                    test_data_handler._test_df = main_data_handler._test_df
                    test_data_handler._train_test_split_ratio = main_data_handler._train_test_split_ratio
                    self.logger.info(f"Copied train/test split to test context: train={len(test_data_handler._train_df)} bars, test={len(test_data_handler._test_df)} bars")
            
            # Configure backtest runner to use test data
            backtest_runner = test_context.container.resolve('backtest_runner')
            if backtest_runner:
                # Set to use test dataset
                backtest_runner.use_test_dataset = True
                
                # Run the test phase backtest
                test_results = backtest_runner.execute()
                
                # Get portfolio for regime performance
                portfolio = test_context.container.resolve('portfolio_manager')
                test_regime_performance = {}
                if portfolio and hasattr(portfolio, 'get_performance_by_regime'):
                    test_regime_performance = portfolio.get_performance_by_regime()
                
                # Calculate overall return
                test_metric_value = 0.0
                if 'total_return' in test_results:
                    test_metric_value = test_results['total_return']
                
                # Log test results
                self.logger.info(f"\nTest set overall return: {test_metric_value:.2%}")
                
                # Get portfolio final summary
                if portfolio:
                    # Log portfolio summary
                    self.logger.info("\n=== TEST PHASE PORTFOLIO SUMMARY ===")
                    if hasattr(portfolio, 'initial_cash') and hasattr(portfolio, 'get_final_portfolio_value'):
                        initial_cash = portfolio.initial_cash
                        final_value = portfolio.get_final_portfolio_value()
                        self.logger.info(f"Initial Cash: ${initial_cash:,.2f}")
                        self.logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
                        self.logger.info(f"Total Return: {((final_value / initial_cash) - 1) * 100:.2f}%")
                    
                    if hasattr(portfolio, 'get_trade_history'):
                        trades = portfolio.get_trade_history()
                        self.logger.info(f"Total Trades: {len(trades)}")
                    
                    if hasattr(portfolio, 'get_realized_pnl'):
                        realized_pnl = portfolio.get_realized_pnl()
                        self.logger.info(f"Realized P&L: ${realized_pnl:,.2f}")
                
                if test_regime_performance:
                    self.logger.info("\n=== TEST SET PERFORMANCE BY REGIME ===")
                    total_trades = 0
                    for regime, perf in test_regime_performance.items():
                        if regime.startswith('_'):  # Skip internal keys
                            continue
                        trade_count = perf.get('count', 0)
                        total_trades += trade_count
                        if trade_count > 0:
                            self.logger.info(f"\n{regime.upper()}:")
                            self.logger.info(f"  Trades: {trade_count}")
                            self.logger.info(f"  Net P&L: ${perf.get('net_pnl', 0):.2f}")
                            self.logger.info(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
                            self.logger.info(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                    self.logger.info(f"\nTotal trades across all regimes: {total_trades}")
                
                test_results = {
                    'total_return': test_metric_value,
                    'regime_performance': test_regime_performance
                }
            else:
                test_results = {'error': 'Backtest runner not available'}
                    
        except Exception as e:
            self.logger.error(f"Error during test phase: {e}")
            test_results = {'error': str(e)}
        finally:
            # Clean up temporary file
            if 'regime_params_path' in locals():
                try:
                    os.unlink(regime_params_path)
                except:
                    pass
            
        # Restore original configs
        self._base_config['components']['data_handler'] = original_data_config
        self._base_config['components']['backtest_runner'] = original_backtest_config
        
        # Restart the main regime detector if we stopped it
        if self._bootstrap:
            main_regime_detector = self._bootstrap.get_component('regime_detector')
            if main_regime_detector and hasattr(main_regime_detector, 'start'):
                main_regime_detector.start()
                self.logger.info("Restarted main regime detector after optimization")
        
        # Return combined results
        return {
            'train_results': train_results,
            'test_results': test_results,
            'regime_parameters': regime_parameters,
            'train_ratio': train_ratio
        }
        
    def _save_regime_results(
        self,
        strategy_name: str,
        regime: str,
        result: OptimizationResult,
        method_name: str,
        metric_name: str
    ) -> None:
        """Save optimization results for a specific regime."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{regime}_{method_name}_{timestamp}.json"
        filepath = os.path.join(self._results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'strategy': strategy_name,
            'regime': regime,
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
            
        self.logger.info(f"Saved {regime} regime optimization results to {filepath}")
        
    def _save_combined_regime_results(
        self,
        strategy_name: str,
        results: Dict[str, OptimizationResult],
        method_name: str,
        metric_name: str
    ) -> None:
        """Save combined results for all regimes."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_regime_combined_{method_name}_{timestamp}.json"
        filepath = os.path.join(self._results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'strategy': strategy_name,
            'method': method_name,
            'metric': metric_name,
            'timestamp': timestamp,
            'regimes': {}
        }
        
        for regime, result in results.items():
            save_data['regimes'][regime] = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'metadata': result.metadata
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        self.logger.info(f"Saved combined regime optimization results to {filepath}")