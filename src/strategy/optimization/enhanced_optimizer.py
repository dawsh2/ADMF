# src/strategy/optimization/enhanced_optimizer.py
import logging
import datetime
import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple
import copy

from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.risk.basic_portfolio import BasicPortfolio
from src.strategy.ma_strategy import MAStrategy

class EnhancedOptimizer(BasicOptimizer):
    """
    Enhanced version of the BasicOptimizer that performs regime-specific optimization.
    
    This optimizer finds the best parameters for each detected market regime instead of
    a single global set of parameters. It leverages the regime detection system and the
    per-regime performance metrics from BasicPortfolio.
    """
    
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container):
        super().__init__(instance_name, config_loader, event_bus, component_config_key, container)
        
        # Settings specific to enhanced optimizer
        self._min_trades_per_regime = self.get_specific_config("min_trades_per_regime", 5)
        self._regime_metric = self.get_specific_config("regime_metric", "sharpe_ratio")
        self._output_file_path = self.get_specific_config("output_file_path", "regime_optimized_parameters.json")
        
        # Results storage
        self._best_params_per_regime: Dict[str, Dict[str, Any]] = {}
        self._best_metric_per_regime: Dict[str, float] = {}
        self._regimes_encountered: Set[str] = set()
        
        self.logger.info(
            f"{self.name} initialized as EnhancedOptimizer. Will optimize strategy '{self._strategy_service_name}' "
            f"per-regime using metric '{self._regime_metric}' from '{self._portfolio_service_name}'. "
            f"Higher is better: {self._higher_metric_is_better}. "
            f"Min trades per regime: {self._min_trades_per_regime}"
        )
    
    def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Overridden to also return regime-specific performance metrics.
        """
        # Get the overall performance metric from parent class
        overall_metric = super()._perform_single_backtest_run(params_to_test, dataset_type)
        
        # If the run failed or there's no portfolio manager, return None for regime performance
        if overall_metric is None:
            return overall_metric, None
            
        # Get the regime-specific performance metrics
        try:
            portfolio_manager: BasicPortfolio = self._container.resolve(self._portfolio_service_name)
            regime_performance = portfolio_manager.get_performance_by_regime()
            return overall_metric, regime_performance
        except Exception as e:
            self.logger.error(f"Error getting regime performance metrics: {e}", exc_info=True)
            return overall_metric, None
    
    def _process_regime_performance(self, params: Dict[str, Any], regime_performance: Dict[str, Dict[str, Any]]) -> None:
        """
        Process regime-specific performance metrics and update best parameters per regime.
        Takes into account boundary trades and their impact on performance.
        """
        if not regime_performance:
            self.logger.warning(f"No regime performance data available for parameters: {params}")
            return
            
        # Special handling for the boundary trades summary section
        boundary_trades_summary = regime_performance.get('_boundary_trades_summary', {})
        
        # Process each regime's metrics
        for regime, metrics in regime_performance.items():
            # Skip the boundary trades summary section, we'll process it separately
            if regime == '_boundary_trades_summary':
                continue
                
            # Track all regimes we encounter
            self._regimes_encountered.add(regime)
            
            # Log boundary trade information if available
            boundary_trade_count = metrics.get('boundary_trade_count', 0)
            pure_regime_count = metrics.get('pure_regime_count', 0)
            total_count = metrics.get('count', 0)
            
            # Skip regimes with too few trades
            if total_count < self._min_trades_per_regime:
                self.logger.debug(f"Regime '{regime}' had only {total_count} trades with params {params}, which is less than the minimum {self._min_trades_per_regime} required.")
                continue
            
            # Log boundary trade percentage for debugging
            if total_count > 0:
                boundary_pct = (boundary_trade_count / total_count) * 100
                self.logger.debug(f"Regime '{regime}': {boundary_trade_count} boundary trades out of {total_count} total ({boundary_pct:.1f}%)")
            
            # Get the metric value for this regime
            metric_value = metrics.get(self._regime_metric)
            if metric_value is None:
                self.logger.warning(f"Metric '{self._regime_metric}' not available for regime '{regime}' with params {params}")
                continue
                
            # Special handling for infinite Sharpe ratios (positive returns with zero volatility)
            if self._regime_metric == 'sharpe_ratio' and metric_value == float('inf'):
                # Use a very large finite value instead
                metric_value = 1000.0  # Arbitrary large value
            
            # If this regime has a high proportion of boundary trades (>30%), we might want to
            # use a more conservative approach to optimization
            boundary_trade_ratio = boundary_trade_count / total_count if total_count > 0 else 0
            boundary_trade_warning = ""
            
            if boundary_trade_ratio > 0.3:  # More than 30% of trades are boundary trades
                boundary_trade_warning = f" (Warning: {boundary_trade_count}/{total_count} boundary trades)"
                
                # Optionally, we could apply a penalty to the metric value based on boundary trade ratio
                # metric_value = metric_value * (1 - (boundary_trade_ratio * 0.2))  # Example: 20% max penalty
            
            # Check if this is the best metric value for this regime so far
            if (regime not in self._best_metric_per_regime or 
                (self._higher_metric_is_better and metric_value > self._best_metric_per_regime[regime]) or
                (not self._higher_metric_is_better and metric_value < self._best_metric_per_regime[regime])):
                
                self._best_metric_per_regime[regime] = metric_value
                self._best_params_per_regime[regime] = copy.deepcopy(params)
                
                self.logger.info(f"New best parameters for regime '{regime}': {params}{boundary_trade_warning}")
                self.logger.info(f"Metric '{self._regime_metric}' value: {metric_value}")
                if pure_regime_count > 0:
                    self.logger.info(f"Based on {pure_regime_count} pure regime trades and {boundary_trade_count} boundary trades")
    
    def run_grid_search(self) -> Optional[Dict[str, Any]]:
        """
        Overridden to perform regime-specific optimization.
        """
        self.logger.info(f"--- {self.name}: Starting Enhanced Grid Search Optimization with Train/Test Split ---")
        self.state = BasicOptimizer.STATE_STARTED
        
        # Reset all tracking variables
        self._best_params_from_train = None
        self._best_training_metric_value = -float('inf') if self._higher_metric_is_better else float('inf')
        self._test_metric_for_best_params = None
        
        self._best_params_per_regime = {}
        self._best_metric_per_regime = {}
        self._regimes_encountered = set()
        
        results_summary = {
            "best_parameters_on_train": None,
            "best_training_metric_value": None,
            "test_set_metric_value_for_best_params": None,
            "best_parameters_per_regime": {},
            "best_metric_per_regime": {},
            "all_training_results": [],
            "regimes_encountered": []
        }
        
        try:
            strategy_to_optimize: MAStrategy = self._container.resolve(self._strategy_service_name)
            data_handler_instance = self._container.resolve(self._data_handler_service_name)
            
            param_space = strategy_to_optimize.get_parameter_space()
            current_strategy_params = strategy_to_optimize.get_parameters()
            
            if not param_space:
                self.logger.warning("Optimizer: Parameter space is empty. Running one iteration with current/default strategy parameters for training and testing.")
                param_combinations = [current_strategy_params] if current_strategy_params else [{}]
            else:
                param_combinations = self._generate_parameter_combinations(param_space)
                
            if not param_combinations:
                self.logger.warning("No parameter combinations to test (parameter space might be empty or produced no combinations).")
                self.state = BasicOptimizer.STATE_STOPPED
                return results_summary
                
            self.logger.info(f"--- Training Phase: Testing {len(param_combinations)} parameter combinations ---")
            
            for i, params in enumerate(param_combinations):
                self.logger.info(f"Training: Combination {i+1}/{len(param_combinations)} with params: {params}")
                
                # Run backtest and get both overall and regime-specific metrics
                training_metric_value, regime_performance = self._perform_single_backtest_run(params, dataset_type="train")
                results_summary["all_training_results"].append({
                    "parameters": params, 
                    "metric_value": training_metric_value,
                    "regime_performance": regime_performance
                })
                
                # Process both the overall and regime-specific metrics
                if training_metric_value is not None:
                    # Process overall metrics (for backward compatibility)
                    # Type assertion for mypy, as -float('inf') is a float
                    current_best_metric = self._best_training_metric_value if isinstance(self._best_training_metric_value, float) else (-float('inf') if self._higher_metric_is_better else float('inf'))
                    
                    if self._higher_metric_is_better:
                        if training_metric_value > current_best_metric:
                            self._best_training_metric_value = training_metric_value
                            self._best_params_from_train = params
                            self.logger.info(f"New best training metric (overall): {self._best_training_metric_value:.4f} with params: {self._best_params_from_train}")
                    else:
                        if training_metric_value < current_best_metric:
                            self._best_training_metric_value = training_metric_value
                            self._best_params_from_train = params
                            self.logger.info(f"New best training metric (overall): {self._best_training_metric_value:.4f} with params: {self._best_params_from_train}")
                    
                    # Process regime-specific metrics
                    if regime_performance:
                        self._process_regime_performance(params, regime_performance)
                else:
                    self.logger.warning(f"Training run failed or returned no metric for params {params}.")
            
            # Update results summary with both overall and regime-specific results
            results_summary["best_parameters_on_train"] = self._best_params_from_train
            results_summary["best_training_metric_value"] = self._best_training_metric_value
            results_summary["best_parameters_per_regime"] = self._best_params_per_regime
            results_summary["best_metric_per_regime"] = self._best_metric_per_regime
            results_summary["regimes_encountered"] = list(self._regimes_encountered)
            
            # Run the test phase with the overall best parameters (for backward compatibility)
            if self._best_params_from_train:
                self.logger.info(
                    f"--- Testing Phase: Evaluating best overall training parameters {self._best_params_from_train} on test data ---"
                )
                if data_handler_instance.test_df_exists_and_is_not_empty:
                    test_metric_value, _ = self._perform_single_backtest_run(
                        self._best_params_from_train, dataset_type="test"
                    )
                    self._test_metric_for_best_params = test_metric_value
                    results_summary["test_set_metric_value_for_best_params"] = test_metric_value
                else:
                    self.logger.warning("No test data available. Skipping testing phase.")
                    results_summary["test_set_metric_value_for_best_params"] = "N/A (No test data)"
            
            # Final logging
            self._log_optimization_results(results_summary)
            
            # Save results to file
            self._save_results_to_file(results_summary)
            
            self.state = BasicOptimizer.STATE_STOPPED
            return results_summary
            
        except Exception as e:
            self.logger.error(f"Critical error during enhanced grid search optimization: {e}", exc_info=True)
            self.state = BasicOptimizer.STATE_FAILED
            # Ensure results_summary still reflects any partial progress or error state
            results_summary["error"] = str(e)
            return results_summary
        finally:
            if self.state not in [BasicOptimizer.STATE_STOPPED, BasicOptimizer.STATE_FAILED]:
                self.state = BasicOptimizer.STATE_STOPPED
            self.logger.info(f"--- {self.name} Enhanced Grid Search with Train/Test Ended. State: {self.state} ---")
    
    def _log_optimization_results(self, results: Dict[str, Any]) -> None:
        """
        Log the optimization results, including regime-specific optimizations.
        """
        # Format metric values for display
        best_train_metric_str = f"{results['best_training_metric_value']:.4f}" if isinstance(results['best_training_metric_value'], float) else "N/A"
        test_metric_str = f"{results['test_set_metric_value_for_best_params']:.4f}" if isinstance(results['test_set_metric_value_for_best_params'], float) else "N/A"
        
        if results['test_set_metric_value_for_best_params'] == "N/A (No test data)":
            test_metric_str = "N/A (No test data)"
        
        # Log overall results (backward compatibility)
        self.logger.info(
            f"--- Enhanced Grid Search Results ---\n"
            f"  Best Overall Parameters (from Training): {results['best_parameters_on_train']}\n"
            f"  Best Training Metric ('{self._metric_to_optimize}'): {best_train_metric_str}\n"
            f"  Test Set Metric for these parameters ('{self._metric_to_optimize}'): {test_metric_str}"
        )
        
        # Log regime-specific results
        if results['best_parameters_per_regime']:
            self.logger.info("--- Regime-Specific Optimization Results ---")
            
            for regime in sorted(results['best_parameters_per_regime'].keys()):
                best_params = results['best_parameters_per_regime'][regime]
                best_metric = results['best_metric_per_regime'].get(regime, "N/A")
                
                metric_str = f"{best_metric:.4f}" if isinstance(best_metric, float) else str(best_metric)
                
                self.logger.info(f"  Regime: {regime}")
                self.logger.info(f"    Best Parameters: {best_params}")
                self.logger.info(f"    Best Metric ('{self._regime_metric}'): {metric_str}")
        else:
            self.logger.warning("No regime-specific optimization results available.")
    
    def _save_results_to_file(self, results: Dict[str, Any]) -> None:
        """
        Save the optimization results to a JSON file.
        """
        try:
            # Create a simplified version of the results for saving
            save_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "overall_best_parameters": results["best_parameters_on_train"],
                "overall_best_metric": {
                    "name": self._metric_to_optimize,
                    "value": results["best_training_metric_value"],
                    "higher_is_better": self._higher_metric_is_better
                },
                "regime_best_parameters": {
                    regime: {
                        "parameters": results["best_parameters_per_regime"][regime],
                        "metric": {
                            "name": self._regime_metric,
                            "value": results["best_metric_per_regime"][regime],
                            "higher_is_better": self._higher_metric_is_better
                        }
                    }
                    for regime in results["best_parameters_per_regime"]
                },
                "regimes_encountered": results["regimes_encountered"],
                "min_trades_per_regime": self._min_trades_per_regime
            }
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self._output_file_path)) or '.', exist_ok=True)
            
            # Write the data to file
            with open(self._output_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Optimization results saved to {self._output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results to file: {e}", exc_info=True)