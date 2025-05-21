# src/strategy/optimization/enhanced_optimizer.py
import logging
import datetime
import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple
import copy

from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.risk.basic_portfolio import BasicPortfolio
# Use loose coupling - avoid importing specific implementations
# This optimizer should work with any strategy, data handler, portfolio manager, or regime detector
# that implements the appropriate interface methods

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
        self._top_n_to_test = self.get_specific_config("top_n_to_test", 1)
        
        # Memory optimization settings
        self._clear_regime_performance = self.get_specific_config("clear_regime_performance", True)
        
        # Results storage
        self._best_params_per_regime: Dict[str, Dict[str, Any]] = {}
        self._best_metric_per_regime: Dict[str, float] = {}
        self._regimes_encountered: Set[str] = set()
        
        self.logger.info(
            f"{self.name} initialized as EnhancedOptimizer. Will optimize strategy '{self._strategy_service_name}' "
            f"per-regime using metric '{self._regime_metric}' from '{self._portfolio_service_name}'. "
            f"Higher is better: {self._higher_metric_is_better}. "
            f"Min trades per regime: {self._min_trades_per_regime}. "
            f"Testing top {self._top_n_to_test} performers from training on test set."
        )
    
    def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Overridden to also return regime-specific performance metrics.
        First resets the portfolio AND regime detector to ensure each run starts from a clean state.
        Directly processes regimes like test_regime_detection.py does.
        """
        # Portfolio reset will be handled by parent class - no need to duplicate
        
        # Ensure RegimeDetector is available and reset it properly
        try:
            regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
            self.logger.info(f"Found regime detector for optimization: {regime_detector.name}")
            
            # IMPORTANT: Reset the regime detector to ensure clean state for each run
            if hasattr(regime_detector, 'reset') and callable(getattr(regime_detector, 'reset')):
                self.logger.debug(f"Resetting regime detector state before {dataset_type} run")
                regime_detector.reset()
            else:
                # If no reset method exists, try stopping and starting for clean state
                if hasattr(regime_detector, 'stop') and callable(getattr(regime_detector, 'stop')):
                    regime_detector.stop()
                if hasattr(regime_detector, 'setup') and callable(getattr(regime_detector, 'setup')):
                    regime_detector.setup()
                if hasattr(regime_detector, 'start') and callable(getattr(regime_detector, 'start')):
                    regime_detector.start()
                self.logger.debug(f"Restarted regime detector for clean state before {dataset_type} run")
        except Exception as e:
            self.logger.warning(f"RegimeDetector not available or could not be reset for run: {e}. Regime-specific optimization may be limited.")
        
        # Get the overall performance metric from parent class
        overall_metric = super()._perform_single_backtest_run(params_to_test, dataset_type)
        
        # If the run failed or there's no portfolio manager, return None for regime performance
        if overall_metric is None:
            return overall_metric, None
            
        # Get the regime-specific performance metrics
        try:
            portfolio_manager: BasicPortfolio = self._container.resolve(self._portfolio_service_name)
            regime_performance = portfolio_manager.get_performance_by_regime()
            
            # Log the regimes encountered for debugging
            trade_regimes = [r for r in regime_performance.keys() if not r.startswith('_')]
            self.logger.debug(f"Regimes with trades in this run: {trade_regimes}")
            
            # Also log regimes detected by detector but with no trades
            try:
                regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
                if regime_detector and hasattr(regime_detector, 'get_statistics'):
                    detector_regimes = list(regime_detector.get_statistics().get('regime_counts', {}).keys())
                    all_regimes = set(trade_regimes)
                    for r in detector_regimes:
                        if r != 'default':
                            all_regimes.add(r)
                    self.logger.debug(f"All regimes detected in this run: {list(all_regimes)}")
            except Exception as e:
                self.logger.debug(f"Could not get all detected regimes: {e}")
            
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
        
        # Log information about available regimes
        trade_regimes = [r for r in regime_performance.keys() if not r.startswith('_')]
        self.logger.info(f"Processing regimes with trades: {trade_regimes}")
        
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
            
            # Get trade PnL information
            total_pnl = metrics.get('net_pnl', metrics.get('pnl', 0))
            gross_pnl = metrics.get('gross_pnl', 0)
            
            self.logger.info(f"Regime '{regime}' trades: {total_count}, PnL: {total_pnl:.2f}, Pure regime trades: {pure_regime_count}")
            
            # Skip regimes with too few trades (but log it for analysis)
            if total_count < self._min_trades_per_regime:
                self.logger.info(f"Regime '{regime}' had only {total_count} trades with params {params}, which is less than the minimum {self._min_trades_per_regime} required.")
                continue
            
            # Log boundary trade percentage for debugging
            if total_count > 0:
                boundary_pct = (boundary_trade_count / total_count) * 100
                self.logger.info(f"Regime '{regime}': {boundary_trade_count} boundary trades out of {total_count} total ({boundary_pct:.1f}%)")
            
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
        Uses a component-agnostic approach with loose coupling.
        """
        # Import here to avoid circular imports
        from src.core.logging_setup import create_optimization_logger
        
        # Create specialized logger for optimization output
        self.opt_logger = create_optimization_logger("optimization")
        
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
            # Ensure the RegimeDetector is available
            regime_detector = None
            try:
                regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
                self.logger.info(f"EnhancedOptimizer found RegimeDetector: {regime_detector.name}")
                
                # Log the regime detector thresholds for debugging
                if hasattr(regime_detector, '_regime_thresholds'):
                    self.logger.info(f"Regime thresholds: {regime_detector._regime_thresholds}")
            except Exception as e:
                self.logger.warning(f"Could not resolve RegimeDetector: {e}. Regime-specific optimization may not work correctly.")
            
            strategy_to_optimize = self._container.resolve(self._strategy_service_name)
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
                
            total_combinations = len(param_combinations)
            self.logger.info(f"--- Training Phase: Testing {total_combinations} parameter combinations ---")
            
            for i, params in enumerate(param_combinations):
                # Create a compact parameter string for concise logging
                param_str = ", ".join([f"{k.split('.')[-1]}: {v}" for k, v in params.items()])
                
                # Log using specialized optimizer logger (minimal format)
                self.opt_logger.info(f"Running backtest for parameter combination {i+1}/{total_combinations}: {param_str}")
                
                # Run backtest and get both overall and regime-specific metrics
                training_metric_value, regime_performance = self._perform_single_backtest_run(params, dataset_type="train")
                
                # Log a concise summary of the results
                metric_name = self._metric_to_optimize.split('_')[-1] if '_' in self._metric_to_optimize else self._metric_to_optimize
                metric_value_str = f"{training_metric_value:.4f}" if isinstance(training_metric_value, float) else str(training_metric_value)
                
                # Get regimes detected in this run
                regimes_detected = []
                if regime_performance:
                    # Include regimes where trades occurred
                    trade_regimes = [r for r in regime_performance.keys() if not r.startswith('_')]
                    regimes_detected.extend(trade_regimes)
                    
                    # Also get regimes detected by detector but with no trades
                    try:
                        regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
                        if regime_detector and hasattr(regime_detector, 'get_statistics'):
                            detector_regimes = list(regime_detector.get_statistics().get('regime_counts', {}).keys())
                            for r in detector_regimes:
                                if r not in regimes_detected and r != 'default':
                                    regimes_detected.append(r)
                    except Exception as e:
                        self.logger.debug(f"Could not get all detected regimes: {e}")
                regimes_str = ', '.join(regimes_detected) if regimes_detected else "none"
                
                # Use optimizer logger for concise results
                self.opt_logger.info(f"Results for combination {i+1}/{total_combinations}: {metric_name}={metric_value_str}, regimes={regimes_str}")
                results_summary["all_training_results"].append({
                    "parameters": params, 
                    "metric_value": training_metric_value,
                    "regime_performance": regime_performance
                })
                
                # Process both the overall and regime-specific metrics
                if training_metric_value is not None:
                    # Store result for top-N performers tracking
                    results_summary["all_training_results"][-1]["metric_value"] = training_metric_value
                    
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
            
            # Run the test phase with the top N parameters
            if data_handler_instance.test_df_exists_and_is_not_empty:
                # Get top N performers from training
                top_n_performers = self._get_top_n_performers(results_summary["all_training_results"], 
                                                           n=self._top_n_to_test, 
                                                           higher_is_better=self._higher_metric_is_better)
                
                # Initialize test results container
                results_summary["top_n_test_results"] = []
                
                self.logger.info(f"--- Testing Phase: Evaluating top {len(top_n_performers)} parameter sets on test data ---")
                
                for rank, (params, train_metric) in enumerate(top_n_performers):
                    if rank == 0:  # The best performer (for backward compatibility)
                        self._best_params_from_train = params
                        self._best_training_metric_value = train_metric
                    
                    # Create a compact parameter string for concise logging
                    param_str = ", ".join([f"{k.split('.')[-1]}: {v}" for k, v in params.items()])
                    
                    # Use optimizer logger for test phase
                    self.opt_logger.info(f"Testing rank #{rank+1} parameters: {param_str} (train score: {train_metric:.4f})")
                    
                    test_metric_value, _ = self._perform_single_backtest_run(params, dataset_type="test")
                    
                    if rank == 0:  # Store best performer result for backward compatibility
                        self._test_metric_for_best_params = test_metric_value
                        results_summary["test_set_metric_value_for_best_params"] = test_metric_value
                    
                    # Store in top-N results
                    results_summary["top_n_test_results"].append({
                        "rank": rank + 1,
                        "parameters": params,
                        "train_metric": train_metric,
                        "test_metric": test_metric_value
                    })
                    
                    metric_name = self._metric_to_optimize.split('_')[-1] if '_' in self._metric_to_optimize else self._metric_to_optimize
                    self.opt_logger.info(f"Test results for rank #{rank+1}: {metric_name}={test_metric_value:.4f} (train: {train_metric:.4f})")
                    
                # Sort test results by test metric before reporting
                if "top_n_test_results" in results_summary and results_summary["top_n_test_results"]:
                    results_summary["top_n_test_results"].sort(
                        key=lambda x: x["test_metric"] if x["test_metric"] is not None else (-float('inf') if self._higher_metric_is_better else float('inf')),
                        reverse=self._higher_metric_is_better
                    )
                    
                    # Update ranks after sorting
                    for i, result in enumerate(results_summary["top_n_test_results"]):
                        result["rank"] = i + 1
                        # Log the updated rankings
                        metric_name = self._metric_to_optimize.split('_')[-1] if '_' in self._metric_to_optimize else self._metric_to_optimize
                        self.opt_logger.info(f"Updated rank #{i+1} (sorted by test metric): {metric_name}={result['test_metric']:.4f} (train: {result['train_metric']:.4f})")
                
                # Add a separator after test results
                self.logger.info("--- Test Phase Complete ---")
            else:
                self.logger.warning("No test data available. Skipping testing phase.")
                results_summary["test_set_metric_value_for_best_params"] = "N/A (No test data)"
                
            # Memory optimization: clear regime performance data to free memory
            if self._clear_regime_performance and "all_training_results" in results_summary:
                self.logger.info("Clearing regime performance data to free memory...")
                for result in results_summary["all_training_results"]:
                    if "regime_performance" in result:
                        result["regime_performance"] = "<cleared to free memory>"
            
            # Check if regime_adaptive_test_results exists before passing to log function
            self.logger.warning(f"DEBUG - Before logging results, regime_adaptive_test_results exists: {'regime_adaptive_test_results' in results_summary}")
            if 'regime_adaptive_test_results' in results_summary:
                self.logger.warning(f"DEBUG - Keys in regime_adaptive_test_results: {list(results_summary['regime_adaptive_test_results'].keys())}")
            
            # Run regime-adaptive strategy on test set if test data is available - MOVED THIS BEFORE LOGGING
            if data_handler_instance.test_df_exists_and_is_not_empty:
                self.logger.warning("DEBUG - About to run regime-adaptive test")
                self._run_regime_adaptive_test(results_summary)
                self.logger.warning(f"DEBUG - After adaptive test, keys in results: {list(results_summary.keys())}")
                if "regime_adaptive_test_results" in results_summary:
                    self.logger.warning(f"DEBUG - Adaptive results keys: {list(results_summary['regime_adaptive_test_results'].keys())}")
            else:
                self.logger.warning("No test data available. Skipping regime-adaptive strategy test.")
                
            # Final logging - MOVED AFTER ADAPTIVE TEST
            self._log_optimization_results(results_summary)
            
            # Save results to file
            if 'regime_adaptive_test_results' in results_summary:
                self.logger.warning(f"DEBUG - Saving results with regime adaptive test: {list(results_summary['regime_adaptive_test_results'].keys())}")
            else:
                self.logger.warning("DEBUG - No regime adaptive test results to save")
            
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
    
    def _get_top_n_performers(self, training_results: List[Dict[str, Any]], n: int, higher_is_better: bool) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get the top N performing parameter sets from training results.
        
        Args:
            training_results: List of training result dictionaries
            n: Number of top performers to return
            higher_is_better: Whether higher metric values are better
            
        Returns:
            List of tuples (parameters, metric_value) for top N performers
        """
        # Filter out results with None metrics
        valid_results = [
            (result["parameters"], result["metric_value"])
            for result in training_results
            if result.get("metric_value") is not None
        ]
        
        if not valid_results:
            self.logger.warning("No valid training results with metrics found")
            return []
        
        # Sort by metric value (descending if higher is better, ascending otherwise)
        sorted_results = sorted(
            valid_results,
            key=lambda x: x[1],
            reverse=higher_is_better
        )
        
        # Return top N (or all if fewer than N available)
        return sorted_results[:min(n, len(sorted_results))]
    
    def _log_optimization_results(self, results: Dict[str, Any]) -> None:
        """
        Log the optimization results, including regime-specific optimizations.
        
        DEBUG: Added extensive logging to troubleshoot missing regime-adaptive results.
        """
        # Import here to avoid circular imports
        from src.core.logging_setup import create_optimization_logger
        
        # Create specialized logger for optimization summary
        summary_logger = create_optimization_logger("summary")
        
        # Get a shortened metric name for display
        metric_name = self._metric_to_optimize.split('_')[-1] if '_' in self._metric_to_optimize else self._metric_to_optimize
        
        # Format metric values for display
        best_train_metric_str = f"{results['best_training_metric_value']:.4f}" if isinstance(results['best_training_metric_value'], float) else "N/A"
        test_metric_str = f"{results['test_set_metric_value_for_best_params']:.4f}" if isinstance(results['test_set_metric_value_for_best_params'], float) else "N/A"
        
        if results['test_set_metric_value_for_best_params'] == "N/A (No test data)":
            test_metric_str = "N/A (No test data)"
        
        # Create a clean parameter presentation
        best_params_str = ", ".join([f"{k.split('.')[-1]}: {v}" for k, v in results['best_parameters_on_train'].items()])
        
        # Log overall results with a clear separator
        summary_logger.info("\n" + "=" * 80)
        summary_logger.info(f"OPTIMIZATION RESULTS SUMMARY")
        summary_logger.info("=" * 80)
        summary_logger.info(f"Best overall parameters: {best_params_str}")
        summary_logger.info(f"Training {metric_name}: {best_train_metric_str} | Test {metric_name}: {test_metric_str}")
        
        # Log top-N test results if available with better formatting
        if "top_n_test_results" in results and results["top_n_test_results"]:
            summary_logger.info("\nTOP PERFORMERS ON TEST SET:")
            for result in results["top_n_test_results"]:
                params_str = ", ".join([f"{k.split('.')[-1]}: {v}" for k, v in result['parameters'].items()])
                summary_logger.info(
                    f"  #{result['rank']}: Test {metric_name}: {result['test_metric']:.4f} | "
                    f"Train {metric_name}: {result['train_metric']:.4f} | Params: {params_str}"
                )
        
        # Log regime-specific results in a cleaner format
        if results['best_parameters_per_regime']:
            summary_logger.info("\nREGIME-SPECIFIC OPTIMAL PARAMETERS:")
            for regime in sorted(results['best_parameters_per_regime'].keys()):
                # Check if we have the expected structure
                if not isinstance(results['best_parameters_per_regime'][regime], dict):
                    continue
                    
                # For backward compatibility with different result formats
                if 'parameters' in results['best_parameters_per_regime'][regime]:
                    regime_params = results['best_parameters_per_regime'][regime]['parameters']
                    if 'metric' in results['best_parameters_per_regime'][regime]:
                        regime_metric = results['best_parameters_per_regime'][regime]['metric']
                        metric_name = regime_metric.get('name', self._regime_metric)
                        metric_val = regime_metric.get('value', results['best_metric_per_regime'].get(regime, "N/A"))
                    else:
                        metric_name = self._regime_metric
                        metric_val = results['best_metric_per_regime'].get(regime, "N/A")
                else:
                    # Fallback to just the parameters themselves
                    regime_params = results['best_parameters_per_regime'][regime]
                    metric_name = self._regime_metric
                    metric_val = results['best_metric_per_regime'].get(regime, "N/A")
                
                # Format parameters for display
                if isinstance(regime_params, dict):
                    params_str = ", ".join([f"{k.split('.')[-1]}: {v}" for k, v in regime_params.items()])
                else:
                    params_str = str(regime_params)
                    
                # Format metric value
                metric_val_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
                
                summary_logger.info(f"  {regime}: {metric_name}={metric_val_str} | Params: {params_str}")
        
        # Log information about regimes encountered
        if 'regimes_encountered' in results and results['regimes_encountered']:
            summary_logger.info(f"\nRegimes encountered: {', '.join(results['regimes_encountered'])}")
            
        # Log regime-adaptive strategy test results if available
        # First log whether the key exists in the results dictionary
        # Check if adapter results are available - use warning level to make sure it shows
        self.logger.warning(f"FINAL CHECK - Adaptive test results exist: {'regime_adaptive_test_results' in results}")
        if 'regime_adaptive_test_results' in results:
            self.logger.warning(f"FINAL CHECK - Adaptive test results keys: {list(results['regime_adaptive_test_results'].keys())}")
            
        # Use direct print to ensure our formatted output is visible
        print("\n================ REGIME-ADAPTIVE STRATEGY TEST RESULTS ================")
            
        if 'regime_adaptive_test_results' in results and results['regime_adaptive_test_results']:
            adaptive_results = results['regime_adaptive_test_results']
            
            # Print the directly accessed metrics
            self.logger.warning(f"SUMMARY_DEBUG: Available keys in adaptive_results: {list(adaptive_results.keys())}")
            best_metric = adaptive_results.get('best_overall_metric')
            adaptive_metric = adaptive_results.get('adaptive_metric')
            self.logger.warning(f"SUMMARY_DEBUG: best_metric = {best_metric}, adaptive_metric = {adaptive_metric}")
            
            # Format the values
            best_str = f"{best_metric:.2f}" if isinstance(best_metric, (int, float)) else "N/A"
            adaptive_str = f"{adaptive_metric:.2f}" if isinstance(adaptive_metric, (int, float)) else "N/A"
            
            # Calculate improvement
            improvement = ""
            if isinstance(best_metric, (int, float)) and isinstance(adaptive_metric, (int, float)) and best_metric != 0:
                pct = ((adaptive_metric - best_metric) / abs(best_metric)) * 100
                improvement = f" ({'+' if pct >= 0 else ''}{pct:.2f}%)"
                
            # Display metric name
            display_metric = self._metric_to_optimize
            if display_metric.startswith("get_"):
                display_metric = display_metric[4:]
                
            # Print the formatted metrics
            print("-" * 50)
            print(f"Best Overall Static Params Test {display_metric}: {best_str}")
            print(f"Dynamic Regime-Adaptive Strategy Test {display_metric}: {adaptive_str}{improvement}")
            print("-" * 50)
            
            # Print regime info
            if 'regimes_detected' in adaptive_results:
                regimes = adaptive_results['regimes_detected']
                print(f"\nTEST SET REGIME ANALYSIS:")
                print(f"Regimes detected in test set: {', '.join(regimes)}")
                
            # Print coverage info
            if 'regimes_info' in adaptive_results:
                regimes_info = adaptive_results['regimes_info']
                optimized = regimes_info.get('regimes_with_optimized_params', [])
                fallback = regimes_info.get('would_use_default_for', [])
                
                print("\nREGIME PARAMETER COVERAGE:")
                if optimized:
                    print(f"- Regimes with optimized parameters: {', '.join(optimized)}")
                if fallback:
                    print(f"- Regimes using fallback parameters: {', '.join(fallback)}")
                    
            # Print methodology
            if adaptive_results.get('method') == 'true_adaptive':
                print("\nMETHODOLOGY:")
                print("True regime-adaptive strategy with dynamic parameter switching")
                
            print("=" * 80)
            
            # Also include a short summary in the logger
            self.logger.warning(f"ADAPTIVE RESULTS SUMMARY: best={best_str}, adaptive={adaptive_str}{improvement}")
            
            # Check if there was an error
            if 'error' in adaptive_results:
                summary_logger.info(f"  Error: {adaptive_results['error']}")
            elif adaptive_results.get('simulated', False):
                summary_logger.info("  [SIMULATED REGIME-ADAPTIVE RESULTS]")
                
            # Log trade count info, an important sign of success
            if 'adaptive_regime_performance' in adaptive_results:
                # Count total trades across all regimes
                total_trades = 0
                for regime, data in adaptive_results['adaptive_regime_performance'].items():
                    if regime != '_boundary_trades_summary' and isinstance(data, dict) and 'count' in data:
                        total_trades += data['count']
                        
                summary_logger.info(f"\nTOTAL TRADES IN ADAPTIVE TEST: {total_trades}")
                if total_trades == 0:
                    summary_logger.info("WARNING: No trades were generated in the adaptive test!")
                    summary_logger.info("This suggests an issue with event flow or parameter application during regime changes.")
                
                summary_logger.info("\nREGIME-ADAPTIVE STRATEGY TEST RESULTS:")
                
                # Format the best overall metric value
                best_overall = adaptive_results.get('best_overall_metric')
                best_overall_str = f"{best_overall:.4f}" if isinstance(best_overall, float) else "N/A"
                
                # Get the true adaptive strategy performance 
                adaptive_metric = adaptive_results.get('adaptive_metric')
                adaptive_str = f"{adaptive_metric:.4f}" if isinstance(adaptive_metric, float) else "N/A"
                
                # Calculate improvement percentage from explicit field or compute it
                improvement = ""
                if 'improvement_pct' in adaptive_results:
                    pct_change = adaptive_results['improvement_pct']
                    improvement = f" ({'+' if pct_change >= 0 else ''}{pct_change:.2f}%)"
                elif isinstance(adaptive_metric, float) and isinstance(best_overall, float) and best_overall != 0:
                    pct_change = ((adaptive_metric - best_overall) / abs(best_overall)) * 100
                    improvement = f" ({'+' if pct_change >= 0 else ''}{pct_change:.2f}%)"
                
                # Display test results with a more prominent header
                summary_logger.info("\nTRUE REGIME-ADAPTIVE STRATEGY TEST RESULTS:")
                summary_logger.info("=" * 80)
                
                # Get the metric name we should display - use final_portfolio_value if metric_to_optimize is a method name
                display_metric_name = self._metric_to_optimize
                if display_metric_name.startswith("get_"):
                    display_metric_name = display_metric_name[4:]  # Remove 'get_'
                
                self.logger.warning(f"DEBUG - Final metrics for summary - Best Overall: {best_overall_str}, Adaptive: {adaptive_str}")
                self.logger.warning(f"DEBUG - Using display metric name: {display_metric_name}")
                
                # Get values directly from results dictionary
                best_metric_direct = adaptive_results.get('best_overall_metric')
                adaptive_metric_direct = adaptive_results.get('adaptive_metric')
                
                best_direct_str = f"{best_metric_direct:.2f}" if isinstance(best_metric_direct, (int, float)) else "N/A"
                adaptive_direct_str = f"{adaptive_metric_direct:.2f}" if isinstance(adaptive_metric_direct, (int, float)) else "N/A"
                
                # Calculate improvement
                direct_improvement = ""
                if isinstance(best_metric_direct, (int, float)) and isinstance(adaptive_metric_direct, (int, float)) and best_metric_direct != 0:
                    pct = ((adaptive_metric_direct - best_metric_direct) / abs(best_metric_direct)) * 100
                    direct_improvement = f" ({'+' if pct >= 0 else ''}{pct:.2f}%)"
                
                summary_logger.info("-" * 50)
                summary_logger.info(f"Best Overall Static Params Test {display_metric_name}: {best_direct_str}")
                summary_logger.info(f"Dynamic Regime-Adaptive Strategy Test {display_metric_name}: {adaptive_direct_str}{direct_improvement}")
                summary_logger.info("-" * 50)
                
                # Show regimes in the test set
                if 'regimes_detected' in adaptive_results:
                    test_regimes = adaptive_results['regimes_detected']
                    summary_logger.info(f"\nTEST SET REGIME ANALYSIS:")
                    summary_logger.info(f"Regimes detected in test set: {', '.join(test_regimes)}")
                
                # Show which regimes have optimized parameters
                optimized_regimes = []
                default_regimes = []
                
                if 'regimes_info' in adaptive_results:
                    if 'regimes_with_optimized_params' in adaptive_results['regimes_info']:
                        optimized_regimes = adaptive_results['regimes_info']['regimes_with_optimized_params']
                    if 'would_use_default_for' in adaptive_results['regimes_info']:
                        default_regimes = adaptive_results['regimes_info']['would_use_default_for']
                
                summary_logger.info("\nREGIME PARAMETER COVERAGE:")    
                if optimized_regimes:
                    summary_logger.info(f"- Regimes with optimized parameters: {', '.join(optimized_regimes)}")
                if default_regimes:
                    summary_logger.info(f"- Regimes using fallback parameters: {', '.join(default_regimes)}")
                    
                # Show regime-specific performance if available
                if 'adaptive_regime_performance' in adaptive_results:
                    regime_performance = adaptive_results['adaptive_regime_performance']
                    if regime_performance:
                        # Filter out special keys
                        trade_regimes = [r for r in regime_performance.keys() if not r.startswith('_')]
                        if trade_regimes:
                            summary_logger.info("\nPERFORMANCE BY REGIME IN ADAPTIVE TEST:")
                            for regime in trade_regimes:
                                regime_data = regime_performance[regime]
                                # Try to get the metric value
                                metric_val = regime_data.get(self._regime_metric)
                                metric_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else "N/A"
                                
                                # Get trade count
                                trade_count = regime_data.get('count', 0)
                                
                                # Get profit info if available
                                total_net_profit = regime_data.get('total_net_profit', None) 
                                profit_str = f", profit: {total_net_profit:.2f}" if isinstance(total_net_profit, (float, int)) else ""
                                
                                summary_logger.info(f"{regime}: {self._regime_metric}={metric_str}{profit_str} ({trade_count} trades)")
                                
                # Add method information at the end
                method = adaptive_results.get('method', 'simulated')
                if method == 'true_adaptive':
                    summary_logger.info("\nMETHODOLOGY:")
                    summary_logger.info("True regime-adaptive strategy with dynamic parameter switching")
                    
                # Add a final separator
                summary_logger.info("=" * 80)
                
                # Add detailed regime information if available
                if 'regimes_info' in adaptive_results:
                    regimes_info = adaptive_results['regimes_info']
                    if 'regimes_with_optimized_params' in regimes_info and regimes_info['regimes_with_optimized_params']:
                        summary_logger.info(f"\n  Regimes with optimized parameters: {', '.join(regimes_info['regimes_with_optimized_params'])}")
                    
                    if 'would_use_default_for' in regimes_info and regimes_info['would_use_default_for']:
                        summary_logger.info(f"  Regimes that would use default parameters: {', '.join(regimes_info['would_use_default_for'])}")
                
                # Add any message
                if 'message' in adaptive_results:
                    summary_logger.info(f"\n  Note: {adaptive_results['message']}")
                
            else:
                # Format the metric value
                metric_val = adaptive_results.get('metric_value')
                metric_val_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
                
                # Compare with best static parameters
                best_static_metric = results.get('top_n_test_results', [{}])[0].get('test_metric')
                best_static_str = f"{best_static_metric:.4f}" if isinstance(best_static_metric, float) else "N/A"
                
                # Calculate improvement percentage if both metrics are valid numbers
                improvement = ""
                if isinstance(metric_val, float) and isinstance(best_static_metric, float) and best_static_metric != 0:
                    pct_change = ((metric_val - best_static_metric) / abs(best_static_metric)) * 100
                    improvement = f" ({'+' if pct_change >= 0 else ''}{pct_change:.2f}%)"
                
                summary_logger.info(f"  Adaptive Strategy {metric_name}: {metric_val_str} vs Best Static: {best_static_str}{improvement}")
            
            # Log regimes detected during test
            if 'regimes_detected' in adaptive_results:
                summary_logger.info(f"  Regimes detected in test: {', '.join(adaptive_results['regimes_detected'])}")
        
        summary_logger.info("=" * 80 + "\n")
        
        # No verbose regime-specific results to avoid clutter
        if not results['best_parameters_per_regime']:
            self.logger.debug("No regime-specific optimization results available.")
    
    def _run_regime_adaptive_test(self, results: Dict[str, Any]) -> None:
        """
        Run a test of the regime-adaptive strategy on the test set.
        
        This is a completely rewritten implementation to fix issues with:
        - Component lifecycle management (proper stopping and starting)
        - Regime detection and classification
        - Trade generation
        - Performance measurement
        
        The method ensures proper component ordering and event flow by:
        1. Properly stopping all components before reconfiguring
        2. Setting up components in the correct order
        3. Triggering manual classification events to ensure regime changes
        4. Implementing clean lifecycle management for all components
        5. Adding trade generation verification
        6. Ensuring reliable portfolio value calculations
        
        Args:
            results: The optimization results dictionary
        """
        self.logger.info("=== Running Regime-Adaptive Strategy Test on Test Set ===")
        self.logger.debug("Detailed debug logging enabled for regime-adaptive test")
        
        # Initialize results container
        if "regime_adaptive_test_results" not in results:
            results["regime_adaptive_test_results"] = {}
            self.logger.debug("Initialized empty regime_adaptive_test_results dictionary")
            
        # Define component dependencies for proper lifecycle management
        component_dependencies = [
            self._portfolio_service_name, 
            self._strategy_service_name, 
            self._risk_manager_service_name, 
            "MyPrimaryRegimeDetector", 
            "execution_handler",  # Missing execution handler!
            self._data_handler_service_name
        ]
            
        try:
            # Get all necessary components with proper error handling
            self.logger.debug("Resolving all required components for regime-adaptive test")
            data_handler = self._container.resolve(self._data_handler_service_name)
            self.logger.debug(f"Resolved data handler: {data_handler.name if hasattr(data_handler, 'name') else 'unknown'}")
            
            portfolio_manager = self._container.resolve(self._portfolio_service_name)
            self.logger.debug(f"Resolved portfolio manager: {portfolio_manager.name if hasattr(portfolio_manager, 'name') else 'unknown'}")
            
            strategy_to_optimize = self._container.resolve(self._strategy_service_name)
            self.logger.debug(f"Resolved strategy: {strategy_to_optimize.name if hasattr(strategy_to_optimize, 'name') else 'unknown'}")
            
            risk_manager = None
            
            # Try to get risk manager if it exists
            try:
                risk_manager = self._container.resolve(self._risk_manager_service_name)
                self.logger.info(f"Found risk manager for adaptive test: {risk_manager.name if hasattr(risk_manager, 'name') else 'unknown'}")
            except Exception as e:
                self.logger.warning(f"Risk manager not available: {e}. Will proceed without explicit risk management.")
            
            # Get regime detector with proper error handling
            try:
                regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
                self.logger.info(f"Found regime detector for adaptive test: {regime_detector.name if hasattr(regime_detector, 'name') else 'unknown'}")
                
                # Validate that the detector has required methods
                if not hasattr(regime_detector, 'get_current_classification') or not callable(getattr(regime_detector, 'get_current_classification')):
                    raise ValueError("Regime detector doesn't have required get_current_classification method")
                self.logger.debug("Regime detector has required get_current_classification method")
                
                # Additional detailed logging for regime detector configuration
                if hasattr(regime_detector, '_regime_thresholds'):
                    self.logger.info(f"Regime detector thresholds: {regime_detector._regime_thresholds}")
                if hasattr(regime_detector, '_lookback_period'):
                    self.logger.debug(f"Regime detector lookback period: {regime_detector._lookback_period}")
            except Exception as e:
                self.logger.error(f"Could not resolve valid RegimeDetector: {e}. Cannot run regime-adaptive test.", exc_info=True)
                results["regime_adaptive_test_results"] = {"error": f"Missing or invalid regime detector: {str(e)}"}
                return
            
            # Ensure we have regime-specific parameters to work with
            if not results["best_parameters_per_regime"]:
                self.logger.warning("No regime-specific parameters available. Can't run adaptive test.")
                results["regime_adaptive_test_results"] = {"error": "No regime-specific parameters available"}
                return
            
            self.logger.debug(f"Available optimized regimes: {list(results['best_parameters_per_regime'].keys())}")
                
            # Ensure the output file exists for future use
            if not os.path.exists(self._output_file_path):
                self.logger.info(f"Output file {self._output_file_path} doesn't exist. Saving results first.")
                self._save_results_to_file(results)
            
            # ==========================================
            # Reset components before adaptive test to clear state from previous test runs
            # ==========================================
            self.logger.info("Resetting components before adaptive test to clear previous test state...")
            
            # Get component dependencies for proper reset order
            component_dependencies = [
                "MyPrimaryRegimeDetector", 
                "execution_handler",
                self._risk_manager_service_name, 
                self._strategy_service_name, 
                self._portfolio_service_name,
                self._data_handler_service_name
            ]
            
            # Stop all components in reverse dependency order to clear state
            self.logger.debug("Stopping all components to clear state from previous test runs")
            for component_name in component_dependencies:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "stop") and callable(getattr(component, "stop")):
                        self.logger.debug(f"Stopping component: {component.name if hasattr(component, 'name') else 'unknown'}")
                        component.stop()
                        self.logger.debug(f"Successfully stopped component: {component.name if hasattr(component, 'name') else 'unknown'}")
                except Exception as e:
                    self.logger.debug(f"Could not stop component {component_name}: {e}")
            
            # Reset portfolio for clean state
            portfolio_manager = self._container.resolve(self._portfolio_service_name)
            self.logger.debug("Resetting portfolio to clean state before adaptive test")
            portfolio_manager.reset()
            self.logger.debug("Portfolio reset complete")
            
            # Reset regime detector
            regime_detector = self._container.resolve("MyPrimaryRegimeDetector")
            if hasattr(regime_detector, "reset") and callable(getattr(regime_detector, "reset")):
                self.logger.debug("Resetting regime detector")
                regime_detector.reset()
                self.logger.debug("Regime detector reset complete")
            
            # ==========================================
            # Step 1: Run the best overall strategy on test set for comparison baseline
            # ==========================================
            # TEMPORARILY COMMENTED OUT FOR DEBUGGING - focus only on adaptive test
            self.logger.info("SKIPPING best overall strategy test - focusing on adaptive test only...")
            
            # Comment out entire best overall test section to simplify debugging
            """
            
            # PROPER COMPONENT LIFECYCLE MANAGEMENT:
            # First stop all components in reverse dependency order
            self.logger.debug("Stopping all components in reverse dependency order")
            for component_name in component_dependencies:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "stop") and callable(getattr(component, "stop")):
                        self.logger.debug(f"Stopping component: {component.name if hasattr(component, 'name') else 'unknown'}")
                        component.stop()
                        self.logger.debug(f"Successfully stopped component: {component.name if hasattr(component, 'name') else 'unknown'}")
                except Exception as e:
                    self.logger.debug(f"Could not stop component {component_name}: {e}")
            
            # Reset portfolio for clean state
            self.logger.debug("Resetting portfolio to clean state")
            portfolio_manager.reset()
            self.logger.debug("Portfolio reset complete")
            
            # Reset regime detector
            if hasattr(regime_detector, "reset") and callable(getattr(regime_detector, "reset")):
                self.logger.debug("Resetting regime detector")
                regime_detector.reset()
                self.logger.debug("Regime detector reset complete")
            
            # Set dataset to test
            self.logger.debug("Setting data handler to use test dataset")
            if hasattr(data_handler, "set_active_dataset") and callable(getattr(data_handler, "set_active_dataset")):
                data_handler.set_active_dataset("test")
                self.logger.debug("Set active dataset to 'test'")
                
                # DEBUG: Verify the test dataset was actually set
                if hasattr(data_handler, '_active_df'):
                    df_size = len(data_handler._active_df) if data_handler._active_df is not None else 0
                    self.logger.warning(f"ADAPTIVE_DEBUG: After setting test dataset, active_df size: {df_size}")
                    
                # DEBUG: Check what datasets are available
                if hasattr(data_handler, 'test_df'):
                    test_size = len(data_handler.test_df) if data_handler.test_df is not None else 0
                    self.logger.warning(f"ADAPTIVE_DEBUG: test_df size: {test_size}")
                if hasattr(data_handler, 'train_df'):
                    train_size = len(data_handler.train_df) if data_handler.train_df is not None else 0
                    self.logger.warning(f"ADAPTIVE_DEBUG: train_df size: {train_size}")
                    
            else:
                self.logger.warning("Data handler does not have set_active_dataset method, trying fallbacks...")
                if hasattr(data_handler, "use_test_data") and callable(getattr(data_handler, "use_test_data")):
                    data_handler.use_test_data()
                    self.logger.debug("Called use_test_data() on data handler")
            
            # Ensure data handler has proper test data
            if hasattr(data_handler, "test_df") and hasattr(data_handler.test_df, "empty"):
                empty = data_handler.test_df.empty
                self.logger.debug(f"Test dataframe empty status: {empty}")
                if empty:
                    self.logger.error("Test dataframe is empty! Cannot continue with adaptive test.")
                    results["regime_adaptive_test_results"] = {"error": "Test dataframe is empty"}
                    return
                self.logger.debug(f"Test dataframe shapes: {data_handler.test_df.shape}")
                    
            # Configure strategy with best overall parameters
            if "best_parameters_on_train" in results and results["best_parameters_on_train"]:
                best_overall_params = results["best_parameters_on_train"]
                if hasattr(strategy_to_optimize, "set_parameters") and callable(getattr(strategy_to_optimize, "set_parameters")):
                    self.logger.warning(f"BEST_OVERALL_DEBUG: Applying best overall parameters to strategy: {best_overall_params}")
                    strategy_to_optimize.set_parameters(best_overall_params)
                    # Verify what parameters were actually applied
                    current_params = strategy_to_optimize.get_parameters()
                    self.logger.warning(f"BEST_OVERALL_DEBUG: Current strategy params after application: {current_params}")
            else:
                self.logger.warning("No best_parameters_on_train available in results")
                
            # Start consumers first, then data publisher
            # Data handler should be LAST so all consumers are ready for events
            component_start_order = [
                "MyPrimaryRegimeDetector", 
                "execution_handler",  # Handles ORDER events from risk manager
                self._risk_manager_service_name, 
                self._strategy_service_name, 
                self._portfolio_service_name,
                self._data_handler_service_name  # LAST - publishes events after all consumers ready
            ]
            
            self.logger.debug("Starting all components in correct dependency order")
            for component_name in component_start_order:
                try:
                    component = self._container.resolve(component_name)
                    # Components are already initialized - no need to call setup() again
                    # Calling setup() would reset the CSVDataHandler's _active_df = None
                    
                    if hasattr(component, "start") and callable(getattr(component, "start")):
                        self.logger.warning(f"OPTIMIZER_DEBUG: Starting component: {component.name if hasattr(component, 'name') else component_name} (State: {component.state if hasattr(component, 'state') else 'N/A'})")
                        component.start()
                        self.logger.warning(f"OPTIMIZER_DEBUG: Successfully started component: {component.name if hasattr(component, 'name') else component_name} (New State: {component.state if hasattr(component, 'state') else 'N/A'})")
                    else:
                        self.logger.debug(f"Component {component_name} does not have a callable start method.")
                except Exception as e:
                    self.logger.warning(f"Could not start or process component {component_name}: {e}", exc_info=True)
            
            # Verify component states
            for component_name in component_start_order:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "state"):
                        self.logger.debug(f"Component {component_name} state: {component.state}")
                except Exception:
                    pass
                    
            # Let data flow naturally - the data handler will automatically stream when started
            self.logger.info("Starting data flow with best overall parameters...")
            self.logger.debug("Data streaming will occur automatically via component lifecycle")
            # No explicit run_simulation call needed - data streams when components are started
            
            # DEBUG: Check if data handler actually has test data and will stream it
            if hasattr(data_handler, '_active_df'):
                df_size = len(data_handler._active_df) if data_handler._active_df is not None else 0
                self.logger.warning(f"ADAPTIVE_DEBUG: Data handler active dataset size: {df_size}")
                if df_size == 0:
                    self.logger.error("ADAPTIVE_DEBUG: Active dataset is empty! No data will stream!")
            
            # DEBUG: Verify data handler actually started and will publish data
            if hasattr(data_handler, 'state'):
                self.logger.warning(f"ADAPTIVE_DEBUG: Data handler state: {data_handler.state}")
                
            # DEBUG: Force data streaming if needed
            if hasattr(data_handler, 'start') and callable(getattr(data_handler, 'start')):
                self.logger.warning("ADAPTIVE_DEBUG: Explicitly calling data_handler.start() to ensure streaming")
                data_handler.start()
            
            # IMPORTANT: Allow time for data to stream and trades to be generated
            import time
            time.sleep(0.1)  # Brief pause to allow event processing
            
            # Check if any trades were generated
            trade_count_best = 0
            if hasattr(portfolio_manager, "_trade_log"):
                trade_count_best = len(portfolio_manager._trade_log)
                self.logger.info(f"Generated {trade_count_best} trades with best overall parameters")
                
                # Log some details of trades if available
                if trade_count_best > 0 and len(portfolio_manager._trade_log) > 0:
                    first_trade = portfolio_manager._trade_log[0]
                    last_trade = portfolio_manager._trade_log[-1]
                    self.logger.debug(f"First trade: {first_trade}")
                    self.logger.debug(f"Last trade: {last_trade}")
                
                if trade_count_best == 0:
                    self.logger.warning("No trades generated with best overall parameters! This indicates a problem.")
                    
                    # Additional debugging for no trades
                    if hasattr(strategy_to_optimize, "get_parameters"):
                        self.logger.debug(f"Current strategy parameters: {strategy_to_optimize.get_parameters()}")
                    if hasattr(data_handler, "current_index"):
                        self.logger.debug(f"Data handler current index: {data_handler.current_index}")
            
            # Get performance metrics
            best_overall_metric = None
            try:
                self.logger.warning(f"BEST_OVERALL_DEBUG: Portfolio object ID: {id(portfolio_manager)}, trade count: {len(portfolio_manager._trade_log) if hasattr(portfolio_manager, '_trade_log') else 'N/A'}")
                self.logger.debug(f"Attempting to get performance metric: {self._metric_to_optimize}")
                
                # Try multiple approaches to get the metric value
                if hasattr(portfolio_manager, self._metric_to_optimize) and callable(getattr(portfolio_manager, self._metric_to_optimize)):
                    # Direct method call
                    method = getattr(portfolio_manager, self._metric_to_optimize)
                    best_overall_metric = method()
                    self.logger.debug(f"Got metric via direct method call: {self._metric_to_optimize}() = {best_overall_metric}")
                elif hasattr(portfolio_manager, "get_performance") and callable(getattr(portfolio_manager, "get_performance")):
                    # Via performance dictionary
                    performance = portfolio_manager.get_performance()
                    self.logger.debug(f"Performance dict keys: {list(performance.keys())}")
                    if self._metric_to_optimize in performance:
                        best_overall_metric = performance[self._metric_to_optimize]
                        self.logger.debug(f"Got metric via performance dict: {self._metric_to_optimize} = {best_overall_metric}")
                
                # Fallback for final portfolio value
                if best_overall_metric is None and self._metric_to_optimize == "get_final_portfolio_value":
                    self.logger.debug("Trying fallbacks for final portfolio value")
                    if hasattr(portfolio_manager, "current_total_value"):
                        best_overall_metric = portfolio_manager.current_total_value
                        self.logger.debug(f"Got metric via current_total_value attribute: {best_overall_metric}")
                    elif hasattr(portfolio_manager, "calculate_total_value") and callable(getattr(portfolio_manager, "calculate_total_value")):
                        best_overall_metric = portfolio_manager.calculate_total_value()
                        self.logger.debug(f"Got metric via calculate_total_value() method: {best_overall_metric}")
                
                if best_overall_metric is None:
                    self.logger.warning(f"Could not get {self._metric_to_optimize} value through any method")
            except Exception as e:
                self.logger.error(f"Error getting best overall metric: {e}", exc_info=True)
                
            # Store metric
            results["regime_adaptive_test_results"]["best_overall_metric"] = best_overall_metric
            self.logger.info(f"Best overall parameters performance on test: {best_overall_metric}")
            
            # Get regime-specific performance
            best_overall_regime_performance = None
            if hasattr(portfolio_manager, "get_performance_by_regime") and callable(getattr(portfolio_manager, "get_performance_by_regime")):
                self.logger.debug("Getting regime-specific performance for best overall run")
                best_overall_regime_performance = portfolio_manager.get_performance_by_regime()
                if best_overall_regime_performance:
                    regimes_with_trades = [r for r in best_overall_regime_performance.keys() if not r.startswith('_')]
                    self.logger.debug(f"Regimes with trades in best overall run: {regimes_with_trades}")
                
            results["regime_adaptive_test_results"]["best_overall_regime_performance"] = best_overall_regime_performance
            """
            
            # Initialize results without best overall test
            results["regime_adaptive_test_results"]["best_overall_performance"] = {"final_value": "N/A - test skipped"}
            results["regime_adaptive_test_results"]["best_overall_regime_performance"] = None
            
            # ==========================================
            # Step 2: Run the adaptive strategy with regime-specific parameters
            # ==========================================
            self.logger.info("\n==========================================")
            self.logger.info("STEP 3: Running true regime-adaptive strategy simulation")
            self.logger.info("==========================================\n")
            
            # Stop all components again in reverse dependency order
            self.logger.debug("Stopping all components again for adaptive strategy test")
            for component_name in component_dependencies:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "stop") and callable(getattr(component, "stop")):
                        self.logger.debug(f"Stopping component: {component.name if hasattr(component, 'name') else 'unknown'}")
                        component.stop()
                except Exception as e:
                    self.logger.debug(f"Could not stop component {component_name}: {e}")
            
            # Store available regimes for reporting
            if hasattr(regime_detector, "get_statistics") and callable(getattr(regime_detector, "get_statistics")):
                self.logger.debug("Getting statistics from regime detector")
                regime_stats = regime_detector.get_statistics()
                regimes_detected = list(regime_stats.get("regime_counts", {}).keys())
                results["regime_adaptive_test_results"]["regimes_detected"] = regimes_detected
                self.logger.info(f"Regimes detected in test set: {regimes_detected}")
            
            # CRITICAL FIX: Reset portfolio between best overall test and adaptive test
            # This ensures each test runs independently and accumulates its own trades
            # COMMENTED OUT: Since we're only running adaptive test now, this reset wipes out the results
            # self.logger.warning("PORTFOLIO_RESET_DEBUG: Resetting portfolio between best overall and adaptive tests")
            # portfolio_manager.reset()
            # self.logger.warning("PORTFOLIO_RESET_DEBUG: Portfolio reset complete for independent adaptive test")
            
            if hasattr(regime_detector, "reset") and callable(getattr(regime_detector, "reset")):
                self.logger.debug("Resetting regime detector")
                regime_detector.reset()
                self.logger.debug("Regime detector reset complete")
                
            # Set dataset to test again
            self.logger.debug("Setting data handler to use test dataset again")
            if hasattr(data_handler, "set_active_dataset") and callable(getattr(data_handler, "set_active_dataset")):
                data_handler.set_active_dataset("test")
                self.logger.debug("Set active dataset to 'test'")
            else:
                if hasattr(data_handler, "use_test_data") and callable(getattr(data_handler, "use_test_data")):
                    data_handler.use_test_data()
                    self.logger.debug("Called use_test_data() on data handler")
            
            # Define the event handler for regime classification changes
            self.logger.debug("Setting up regime classification change handler")
            
            def on_classification_change(event):
                """Handle regime classification change events and update strategy parameters"""
                self.logger.warning("REGIME_DEBUG: Classification event received in optimizer")
                if not hasattr(event, 'payload') or not event.payload:
                    self.logger.warning("No payload in classification event")
                    return
                    
                payload = event.payload
                if not isinstance(payload, dict):
                    self.logger.warning(f"Payload is not a dictionary, got {type(payload)}")
                    return
                    
                new_regime = payload.get('classification')
                if not new_regime:
                    self.logger.warning("No classification in payload")
                    return
                    
                self.logger.warning(f"REGIME_DEBUG: Regime changed to: {new_regime}")
                
                # Log timestamp if available
                if 'timestamp' in payload:
                    self.logger.debug(f"Classification timestamp: {payload['timestamp']}")
                
                # Apply regime-specific parameters
                self.logger.warning(f"REGIME_DEBUG: Available optimized regimes in results: {list(results['best_parameters_per_regime'].keys())}")
                if new_regime in results["best_parameters_per_regime"]:
                    params = results["best_parameters_per_regime"][new_regime]
                    self.logger.warning(f"REGIME_DEBUG: Applying optimized parameters for regime {new_regime}: {params}")
                    try:
                        strategy_to_optimize.set_parameters(params)
                        # Verify the parameters were actually applied
                        current_params = strategy_to_optimize.get_parameters()
                        self.logger.warning(f"REGIME_DEBUG: Successfully applied parameters for regime {new_regime}. Current strategy params: {current_params}")
                    except Exception as e:
                        self.logger.error(f"Failed to apply parameters for regime {new_regime}: {e}", exc_info=True)
                elif "default" in results["best_parameters_per_regime"]:
                    params = results["best_parameters_per_regime"]["default"]
                    self.logger.info(f"No specific parameters for {new_regime}, using default regime parameters")
                    try:
                        strategy_to_optimize.set_parameters(params)
                        self.logger.debug("Successfully applied default parameters")
                    except Exception as e:
                        self.logger.error(f"Failed to apply default parameters: {e}", exc_info=True)
                else:
                    self.logger.info(f"No specific parameters for {new_regime}, using best overall parameters")
                    try:
                        strategy_to_optimize.set_parameters(results["best_parameters_on_train"])
                        self.logger.debug("Successfully applied best overall parameters")
                    except Exception as e:
                        self.logger.error(f"Failed to apply best overall parameters: {e}", exc_info=True)
            
            # Subscribe to classification events
            from src.core.event import EventType
            self.logger.info("Subscribing to CLASSIFICATION events")
            self._event_bus.subscribe(EventType.CLASSIFICATION, on_classification_change)
            self.logger.debug("Successfully subscribed to CLASSIFICATION events")
            
            # Define component start order for adaptive test
            component_start_order = [
                "MyPrimaryRegimeDetector", 
                "execution_handler",  # Handles ORDER events from risk manager
                self._risk_manager_service_name, 
                self._strategy_service_name, 
                self._portfolio_service_name,
                self._data_handler_service_name  # LAST - publishes events after all consumers ready
            ]
            
            # Start components in dependency order
            self.logger.debug("Starting all components for adaptive test in dependency order")
            for component_name in component_start_order:
                try:
                    component = self._container.resolve(component_name)
                    # Components are already initialized - no need to call setup() again
                    # Calling setup() would reset the CSVDataHandler's _active_df = None
                    
                    if hasattr(component, "start") and callable(getattr(component, "start")):
                        self.logger.warning(f"OPTIMIZER_DEBUG: Starting component: {component.name if hasattr(component, 'name') else component_name} (State: {component.state if hasattr(component, 'state') else 'N/A'})")
                        component.start()
                        self.logger.warning(f"OPTIMIZER_DEBUG: Successfully started component: {component.name if hasattr(component, 'name') else component_name} (New State: {component.state if hasattr(component, 'state') else 'N/A'})")
                    else:
                        self.logger.debug(f"Component {component_name} does not have a callable start method.")
                except Exception as e:
                    self.logger.warning(f"Could not start or process component {component_name}: {e}", exc_info=True)
            
            # Verify all components are properly started
            for component_name in component_start_order:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "state"):
                        self.logger.debug(f"Component {component_name} state: {component.state}")
                except Exception:
                    pass
            
            # Apply initial parameters - use default or best overall
            initial_params = results["best_parameters_per_regime"].get("default", results["best_parameters_on_train"])
            if hasattr(strategy_to_optimize, "set_parameters") and callable(getattr(strategy_to_optimize, "set_parameters")):
                self.logger.info(f"Setting initial parameters for adaptive strategy: {initial_params}")
                strategy_to_optimize.set_parameters(initial_params)
                self.logger.debug("Successfully set initial parameters")
            
            # Immediately force a classification to ensure parameter switching works
            current_regime = None
            try:
                self.logger.warning("REGIME_DEBUG: Requesting current classification from regime detector")
                current_regime = regime_detector.get_current_classification()
                self.logger.warning(f"REGIME_DEBUG: Initial regime detected: {current_regime}")
            except Exception as e:
                self.logger.error(f"REGIME_DEBUG: Error getting current classification: {e}", exc_info=True)
            
            # Manually trigger an initial classification event
            if current_regime:
                try:
                    from src.core.event import Event
                    self.logger.warning(f"REGIME_DEBUG: Creating manual classification event for regime: {current_regime}")
                    classification_payload = {
                        'classification': current_regime,
                        'timestamp': datetime.datetime.now(datetime.timezone.utc),
                        'detector_name': regime_detector.name if hasattr(regime_detector, 'name') else 'unknown',
                        'source': 'manual_trigger_from_optimizer'
                    }
                    classification_event = Event(EventType.CLASSIFICATION, classification_payload)
                    self.logger.warning("REGIME_DEBUG: Publishing manual classification event to event bus")
                    self._event_bus.publish(classification_event)
                    self.logger.warning(f"REGIME_DEBUG: Manually published initial CLASSIFICATION event for regime '{current_regime}'")
                except Exception as e:
                    self.logger.error(f"Error publishing manual classification event: {e}", exc_info=True)
            else:
                self.logger.warning("No initial regime classification available for manual trigger")
            
            # The simulation will run automatically - data streams when components are started
            self.logger.info("Data flow initiated with adaptive regime-switching...")
            self.logger.debug("Adaptive simulation will occur automatically via component lifecycle")
            # No explicit run_simulation call needed - data streams when components are started
            
            # IMPORTANT: Wait for data streaming to complete
            import time
            time.sleep(0.1)  # Brief pause to ensure all events are processed
            
            self.logger.warning("=== CRITICAL_TIMING_DEBUG: Data streaming should be complete ===")
            if hasattr(portfolio_manager, "_trade_log"):
                immediate_count = len(portfolio_manager._trade_log)
                self.logger.warning(f"CRITICAL_TIMING_DEBUG: Immediate post-streaming trade count: {immediate_count}")
                
            # Check if any trades were generated
            self.logger.warning("=== CRITICAL_MEASUREMENT_DEBUG: About to check adaptive trade count ===")
            if hasattr(portfolio_manager, "_trade_log"):
                actual_count = len(portfolio_manager._trade_log)
                self.logger.warning(f"CRITICAL_MEASUREMENT_DEBUG: Portfolio _trade_log has {actual_count} trades")
                
            trade_count_adaptive = 0
            if hasattr(portfolio_manager, "_trade_log"):
                trade_count_adaptive = len(portfolio_manager._trade_log)
                self.logger.warning(f"CRITICAL_MEASUREMENT_DEBUG: Measured adaptive strategy trades: {trade_count_adaptive}")
                self.logger.info(f"Adaptive strategy generated {trade_count_adaptive} trades in total")
                
                # Log trade details if available
                if trade_count_adaptive > 0 and len(portfolio_manager._trade_log) > 0:
                    first_trade = portfolio_manager._trade_log[0]
                    last_trade = portfolio_manager._trade_log[-1]
                    self.logger.debug(f"First adaptive trade: {first_trade}")
                    self.logger.debug(f"Last adaptive trade: {last_trade}")
                
                if trade_count_adaptive == 0:
                    self.logger.error("No trades generated during adaptive test! Something is wrong with the implementation.")
                    results["regime_adaptive_test_results"]["warning"] = "No trades were generated during adaptive test"
                    
                    # Additional debugging for no trades
                    if hasattr(regime_detector, "get_statistics") and callable(getattr(regime_detector, "get_statistics")):
                        stats = regime_detector.get_statistics()
                        self.logger.debug(f"Regime detector statistics: {stats}")
                    if hasattr(strategy_to_optimize, "get_parameters"):
                        self.logger.debug(f"Current strategy parameters: {strategy_to_optimize.get_parameters()}")
                    if hasattr(data_handler, "current_index"):
                        self.logger.debug(f"Data handler current index: {data_handler.current_index}")
            
            # CRITICAL DEBUG: Check portfolio state IMMEDIATELY after data streaming is complete
            self.logger.warning("=== ADAPTIVE_FINAL_DEBUG: Checking portfolio state immediately after data streaming ===")
            if hasattr(portfolio_manager, "_trade_log"):
                trade_count = len(portfolio_manager._trade_log)
                self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Portfolio has {trade_count} trades in _trade_log")
                if trade_count > 0:
                    self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: First trade: {portfolio_manager._trade_log[0]}")
                    self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Last trade: {portfolio_manager._trade_log[-1]}")
            if hasattr(portfolio_manager, "current_cash"):
                self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Portfolio current_cash: {portfolio_manager.current_cash}")
            if hasattr(portfolio_manager, "current_total_value"):
                self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Portfolio current_total_value: {portfolio_manager.current_total_value}")
            
            # Get adaptive strategy performance
            adaptive_metric = None
            try:
                self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Portfolio object ID: {id(portfolio_manager)}, trade count: {len(portfolio_manager._trade_log) if hasattr(portfolio_manager, '_trade_log') else 'N/A'}")
                self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Attempting to get performance metric for adaptive strategy: {self._metric_to_optimize}")
                
                # Try multiple approaches to get the metric value
                # IMPORTANT: Don't call methods that might reset the portfolio!
                if hasattr(portfolio_manager, self._metric_to_optimize) and callable(getattr(portfolio_manager, self._metric_to_optimize)):
                    # Direct method call
                    method = getattr(portfolio_manager, self._metric_to_optimize)
                    adaptive_metric = method()
                    self.logger.warning(f"ADAPTIVE_FINAL_DEBUG: Got adaptive metric via direct method call: {self._metric_to_optimize}() = {adaptive_metric}")
                elif hasattr(portfolio_manager, "get_performance") and callable(getattr(portfolio_manager, "get_performance")):
                    # Via performance dictionary
                    performance = portfolio_manager.get_performance()
                    self.logger.debug(f"Performance dict keys for adaptive run: {list(performance.keys())}")
                    if self._metric_to_optimize in performance:
                        adaptive_metric = performance[self._metric_to_optimize]
                        self.logger.debug(f"Got adaptive metric via performance dict: {self._metric_to_optimize} = {adaptive_metric}")
                
                # Fallback for final portfolio value
                if adaptive_metric is None and self._metric_to_optimize == "get_final_portfolio_value":
                    self.logger.debug("Trying fallbacks for adaptive final portfolio value")
                    if hasattr(portfolio_manager, "current_total_value"):
                        adaptive_metric = portfolio_manager.current_total_value
                        self.logger.debug(f"Got adaptive metric via current_total_value attribute: {adaptive_metric}")
                    elif hasattr(portfolio_manager, "calculate_total_value") and callable(getattr(portfolio_manager, "calculate_total_value")):
                        adaptive_metric = portfolio_manager.calculate_total_value()
                        self.logger.debug(f"Got adaptive metric via calculate_total_value() method: {adaptive_metric}")
                    
                # Last resort fallback - calculate from positions and cash
                if adaptive_metric is None and hasattr(portfolio_manager, "positions") and hasattr(portfolio_manager, "cash"):
                    self.logger.debug("Using last resort calculation from positions and cash")
                    total_position_value = sum(position.current_value for position in portfolio_manager.positions.values())
                    adaptive_metric = total_position_value + portfolio_manager.cash
                    self.logger.debug(f"Calculated adaptive metric from positions and cash: {adaptive_metric}")
                
                if adaptive_metric is None:
                    self.logger.warning(f"Could not get {self._metric_to_optimize} value for adaptive run through any method")
            except Exception as e:
                self.logger.error(f"Error getting adaptive strategy metric: {e}", exc_info=True)
                
            # Store adaptive metric
            results["regime_adaptive_test_results"]["adaptive_metric"] = adaptive_metric
            self.logger.info(f"Adaptive strategy performance on test: {adaptive_metric}")
            
            # Get regime-specific performance for the adaptive strategy
            adaptive_regime_performance = None
            if hasattr(portfolio_manager, "get_performance_by_regime") and callable(getattr(portfolio_manager, "get_performance_by_regime")):
                self.logger.debug("Getting regime-specific performance for adaptive strategy")
                adaptive_regime_performance = portfolio_manager.get_performance_by_regime()
                if adaptive_regime_performance:
                    adaptive_regimes = [r for r in adaptive_regime_performance.keys() if not r.startswith('_')]
                    self.logger.debug(f"Regimes with trades in adaptive run: {adaptive_regimes}")
                    
                    # Log detailed performance metrics for each regime
                    for regime in adaptive_regimes:
                        metrics = adaptive_regime_performance[regime]
                        self.logger.debug(f"Performance for regime {regime}: {metrics}")
            
            results["regime_adaptive_test_results"]["adaptive_regime_performance"] = adaptive_regime_performance
            
            # Unsubscribe from classification events
            self.logger.debug("Unsubscribing from CLASSIFICATION events")
            self._event_bus.unsubscribe(EventType.CLASSIFICATION, on_classification_change)
            self.logger.debug("Successfully unsubscribed from CLASSIFICATION events")
            
            # Calculate improvement over best overall (skipped since we're only running adaptive test)
            # if isinstance(adaptive_metric, (int, float)) and isinstance(best_overall_metric, (int, float)) and best_overall_metric != 0:
            #     improvement_pct = ((adaptive_metric - best_overall_metric) / abs(best_overall_metric)) * 100
            #     results["regime_adaptive_test_results"]["improvement_pct"] = improvement_pct
            #     self.logger.info(f"Improvement over best overall parameters: {improvement_pct:.2f}%")
            
            # Add detailed results
            results["regime_adaptive_test_results"]["method"] = "true_adaptive" 
            results["regime_adaptive_test_results"]["message"] = "Used true regime-adaptive strategy with dynamic parameter switching"
            results["regime_adaptive_test_results"]["trade_counts"] = {
                "best_overall": "N/A - test skipped",
                "adaptive": trade_count_adaptive
            }
            
            # Add information about which regimes have optimized parameters
            self.logger.debug("Analyzing regime parameter coverage")
            regimes_with_params = []
            for regime in regimes_detected:
                if regime in results["best_parameters_per_regime"]:
                    regimes_with_params.append(regime)
            
            missing_regimes = [r for r in regimes_detected if r not in regimes_with_params and r != "default"]
            
            regimes_info = {
                "regimes_with_optimized_params": regimes_with_params,
                "regimes_without_params": missing_regimes,
                "would_use_default_for": missing_regimes
            }
            
            results["regime_adaptive_test_results"]["regimes_info"] = regimes_info
            
            # Summary logging
            self.logger.info("\n=== Adaptive Strategy Test Results Summary ===")
            self.logger.info(f"Regimes detected in test: {regimes_detected}")
            self.logger.info(f"Regimes with optimized parameters: {regimes_with_params}")
            if missing_regimes:
                self.logger.info(f"Regimes using fallback parameters: {missing_regimes}")
            # self.logger.info(f"Best overall metric: {best_overall_metric}")  # Skipped since we're only running adaptive test
            self.logger.info(f"Adaptive strategy metric: {adaptive_metric}")
            # if isinstance(adaptive_metric, (int, float)) and isinstance(best_overall_metric, (int, float)) and best_overall_metric != 0:
            #     self.logger.info(f"Improvement: {improvement_pct:.2f}%")
            # self.logger.info(f"Trades with best overall: {trade_count_best}")  # Skipped
            self.logger.info(f"Trades with adaptive: {trade_count_adaptive}")
            
            # Verify results dictionary is complete
            self.logger.debug(f"Keys in regime_adaptive_test_results: {list(results['regime_adaptive_test_results'].keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to run regime-adaptive strategy test: {e}", exc_info=True)
            results["regime_adaptive_test_results"] = {"error": str(e)}
        finally:
            # Clean-up: stop all components
            self.logger.debug("Performing final cleanup - stopping all components")
            for component_name in component_dependencies:
                try:
                    component = self._container.resolve(component_name)
                    if hasattr(component, "stop") and callable(getattr(component, "stop")):
                        self.logger.debug(f"Stopping component: {component.name if hasattr(component, 'name') else 'unknown'}")
                        component.stop()
                except Exception as e:
                    self.logger.debug(f"Error during cleanup for component {component_name}: {e}")
                    
            self.logger.info("=== Regime-Adaptive Strategy Test Complete ===")
            
    def _save_results_to_file(self, results: Dict[str, Any]) -> None:
        """
        Save the optimization results to a JSON file.
        
        DEBUG: Ensure we're saving the regime-adaptive test results.
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
            
            # Add test results data if available
            if "top_n_test_results" in results and results["top_n_test_results"]:
                save_data["test_results"] = results["top_n_test_results"]
                
            # Add regime-adaptive test results if available
            self.logger.warning(f"DEBUG - In save_results_to_file, regime_adaptive_test_results exists: {'regime_adaptive_test_results' in results}")
            if "regime_adaptive_test_results" in results and results["regime_adaptive_test_results"]:
                save_data["regime_adaptive_test_results"] = results["regime_adaptive_test_results"]
                self.logger.warning(f"DEBUG - Saving adaptive test results: {list(results['regime_adaptive_test_results'].keys())}")
            else:
                self.logger.warning("DEBUG - No regime_adaptive_test_results available to save")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self._output_file_path)) or '.', exist_ok=True)
            
            # Write the data to file
            with open(self._output_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            self.logger.info(f"Optimization results saved to {self._output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results to file: {e}", exc_info=True)