"""
EnhancedOptimizer V2 - Modified to use modular components (BacktestEngine, ResultsManager, ParameterManager).

This is a transitional implementation showing how to integrate the new components
while maintaining backward compatibility.
"""

import json
import tempfile
import os
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.strategy.optimization.engines import BacktestEngine
from src.strategy.optimization.results import ResultsManager
from src.strategy.optimization.core import ParameterManager


class EnhancedOptimizerV2(EnhancedOptimizer):
    """
    Enhanced optimizer using modular components for better maintainability
    and consistency with production runs.
    """
    
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container):
        """Initialize the enhanced optimizer with new components."""
        super().__init__(instance_name, config_loader, event_bus, component_config_key, container)
        
        # Initialize new modular components
        self.backtest_engine = BacktestEngine(container, config_loader, event_bus)
        self.results_manager = ResultsManager()
        self.parameter_manager = ParameterManager()
        
        self.logger.info("Initialized EnhancedOptimizerV2 with modular components")
        
    def _perform_single_backtest_run(
        self, 
        params_to_test: Dict[str, Any], 
        dataset_type: str
    ) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Use BacktestEngine for consistent backtest execution.
        
        This ensures identical behavior between optimization and production runs.
        """
        self.logger.debug(f"Running backtest with BacktestEngine on {dataset_type} dataset")
        
        # Determine strategy type based on current configuration
        strategy_type = "ensemble"  # Default for optimization
        
        # Run backtest using the engine
        metric_value, regime_performance = self.backtest_engine.run_backtest(
            parameters=params_to_test,
            dataset_type=dataset_type,
            strategy_type=strategy_type,
            use_regime_adaptive=False  # Grid search uses fixed parameters
        )
        
        return metric_value, regime_performance
        
    def run_adaptive_test(self, results_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the adaptive test using BacktestEngine for consistency.
        
        This ensures the OOS test behaves identically to production runs.
        """
        self.logger.info("Running regime-adaptive test with BacktestEngine")
        
        # Save regime parameters to a temporary file for the adaptive strategy
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Extract and format regime parameters
            regime_params = self._extract_regime_parameters(results_summary)
            json.dump(regime_params, f)
            params_file = f.name
            
        try:
            # Run adaptive test using BacktestEngine
            metric_value, regime_performance = self.backtest_engine.run_backtest(
                parameters={},  # Not used for regime-adaptive
                dataset_type="test",
                strategy_type="regime_adaptive",
                use_regime_adaptive=True,
                adaptive_params_path=params_file
            )
            
            # Process and store results
            if metric_value is not None:
                adaptive_results = {
                    "adaptive_metric": metric_value,
                    "method": "true_adaptive",
                    "regimes_detected": list(regime_performance.keys()) if regime_performance else [],
                    "adaptive_regime_performance": regime_performance
                }
                
                # Add regime coverage info
                adaptive_results["regimes_info"] = self._analyze_regime_coverage(
                    results_summary, 
                    regime_performance
                )
                
                results_summary["regime_adaptive_test_results"] = adaptive_results
                self.logger.info(f"Adaptive test completed. Final metric: {metric_value}")
            else:
                self.logger.error("Adaptive test failed - no metric value returned")
                results_summary["regime_adaptive_test_results"] = {"error": "Test failed"}
                
        finally:
            # Clean up temporary file
            Path(params_file).unlink(missing_ok=True)
            
        # Log and save results using new components
        self._log_optimization_results_v2(results_summary)
        self._save_results_with_versioning(results_summary)
        
        return results_summary
        
    def _extract_regime_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format regime parameters for adaptive strategy."""
        regime_parameters = {}
        
        for regime, regime_data in results.get('best_parameters_per_regime', {}).items():
            if isinstance(regime_data, dict) and 'parameters' in regime_data:
                regime_parameters[regime] = regime_data['parameters'].copy()
            else:
                regime_parameters[regime] = regime_data.copy()
                
            # Merge GA-optimized weights if available
            if 'best_weights_per_regime' in results and regime in results['best_weights_per_regime']:
                ga_weights = results['best_weights_per_regime'][regime].get('weights', {})
                if ga_weights:
                    regime_parameters[regime].update(ga_weights)
                    self.logger.debug(f"Merged GA weights for regime '{regime}'")
                    
        return regime_parameters
        
    def _analyze_regime_coverage(
        self, 
        results: Dict[str, Any], 
        regime_performance: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze which regimes have optimized parameters vs defaults."""
        optimized_regimes = set(results.get('best_parameters_per_regime', {}).keys())
        detected_regimes = set(regime_performance.keys()) if regime_performance else set()
        
        regimes_with_params = list(optimized_regimes & detected_regimes)
        regimes_with_defaults = list(detected_regimes - optimized_regimes)
        
        return {
            "regimes_with_optimized_params": regimes_with_params,
            "would_use_default_for": regimes_with_defaults
        }
        
    def _log_optimization_results_v2(self, results: Dict[str, Any]) -> None:
        """Use ResultsManager for consistent result logging."""
        # Prevent duplicate logging
        if hasattr(self, '_results_already_logged') and self._results_already_logged:
            return
        self._results_already_logged = True
        
        # Generate and print summary
        summary = self.results_manager.generate_summary(results)
        print(summary)
        
    def _save_results_with_versioning(self, results: Dict[str, Any]) -> None:
        """Save results with both ResultsManager and ParameterManager."""
        # Save full results
        self.results_manager.save_results(
            results,
            optimization_type="grid_search",
            version_metadata={
                "dataset_info": {
                    "train_size": len(self.train_df) if hasattr(self, 'train_df') else 0,
                    "test_size": len(self.test_df) if hasattr(self, 'test_df') else 0,
                    "split_ratio": 0.8  # Default value, could get from data handler if needed
                }
            }
        )
        
        # Save versioned parameters
        self._save_versioned_parameters(results)
        
    def _save_versioned_parameters(self, results: Dict[str, Any]) -> None:
        """Save parameters with full versioning support."""
        # Save overall best parameters
        if "best_parameters_on_train" in results:
            self.parameter_manager.create_version(
                parameters=results["best_parameters_on_train"],
                strategy_name="ensemble_strategy",
                optimization_method="grid_search",
                training_period={
                    "start": str(self.train_df.index[0]) if hasattr(self, 'train_df') else "unknown",
                    "end": str(self.train_df.index[-1]) if hasattr(self, 'train_df') else "unknown"
                },
                performance_metrics={
                    "training_metric": results.get("best_training_metric_value", 0),
                    "test_metric": results.get("test_set_metric_value_for_best_params", 0)
                },
                dataset_info={
                    "symbol": self._config_loader.get("data.symbol", "Unknown"),
                    "train_size": len(self.train_df) if hasattr(self, 'train_df') else 0,
                    "test_size": len(self.test_df) if hasattr(self, 'test_df') else 0
                }
            )
            
        # Save regime-specific parameters with versioning
        if "best_parameters_per_regime" in results:
            for regime, regime_data in results["best_parameters_per_regime"].items():
                if isinstance(regime_data, dict) and 'parameters' in regime_data:
                    params = regime_data['parameters']
                    metric_info = regime_data.get('metric', {})
                    metric_value = metric_info.get('value', 0)
                else:
                    params = regime_data
                    metric_value = results.get('best_metric_per_regime', {}).get(regime, 0)
                    
                self.parameter_manager.create_version(
                    parameters=params,
                    strategy_name="ensemble_strategy",
                    optimization_method="grid_search",
                    training_period={
                        "start": str(self.train_df.index[0]) if hasattr(self, 'train_df') else "unknown",
                        "end": str(self.train_df.index[-1]) if hasattr(self, 'train_df') else "unknown"
                    },
                    performance_metrics={
                        self._regime_metric: metric_value
                    },
                    dataset_info={
                        "symbol": self._config_loader.get("data.symbol", "Unknown"),
                        "regime": regime
                    },
                    regime=regime
                )
                
        # Export for production use
        self.parameter_manager.export_for_production(
            "ensemble_strategy",
            "regime_optimized_parameters.json"
        )
        
    # Note: Other methods from EnhancedOptimizer would be inherited or gradually
    # refactored to use the new components. This shows the key integration points.