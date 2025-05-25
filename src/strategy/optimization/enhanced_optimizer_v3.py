"""
EnhancedOptimizer V3 - Uses CleanBacktestEngine for complete state isolation.

This version ensures every backtest (optimization and adaptive test) runs
with completely fresh components, preventing any state leakage.
"""

import json
import tempfile
import os
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.strategy.optimization.engines.clean_backtest_engine import CleanBacktestEngine
from src.strategy.optimization.results import ResultsManager
from src.strategy.optimization.core import ParameterManager


class EnhancedOptimizerV3(EnhancedOptimizer):
    """
    Enhanced optimizer with complete state isolation between runs.
    
    Key improvements:
    - Uses CleanBacktestEngine for every backtest
    - No state leakage between optimization runs
    - Consistent results between optimization and production
    """
    
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container):
        """Initialize V3 with clean backtest engine."""
        # Call parent's __init__ properly with all required arguments
        super().__init__(instance_name, config_loader, event_bus, component_config_key, container)
        
        # Override the backtest engine with clean version
        # (parent class sets up everything else we need)
        
        # Initialize clean backtest engine (only needs config)
        self.clean_engine = CleanBacktestEngine(config_loader)
        
        # Initialize modular components
        self.results_manager = ResultsManager()
        self.parameter_manager = ParameterManager()
        
        self.logger.info("Initialized EnhancedOptimizerV3 with CleanBacktestEngine")
        
    def _perform_single_backtest_run(
        self, 
        params_to_test: Dict[str, Any], 
        dataset_type: str
    ) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Execute a single backtest with complete state isolation.
        
        Every run gets fresh components, ensuring no state leakage.
        """
        self.logger.debug(f"Running clean backtest on {dataset_type} dataset with params: {params_to_test}")
        
        # Determine strategy type
        strategy_type = self.get_specific_config("strategy_type", "ensemble")
        
        # Run backtest with clean state
        metric_value, regime_performance = self.clean_engine.run_backtest(
            parameters=params_to_test,
            dataset_type=dataset_type,
            strategy_type=strategy_type,
            use_regime_adaptive=False  # Grid search uses fixed parameters
        )
        
        if metric_value is not None:
            self.logger.debug(f"Backtest completed. Metric: {metric_value}")
        else:
            self.logger.warning("Backtest failed - no metric returned")
            
        return metric_value, regime_performance
        
    def run_adaptive_test(self, results_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run adaptive test with complete state isolation.
        
        This ensures the adaptive test produces identical results to production.
        """
        self.logger.info("Running regime-adaptive test with CleanBacktestEngine")
        
        # Save regime parameters to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Extract and format regime parameters
            regime_params = self._extract_regime_parameters(results_summary)
            
            # Merge with GA-optimized weights if available
            if 'best_weights_per_regime' in results_summary:
                regime_params = self._merge_ga_weights(regime_params, results_summary['best_weights_per_regime'])
                
            json.dump(regime_params, f)
            params_file = f.name
            
        try:
            # Run adaptive test with clean state
            metric_value, regime_performance = self.clean_engine.run_backtest(
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
                    "method": "clean_adaptive",
                    "regimes_detected": list(regime_performance.keys()) if regime_performance else [],
                    "adaptive_regime_performance": regime_performance
                }
                
                # Add regime coverage info
                adaptive_results["regimes_info"] = self._analyze_regime_coverage(
                    results_summary, 
                    regime_performance
                )
                
                results_summary["regime_adaptive_test_results"] = adaptive_results
                self.logger.info(f"Clean adaptive test completed. Final metric: {metric_value}")
            else:
                self.logger.error("Adaptive test failed - no metric value returned")
                results_summary["regime_adaptive_test_results"] = {"error": "Test failed"}
                
        finally:
            # Clean up temporary file
            Path(params_file).unlink(missing_ok=True)
            
        # Log and save results
        self._log_optimization_results(results_summary)
        if self._output_file_path:
            self._save_results_to_file(results_summary)
        
        return results_summary
        
    def _extract_regime_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format regime parameters for adaptive strategy."""
        regime_parameters = {}
        
        for regime, regime_data in results.get('best_parameters_per_regime', {}).items():
            if isinstance(regime_data, dict) and 'parameters' in regime_data:
                regime_parameters[regime] = regime_data['parameters'].copy()
            else:
                regime_parameters[regime] = regime_data.copy()
                
        return regime_parameters
        
    def _merge_ga_weights(self, regime_params: Dict[str, Any], ga_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Merge GA-optimized weights into regime parameters."""
        merged = regime_params.copy()
        
        for regime, weight_data in ga_weights.items():
            if regime in merged and isinstance(weight_data, dict) and 'weights' in weight_data:
                # Update the regime parameters with GA weights
                for weight_key, weight_value in weight_data['weights'].items():
                    merged[regime][weight_key] = weight_value
                    
        return merged
        
    def _analyze_regime_coverage(
        self, 
        results_summary: Dict[str, Any], 
        regime_performance: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze regime coverage in adaptive test."""
        if not regime_performance:
            return {}
            
        optimized_regimes = set(results_summary.get('best_parameters_per_regime', {}).keys())
        detected_regimes = set(k for k in regime_performance.keys() if not k.startswith('_'))
        
        return {
            "optimized_regimes": list(optimized_regimes),
            "detected_regimes": list(detected_regimes),
            "coverage": len(detected_regimes.intersection(optimized_regimes)) / len(optimized_regimes) if optimized_regimes else 0
        }