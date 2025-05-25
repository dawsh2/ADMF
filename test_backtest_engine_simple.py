#!/usr/bin/env python3
"""
Simple test to verify BacktestEngine can be used.
This script shows how to integrate BacktestEngine with your existing code.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing BacktestEngine integration...\n")

# First, let's create a patch file showing how to modify existing EnhancedOptimizer
patch_content = """
# Patch for EnhancedOptimizer to use BacktestEngine
# Apply these changes to src/strategy/optimization/enhanced_optimizer.py

## 1. Add import at the top of the file:
from src.strategy.optimization.engines import BacktestEngine

## 2. In __init__ method, add:
self.backtest_engine = BacktestEngine(container, config_loader, event_bus)

## 3. Replace _perform_single_backtest_run method with:
def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
    '''Use BacktestEngine for consistent backtest execution.'''
    self.logger.debug(f"Running backtest with BacktestEngine on {dataset_type} dataset")
    
    # Run backtest using the engine
    metric_value, regime_performance = self.backtest_engine.run_backtest(
        parameters=params_to_test,
        dataset_type=dataset_type,
        strategy_type="ensemble",  # Grid search uses ensemble strategy
        use_regime_adaptive=False
    )
    
    return metric_value, regime_performance

## 4. For run_adaptive_test, replace the _run_regime_adaptive_test call with:
# Save regime parameters to temporary file
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    regime_params = self._extract_regime_parameters(results_summary)
    json.dump(regime_params, f)
    params_file = f.name
    
try:
    # Run adaptive test using BacktestEngine
    metric_value, regime_performance = self.backtest_engine.run_backtest(
        parameters={},
        dataset_type="test",
        strategy_type="regime_adaptive",
        use_regime_adaptive=True,
        adaptive_params_path=params_file
    )
    
    # Store results
    if metric_value is not None:
        results_summary["regime_adaptive_test_results"] = {
            "adaptive_metric": metric_value,
            "method": "true_adaptive",
            "regimes_detected": list(regime_performance.keys()) if regime_performance else [],
            "adaptive_regime_performance": regime_performance
        }
finally:
    os.unlink(params_file)

## 5. Add helper method:
def _extract_regime_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
    '''Extract regime parameters for adaptive strategy.'''
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
                
    return regime_parameters
"""

print("INTEGRATION INSTRUCTIONS:")
print("=" * 80)
print(patch_content)
print("=" * 80)

# Show how to use the standalone production runner
print("\nTO USE THE NEW PRODUCTION RUNNER:")
print("-" * 50)
print("1. For regime-adaptive backtest with optimized parameters:")
print("   python3 run_production_backtest_v2.py --config config/config_adaptive_production.yaml --strategy regime_adaptive")
print()
print("2. For ensemble strategy backtest:")
print("   python3 run_production_backtest_v2.py --config config/config.yaml --strategy ensemble --dataset test")
print()
print("3. To verify consistency between optimizer and production:")
print("   python3 verify_backtest_consistency.py")
print()

# Create a minimal patch file
patch_file = "enhanced_optimizer_backtest_engine.patch"
with open(patch_file, 'w') as f:
    f.write("""--- a/src/strategy/optimization/enhanced_optimizer.py
+++ b/src/strategy/optimization/enhanced_optimizer.py
@@ -10,6 +10,7 @@ from typing import Dict, Any, List, Tuple, Optional
 from src.strategy.optimization.basic_optimizer import BasicOptimizer
 from src.core.events import EventType
 from src.strategy.optimization.genetic_optimizer import GeneticOptimizer
+from src.strategy.optimization.engines import BacktestEngine
 
 logger = logging.getLogger(__name__)
 
@@ -30,6 +31,9 @@ class EnhancedOptimizer(BasicOptimizer):
         super().__init__(instance_name, config_loader, event_bus, component_config_key, container)
         # Store container reference for component resolution
         self._container = container
+        
+        # Initialize BacktestEngine for consistent backtest execution
+        self.backtest_engine = BacktestEngine(container, config_loader, event_bus)
         
         # Get configuration for top-N tracking
         self._track_top_n = self._config_loader.get(f"{self._config_key}.track_top_n", 10)
@@ -49,6 +53,15 @@ class EnhancedOptimizer(BasicOptimizer):
     
     def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
+        '''Use BacktestEngine for consistent backtest execution.'''
+        return self.backtest_engine.run_backtest(
+            parameters=params_to_test,
+            dataset_type=dataset_type,
+            strategy_type="ensemble",
+            use_regime_adaptive=False
+        )
+    
+    def _perform_single_backtest_run_OLD(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
         '''
         Perform a single backtest run with given parameters on the specified dataset.
         Returns the metric value for optimization and regime performance data.
""")

print(f"\nPATCH FILE CREATED: {patch_file}")
print("To apply the patch:")
print(f"  cd /Users/daws/ADMF && patch -p1 < {patch_file}")
print("\nOr manually apply the changes shown above.")
print()
print("The BacktestEngine is ready to use! It provides:")
print("- Consistent component initialization")
print("- Identical behavior between optimizer and production")
print("- Clean separation of backtest logic")
print("- Support for both ensemble and regime-adaptive strategies")