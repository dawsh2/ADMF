#!/usr/bin/env python3
"""
BacktestEngine Integration Demo

This shows how to integrate BacktestEngine into your existing code
without running it (to avoid dependency issues).
"""

print("BacktestEngine Integration Demo")
print("=" * 80)
print()

# Show the key integration points
print("1. MODIFY EnhancedOptimizer.__init__ to add:")
print("-" * 50)
print("""
from src.strategy.optimization.engines import BacktestEngine

class EnhancedOptimizer(BasicOptimizer):
    def __init__(self, ...):
        super().__init__(...)
        # Add this line:
        self.backtest_engine = BacktestEngine(container, config_loader, event_bus)
""")

print("\n2. REPLACE _perform_single_backtest_run with:")
print("-" * 50)
print("""
def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
    '''Use BacktestEngine for consistent backtest execution.'''
    return self.backtest_engine.run_backtest(
        parameters=params_to_test,
        dataset_type=dataset_type,
        strategy_type="ensemble",
        use_regime_adaptive=False
    )
""")

print("\n3. FOR ADAPTIVE TEST in run_adaptive_test:")
print("-" * 50)
print("""
# Replace the call to self._run_regime_adaptive_test(results_summary) with:

import tempfile
import json

# Extract regime parameters
regime_params = {}
for regime, regime_data in results_summary.get('best_parameters_per_regime', {}).items():
    if isinstance(regime_data, dict) and 'parameters' in regime_data:
        regime_params[regime] = regime_data['parameters'].copy()
    else:
        regime_params[regime] = regime_data.copy()
        
    # Merge GA weights if available
    if 'best_weights_per_regime' in results_summary and regime in results_summary['best_weights_per_regime']:
        ga_weights = results_summary['best_weights_per_regime'][regime].get('weights', {})
        if ga_weights:
            regime_params[regime].update(ga_weights)

# Save to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
    import os
    os.unlink(params_file)
""")

print("\n4. BENEFITS OF THIS APPROACH:")
print("-" * 50)
print("✓ Single source of truth for backtest execution")
print("✓ Identical component initialization order")
print("✓ Guaranteed cold starts (fresh components)")
print("✓ Same behavior in optimizer and production")
print("✓ Easier to debug and maintain")

print("\n5. WHAT THIS SOLVES:")
print("-" * 50)
print("✓ Part 1.1: Cold starts are guaranteed by BacktestEngine")
print("✓ OOS alignment: Same code path for optimizer and production")
print("✓ Component order: Consistent initialization sequence")
print("✓ State isolation: No state leakage between runs")

print("\n6. KEY FEATURES OF BacktestEngine:")
print("-" * 50)
print("• Handles component resolution and setup")
print("• Configures dataset (train/test/full)")
print("• Initializes components in correct order:")
print("  1. RegimeDetector (first)")
print("  2. ExecutionHandler")
print("  3. RiskManager")
print("  4. Strategy")
print("  5. PortfolioManager")
print("  6. DataHandler (last - publishes events)")
print("• Collects results and performance metrics")
print("• Properly cleans up components after run")

print("\n" + "=" * 80)
print("The BacktestEngine is ready to integrate!")
print("Apply the changes above to start using it.")
print("=" * 80)