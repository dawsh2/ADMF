# Integration Guide for New Optimization Components

## Overview
This guide shows how to integrate the new modular components (BacktestEngine, ResultsManager, ParameterManager) with the existing EnhancedOptimizer.

## Quick Integration Steps

### 1. Update EnhancedOptimizer to use BacktestEngine

Replace the `_perform_single_backtest_run` method in EnhancedOptimizer with:

```python
from src.strategy.optimization.engines import BacktestEngine

class EnhancedOptimizer(BasicOptimizer):
    def __init__(self, ...):
        # ... existing init code ...
        self.backtest_engine = BacktestEngine(
            self.container, 
            self.config_loader, 
            self.event_bus
        )
        
    def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """Use BacktestEngine for consistent backtest execution."""
        return self.backtest_engine.run_backtest(
            parameters=params_to_test,
            dataset_type=dataset_type,
            strategy_type="ensemble"  # or "regime_adaptive" based on config
        )
```

### 2. Update Results Handling

Replace results saving and logging methods with ResultsManager:

```python
from src.strategy.optimization.results import ResultsManager

class EnhancedOptimizer(BasicOptimizer):
    def __init__(self, ...):
        # ... existing init code ...
        self.results_manager = ResultsManager()
        
    def _save_results_to_file(self, results: Dict[str, Any]) -> None:
        """Use ResultsManager for saving results."""
        version_metadata = {
            "dataset_info": {
                "train_size": len(self.train_df) if hasattr(self, 'train_df') else 0,
                "test_size": len(self.test_df) if hasattr(self, 'test_df') else 0,
                "split_ratio": self.train_test_split_ratio
            }
        }
        
        self.results_manager.save_results(
            results,
            optimization_type="grid_search",
            version_metadata=version_metadata
        )
        
    def _log_optimization_results(self, results: Dict[str, Any]) -> None:
        """Use ResultsManager for generating summaries."""
        summary = self.results_manager.generate_summary(results)
        print(summary)
```

### 3. Add Parameter Versioning

Integrate ParameterManager for versioned parameter storage:

```python
from src.strategy.optimization.core import ParameterManager

class EnhancedOptimizer(BasicOptimizer):
    def __init__(self, ...):
        # ... existing init code ...
        self.parameter_manager = ParameterManager()
        
    def _save_versioned_parameters(self, results: Dict[str, Any]) -> None:
        """Save parameters with versioning."""
        # Save overall best parameters
        if "best_parameters_on_train" in results:
            self.parameter_manager.create_version(
                parameters=results["best_parameters_on_train"],
                strategy_name="ensemble_strategy",
                optimization_method="grid_search",
                training_period={
                    "start": str(self.train_df.index[0]),
                    "end": str(self.train_df.index[-1])
                },
                performance_metrics={
                    "training_metric": results.get("best_training_metric_value", 0),
                    "test_metric": results.get("test_set_metric_value_for_best_params", 0)
                },
                dataset_info={
                    "symbol": self.config_loader.get("data.symbol", "Unknown"),
                    "train_size": len(self.train_df),
                    "test_size": len(self.test_df)
                }
            )
            
        # Save regime-specific parameters
        if "best_parameters_per_regime" in results:
            for regime, regime_data in results["best_parameters_per_regime"].items():
                params = regime_data.get("parameters", regime_data)
                metrics = regime_data.get("metric", {})
                
                self.parameter_manager.create_version(
                    parameters=params,
                    strategy_name="ensemble_strategy",
                    optimization_method="grid_search",
                    training_period={
                        "start": str(self.train_df.index[0]),
                        "end": str(self.train_df.index[-1])
                    },
                    performance_metrics={
                        metrics.get("name", "metric"): metrics.get("value", 0)
                    },
                    dataset_info={
                        "symbol": self.config_loader.get("data.symbol", "Unknown"),
                        "regime": regime
                    },
                    regime=regime
                )
```

## Using Components for OOS Test Alignment

To ensure consistent behavior between optimizer and production:

### 1. Use BacktestEngine in Adaptive Test

```python
def _run_regime_adaptive_test(self, results: Dict[str, Any]) -> None:
    """Run adaptive test using BacktestEngine for consistency."""
    # Save parameters to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results["best_parameters_per_regime"], f)
        params_file = f.name
        
    # Run backtest with regime-adaptive strategy
    metric_value, regime_performance = self.backtest_engine.run_backtest(
        parameters={},  # Not used for regime-adaptive
        dataset_type="test",
        strategy_type="regime_adaptive",
        use_regime_adaptive=True,
        adaptive_params_path=params_file
    )
    
    # Store results
    results["regime_adaptive_test_results"] = {
        "adaptive_metric": metric_value,
        "regime_performance": regime_performance,
        "method": "true_adaptive"
    }
```

### 2. Ensure Production Uses Same BacktestEngine

In your production run script:

```python
from src.strategy.optimization.engines import BacktestEngine

# Create backtest engine
backtest_engine = BacktestEngine(container, config_loader, event_bus)

# Run production backtest
metric_value, regime_performance = backtest_engine.run_backtest(
    parameters={},  # Use defaults from config
    dataset_type="full",  # or "test" for validation
    strategy_type="regime_adaptive",
    use_regime_adaptive=True,
    adaptive_params_path="regime_optimized_parameters.json"
)
```

## Benefits of This Approach

1. **Consistency**: Same backtest logic used everywhere
2. **Modularity**: Each component has a single responsibility
3. **Testability**: Components can be tested in isolation
4. **Versioning**: Parameters are tracked with full metadata
5. **Extensibility**: Easy to add new optimization methods

## Next Steps

1. Gradually migrate methods from EnhancedOptimizer to use these components
2. Add unit tests for each component
3. Create integration tests to ensure behavior consistency
4. Eventually refactor EnhancedOptimizer to be just an orchestrator