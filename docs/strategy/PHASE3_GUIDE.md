# Phase 3: Regime-Specific Optimization Guide

This guide provides step-by-step instructions for implementing and using the Phase 3 regime-specific optimization capabilities in ADMF-Trader.

## Overview

Phase 3 extends ADMF-Trader with the ability to:
1. Track performance metrics separately for each market regime
2. Find optimal strategy parameters for each regime
3. Automatically switch parameters when the market regime changes

## Quick Start

### Step 1: Update Codebase

Ensure you have the following files in your project:
- `/src/risk/basic_portfolio.py` - Enhanced with regime performance tracking
- `/src/strategy/optimization/enhanced_optimizer.py` - New optimizer for regime-specific optimization
- `/src/strategy/regime_adaptive_strategy.py` - New strategy that adapts parameters based on regime

### Step 2: Update Configuration

1. Copy examples from `config/phase3_config_examples.yaml` to your main config file or use as reference
2. Register the new components in `main.py`

### Step 3: Run Optimization

```bash
python main.py --optimize --bars 5000
```

This will:
- Run grid search optimization across parameter combinations
- Find optimal parameters for each detected regime
- Save the results to `regime_optimized_parameters.json`

### Step 4: Run with Regime-Adaptive Strategy

```bash
python main.py
```

This will:
- Load the optimal parameters for each regime
- Apply different parameters as the market regime changes

## Detailed Implementation Steps

### 1. Implementing Regime Performance Tracking

The `BasicPortfolio` class now tracks trade performance by regime:

```python
# In BasicPortfolio.get_performance_by_regime()
regime_performance = {}
for trade in self._trade_log:
    regime = trade.get('regime', 'unknown')
    # ... accumulate performance metrics by regime
return regime_performance
```

### 2. Implementing Regime-Specific Optimization

The `EnhancedOptimizer` extends `BasicOptimizer` to track the best parameters for each regime:

```python
# Registration in main.py
optimizer_args = {
    "instance_name": "EnhancedOptimizer", 
    "config_loader": config_loader,
    "event_bus": event_bus, 
    "component_config_key": "components.enhanced_optimizer",
    "container": container
}
container.register_type(
    "optimizer_service", 
    EnhancedOptimizer, 
    True, 
    constructor_kwargs=optimizer_args
)
```

### 3. Implementing Regime-Adaptive Strategy

The `RegimeAdaptiveStrategy` extends `MAStrategy` to switch parameters based on the current regime:

```python
# Registration in main.py
strategy_args = {
    "instance_name": "RegimeAdaptiveStrategy", 
    "config_loader": config_loader,
    "event_bus": event_bus, 
    "container": container,
    "component_config_key": "components.regime_adaptive_strategy"
}
container.register_type(
    "strategy", 
    RegimeAdaptiveStrategy, 
    True, 
    constructor_kwargs=strategy_args
)
```

## Configuration Reference

### Enhanced Optimizer Configuration

```yaml
enhanced_optimizer:
  strategy_service_name: "strategy"
  portfolio_service_name: "portfolio_manager"
  data_handler_service_name: "data_handler"
  metric_to_optimize: "get_final_portfolio_value"
  higher_metric_is_better: true
  min_trades_per_regime: 5
  regime_metric: "sharpe_ratio"
  output_file_path: "regime_optimized_parameters.json"
```

### Regime-Adaptive Strategy Configuration

```yaml
regime_adaptive_strategy:
  symbol: "SPY"
  short_window_default: 10
  long_window_default: 20
  regime_detector_service_name: "MyPrimaryRegimeDetector"
  regime_params_file_path: "regime_optimized_parameters.json"
  fallback_to_overall_best: true
```

## Output File Format

The optimization results are saved in JSON format:

```json
{
  "timestamp": "2025-05-17T18:58:04.835",
  "overall_best_parameters": {
    "short_window": 10,
    "long_window": 20
  },
  "regime_best_parameters": {
    "ranging_low_vol": {
      "parameters": {
        "short_window": 5,
        "long_window": 15
      },
      "metric": {
        "name": "sharpe_ratio",
        "value": 1.87,
        "higher_is_better": true
      }
    },
    "trending_up_volatile": {
      "parameters": {
        "short_window": 15,
        "long_window": 30
      },
      "metric": {
        "name": "sharpe_ratio",
        "value": 2.43,
        "higher_is_better": true
      }
    }
  }
}
```

## Troubleshooting

### Handling Boundary Trades

Trades that open in one regime and close in another (boundary trades) are handled specially:

1. **In BasicPortfolio**:
   ```python
   # Each trade is tagged with both entry and exit regimes
   trade_entry = {
       'entry_regime': entry_regime,
       'exit_regime': exit_regime,
       'is_boundary_trade': (entry_regime != exit_regime),
       # Other trade data...
   }
   ```

2. **Performance Metrics**:
   - Boundary trades are primarily attributed to their entry regime
   - Performance summaries show both "pure regime trades" and "boundary trades"
   - A special "_boundary_trades_summary" section shows statistics for each type of regime transition

3. **Optimization Considerations**:
   - The optimizer logs the percentage of boundary trades for each regime
   - Regimes with >30% boundary trades get flagged with a warning
   - You can configure a penalty factor for regimes with many boundary trades

### Common Issues

1. **No regime-specific parameters found**:
   - Check that your regime detector is correctly identifying regimes
   - Ensure there are enough trades in each regime (min_trades_per_regime)
   - Verify the optimizer is accessing the portfolio's get_performance_by_regime method

2. **Parameters not switching on regime change**:
   - Verify that the regime detector is publishing CLASSIFICATION events
   - Check that RegimeAdaptiveStrategy is subscribed to these events
   - Ensure the parameter file is correctly loaded and formatted

3. **Optimizer fails**:
   - Check for errors in the parameter space definition
   - Ensure all required services are registered and accessible
   - Verify that BasicPortfolio correctly tracks regime performance

### Logging

Enable DEBUG level logging to see detailed information about regime changes and parameter switching:

```yaml
logging:
  level: "DEBUG"  # Change from INFO to DEBUG
```

## Performance Tips

1. **Minimum Trades Threshold**: Set `min_trades_per_regime` to at least 5-10 trades for reliable optimization
2. **Metric Selection**: Consider using `sharpe_ratio` instead of raw P&L for more stable optimization
3. **Regime Definitions**: Define regimes that represent truly different market behaviors
4. **Parameter Space**: Start with a wide parameter space, then narrow once you find promising regions

## Additional Resources

- See `docs/strategy/REGIME_OPTIMIZATION.md` for a detailed explanation of the implementation
- Check `config/phase3_config_examples.yaml` for configuration examples