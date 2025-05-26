# Regime-Specific Strategy Optimization

## Overview

This document describes the Phase 3 implementation of regime-specific parameter optimization for the ADMF-Trader system. This feature allows the system to use different optimal parameters for different market regimes, enhancing performance across varying market conditions.

## Components

The implementation consists of three main components:

1. **Regime Performance Tracking**: The `BasicPortfolio` class tracks performance metrics segmented by market regime.
2. **Regime-Specific Optimization**: The `EnhancedOptimizer` class finds optimal parameters for each detected market regime.
3. **Regime-Adaptive Strategy**: The `RegimeAdaptiveStrategy` class switches parameters automatically based on the detected market regime.

## Implementation Details

### 1. Regime Performance Tracking

The `BasicPortfolio` class has been enhanced to track trading performance segmented by market regime:

- Each trade is tagged with the active regime at the time of execution
- Performance metrics are calculated separately for each regime
- A new `get_performance_by_regime()` method allows other components to access regime-specific metrics

Key metrics tracked per regime:
- Total P&L
- Win/loss count
- Win rate
- Sharpe ratio
- Average P&L per trade

### 2. Regime-Specific Optimization

The `EnhancedOptimizer` class extends `BasicOptimizer` to find optimal parameters for each market regime:

```python
# Example usage in config.yaml
components:
  optimizer:
    class_path: "src.strategy.optimization.enhanced_optimizer.EnhancedOptimizer"
    strategy_service_name: "strategy"
    portfolio_service_name: "portfolio_manager"
    data_handler_service_name: "data_handler"
    min_trades_per_regime: 5  # Minimum number of trades required to consider a regime
    regime_metric: "sharpe_ratio"  # Metric to optimize for each regime
    output_file_path: "regime_optimized_parameters.json"  # Where to save results
```

Key features:
- Grid search operates across the parameter space as in `BasicOptimizer`
- Tracks best parameters separately for each detected market regime
- Requires a minimum number of trades in a regime for reliable optimization
- Saves optimal parameters for each regime to a configuration file

Example output file structure:
```json
{
  "timestamp": "2025-05-17T18:58:04.835",
  "overall_best_parameters": {
    "short_window": 10,
    "long_window": 20
  },
  "overall_best_metric": {
    "name": "get_final_portfolio_value",
    "value": 100019.31,
    "higher_is_better": true
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
  },
  "regimes_encountered": [
    "ranging_low_vol",
    "trending_up_volatile",
    "oversold_in_uptrend"
  ],
  "min_trades_per_regime": 5
}
```

### 3. Regime-Adaptive Strategy

The `RegimeAdaptiveStrategy` class extends `MAStrategy` to apply different parameters based on the current market regime:

```python
# Example usage in config.yaml
components:
  strategy:
    class_path: "src.strategy.regime_adaptive_strategy.RegimeAdaptiveStrategy"
    regime_detector_service_name: "MyPrimaryRegimeDetector"
    regime_params_file_path: "regime_optimized_parameters.json"
    fallback_to_overall_best: true
```

Key features:
- Loads regime-specific parameters from a configuration file
- Subscribes to classification events to detect regime changes
- Automatically applies optimal parameters when the regime changes
- Falls back to overall best parameters if no regime-specific parameters exist

## Workflow

1. **Training Phase**:
   - Run the `EnhancedOptimizer` in training mode
   - Grid search across parameter combinations
   - Evaluate parameters separately for each detected regime
   - Save optimal parameters for each regime to a configuration file

2. **Live Trading/Backtest Phase**:
   - Deploy the `RegimeAdaptiveStrategy` with the saved parameter file
   - Strategy automatically adapts parameters when regime changes

## Usage Example

1. **Run Optimization**:
   ```bash
   python main.py --optimize --bars 5000
   ```
   This will:
   - Run grid search optimization across parameter combinations
   - Track performance by regime using portfolio's `get_performance_by_regime()` method
   - Find best parameters for each regime
   - Save regime-specific parameters to `regime_optimized_parameters.json`

2. **Run Regime-Adaptive Strategy**:
   ```bash
   python main.py
   ```
   This will:
   - Load the regime-specific parameters from the saved file
   - Apply different parameters based on the detected market regime
   - Switch parameters automatically when the regime changes

## Performance Implications

Using regime-specific optimized parameters has several advantages over a single set of global parameters:

1. **Improved Performance**: Different market conditions require different strategy parameters
2. **Reduced Drawdowns**: Parameters optimized for ranging markets can reduce losses during non-trending periods
3. **Higher Sharpe Ratio**: More consistent performance across market conditions

## Handling of Boundary Trades

Boundary trades (trades that open in one regime and close in another) require special attention in regime-specific optimization. The implementation handles these cases as follows:

### 1. Trade Attribution and Tracking

- Each trade is tagged with both entry and exit regimes
- The system tracks whether a trade is a "boundary trade" (entry regime != exit regime)
- Performance metrics track both "pure regime trades" and "boundary trades" separately

Example log output:
```
Boundary trade detected: LONG 50.00 SPY opened in 'trending_up_volatile' and closed in 'ranging_low_vol'. PnL: 125.50, Segment ID: 7b8c9d0e-1f2a-3b4c-5d6e-7f8a9b0c1d2e
```

### 2. Performance Reporting

Performance summaries include detailed information about boundary trades:

```
Regime: trending_up_volatile
  Total Gross Pnl: 2450.75
  Trade Segments: 45
  - Pure Regime Trades: 38 (PnL: 2150.25)
  - Boundary Trades: 7 (PnL: 300.50)
  ...

--- Boundary Trades Details ---
  Total Boundary Trades: 12
  Total Boundary Trades PnL: 525.75
  
  Transition: trending_up_volatile_to_ranging_low_vol
    Count: 7
    PnL: 300.50
    Win Rate: 0.71
    ...
```

### 3. Optimization Considerations

The `EnhancedOptimizer` takes boundary trades into account when selecting optimal parameters:

- Logs the percentage of boundary trades for each regime
- Issues warnings when a regime has a high proportion of boundary trades (>30%)
- Provides detailed statistics on the contribution of boundary trades to overall performance
- Can optionally apply a penalty to metrics for regimes with many boundary trades

### 4. Primary Attribution Policy

For optimization purposes, a trade is primarily attributed to its entry regime, but:

- The optimizer checks what percentage of trades in each regime are boundary trades
- If a high percentage of trades in a regime are boundary trades, the optimizer flags this as potentially less reliable data
- Regimes with mostly boundary trades might be considered less reliable for parameter optimization

## Limitations

1. **Overfitting Risk**: Optimizing separately for each regime increases the risk of overfitting
2. **Regime Transition Lag**: There may be a lag between regime change detection and parameter switching
3. **Minimum Regime Duration**: Requires enough data in each regime to reliably optimize parameters
4. **Boundary Trade Attribution**: Trades spanning multiple regimes create attribution challenges

## Future Enhancements

1. **Walk-Forward Optimization**: Implement regime-specific walk-forward optimization
2. **Ensemble Approaches**: Use multiple parameter sets within each regime with weighting
3. **Auto-Regime Detection**: Use unsupervised learning to detect regimes automatically