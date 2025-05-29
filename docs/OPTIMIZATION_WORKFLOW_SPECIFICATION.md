# ADMF-Trader Optimization Workflow Specification

## Overview

The `--optimize` flag triggers a sophisticated multi-stage optimization process that determines optimal parameters and weights for a regime-adaptive ensemble trading strategy. This document formalizes the optimization workflow implemented in the ADMF-Trader system.

## Optimization Process Stages

### 1. Isolated Rule Optimization (Grid Search)

**Purpose**: Determine optimal parameters for each trading rule independently, without influence from other rules.

**Process**:
- Each rule (e.g., MA Crossover, RSI) is evaluated in isolation as a single-rule strategy
- Grid search is performed across the parameter space defined in the configuration
- Performance is evaluated on the TRAINING dataset only
- Metrics are tracked separately for each detected market regime

**Example**:
- MA Crossover Rule tested with parameters:
  - fast_ma: [5, 10, 15]
  - slow_ma: [20, 30, 40]
  - min_separation: [0.0, 0.001, 0.002]
- RSI Rule tested with parameters:
  - lookback_period: [9, 14, 21, 30]
  - oversold_threshold: [20, 30]
  - overbought_threshold: [60, 70]

**Output**: Best parameter combination for each rule in each regime

### 2. Regime-Specific Parameter Selection

**Purpose**: Identify the best-performing parameters for each rule within each market regime.

**Process**:
- Results from isolated optimization are analyzed per regime
- For each regime (e.g., trending_up, trending_down, volatile, default):
  - The parameter combination with the best performance metric (e.g., Sharpe ratio) is selected
  - Only trades that occurred during that specific regime are considered

**Example Output**:
```json
{
  "trending_up": {
    "ma_crossover": {"fast": 5, "slow": 20, "min_separation": 0.0},
    "rsi": {"lookback_period": 30, "oversold": 20, "overbought": 70}
  },
  "trending_down": {
    "ma_crossover": {"fast": 10, "slow": 30, "min_separation": 0.001},
    "rsi": {"lookback_period": 14, "oversold": 30, "overbought": 60}
  }
}
```

### 3. Ensemble Weight Optimization

**Purpose**: Determine optimal weights for combining signals from multiple rules.

**Important Note**: Weights are optimized globally across all regimes, not per-regime. This design choice significantly reduces the number of required backtests while still allowing regime-adaptive behavior through parameter switching.

**Process**:
- Form an ensemble strategy using the regime-specific parameters from Stage 2
- Run a single backtest on the TRAINING dataset with dynamic parameter switching:
  - When regime changes are detected, the strategy switches to the pre-optimized parameters for that regime
  - The weights remain constant throughout
- Test different weight combinations (e.g., [0.3, 0.7], [0.5, 0.5], [0.7, 0.3])
- Select the weight combination that produces the best overall performance

**Example**:
- Test weight combinations for [MA_weight, RSI_weight]:
  - [0.3, 0.7]: Sharpe ratio = 1.2
  - [0.5, 0.5]: Sharpe ratio = 0.9
  - [0.7, 0.3]: Sharpe ratio = 1.5 (selected)

### 4. Final Test Set Evaluation

**Purpose**: Evaluate the fully optimized strategy on unseen data.

**Process**:
- Apply the complete optimization results:
  - Regime-specific parameters from Stage 2
  - Global ensemble weights from Stage 3
- Run the strategy on the TEST dataset
- The strategy dynamically switches parameters based on detected regimes
- Performance metrics are calculated and reported

**Behavior**:
- When a regime change is detected, the strategy:
  - Loads the pre-optimized parameters for the new regime
  - Applies the static ensemble weights
  - Continues trading with the new configuration

## Configuration Structure

The optimization process is driven by the configuration file:

```yaml
optimization:
  workflow:
    type: "sequential"
    steps:
      - name: "optimize_ma_isolated"
        component: "strategy_ma_crossover"
        method: "grid_search"
        evaluator: "isolated"
        
      - name: "optimize_rsi_isolated"
        component: "strategy_rsi"
        method: "grid_search"
        evaluator: "isolated"
        
      - name: "optimize_weights"
        method: "ensemble_weights"
        weight_combinations:
          - [0.3, 0.7]
          - [0.5, 0.5]
          - [0.7, 0.3]
```

## Output Files

The optimization process generates several output files:

1. **Individual Component Results**: `optimize_ma_isolated_*.json`, `optimize_rsi_isolated_*.json`
   - Contains performance metrics for each parameter combination
   - Includes regime-specific breakdowns

2. **Regime Analysis**: `regime_analysis_*.json`
   - Detailed analysis of performance within each regime

3. **Weight Optimization Results**: `optimize_weights_*.json`
   - Performance metrics for each weight combination tested

4. **Final Parameters**: `regime_optimized_parameters_*.json`
   - Complete set of optimized parameters organized by regime
   - Includes both rule parameters and ensemble weights

5. **Workflow Summary**: `workflow_*.json`
   - Comprehensive record of the entire optimization process
   - Includes all intermediate results and final selections

## Key Design Decisions

1. **Isolated Evaluation**: Rules are optimized independently to find their best parameters without interference from other rules.

2. **Regime-Aware Optimization**: Parameters are optimized separately for each market regime, allowing the strategy to adapt to different market conditions.

3. **Global Weight Optimization**: Ensemble weights are kept constant across regimes to reduce computational complexity while maintaining adaptive behavior through parameter switching.

4. **Train/Test Split**: The TEST dataset is strictly reserved for final evaluation and is never touched during the optimization process.

## Benefits

- **Computational Efficiency**: By using global weights instead of per-regime weights, the number of required backtests is significantly reduced.
- **Regime Adaptability**: The strategy can still adapt to market conditions by switching parameters.
- **Robust Evaluation**: Isolated optimization ensures each component is properly tuned.
- **Prevent Overfitting**: Clear separation between training and test data.

## Usage

To run the optimization workflow:

```bash
python main_ultimate.py --config config/test_ensemble_optimization.yaml --optimize --bars 1000
```

This command will execute the complete optimization workflow as specified in the configuration file.