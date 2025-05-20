# Portfolio Reset Fix

## Issue Description

The optimization process in ADMF was encountering an issue where the portfolio state was not being reset between optimization runs. This resulted in two major problems:

1. **Persistent Portfolio State**: The final portfolio state from one optimization run was carrying over to the next run. This meant that each new parameter combination was not being evaluated under identical initial conditions, making comparisons between parameter sets invalid.

2. **No Trades in Test Phase**: The test phase, which should evaluate the best parameters found during training on unseen data, was starting with an already-modified portfolio state. This was causing no new trades to be executed in the test phase, as the portfolio already had a significant realized P&L from previous runs.

## Root Causes

1. The `BasicPortfolio` class did not have a dedicated reset method to restore its initial state.

2. The `_perform_single_backtest_run` method in both `BasicOptimizer` and `EnhancedOptimizer` classes did not reset the portfolio state between runs.

## Solution Implemented

### 1. Added a `reset()` method to `BasicPortfolio`

The new method performs the following actions:
- Closes any open positions
- Resets cash to the initial value
- Clears all trades history
- Resets all performance metrics (realized/unrealized P&L)
- Clears portfolio value history
- Resets market regime to default

```python
def reset(self):
    """Reset the portfolio to its initial state for a fresh backtest run."""
    self.logger.info(f"Resetting portfolio '{self.name}' to initial state")
    
    # Close any open positions
    if self.open_positions:
        now = datetime.datetime.now(datetime.timezone.utc)
        self.close_all_positions(now)
        
    # Reset cash and positions
    self.current_cash = self.initial_cash
    self.open_positions = {}
    self._trade_log = []
    
    # Reset performance metrics
    self.realized_pnl = 0.0
    self.unrealized_pnl = 0.0
    self.current_holdings_value = 0.0
    self.current_total_value = self.initial_cash
    
    # Reset market data
    self._last_bar_prices = {}
    
    # Reset history
    self._portfolio_value_history = []
    
    # Reset market regime to default
    self._current_market_regime = "default"
    
    self.logger.info(f"Portfolio '{self.name}' reset successfully. Cash: {self.current_cash:.2f}, Total Value: {self.current_total_value:.2f}")
```

### 2. Updated `_perform_single_backtest_run` in both optimizer classes

Added code to both `BasicOptimizer` and `EnhancedOptimizer` classes to call this reset method before each backtest run:

```python
# Reset portfolio state to ensure a clean start
try:
    self.logger.info(f"Optimizer: Resetting portfolio state before {dataset_type} run with params: {params_to_test}")
    if hasattr(portfolio_manager, 'reset') and callable(portfolio_manager.reset):
        portfolio_manager.reset()
    else:
        self.logger.warning("Portfolio does not have a reset method. State may persist between runs.")
except Exception as e:
    self.logger.error(f"Error resetting portfolio before backtest run: {e}", exc_info=True)
```

## Expected Impact

1. **Consistent Evaluation**: Each parameter combination will now be evaluated under identical initial conditions, ensuring fair comparison between different parameter sets.

2. **Accurate Testing**: The test phase will start with a fresh portfolio state, allowing proper evaluation of the best parameters on the test dataset.

3. **Proper Market Regime Attribution**: Performance attribution by market regime will be more accurate, as each run will properly track trades from the beginning.

## Testing Recommendations

To verify the fix:
1. Run the optimization process and check that each run starts with the same initial portfolio value (should be 100,000.00)
2. Verify that the test phase executes trades properly
3. Confirm that market regime-specific performance metrics are being calculated correctly across multiple runs

This fix ensures the integrity of the optimization process and will lead to more reliable and accurate optimization results.