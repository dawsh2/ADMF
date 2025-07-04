# Performance Debug Checklist

## 1. Trade Execution Validation
- [ ] Verify trade log contains actual trades
  - Check `_trade_log` in portfolio has entries
  - Validate trade count matches regime performance totals
  - Audit individual trades for reasonable P&L
- [ ] Trace signal flow end-to-end
  - Signal generated by strategy
  - Order created by risk manager
  - Fill executed by execution handler
  - Portfolio updated with position
  - Trade logged when position closed
- [ ] Check for dropped signals
  - Count signals generated vs orders created
  - Count orders created vs fills executed
  - Identify any breaks in the chain

## 2. Portfolio Value Tracking
- [ ] Validate portfolio value calculations
  - Initial value = $100,000
  - Cash changes match trades
  - Holdings value = sum(position * current_price)
  - Total value = cash + holdings
- [ ] Check for value resets
  - Portfolio value shouldn't reset on regime change
  - Value history should be continuous
  - No sudden jumps except from trades
- [ ] Verify P&L calculations
  - Realized P&L accumulates correctly
  - Unrealized P&L matches open positions
  - Commission deducted properly

## 3. Parameter Management
- [ ] Verify parameter files
  - Check `test_regime_parameters.json` exists
  - Validate parameter values are reasonable
  - Ensure all regimes have parameters
- [ ] Trace parameter loading
  - Parameters loaded at startup
  - Correct file loaded for test dataset
  - All components receive parameters
- [ ] Monitor regime adaptation
  - Log when regime changes occur
  - Verify `_on_regime_change` is called
  - Check parameters update correctly
  - Ensure indicators/rules use new parameters

## 4. State Management
- [ ] Check reset behavior
  - Portfolio state persists across bars
  - Indicators maintain proper history
  - Only reset between separate runs
- [ ] Validate indicator state
  - Buffers have correct length
  - Values update each bar
  - No buffer overflow/underflow
- [ ] Monitor regime transitions
  - State preserved during regime change
  - Only parameters change, not data
  - No position closures on regime change

## 5. Signal Generation
- [ ] Audit signal history
  - Log all signals with timestamp
  - Match signals to trades taken
  - Identify missed opportunities
- [ ] Validate signal logic
  - Indicators calculate correctly
  - Rules evaluate properly
  - Weights applied as expected
- [ ] Check signal timing
  - Signals use current bar data
  - No look-ahead bias
  - Proper indicator warmup

## 6. Performance Metrics
- [ ] Fix Sharpe ratio calculation
  - Negative returns = negative Sharpe
  - Check calculation in portfolio
  - Validate against manual calculation
- [ ] Verify trade counts
  - Summary matches detail
  - Regime counts sum correctly
  - No double counting
- [ ] Validate return calculations
  - (Final - Initial) / Initial
  - Matches sum of trade P&Ls
  - Includes commission costs

## 7. Synthetic Testing
- [ ] Create controlled test data
  - Known price patterns
  - Predictable indicator values
  - Expected signal outcomes
- [ ] Implement simple test rules
  - Buy when price > MA
  - Sell when price < MA
  - Known profit targets
- [ ] Verify expected results
  - Correct number of trades
  - Expected P&L per trade
  - Predictable total return

## 8. Event System Validation
- [ ] Trace event flow
  - BAR → Strategy → SIGNAL
  - SIGNAL → Risk Manager → ORDER
  - ORDER → Execution → FILL
  - FILL → Portfolio → Update
- [ ] Check event timing
  - Events processed in order
  - No dropped events
  - Proper event payload
- [ ] Monitor event bus
  - Subscribers registered
  - Events published correctly
  - No duplicate processing

## 9. Data Integrity
- [ ] Validate data loading
  - Correct date ranges
  - No missing bars
  - Prices reasonable
- [ ] Check data routing
  - Train vs test split correct
  - No data leakage
  - Proper windowing
- [ ] Verify data consistency
  - Same data each run
  - No randomness
  - Deterministic results

## 10. Quick Fixes Needed

### Fix Sharpe Ratio Sign
```python
# In basic_portfolio.py - calculate_portfolio_sharpe_ratio()
if avg_return < 0 and sharpe_ratio > 0:
    sharpe_ratio = -abs(sharpe_ratio)  # Ensure negative
```

### Add Trade Execution Logging
```python
# In strategy - after generating signal
self.logger.info(f"SIGNAL: {signal} at bar {bar_num} price {price}")

# In portfolio - after trade complete
self.logger.info(f"TRADE COMPLETE: {symbol} P&L: {pnl}")
```

### Create Debug Mode
```python
# Add --debug flag to enable verbose logging
# Log every decision point
# Save all intermediate calculations
```

## Debug Scripts to Create

1. **validate_trades.py** - Audit trade log for consistency
2. **trace_signal_flow.py** - Track signal from generation to execution
3. **check_parameters.py** - Verify parameter loading and application
4. **synthetic_backtest.py** - Run with known data/rules
5. **event_flow_monitor.py** - Trace all events in the system

## Current Issues Summary

1. **Zero trades showing in summary** despite trades in regime performance
2. **Positive Sharpe with negative returns** - calculation error
3. **Potential parameter switching issues** during test runs
4. **Possible state contamination** between runs

## Next Steps

1. Fix Sharpe ratio calculation immediately
2. Add comprehensive trade logging
3. Create synthetic test case
4. Run full validation suite
5. Compare results with known good run