# FIX_ME.MD - Issues to Address

## 1. Regime Analyzer Trade Count Issue (FIXED)

The regime statistics were showing "Total trades: 0" even though trades were being tracked by regime. This was because the regime analyzer wasn't storing the trade counts from each parameter combination.

**Fix Applied**: Added `trade_counts` list to store trade counts from each parameter combination and updated the statistics calculation to sum these counts.

## 2. Total Bars Checked Misleading

The regime detector summary shows cumulative bars across all optimization iterations:
```
Total bars checked: 40000  # With 4 parameter combinations and 10000 bars each
```

This should be divided by the number of unique parameter combinations to show the actual bars per backtest.

## 3. MyPrimaryRegimeDetector Shows No Checks

At teardown, there's a second regime detector (`MyPrimaryRegimeDetector`) that shows:
```
=== Regime Detection Summary ===
No regime detection checks were performed
=== End of Summary ===
```

This appears to be a duplicate/unused regime detector that should be removed from the configuration.

## 4. Regime-Specific Parameter Selection

With 10,000 bars, we now see some differentiation in parameter selection:
- **Default regime**: Selected fast=10, slow=30 (different from others!)
- **Other regimes**: All selected fast=5, slow=20

This shows the optimization is working, but the limited parameter space (only 4 combinations) may not provide enough diversity for truly regime-specific optimization.

## Next Steps

1. Implement proper trade counting in regime analyzer (or accept current limitation)
2. Fix the bars checked display to show per-backtest count
3. Remove or properly configure MyPrimaryRegimeDetector
4. Consider expanding parameter space for better regime differentiation