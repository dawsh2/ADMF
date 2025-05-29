# Execution Path Findings for --dataset test

## Critical Issue: Different Results with --dataset test

When running with `--dataset test`, you're getting different results than the optimization test phase. This document explains why and what needs to be fixed.

## Current Execution Path

### 1. Entry Point Flow
```
main_ultimate.py 
→ ApplicationLauncher (parses CLI args)
→ Bootstrap (initializes system)
→ BacktestRunner.execute() (NOT AppRunner!)
```

### 2. Key Discovery: BacktestRunner vs AppRunner
- In backtest mode, Bootstrap uses `BacktestRunner` as the entrypoint component
- The `AppRunner` component is NOT used for backtests
- Any code added to `AppRunner` won't affect backtest runs

### 3. Parameter Loading Issue

**The Problem:**
- When using `--dataset test`, the system correctly loads the test dataset
- BUT it does NOT automatically load the optimized parameters from the optimization run
- The strategy uses whatever parameters are in the config file or `test_regime_parameters.json`

**What's Actually Happening:**
1. Strategy loads from `test_regime_parameters.json` during initialization (if it exists)
2. These may NOT be the same parameters used during optimization
3. The `regime_optimized_parameters.json` file is NOT loaded

## Why You're Getting Different Results

### During Optimization:
1. Optimization runs with specific parameters for each regime
2. Test phase uses those optimized parameters
3. Results are saved to `optimization_results/regime_optimized_parameters.json`

### During --dataset test Run:
1. Strategy loads from `test_regime_parameters.json` (NOT the optimization results)
2. Uses potentially different parameters
3. Gets different trading results

## Fix Required

To reproduce optimization test results, we need to:

1. **Modify BacktestRunner** to load optimized parameters when `--dataset test` is specified
2. **Load from the correct file**: `optimization_results/regime_optimized_parameters.json`
3. **Apply parameters dynamically** before the backtest starts

## Current Workaround

The strategy is loading parameters from `test_regime_parameters.json`. To get matching results:
1. Copy the optimized parameters to this file
2. Or update the strategy config to point to the optimization results

## Code That Needs Implementation

```python
# In BacktestRunner.execute()
if self.dataset_override == 'test':
    # Load from optimization_results/regime_optimized_parameters.json
    # Apply to strategy before running
```

## Summary

The execution path is:
- ✅ Correctly using test dataset
- ❌ NOT loading optimized parameters from the optimization run
- ❌ Using different parameters than the optimization test phase
- ✅ The display code is now in place (in BacktestRunner)

This explains why you're seeing different results when running with `--dataset test`.