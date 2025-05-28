# Critical Bug Found: Portfolio State Persistence

## The Issue
The portfolio is NOT being properly reset between isolated component evaluations, causing:
1. **Passive drift returns** - Previous trades' P&L carries over
2. **Identical scores** - All components see the same final portfolio value
3. **No isolation** - Components are evaluated with contaminated state

## Evidence
- RSI evaluation shows Total Value: $100,470.90 with Realized P&L: $470.90
- This $470.90 came from MA Crossover trades that ran BEFORE the RSI evaluation
- RSI generated 0 trades but still shows a 0.47% return
- All subsequent evaluations see this same contaminated portfolio state

## The Root Cause
The portfolio.reset() is being called, but the same portfolio instance is being reused across multiple strategy evaluations. When the isolated strategy runs, it's seeing the accumulated P&L from previous strategies.

## The Fix
The isolated evaluator needs to ensure complete portfolio isolation by either:
1. Creating a fresh portfolio instance for each evaluation
2. Ensuring portfolio.reset() is called AND verified before each run
3. Using a separate portfolio instance for isolated evaluations

## Temporary Workaround
Until fixed, optimization results for rules that don't generate many trades will be unreliable. The scores reflect portfolio drift, not actual strategy performance.