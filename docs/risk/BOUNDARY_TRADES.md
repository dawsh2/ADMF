# Boundary Trades Documentation

## Overview

This document explains how boundary trades are handled during the trading phase and how their performance is attributed across different market regimes.

## Boundary Trade Attribution Logic

### Definition of Boundary Trades

A boundary trade is one where the **entry regime** (when position was opened) differs from the **exit regime** (when position was closed). This is determined by the condition: `entry_regime != exit_regime`

### Performance Attribution Rules

#### 1. Primary Attribution: Entry Regime

All boundary trades are **attributed to the entry regime** for performance calculation. This is the regime that was active when the position was first opened. The trade's P&L, commission, and statistics are added to the entry regime's metrics.

#### 2. Dual Tracking System

```python
# Primary attribution to entry regime
regime_data = regime_performance[entry_regime]
regime_data['pnl'] += trade_pnl
regime_data['count'] += 1

# Separate tracking for boundary analysis
if is_boundary_trade:
    regime_data['boundary_trade_count'] += 1
    regime_data['boundary_trades_pnl'] += trade_pnl
```

#### 3. Boundary Trade Analysis

- Boundary trades are also tracked separately in a `_boundary_trades_summary` section
- These are categorized by regime transition patterns like `"regime_A_to_regime_B"`
- This allows analysis of which regime transitions are most/least profitable

## Implementation Details

### Regime Tracking During Trade Lifecycle

```python
# When opening position
pos['current_segment_regime'] = active_regime
pos['initial_entry_regime_for_trade'] = active_regime

# When closing position  
entry_regime = pos['current_segment_regime']
exit_regime = self._get_current_regime()
is_boundary_trade = (entry_regime != exit_regime)
```

### Performance Impact

- **Conservative Approach**: Performance is attributed to the regime that made the trading decision (entry regime)
- **Boundary Trade Penalties**: The optimizer considers boundary trade ratios when evaluating regime performance
- **Dual Metrics**: Each regime shows both "pure regime trades" and "boundary trades" separately

## Example Scenario

1. Position opened in "trending" regime at $100
2. Regime changes to "volatile" during the hold period  
3. Position closed in "volatile" regime at $105
4. **Result**: +$5 P&L is attributed to "trending" regime (entry), but marked as a boundary trade
5. **Analysis**: "trending" regime gets credit for the decision, but boundary trade ratio affects regime quality assessment

## Benefits of This Approach

This approach ensures that regime-specific optimization doesn't get distorted by regime changes that happen during position holding periods, while still tracking the impact of boundary trades for analysis.

The conservative attribution method prevents regime performance metrics from being artificially inflated or deflated by market regime changes that occur after trading decisions are made.

## Code Location

The boundary trade logic is implemented in:
- `/src/risk/basic_portfolio.py` - Main boundary trade detection and attribution logic
- `/src/strategy/optimization/enhanced_optimizer.py` - Boundary trade analysis during optimization