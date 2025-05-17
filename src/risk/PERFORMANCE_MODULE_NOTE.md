# Performance Module Note

This directory should include a `performance` sub-module that handles portfolio performance tracking and analytics.

## Rationale

Analytics functionality related to portfolio performance tracking should live in the risk module rather than creating a separate analytics directory elsewhere. This ensures:

1. Close proximity to the portfolio code that generates the performance data
2. Clear separation from the analytics components in `strategy/analytics` which serve a different purpose
3. Logical organization of risk management and performance measurement in one place

## Suggested Structure

```
src/risk/
├── basic_portfolio.py           (Already exists)
├── basic_risk_manager.py        (Already exists)
├── performance/                 (To be created)
│   ├── __init__.py
│   ├── performance_tracker.py   (Portfolio performance metrics)
│   ├── analysis.py              (Statistical analysis of performance)
│   └── reporting.py             (Report generation)
└── ... (other risk files)
```

This structure keeps all risk and performance-related functionality together while avoiding naming conflicts with the strategy analytics framework.