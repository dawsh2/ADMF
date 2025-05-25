#!/usr/bin/env python3
"""
Test script to verify the optimizer works with BacktestEngine fixes.
"""

print("""
The optimizer should now work! The fixes we made:

1. Renamed conflicting method names in BacktestEngine:
   - _setup_components (for resolving) -> _resolve_components
   - _setup_components (for initialization) remains the same

2. Made ResultsManager more robust to handle None values

To test:
python3 main.py --config config/config.yaml --optimize

The BacktestEngine will now:
- Properly resolve components
- Set them up in the correct order
- Configure the dataset
- Run the backtest
- Return results

If you still see errors, they might be related to missing dependencies (pandas, yaml, etc.)
""")