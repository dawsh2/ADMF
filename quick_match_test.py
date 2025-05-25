#!/usr/bin/env python3
"""
Quick test to verify OOS and production matching.
"""

import json
import sys
from pathlib import Path

print("""
To verify OOS test matches production:

1. First, run optimization to generate parameters:
   python main.py --config config/config.yaml --optimize

2. Note the adaptive test result from the optimization output:
   Look for "Adaptive GA Ensemble Strategy Test final_portfolio_value: XXXXX"

3. Run standalone production with same parameters:
   python run_production_backtest_v2.py --config config/config_adaptive_production.yaml \\
       --strategy regime_adaptive --dataset test

4. Compare the final portfolio values - they should match!

Key factors for matching:
- Both use BacktestEngine
- Both use 'test' dataset
- Both use regime_optimized_parameters.json
- Both use same component initialization order

If they don't match, check:
- Are regime detector indicators resetting? (Part 1.2)
- Are fallback parameters consistent? (Part 1.3)
- Are components in same state at start?
""")