#!/usr/bin/env python3
"""
Example of using EnhancedOptimizerV2 with BacktestEngine.
"""

# To use EnhancedOptimizerV2, modify main.py:
# Change line 26:
# from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
# To:
# from src.strategy.optimization.enhanced_optimizer_v2 import EnhancedOptimizerV2

# And change line 216:
# container.register_type("optimizer_service", EnhancedOptimizer, True, constructor_kwargs=optimizer_args)
# To:
# container.register_type("optimizer_service", EnhancedOptimizerV2, True, constructor_kwargs=optimizer_args)

print("""
To use EnhancedOptimizerV2 (which already includes BacktestEngine):

1. Edit main.py and change the import:
   FROM: from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
   TO:   from src.strategy.optimization.enhanced_optimizer_v2 import EnhancedOptimizerV2

2. Change the registration:
   FROM: container.register_type("optimizer_service", EnhancedOptimizer, True, ...)
   TO:   container.register_type("optimizer_service", EnhancedOptimizerV2, True, ...)

3. Run optimization as normal:
   python3 main.py --config config/config.yaml --optimize

The optimizer will now use BacktestEngine for all backtests!
""")