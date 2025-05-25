#!/usr/bin/env python3
import sys
sys.path.append('.')
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
import json

config_path = 'config/config.yaml'
# Load the best parameters from regime_optimized_parameters.json
with open('regime_optimized_parameters.json', 'r') as f:
    data = json.load(f)

# Use the joint optimization results
best_params = data['best_parameters_on_train']

print("Running optimizer adaptive test with detailed logging...")
optimizer = EnhancedOptimizer(config_path)
results = optimizer._run_regime_adaptive_test(best_params)
print(f'OPTIMIZER TEST: Final Value: {results["final_portfolio_value"]:.2f}, Trades: {results["total_trades"]}')