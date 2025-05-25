#!/usr/bin/env python3
import sys
import os
import json
sys.path.append('.')

# Redirect stdout to capture all logs
import io
from contextlib import redirect_stdout, redirect_stderr

print("Running optimizer test to capture indicator logs...")

# Capture all output
f = io.StringIO()
with redirect_stdout(f), redirect_stderr(f):
    try:
        from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
        
        config_path = 'config/config.yaml'
        with open('regime_optimized_parameters.json', 'r') as file:
            data = json.load(file)
        
        best_params = data['overall_best_parameters']
        print(f"Using parameters: {best_params}")
        
        optimizer = EnhancedOptimizer(config_path)
        results = optimizer._run_regime_adaptive_test(best_params)
        
        print(f'OPTIMIZER TEST RESULTS:')
        print(f'Final Value: {results["final_portfolio_value"]:.2f}')
        print(f'Total Trades: {results["total_trades"]}')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Get the captured output
output = f.getvalue()

# Write to file
with open('optimizer_debug_output.log', 'w') as file:
    file.write(output)

print("Optimizer test completed. Output saved to optimizer_debug_output.log")

# Extract and display indicator logs
lines = output.split('\n')
indicator_lines = [line for line in lines if '📊 BAR_' in line]

print(f"\nFound {len(indicator_lines)} indicator log lines")
print("\nFirst 30 indicator logs from optimizer:")
print("=" * 80)
for i, line in enumerate(indicator_lines[:30]):
    print(f"{i+1:2d}. {line}")