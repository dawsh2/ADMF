#!/usr/bin/env python3
"""
Find the divergence between optimizer and production runs.
"""

import sys
sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.strategy.optimization.enhanced_optimizer_v3 import EnhancedOptimizerV3
from src.core.event_bus import EventBus
from src.core.container import Container
import json

def run_optimizer_test():
    """Run optimizer and return adaptive test result."""
    print("Running EnhancedOptimizerV3 adaptive test...")
    
    # Setup
    config_loader = SimpleConfigLoader("config/config_optimization_exact.yaml")
    event_bus = EventBus()
    container = Container()
    
    # Create optimizer
    optimizer = EnhancedOptimizerV3(
        instance_name="TestOptimizer",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="optimizer",
        container=container
    )
    
    # Just run the adaptive test with known parameters
    params = {
        'ma_weight': 0.6,
        'rsi_weight': 0.1,
        'bb_weight': 0.2,
        'volume_weight': 0.1,
        'ma_short_period': 10,
        'ma_long_period': 20,
        'rsi_period': 21,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
    
    # Run adaptive test
    result = optimizer.test_adaptive_parameters(params)
    
    if result:
        print(f"\nOptimizer Adaptive Test Result: ${result.get('final_portfolio_value', 0):,.2f}")
        print(f"Number of trades: {result.get('total_trades', 0)}")
        return result
    
    return None

def run_production_test():
    """Run production with same parameters."""
    print("\nRunning production test...")
    
    # We'll use subprocess to run the production script
    import subprocess
    
    cmd = [
        sys.executable, 
        "run_production_adaptive.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract final value from output
    import re
    match = re.search(r'Final Portfolio Value: \$?([0-9,]+\.?\d*)', result.stdout)
    if match:
        value = float(match.group(1).replace(',', ''))
        print(f"\nProduction Result: ${value:,.2f}")
        
        # Count trades
        trades = len(re.findall(r'TRADE:', result.stdout))
        print(f"Number of trades: {trades}")
        
        return {'final_portfolio_value': value, 'total_trades': trades}
    
    return None

def main():
    print("="*80)
    print("FINDING DIVERGENCE BETWEEN OPTIMIZER AND PRODUCTION")
    print("="*80)
    
    # Run both
    opt_result = run_optimizer_test()
    prod_result = run_production_test()
    
    if opt_result and prod_result:
        opt_value = opt_result.get('final_portfolio_value', 0)
        prod_value = prod_result.get('final_portfolio_value', 0)
        
        diff = abs(opt_value - prod_value)
        pct_diff = (diff / opt_value) * 100 if opt_value > 0 else 0
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS:")
        print("="*80)
        print(f"Optimizer:  ${opt_value:,.2f} ({opt_result.get('total_trades', 0)} trades)")
        print(f"Production: ${prod_value:,.2f} ({prod_result.get('total_trades', 0)} trades)")
        print(f"Difference: ${diff:,.2f} ({pct_diff:.4f}%)")
        
        if pct_diff > 0.01:
            print(f"\n❌ Discrepancy of {pct_diff:.4f}% detected!")
            print("\nTo investigate:")
            print("1. Check regime detection differences")
            print("2. Compare indicator warmup periods")
            print("3. Verify data indexing alignment")
            print("4. Check for state accumulation")

if __name__ == "__main__":
    main()