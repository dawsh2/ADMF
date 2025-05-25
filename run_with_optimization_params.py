#!/usr/bin/env python3
"""
Run the system with the exact parameters that optimization used for its test phase
"""

import subprocess
import json
import yaml
import shutil
import sys

# The exact parameters from optimization that achieved $111,438.51
OPTIMIZATION_PARAMS = {
    'short_window': 5,
    'long_window': 20,
    'rsi_indicator.period': 20,
    'rsi_rule.oversold_threshold': 20.0,
    'rsi_rule.overbought_threshold': 80.0,
    # These weights produce signal strength 0.22
    'ma_rule.weight': 0.2,
    'rsi_rule.weight': 0.8
}

def create_optimized_config():
    """Create a config file with optimization parameters"""
    
    # Load the base config
    with open('config/config_optimization_exact.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update ensemble_strategy with optimization parameters
    ensemble_config = config['components']['ensemble_strategy']
    
    # Apply the exact optimization parameters
    ensemble_config['short_window'] = OPTIMIZATION_PARAMS['short_window']
    ensemble_config['long_window'] = OPTIMIZATION_PARAMS['long_window']
    ensemble_config['ma_rule.weight'] = OPTIMIZATION_PARAMS['ma_rule.weight']
    ensemble_config['rsi_rule.weight'] = OPTIMIZATION_PARAMS['rsi_rule.weight']
    
    # Update RSI indicator
    ensemble_config['rsi_indicator']['period'] = OPTIMIZATION_PARAMS['rsi_indicator.period']
    
    # Update RSI rule
    ensemble_config['rsi_rule']['oversold_threshold'] = OPTIMIZATION_PARAMS['rsi_rule.oversold_threshold']
    ensemble_config['rsi_rule']['overbought_threshold'] = OPTIMIZATION_PARAMS['rsi_rule.overbought_threshold']
    ensemble_config['rsi_rule']['weight'] = OPTIMIZATION_PARAMS['rsi_rule.weight']
    
    # Save the optimized config
    with open('config/config_optimized_production.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created optimized config with parameters:")
    print(f"  MA: short={OPTIMIZATION_PARAMS['short_window']}, long={OPTIMIZATION_PARAMS['long_window']}, weight={OPTIMIZATION_PARAMS['ma_rule.weight']}")
    print(f"  RSI: period={OPTIMIZATION_PARAMS['rsi_indicator.period']}, oversold={OPTIMIZATION_PARAMS['rsi_rule.oversold_threshold']}, overbought={OPTIMIZATION_PARAMS['rsi_rule.overbought_threshold']}, weight={OPTIMIZATION_PARAMS['rsi_rule.weight']}")
    print(f"  Signal strength should be: {OPTIMIZATION_PARAMS['ma_rule.weight'] * 0.5 + OPTIMIZATION_PARAMS['rsi_rule.weight'] * 0.5:.2f}")

def main():
    # Create the optimized config
    create_optimized_config()
    
    # Run the system with the optimized config
    print("\nRunning system with optimized parameters...")
    cmd = ['python3', 'main.py', '--config', 'config/config_optimized_production.yaml']
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())