#!/usr/bin/env python3
"""
Enable adaptive mode in production to match optimization test behavior
"""

import json
import sys
import os

def extract_regime_parameters_from_json():
    """Extract and format regime parameters correctly"""
    try:
        with open('regime_optimized_parameters.json', 'r') as f:
            data = json.load(f)
        
        # Extract the regime_best_parameters section
        if 'regime_best_parameters' in data:
            regime_params = {}
            for regime, regime_data in data['regime_best_parameters'].items():
                # Navigate the nested structure: regime_data['parameters']['parameters']
                if 'parameters' in regime_data and 'parameters' in regime_data['parameters']:
                    params = regime_data['parameters']['parameters']
                    regime_params[regime] = {'parameters': params}
                    print(f"Loaded parameters for regime '{regime}': {params}")
                else:
                    print(f"Warning: Could not find parameters for regime '{regime}'")
            
            return regime_params
        else:
            print("Error: 'regime_best_parameters' not found in JSON file")
            return {}
            
    except Exception as e:
        print(f"Error loading regime parameters: {e}")
        return {}

def create_adaptive_parameters_file():
    """Create a simplified parameters file for adaptive mode"""
    
    regime_params = extract_regime_parameters_from_json()
    
    if not regime_params:
        print("No regime parameters found!")
        return False
    
    # Create a simplified structure for adaptive mode
    adaptive_params = {
        'best_parameters_per_regime': regime_params
    }
    
    # Save to a file that the strategy can load
    with open('adaptive_regime_parameters.json', 'w') as f:
        json.dump(adaptive_params, f, indent=2)
    
    print(f"Created adaptive_regime_parameters.json with {len(regime_params)} regimes")
    return True

def print_usage():
    print("""
To enable adaptive mode in production:

1. First, run this script to prepare the parameters:
   python3 enable_adaptive_mode_production.py

2. Then modify main.py to enable adaptive mode on the EnsembleSignalStrategy:

   After line where strategy is resolved, add:
   
   # Enable adaptive mode like optimization does
   import json
   with open('adaptive_regime_parameters.json', 'r') as f:
       regime_params = json.load(f)['best_parameters_per_regime']
   
   print("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
   strategy.enable_adaptive_mode(regime_params)

3. Run with:
   python3 main.py --config config/config_optimization_exact_production.yaml

This will replicate the optimization's adaptive test behavior.
""")

def main():
    print("Enabling adaptive mode in production...")
    
    if not os.path.exists('regime_optimized_parameters.json'):
        print("Error: regime_optimized_parameters.json not found!")
        print("Please run optimization first to generate this file.")
        return 1
    
    success = create_adaptive_parameters_file()
    
    if success:
        print_usage()
        return 0
    else:
        print("Failed to create adaptive parameters file")
        return 1

if __name__ == "__main__":
    sys.exit(main())