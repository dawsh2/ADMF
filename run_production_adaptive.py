#!/usr/bin/env python3
"""
Run production with adaptive mode enabled to match optimization test results
"""

import subprocess
import sys
import json
import tempfile
import os

def create_modified_main():
    """Create a modified main.py with adaptive mode enabled"""
    
    # Read the original main.py
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    # Find where to inject the adaptive mode code
    # Look for the line where strategy is resolved
    injection_point = 'strategy = app_container.resolve("strategy")'
    
    if injection_point not in main_content:
        print("Error: Could not find strategy resolution point in main.py")
        return None
    
    # Create the adaptive mode injection code
    adaptive_code = '''
        # Enable adaptive mode like optimization does
        import json
        try:
            with open('adaptive_regime_parameters.json', 'r') as f:
                regime_params = json.load(f)['best_parameters_per_regime']
            
            print("\\n" + "=" * 80)
            print("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
            print(f"Available regimes: {list(regime_params.keys())}")
            print("This will allow the strategy to switch parameters during regime changes")
            print("=" * 80 + "\\n")
            
            logger.warning("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
            strategy.enable_adaptive_mode(regime_params)
            
            # Verify adaptive mode is enabled
            if hasattr(strategy, "get_adaptive_mode_status"):
                status = strategy.get_adaptive_mode_status()
                print("=== ADAPTIVE MODE STATUS ===")
                print(f"Adaptive mode enabled: {status['adaptive_mode_enabled']}")
                print(f"Parameters loaded for regimes: {status['available_regimes']}")
                print(f"Starting regime: {status['current_regime']}")
                print("=" * 80 + "\\n")
                
        except Exception as e:
            print(f"Warning: Could not enable adaptive mode: {e}")
            print("Running in standard mode...")
'''
    
    # Inject the adaptive code after strategy resolution
    modified_content = main_content.replace(
        injection_point,
        injection_point + adaptive_code
    )
    
    # Create a temporary modified main file
    temp_main = tempfile.NamedTemporaryFile(mode='w', suffix='_main_adaptive.py', delete=False)
    temp_main.write(modified_content)
    temp_main.close()
    
    return temp_main.name

def main():
    print("Running production with adaptive mode enabled...")
    
    # Check if adaptive parameters file exists
    if not os.path.exists('adaptive_regime_parameters.json'):
        print("Error: adaptive_regime_parameters.json not found!")
        print("Please run: python3 enable_adaptive_mode_production.py")
        return 1
    
    # Create modified main.py
    temp_main = create_modified_main()
    if not temp_main:
        return 1
    
    try:
        # Run the modified main.py
        cmd = [sys.executable, temp_main, '--config', 'config/config_optimization_exact_production.yaml']
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
        
    finally:
        # Clean up temporary file
        if temp_main and os.path.exists(temp_main):
            os.unlink(temp_main)

if __name__ == "__main__":
    sys.exit(main())