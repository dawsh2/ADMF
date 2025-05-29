#!/usr/bin/env python3
"""
Test script for isolated component optimization.
Demonstrates the efficiency gains from optimizing rules in isolation.
"""

import sys
sys.path.append('/Users/daws/ADMF')

from src.core.app_runner import AppRunner
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def compare_optimization_approaches():
    """Compare standard vs isolated optimization."""
    
    print("=" * 70)
    print("Isolated Component Optimization Test")
    print("=" * 70)
    
    # Load configuration
    config_path = "/Users/daws/ADMF/config/test_ensemble_optimization.yaml"
    
    print(f"\nLoading configuration from: {config_path}")
    
    # Calculate expected backtests
    print("\nExpected backtest counts:")
    print("-" * 50)
    
    # Standard joint optimization
    ma_params = 3 * 3  # short_window * long_window
    rsi_params = 3 * 3 * 3  # period * oversold * overbought
    weight_params = 7 * 7  # ma_weight * rsi_weight
    total_joint = ma_params * rsi_params * weight_params
    
    print(f"Standard joint optimization: {total_joint:,} backtests")
    
    # Isolated optimization
    ma_isolated = ma_params  # 9 backtests
    rsi_isolated = rsi_params  # 27 backtests
    weight_only = weight_params  # 49 backtests
    total_isolated = ma_isolated + rsi_isolated + weight_only
    
    print(f"Isolated optimization:")
    print(f"  - MA rule alone: {ma_isolated} backtests")
    print(f"  - RSI rule alone: {rsi_isolated} backtests")
    print(f"  - Weight optimization: {weight_only} backtests")
    print(f"  - Total: {total_isolated} backtests")
    
    reduction = (1 - total_isolated / total_joint) * 100
    print(f"\nReduction: {reduction:.1f}% fewer backtests!")
    
    # For regime-specific optimization
    num_regimes = 4
    print(f"\nWith {num_regimes} regimes:")
    print(f"  - Standard: {total_joint * num_regimes:,} backtests")
    print(f"  - Isolated: {total_isolated * num_regimes} backtests")
    
    print("\n" + "=" * 70)
    
    # Run actual optimization
    try:
        print("\nStarting optimization with isolated workflow...")
        
        # Use ApplicationLauncher which properly sets up CLI args
        from src.core.application_launcher import ApplicationLauncher
        
        # Create launcher with proper CLI arguments
        argv = [
            '--config', config_path,
            '--optimize',
            '--bars', '1000'  # Use same dataset size as normal test
        ]
        
        launcher = ApplicationLauncher(argv)
        # Run through launcher
        launcher.run()
        
        print("\nOptimization completed!")
        
        # Check for results
        from pathlib import Path
        results_dir = Path("optimization_results")
        if results_dir.exists():
            print(f"\nResults saved to: {results_dir}")
            
            # List result files
            result_files = list(results_dir.glob("*.json"))
            print(f"Found {len(result_files)} result files:")
            for f in sorted(result_files)[-5:]:  # Show last 5
                print(f"  - {f.name}")
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = compare_optimization_approaches()
    sys.exit(0 if success else 1)