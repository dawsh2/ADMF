#!/usr/bin/env python3
"""
Debug why we're seeing 0 trades in both optimization and test runs.
"""

import subprocess
import re

def run_with_debug(dataset_type="test"):
    """Run with extra debug logging for signal generation."""
    cmd = f"""cd /Users/daws/ADMF && source venv/bin/activate && python -c "
import logging
logging.getLogger('src.strategy.base.strategy').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.implementations').setLevel(logging.DEBUG)
logging.getLogger('src.strategy.components.rules').setLevel(logging.DEBUG)
logging.getLogger('src.risk').setLevel(logging.DEBUG)
logging.getLogger('src.execution').setLevel(logging.DEBUG)

from src.core.application_launcher import ApplicationLauncher
launcher = ApplicationLauncher()
launcher.run(['--config', 'config/test_ensemble_optimization.yaml', '--bars', '50', '--dataset', '{dataset_type}', '--log-level', 'DEBUG'])
" 2>&1 | grep -E "(evaluate.*returned|signal.*strength|Generated.*signal|position_size|Risk check|Order rejected|Executing.*order|No signal|rule.*evaluated|Rule.*signal)" | head -100
"""
    
    print(f"\nRunning debug for {dataset_type} dataset...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def analyze_test_params():
    """Check the test_regime_parameters.json file."""
    import json
    
    print("\nAnalyzing test_regime_parameters.json...")
    try:
        with open('/Users/daws/ADMF/test_regime_parameters.json', 'r') as f:
            params = json.load(f)
            
        # Check weights
        print("\nRule weights by regime:")
        for regime, regime_params in params.items():
            if isinstance(regime_params, dict):
                weights = {}
                for key, value in regime_params.items():
                    if key.endswith('.weight'):
                        rule_name = key.replace('.weight', '')
                        weights[rule_name] = value
                
                if weights:
                    print(f"\n{regime}:")
                    for rule, weight in sorted(weights.items()):
                        print(f"  {rule}: {weight}")
                        
        # Check rule parameters
        print("\n\nKey rule parameters:")
        for regime, regime_params in params.items():
            if isinstance(regime_params, dict):
                print(f"\n{regime}:")
                # MA crossover
                min_sep = regime_params.get('strategy_ma_crossover.min_separation', 'N/A')
                print(f"  MA min_separation: {min_sep}")
                
                # RSI thresholds
                oversold = regime_params.get('strategy_rsi_rule.oversold_threshold', 'N/A')
                overbought = regime_params.get('strategy_rsi_rule.overbought_threshold', 'N/A')
                print(f"  RSI thresholds: {oversold} / {overbought}")
                
                # BB filter
                bb_filter = regime_params.get('strategy_bb_rule.band_width_filter', 'N/A')
                print(f"  BB band_width_filter: {bb_filter}")
                
    except Exception as e:
        print(f"Error reading parameters: {e}")

def check_position_sizing():
    """Check position sizing configuration."""
    cmd = """cd /Users/daws/ADMF && grep -E "(position_size|max_positions|risk_per_trade)" config/test_ensemble_optimization.yaml"""
    
    print("\n\nChecking position sizing config...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)

def main():
    print("="*80)
    print("DEBUGGING NO TRADES ISSUE")
    print("="*80)
    
    # 1. Check parameters
    analyze_test_params()
    
    # 2. Check position sizing config
    check_position_sizing()
    
    # 3. Run with debug logging
    test_output = run_with_debug("test")
    
    print("\n\nDEBUG OUTPUT:")
    print(test_output)
    
    # Count different signal types
    buy_signals = test_output.count("BUY signal")
    sell_signals = test_output.count("SELL signal")
    no_signals = test_output.count("No signal")
    
    print(f"\n\nSIGNAL SUMMARY:")
    print(f"BUY signals: {buy_signals}")
    print(f"SELL signals: {sell_signals}")
    print(f"No signal messages: {no_signals}")

if __name__ == "__main__":
    main()