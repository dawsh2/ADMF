#!/usr/bin/env python3
"""
Run the optimizer with exact parameters to verify the 17 signals.
This will help us understand the exact configuration.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.core.config import SimpleConfigLoader

# Setup logging to capture everything
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimizer_exact_match.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run optimizer with exact settings."""
    config = SimpleConfigLoader('config/config.yaml')
    
    logger.info("=== RUNNING OPTIMIZER WITH EXACT SETTINGS ===")
    logger.info("Mode: MA-only optimization")
    logger.info("Split: 80/20 (798 training, 200 test bars)")
    
    # Create optimizer with exact settings
    optimizer = EnhancedOptimizer(
        config_loader=config,
        train_test_split_ratio=0.8,
        mode='test',  # Single test run
        iterations=1,
        population_size=1,
        bars_limit=1000,  # Process first 1000 bars (but data only has 998)
        strategy_type='ensemble',
        strategy_config={
            'instance_name': 'test_strategy',
            'component_config_key': 'components.ensemble_strategy'
        },
        portfolio_config={
            'instance_name': 'test_portfolio', 
            'component_config_key': 'components.basic_portfolio',
            'initial_cash': 100000.0
        },
        risk_manager_config={
            'instance_name': 'test_risk_manager',
            'component_config_key': 'components.basic_risk_manager'
        },
        execution_handler_config={
            'instance_name': 'test_execution',
            'component_config_key': 'components.simulated_execution_handler'
        },
        data_handler_config={
            'instance_name': 'test_data_handler',
            'component_config_key': 'components.data_handler_csv',
            'csv_file_path': 'data/1000_1min.csv'
        },
        enable_per_iteration_plot=False,
        enable_final_plot=False,
        save_results=False,
        result_dir='optimization_results',
        experiment_name='exact_match_test',
        log_level='INFO',
        optimize_ma=True,      # KEY: MA-only optimization
        optimize_rsi=False,    # KEY: Disable RSI
        disable_adaptive_test=True  # Don't run adaptive test
    )
    
    # Run optimization
    best_params, results = optimizer.optimize()
    
    logger.info("\n=== OPTIMIZATION COMPLETE ===")
    
    # Extract detailed results
    if results:
        # Log best parameters
        logger.info(f"Best parameters: {best_params}")
        
        # Check test metrics
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            logger.info(f"\nTest metrics:")
            logger.info(f"  Total trades: {test_metrics.get('total_trades', 'N/A')}")
            logger.info(f"  Final value: {test_metrics.get('final_value', 'N/A')}")
            
        # Check individual results for signal details
        if 'individual_results' in results:
            for idx, individual in enumerate(results['individual_results']):
                if 'test_signal_log' in individual:
                    test_signals = individual['test_signal_log']
                    logger.info(f"\nTest signals for individual {idx}: {len(test_signals)}")
                    
                    # Log each signal
                    for i, signal in enumerate(test_signals):
                        logger.info(f"  Signal {i+1}: {signal.get('timestamp', 'N/A')}, "
                                   f"Type={signal.get('signal_type', 'N/A')}, "
                                   f"Price={signal.get('price', 'N/A')}")
                        
                    # Also check training signals
                    if 'training_signal_log' in individual:
                        train_signals = individual['training_signal_log'] 
                        logger.info(f"Training signals: {len(train_signals)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())