#!/usr/bin/env python3
"""
Test exact match with optimizer by using same data flow.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.core.config import SimpleConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run optimization to verify signal count."""
    config = SimpleConfigLoader('config/config.yaml')
    
    # Create optimizer with MA-only mode
    optimizer = EnhancedOptimizer(
        config_loader=config,
        train_test_split_ratio=0.8,
        mode='test',
        iterations=1,
        population_size=1,
        bars_limit=1000,
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
        experiment_name='test_match',
        log_level='INFO',
        optimize_ma=True,  # Enable MA-only optimization
        optimize_rsi=False
    )
    
    logger.info("=== RUNNING OPTIMIZER IN MA MODE ===")
    
    # Run optimization
    best_params, results = optimizer.optimize()
    
    # Extract test metrics
    if results and 'test_metrics' in results:
        test_trades = results['test_metrics'].get('total_trades', 0)
        logger.info(f"Test period trades: {test_trades}")
        
        # Count signals from individual results
        if 'individual_results' in results:
            for idx, res in enumerate(results['individual_results']):
                if 'test_signal_log' in res:
                    signal_count = len(res['test_signal_log'])
                    logger.info(f"Individual {idx}: {signal_count} test signals")
                    
                    # Show first few signals
                    for i, sig in enumerate(res['test_signal_log'][:3]):
                        logger.info(f"  Signal {i+1}: {sig.get('timestamp', 'N/A')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())