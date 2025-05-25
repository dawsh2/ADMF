#!/usr/bin/env python3
"""
Run just the optimizer's adaptive test without the optimization phase
This should produce identical results to the optimizer's test phase
"""
import logging
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer

def main():
    # Setup logging to match optimizer
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/optimizer_test_only.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Running optimizer test phase only (no optimization)")
    
    # Create optimizer instance
    optimizer = EnhancedOptimizer(config_path='config/config_adaptive_production.yaml')
    
    # Load the adaptive parameters (already optimized)
    import json
    with open('adaptive_regime_parameters.json', 'r') as f:
        params = json.load(f)
    
    best_params = params['best_parameters_per_regime']
    
    logger.info("Running adaptive test with pre-optimized parameters...")
    logger.info(f"Available regimes: {list(best_params.keys())}")
    
    # Run just the adaptive test
    results = optimizer.run_adaptive_test(best_params)
    
    logger.info("\n=== ADAPTIVE TEST RESULTS ===")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Final Equity: ${results['final_equity']:,.2f}")

if __name__ == "__main__":
    main()