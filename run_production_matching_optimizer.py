#!/usr/bin/env python3
"""
Production backtest that matches optimizer's view of test data.
Uses BacktestEngine exactly like the optimizer does.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.append('.')

from src.core.config import SimpleConfigLoader
from src.core.container import Container
from src.core.event_bus import EventBus
from src.core.component import BaseComponent
from src.strategy.optimization.engines.backtest_engine import BacktestEngine
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.regime_detector import RegimeDetector
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler

def setup_container_for_production(config_loader, event_bus, container):
    """
    Set up the DI container with all required components.
    Matches main.py setup for regime adaptive mode.
    """
    logger = logging.getLogger(__name__)
    
    # Register data handler
    data_handler_args = {
        "instance_name": "SPY_CSV_Loader",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.data_handler_csv"
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=data_handler_args)
    
    # Register regime detector
    regime_detector_args = {
        "instance_name": "MyPrimaryRegimeDetector_Instance",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.MyPrimaryRegimeDetector"
    }
    container.register_type("MyPrimaryRegimeDetector", RegimeDetector, True, constructor_kwargs=regime_detector_args)
    
    # Register portfolio manager
    portfolio_args = {
        "instance_name": "BasicPortfolio",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.basic_portfolio"
    }
    container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    
    # Register risk manager
    risk_manager_args = {
        "instance_name": "BasicRiskMan1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.basic_risk_manager",
        "container": container,
        "portfolio_manager_key": "portfolio_manager"
    }
    container.register_type("risk_manager", BasicRiskManager, True, constructor_kwargs=risk_manager_args)
    
    # Register execution handler
    exec_handler_args = {
        "instance_name": "SimExec_1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.simulated_execution_handler"
    }
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=exec_handler_args)
    
    # Register regime adaptive strategy
    strategy_args = {
        "instance_name": "SPY_RegimeAdaptive_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.regime_adaptive_strategy"
    }
    container.register_type("strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=strategy_args)
    
    logger.info("Container setup complete for production run")

def run_production_matching_optimizer(config_path: str, params_path: str, log_level: str = "INFO"):
    """
    Run production backtest matching optimizer's data handling exactly.
    """
    # Setup logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("PRODUCTION RUN - MATCHING OPTIMIZER DATA VIEW")
    print("="*80)
    print("This uses BacktestEngine exactly like the optimizer does.")
    print("Should produce identical results to optimizer's adaptive test.")
    print("")
    
    # Load configuration
    config_loader = SimpleConfigLoader(config_path)
    event_bus = EventBus()
    container = Container()
    
    # Set up container with required components
    setup_container_for_production(config_loader, event_bus, container)
    
    # Create BacktestEngine (same as optimizer uses)
    engine = BacktestEngine(container, config_loader, event_bus)
    
    logger.info("Running production backtest with optimizer-matching setup...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Adaptive params: {params_path}")
    
    # Run backtest with EXACTLY the same parameters as optimizer's adaptive test
    metric_value, regime_performance = engine.run_backtest(
        parameters={},  # Not used for regime-adaptive
        dataset_type="test",  # Uses test portion of full dataset (bars 800-999)
        strategy_type="regime_adaptive",
        use_regime_adaptive=True,
        adaptive_params_path=params_path
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS (Should Match Optimizer)")
    print("="*80)
    
    if metric_value is not None:
        print(f"Final Portfolio Value: ${metric_value:.2f}")
        
        if regime_performance:
            print("\nPERFORMANCE BY REGIME:")
            for regime, metrics in regime_performance.items():
                if isinstance(metrics, dict):
                    trades = metrics.get('total_trades', 0)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    print(f"  {regime}: {trades} trades, Sharpe: {sharpe:.4f}")
    else:
        print("ERROR: No metric value returned")
    
    print("="*80)
    print("\nCompare with optimizer's adaptive test result:")
    print("If they match, the cold start fix is working correctly!")
    print("If they don't match, there's still state leakage somewhere.")
    
    return metric_value

def main():
    parser = argparse.ArgumentParser(description='Run production matching optimizer data view')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--adaptive-params', type=str, default='regime_optimized_parameters.json',
                       help='Path to regime parameters JSON file')
    parser.add_argument('--log-level', type=str, default='ERROR',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    
    args = parser.parse_args()
    
    if not Path(args.adaptive_params).exists():
        print(f"Error: Adaptive parameters file not found: {args.adaptive_params}")
        sys.exit(1)
        
    run_production_matching_optimizer(args.config, args.adaptive_params, args.log_level)

if __name__ == "__main__":
    main()