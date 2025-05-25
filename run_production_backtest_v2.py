#!/usr/bin/env python3
"""
Production Backtest Runner V2 - Uses BacktestEngine for consistency with optimizer.

This script demonstrates how to run a production backtest using the same
BacktestEngine component used by the optimizer, ensuring identical behavior.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.optimization.engines import BacktestEngine

# Import component classes for registration
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy

logger = logging.getLogger(__name__)


def setup_container(config_loader, event_bus, max_bars=None):
    """Set up the dependency injection container with all required components."""
    container = Container()
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Register data handler
    csv_args = {
        "instance_name": "SPY_CSV_Loader",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.data_handler_csv",
        "max_bars": max_bars
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)
    
    # Register regime detector
    regime_detector_args = {
        "instance_name": "MyPrimaryRegimeDetector_Instance",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.MyPrimaryRegimeDetector"
    }
    container.register_type(
        "MyPrimaryRegimeDetector",
        RegimeDetector,
        True,
        constructor_kwargs=regime_detector_args
    )
    
    # Register strategies (both types for flexibility)
    ensemble_args = {
        "instance_name": "SPY_Ensemble_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy",
        "container": container
    }
    container.register_type("ensemble_strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_args)
    # Also register as "strategy" for BacktestEngine compatibility
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_args)
    
    adaptive_args = {
        "instance_name": "RegimeAdaptiveStrategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "container": container,
        "component_config_key": "components.regime_adaptive_strategy"
    }
    container.register_type("regime_adaptive_strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=adaptive_args)
    
    # Register portfolio
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
    sim_exec_args = {
        "instance_name": "SimExec_1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.simulated_execution_handler"
    }
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=sim_exec_args)
    
    return container


def main():
    parser = argparse.ArgumentParser(description="Production Backtest Runner V2")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config_adaptive_production.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Number of bars to process"
    )
    parser.add_argument(
        "--adaptive-params",
        type=str,
        default="regime_optimized_parameters.json",
        help="Path to regime-specific parameters JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["full", "train", "test"],
        default="full",
        help="Which dataset to use"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ensemble", "regime_adaptive"],
        default="regime_adaptive",
        help="Strategy type to use"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_loader = SimpleConfigLoader(config_file_path=args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
        
    # Set up logging
    setup_logging(config_loader, args.log_level)
    logger.info("Starting Production Backtest V2")
    
    # Create event bus and container
    event_bus = EventBus()
    container = setup_container(config_loader, event_bus, args.bars)
    
    # Create backtest engine
    backtest_engine = BacktestEngine(container, config_loader, event_bus)
    
    # Determine if we're using adaptive parameters
    use_adaptive = args.strategy == "regime_adaptive"
    adaptive_params_path = None
    
    if use_adaptive and Path(args.adaptive_params).exists():
        adaptive_params_path = args.adaptive_params
        logger.info(f"Using regime-adaptive parameters from: {adaptive_params_path}")
    elif use_adaptive:
        logger.warning(f"Adaptive parameters file not found: {args.adaptive_params}")
        logger.warning("Will use default parameters from configuration")
    
    # Run the backtest
    logger.info(f"Running {args.strategy} backtest on {args.dataset} dataset")
    
    try:
        metric_value, regime_performance = backtest_engine.run_backtest(
            parameters={},  # Use defaults from config
            dataset_type=args.dataset,
            strategy_type=args.strategy,
            use_regime_adaptive=use_adaptive,
            adaptive_params_path=adaptive_params_path
        )
        
        # Display results
        if metric_value is not None:
            print("\n" + "=" * 80)
            print("PRODUCTION BACKTEST RESULTS")
            print("=" * 80)
            print(f"Final Portfolio Value: {metric_value:.2f}")
            
            if regime_performance:
                print("\nPERFORMANCE BY REGIME:")
                for regime, perf_data in sorted(regime_performance.items()):
                    if not regime.startswith('_'):  # Skip internal keys
                        trades = perf_data.get('count', 0)
                        sharpe = perf_data.get('sharpe_ratio', 'N/A')
                        sharpe_str = f"{sharpe:.4f}" if isinstance(sharpe, (float, int)) else sharpe
                        print(f"  {regime}: {trades} trades, Sharpe: {sharpe_str}")
                        
            print("=" * 80)
        else:
            logger.error("Backtest failed - no results returned")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("Production backtest completed")


if __name__ == "__main__":
    main()