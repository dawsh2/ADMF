#!/usr/bin/env python3
# test_optimizer.py
import argparse
import logging
import sys
from typing import Dict, Any

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging, create_optimization_logger
from src.core.event_bus import EventBus
from src.core.container import Container

# Data and portfolio imports
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio

# Strategy and optimizer imports
from src.strategy.ma_strategy import MAStrategy
from src.strategy.regime_detector import RegimeDetector
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer

def main():
    parser = argparse.ArgumentParser(description="Test EnhancedOptimizer with Regime Detection")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--bars", type=int, default=1000, help="Number of bars to process. Default is 1000."
    )
    args = parser.parse_args()
    config_path = args.config
    max_bars_to_process = args.bars

    # Set up logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("test_optimizer")
    
    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config_loader = SimpleConfigLoader(config_file_path=config_path)
    setup_logging(config_loader)
    
    # Create specialized optimization logger
    optimization_logger = create_optimization_logger("test_optimization")
    optimization_logger.info("Starting optimization test")
    
    # Create event bus and container
    event_bus = EventBus()
    container = Container()
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Data handler
    data_handler = CSVDataHandler(
        instance_name="SPY_CSV_Loader",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.data_handler_csv",
        max_bars=max_bars_to_process
    )
    container.register("SPY_CSV_Loader", data_handler)
    
    # Portfolio
    portfolio = BasicPortfolio(
        instance_name="MyPortfolio",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.portfolio"
    )
    container.register("MyPortfolio", portfolio)
    
    # Strategy
    strategy = MAStrategy(
        instance_name="MovingAverageStrategy",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.strategy"
    )
    container.register("MovingAverageStrategy", strategy)
    
    # Regime detector
    regime_detector = RegimeDetector(
        instance_name="MyPrimaryRegimeDetector",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.MyPrimaryRegimeDetector"
    )
    container.register("MyPrimaryRegimeDetector", regime_detector)
    
    # Optimizer
    optimizer = EnhancedOptimizer(
        instance_name="TestOptimizer",
        config_loader=config_loader,
        event_bus=event_bus,
        component_config_key="components.optimizer",
        container=container
    )
    
    # Setup components
    logger.info("Setting up components...")
    data_handler.setup()
    portfolio.setup()
    strategy.setup()
    regime_detector.setup()
    optimizer.setup()
    
    # Start components
    logger.info("Starting components...")
    portfolio.start()
    strategy.start()
    regime_detector.start()
    
    # Run the optimizer
    logger.info("Running the enhanced optimizer with regime-specific optimization...")
    optimization_results = optimizer.run_grid_search()
    
    if optimization_results:
        # Log summary of the key optimization results
        optimization_logger.info("Optimization completed successfully")
        
        # Log best parameters per regime
        if "best_parameters_per_regime" in optimization_results and optimization_results["best_parameters_per_regime"]:
            optimization_logger.info("\nBest parameters per regime:")
            for regime, params in optimization_results["best_parameters_per_regime"].items():
                optimization_logger.info(f"  - {regime}: {params}")
        
        # Log the regimes encountered
        if "regimes_encountered" in optimization_results:
            optimization_logger.info(f"\nRegimes encountered: {optimization_results['regimes_encountered']}")
        
        # Log the adaptive strategy test results
        if "regime_adaptive_test_results" in optimization_results:
            adaptive_results = optimization_results["regime_adaptive_test_results"]
            optimization_logger.info("\nRegime Adaptive Strategy Test Results:")
            if "error" in adaptive_results:
                optimization_logger.info(f"  Error: {adaptive_results['error']}")
            else:
                optimization_logger.info(f"  Metric value: {adaptive_results.get('metric_value')}")
                optimization_logger.info(f"  Regimes detected: {adaptive_results.get('regimes_detected', [])}")
    else:
        logger.error("Optimization failed or returned no results")
    
    # Clean up and stop components
    logger.info("Stopping components...")
    strategy.stop()
    regime_detector.stop()
    portfolio.stop()
    data_handler.stop()
    
    logger.info("Test completed")

if __name__ == "__main__":
    main()