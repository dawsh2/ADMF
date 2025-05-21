#!/usr/bin/env python3
"""
Standalone test of the regime-adaptive trading with EnsembleSignalStrategy.

This script:
1. Loads the optimized parameters from regime_optimized_parameters.json
2. Sets up the necessary components
3. Runs a simulation with the EnsembleSignalStrategy responding to regime changes
"""

import logging
import sys
import os
import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('adaptive_strategy_test.log')
                    ])
logger = logging.getLogger(__name__)

def run_standalone_test():
    """Run a standalone test of the regime-adaptive strategy"""
    from src.core.config import SimpleConfigLoader
    from src.core.container import Container
    from src.core.event_bus import EventBus
    from src.core.event import Event, EventType
    
    # Load the configuration
    config_loader = SimpleConfigLoader(config_file_path="config/config.yaml")
    
    # Create event bus and container
    event_bus = EventBus()
    container = Container()
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Register components
    logger.info("Setting up components for standalone test...")
    
    # Data handler
    from src.data.csv_data_handler import CSVDataHandler
    csv_args = {
        "instance_name": "SPY_CSV_Loader", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "component_config_key": "components.data_handler_csv",
        "max_bars": 1000  # Limit the number of bars
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)
    
    # Regime detector
    from src.strategy.regime_detector import RegimeDetector
    regime_detector_args = {
        "instance_name": "MyPrimaryRegimeDetector_Instance",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.MyPrimaryRegimeDetector"
    }
    container.register_type("MyPrimaryRegimeDetector", RegimeDetector, True, 
                          constructor_kwargs=regime_detector_args)
    
    # Strategy
    from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
    ensemble_strat_args = {
        "instance_name": "SPY_Ensemble_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy",
        "container": container  # Pass container to allow access to RegimeDetector
    }
    container.register_type("strategy", EnsembleSignalStrategy, True, 
                          constructor_kwargs=ensemble_strat_args)
    
    # Portfolio
    from src.risk.basic_portfolio import BasicPortfolio
    portfolio_args = {
        "instance_name": "BasicPortfolio", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "container": container,
        "component_config_key": "components.basic_portfolio"
    }
    container.register_type("portfolio_manager", BasicPortfolio, True, 
                          constructor_kwargs=portfolio_args)
    
    # Risk manager
    from src.risk.basic_risk_manager import BasicRiskManager
    risk_manager_args = {
        "instance_name": "BasicRiskMan1", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "component_config_key": "components.basic_risk_manager",
        "container": container, 
        "portfolio_manager_key": "portfolio_manager"
    }
    container.register_type("risk_manager", BasicRiskManager, True, 
                          constructor_kwargs=risk_manager_args)
    
    # Execution handler
    from src.execution.simulated_execution_handler import SimulatedExecutionHandler
    sim_exec_args = {
        "instance_name": "SimExec_1", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "component_config_key": "components.simulated_execution_handler"
    }
    container.register_type("execution_handler", SimulatedExecutionHandler, True, 
                          constructor_kwargs=sim_exec_args)
    
    # Resolve components
    data_handler = container.resolve("data_handler")
    regime_detector = container.resolve("MyPrimaryRegimeDetector")
    strategy = container.resolve("strategy")
    portfolio = container.resolve("portfolio_manager")
    risk_manager = container.resolve("risk_manager")
    execution_handler = container.resolve("execution_handler")
    
    # Set up all components
    logger.info("Setting up components...")
    data_handler.setup()
    regime_detector.setup()
    strategy.setup()
    portfolio.setup()
    risk_manager.setup()
    execution_handler.setup()
    
    # Start components in order
    logger.info("Starting components...")
    regime_detector.start()
    portfolio.start()
    risk_manager.start()
    strategy.start()
    execution_handler.start()
    
    # Use test data if available
    if hasattr(data_handler, "set_active_dataset") and callable(getattr(data_handler, "set_active_dataset")):
        data_handler.set_active_dataset("test")
        logger.info("Using test dataset for simulation")
    
    # Start the data handler last to begin event flow
    data_handler.start()
    
    # Wait for processing to complete
    logger.info("Waiting for processing to complete...")
    try:
        # Since CSVDataHandler doesn't have is_running, just a small delay is enough
        # The CSV data handler will have finished by now based on the logs
        import time
        time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    # Stop all components in reverse order
    logger.info("Stopping components...")
    data_handler.stop()
    execution_handler.stop()
    strategy.stop()
    risk_manager.stop()
    portfolio.stop()
    regime_detector.stop()
    
    # Log summary statistics
    logger.info("=== Test Results ===")
    
    # Get trade count
    trade_count = 0
    if hasattr(portfolio, "_trade_log"):
        trade_count = len(portfolio._trade_log)
        logger.info(f"Total trades: {trade_count}")
        
        # Log some trades if available
        if trade_count > 0:
            logger.info("Sample trades:")
            for i, trade in enumerate(portfolio._trade_log[:5]):
                logger.info(f"Trade {i+1}: {trade}")
    
    # Get final portfolio value
    if hasattr(portfolio, "get_final_portfolio_value") and callable(getattr(portfolio, "get_final_portfolio_value")):
        final_value = portfolio.get_final_portfolio_value()
        logger.info(f"Final portfolio value: {final_value}")
    
    # Get regime-specific performance
    if hasattr(portfolio, "get_performance_by_regime") and callable(getattr(portfolio, "get_performance_by_regime")):
        regime_performance = portfolio.get_performance_by_regime()
        if regime_performance:
            logger.info("Performance by regime:")
            for regime, data in regime_performance.items():
                if regime != "_boundary_trades_summary":
                    logger.info(f"  {regime}: {data}")
    
    logger.info("=== End of Test ===")
    
    return trade_count > 0

if __name__ == "__main__":
    logger.info("Starting standalone test of regime-adaptive strategy...")
    if run_standalone_test():
        logger.info("Test successful - trades were generated!")
        sys.exit(0)
    else:
        logger.info("Test failed - no trades were generated")
        sys.exit(1)