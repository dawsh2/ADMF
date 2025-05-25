#!/usr/bin/env python3
"""
Run the system in adaptive mode like the optimization test phase
"""

import json
import yaml
from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.strategy.regime_detector import RegimeDetector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load config
    config_loader = SimpleConfigLoader("config/config_optimization_exact.yaml")
    event_bus = EventBus()
    container = Container(event_bus)
    
    # Register components
    container.register_instance("config_loader", config_loader)
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)
    
    # Register data handler
    data_args = {
        "instance_name": "SPY_CSV_Loader",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.data_handler_csv",
        "max_bars": None
    }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=data_args)
    
    # Register strategy
    strategy_args = {
        "instance_name": "SPY_Ensemble_Strategy",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy",
        "container": container
    }
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=strategy_args)
    
    # Register regime detector
    regime_args = {
        "instance_name": "MyPrimaryRegimeDetector_Instance",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.MyPrimaryRegimeDetector"
    }
    container.register_type("MyPrimaryRegimeDetector", RegimeDetector, True, constructor_kwargs=regime_args)
    
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
    risk_args = {
        "instance_name": "BasicRiskMan1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.basic_risk_manager",
        "container": container,
        "portfolio_manager_key": "portfolio_manager"
    }
    container.register_type("risk_manager", BasicRiskManager, True, constructor_kwargs=risk_args)
    
    # Register execution handler
    exec_args = {
        "instance_name": "SimExec_1",
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.simulated_execution_handler"
    }
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=exec_args)
    
    # Resolve components
    data_handler = container.resolve("data_handler")
    strategy = container.resolve("strategy")
    regime_detector = container.resolve("MyPrimaryRegimeDetector")
    portfolio = container.resolve("portfolio_manager")
    risk_manager = container.resolve("risk_manager")
    execution_handler = container.resolve("execution_handler")
    
    # Setup components
    data_handler.setup()
    strategy.setup()
    regime_detector.setup()
    portfolio.setup()
    risk_manager.setup()
    execution_handler.setup()
    
    # Enable adaptive mode on the strategy
    logger.warning("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
    strategy.set_adaptive_mode_enabled(True)
    
    # Set the regime-specific parameters that optimization found
    regime_params = {
        'trending_down': {
            'parameters': {
                'short_window': 5,
                'long_window': 20,
                'rsi_indicator.period': 21,
                'rsi_rule.oversold_threshold': 20.0,
                'rsi_rule.overbought_threshold': 60.0
            }
        }
    }
    
    # Load regime optimized parameters if available
    try:
        with open('regime_optimized_parameters.json', 'r') as f:
            full_params = json.load(f)
            if 'best_parameters_per_regime' in full_params:
                regime_params = full_params['best_parameters_per_regime']
    except:
        logger.info("Using default trending_down parameters")
    
    strategy._regime_specific_params = regime_params
    
    # Start components
    data_handler.start()
    strategy.start()
    regime_detector.start()
    portfolio.start()
    risk_manager.start()
    execution_handler.start()
    
    # Process test data only
    data_handler.set_mode("test")
    
    logger.info("Processing all bars in adaptive mode...")
    while data_handler.has_next():
        data_handler.next()
    
    # Get final results
    portfolio_value = portfolio.get_final_portfolio_value()
    total_trades = portfolio.get_total_trades()
    
    logger.info(f"=== ADAPTIVE TEST RESULTS ===")
    logger.info(f"Final portfolio value: ${portfolio_value:,.2f}")
    logger.info(f"Total trades: {total_trades}")
    
    # Cleanup
    execution_handler.stop()
    risk_manager.stop()
    portfolio.stop()
    regime_detector.stop()
    strategy.stop()
    data_handler.stop()

if __name__ == "__main__":
    main()