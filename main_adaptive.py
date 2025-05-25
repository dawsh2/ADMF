#!/usr/bin/env python3
"""
Modified main.py with adaptive mode enabled
"""

import argparse
import sys
import json
from datetime import datetime

from src.core.config import SimpleConfigLoader
from src.core.event_bus import EventBus
from src.core.container import Container
from src.core.dummy_component import DummyComponent
from src.core.logging_setup import setup_logging

from src.data.csv_data_handler import CSVDataHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategy.regime_detector import RegimeDetector
from src.execution.simulated_execution_handler import SimulatedExecutionHandler

from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.strategy.optimization.genetic_optimizer import GeneticOptimizer
from src.core.exceptions import ComponentError, DependencyNotFoundError, ConfigurationError

import logging

def main():
    parser = argparse.ArgumentParser(description='Run ADMF Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization instead of trading')
    parser.add_argument('--bars', type=int, default=None,
                        help='Limit the number of bars to process (for testing)')
    args = parser.parse_args()

    config_loader = SimpleConfigLoader(args.config)
    setup_logging(config_loader)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Attempting to load configuration from: {args.config}")

    event_bus = EventBus()
    app_container = Container()
    app_container.register_instance("config_loader", config_loader)
    app_container.register_instance("event_bus", event_bus)
    app_container.register_instance("container", app_container)

    data_handler_instance_name = "SPY_CSV_Loader"
    data_handler_config_key = "components.data_handler_csv"
    data_handler_constructor_kwargs = {
        "instance_name": data_handler_instance_name,
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": data_handler_config_key,
        "max_bars": args.bars
    }
    app_container.register_type(
        "data_handler",
        CSVDataHandler,
        True,
        constructor_kwargs=data_handler_constructor_kwargs
    )

    if args.optimize:
        # Optimization mode - use EnsembleSignalStrategy
        ensemble_strategy_instance_name = "SPY_Ensemble_Strategy"
        ensemble_strategy_config_key = "components.ensemble_strategy"
        ensemble_strategy_constructor_kwargs = {
            "instance_name": ensemble_strategy_instance_name,
            "config_loader": config_loader,
            "event_bus": event_bus,
            "component_config_key": ensemble_strategy_config_key,
            "container": app_container
        }
        app_container.register_type(
            "strategy",
            EnsembleSignalStrategy,
            True,
            constructor_kwargs=ensemble_strategy_constructor_kwargs
        )
        logger.info("EnsembleSignalStrategy registered as 'strategy' for optimization mode.")
    else:
        # Regular mode - use EnsembleSignalStrategy but enable adaptive mode
        ensemble_config = config_loader.get("components.ensemble_strategy", {})
        if ensemble_config:
            ensemble_strategy_instance_name = "SPY_Ensemble_Strategy"
            ensemble_strategy_config_key = "components.ensemble_strategy"
            ensemble_strategy_constructor_kwargs = {
                "instance_name": ensemble_strategy_instance_name,
                "config_loader": config_loader,
                "event_bus": event_bus,
                "component_config_key": ensemble_strategy_config_key,
                "container": app_container
            }
            app_container.register_type(
                "strategy",
                EnsembleSignalStrategy,
                True,
                constructor_kwargs=ensemble_strategy_constructor_kwargs
            )
            logger.info("EnsembleSignalStrategy registered as 'strategy' for regular mode.")
        else:
            # Fallback to RegimeAdaptiveStrategy
            adaptive_strat_args = {
                "instance_name": "RegimeAdaptiveStrategy",
                "config_loader": config_loader,
                "event_bus": event_bus,
                "container": app_container,
                "component_config_key": "components.regime_adaptive_strategy"
            }
            app_container.register_type("strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=adaptive_strat_args)
            logger.info("RegimeAdaptiveStrategy registered as 'strategy' for regular mode.")

    # Register other components
    regime_detector_service_name = "MyPrimaryRegimeDetector"
    regime_detector_instance_name = "MyPrimaryRegimeDetector_Instance"
    regime_detector_config_key = "components.MyPrimaryRegimeDetector"
    regime_detector_constructor_kwargs = {
        "instance_name": regime_detector_instance_name,
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": regime_detector_config_key
    }
    app_container.register_type(
        regime_detector_service_name,
        RegimeDetector,
        True, 
        constructor_kwargs=regime_detector_constructor_kwargs
    )
    logger.info(f"RegimeDetector registered as '{regime_detector_service_name}'.")

    portfolio_args = {
        "instance_name": "BasicPortfolio", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "container": app_container,
        "component_config_key": "components.basic_portfolio"
    }
    app_container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    logger.info("BasicPortfolio registered as 'portfolio_manager'.")

    risk_manager_args = {"instance_name": "BasicRiskMan1", "config_loader": config_loader,
                         "event_bus": event_bus, "component_config_key": "components.basic_risk_manager",
                         "container": app_container, "portfolio_manager_key": "portfolio_manager"}
    app_container.register_type("risk_manager", BasicRiskManager, True, constructor_kwargs=risk_manager_args)
    logger.info("BasicRiskManager registered as 'risk_manager'.")

    sim_exec_args = {"instance_name": "SimExec_1", "config_loader": config_loader,
                     "event_bus": event_bus, "component_config_key": "components.simulated_execution_handler"}
    app_container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=sim_exec_args)

    signal_consumer_args = {"instance_name": "SignalLogger", "config_loader": config_loader,
                            "event_bus": event_bus, "component_config_key": "components.dummy_service",
                            "listen_to_event_type_str": "SIGNAL"}
    app_container.register_type("signal_consumer", DummyComponent, True, constructor_kwargs=signal_consumer_args)

    order_consumer_args = {"instance_name": "OrderLogger", "config_loader": config_loader,
                           "event_bus": event_bus, "component_config_key": "components.dummy_service",
                           "listen_to_event_type_str": "ORDER"}
    app_container.register_type("order_consumer", DummyComponent, True, constructor_kwargs=order_consumer_args)

    if args.optimize:
        logger.info("Optimization mode: Running parameter optimization instead of standard backtest.")
        run_optimization(app_container, config_loader, logger)
    else:
        logger.info("Standard backtest: Processing all available bars from the dataset.")

    app_name = config_loader.get("system.name", "ADMF-Trader-MVP")
    mode_description = "Running Optimization" if args.optimize else "Running Standard Backtest"
    logger.info(f"App Name: {app_name} - {mode_description}")

    logger.info("Bootstrap complete. Starting application logic...")

    if args.optimize:
        logger.info("Optimization complete.")
    else:
        logger.info("Running main application logic (standard backtest)...")
        try:
            run_application_logic(app_container, logger)
        except Exception as e:
            logger.error(f"Error during application logic: {e}", exc_info=True)

    logger.info("ADMF-Trader MVP finished.")

def run_application_logic(app_container, logger):
    data_handler = app_container.resolve("data_handler")
    strategy = app_container.resolve("strategy")
    
    # ENABLE ADAPTIVE MODE FOR PRODUCTION
    try:
        with open('adaptive_regime_parameters.json', 'r') as f:
            regime_params = json.load(f)['best_parameters_per_regime']
        
        print("\n" + "=" * 80)
        print("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
        print(f"Available regimes: {list(regime_params.keys())}")
        print("This will allow the strategy to switch parameters during regime changes")
        print("=" * 80 + "\n")
        
        logger.warning("!!! ENABLING ADAPTIVE MODE - REGIME-SPECIFIC PARAMETERS WILL BE APPLIED !!!")
        strategy.enable_adaptive_mode(regime_params)
        
        # Verify adaptive mode is enabled
        if hasattr(strategy, "get_adaptive_mode_status"):
            status = strategy.get_adaptive_mode_status()
            print("=== ADAPTIVE MODE STATUS ===")
            print(f"Adaptive mode enabled: {status['adaptive_mode_enabled']}")
            print(f"Parameters loaded for regimes: {status['available_regimes']}")
            print(f"Starting regime: {status['current_regime']}")
            print("=" * 80 + "\n")
            
    except Exception as e:
        print(f"Warning: Could not enable adaptive mode: {e}")
        print("Running in standard mode...")
    
    regime_detector = app_container.resolve("MyPrimaryRegimeDetector")
    portfolio_manager = app_container.resolve("portfolio_manager")
    risk_manager = app_container.resolve("risk_manager")
    execution_handler = app_container.resolve("execution_handler")
    signal_consumer = app_container.resolve("signal_consumer")
    order_consumer = app_container.resolve("order_consumer")

    components_to_manage = [
        data_handler, strategy, regime_detector, portfolio_manager, 
        risk_manager, execution_handler, signal_consumer, order_consumer
    ]

    for comp in components_to_manage:
        if isinstance(comp, DummyComponent) or hasattr(comp, 'setup'):
            if hasattr(comp, 'get_state') and comp.get_state() == getattr(comp, 'STATE_CREATED', None):
                logger.info(f"--- Setting up {comp.name if hasattr(comp, 'name') else type(comp).__name__} ---")
                comp.setup()
                if hasattr(comp, 'get_state') and comp.get_state() == getattr(comp, 'STATE_FAILED', None):
                    raise ComponentError(f"Component '{comp.name}' failed to setup.")
            else:
                logger.warning(f"Item {type(comp).__name__} is not a BaseComponent, skipping setup for it directly.")

    if data_handler and isinstance(data_handler, CSVDataHandler) and hasattr(data_handler, "set_active_dataset"):
        if hasattr(data_handler, 'test_df_exists_and_is_not_empty') and data_handler.test_df_exists_and_is_not_empty:
            logger.info("Production validation: Setting active dataset to 'test' (same as optimization test set).")
            data_handler.set_active_dataset("test")
        else:
            logger.info("Standard run: Setting active dataset to 'full' in DataHandler (respects --bars if provided).")
            data_handler.set_active_dataset("full")

    for comp in components_to_manage:
         if hasattr(comp, 'get_state') and comp.get_state() == getattr(comp, 'STATE_INITIALIZED', None):
            logger.info(f"--- Starting {comp.name if hasattr(comp, 'name') else type(comp).__name__} ---")
            comp.start() 
            if hasattr(comp, 'get_state') and comp.get_state() == getattr(comp, 'STATE_FAILED', None):
                raise ComponentError(f"Component '{comp.name}' failed to start.")

    logger.info("All components started. Data processing handled by components automatically.")

    logger.info("--- Initiating application shutdown sequence (standard backtest) ---")

    # Close all open positions before calculating final portfolio value
    try:
        if portfolio_manager and hasattr(portfolio_manager, 'close_all_positions'):
            portfolio_manager.close_all_positions()
        else:
            logger.warning("PortfolioManager not available or not in a valid state to close open positions.")
    except Exception as e:
        logger.warning(f"Error closing positions: {e}")

    # Stop components in reverse order
    for comp in reversed(components_to_manage):
        if hasattr(comp, 'stop'):
            comp.stop()

    logger.info("Main application logic (standard backtest) finished.")

def run_optimization(app_container, config_loader, logger):
    # Implementation would go here
    pass

if __name__ == "__main__":
    main()