# main.py
import argparse
import logging
import sys
# from pathlib import Path # Keep for general utility

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.exceptions import ConfigurationError, ADMFTraderError, ComponentError, DependencyNotFoundError
from src.core.component import BaseComponent 
from src.core.event import Event, EventType 
from src.core.event_bus import EventBus
from src.core.container import Container

# Application Component imports
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.ma_strategy import MAStrategy
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio # UPDATED IMPORT PATH
from src.core.dummy_component import DummyComponent # For SIGNAL and ORDER logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="ADMF-Trader Application")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--bars", type=int, default=None, help="Number of bars to process. Default is all."
    )
    args = parser.parse_args()
    config_path = args.config
    max_bars_to_process = args.bars
    
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    
    logger.info(f"Attempting to load configuration from: {config_path}")
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except ConfigurationError as e:
        logging.critical(f"CRITICAL: Config error - {e}. Path: {config_path}", exc_info=True)
        print(f"CRITICAL: Config error - {e}. Check config: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    setup_logging(config_loader)
    logger.info("Initial configuration loaded and logging system configured.")
    if max_bars_to_process is not None and max_bars_to_process > 0:
        logger.info(f"Limiting data processing to the first {max_bars_to_process} bars.")

    container = Container()
    try:
        container.register_instance("config_loader", config_loader)
        event_bus = EventBus()
        container.register_instance("event_bus", event_bus)

        csv_args = {"instance_name": "SPY_CSV_Loader", "config_loader": config_loader,
                    "event_bus": event_bus, "component_config_key": "components.data_handler_csv",
                    "max_bars": max_bars_to_process}
        container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)

        ma_strat_args = {"instance_name": "SPY_MA_Cross", "config_loader": config_loader,
                         "event_bus": event_bus, "component_config_key": "components.ma_crossover_strategy"}
        container.register_type("strategy", MAStrategy, True, constructor_kwargs=ma_strat_args)

        sim_exec_args = {"instance_name": "SimExec_1", "config_loader": config_loader,
                         "event_bus": event_bus, "component_config_key": "components.simulated_execution_handler"}
        container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=sim_exec_args)
        
        # BasicPortfolio registration
        portfolio_args = {"instance_name": "BasicPortfolio", "config_loader": config_loader,
                          "event_bus": event_bus, "component_config_key": "components.basic_portfolio"}
        container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
        logger.info("BasicPortfolio registered.")

        # DummyComponent for SIGNAL events
        signal_consumer_args = {"instance_name": "SignalLogger", "config_loader": config_loader,
                                "event_bus": event_bus, "component_config_key": "components.dummy_service",
                                "listen_to_event_type_str": "SIGNAL"}
        container.register_type("signal_consumer", DummyComponent, True, constructor_kwargs=signal_consumer_args)
        
        # DummyComponent for ORDER events
        order_consumer_args = {"instance_name": "OrderLogger", "config_loader": config_loader,
                               "event_bus": event_bus, "component_config_key": "components.dummy_service", 
                               "listen_to_event_type_str": "ORDER"}
        container.register_type("order_consumer", DummyComponent, True, constructor_kwargs=order_consumer_args)
        # Note: FillLogger DummyComponent is now replaced by BasicPortfolio for consuming FILL events.

        logger.info(f"App Name: {config_loader.get('system.name', 'N/A')}")
        logger.info("Bootstrap complete. Starting application logic...")
        run_application_logic(container)
        logger.info("ADMF-Trader MVP finished.")

    except DependencyNotFoundError as e:
        logger.critical(f"CRITICAL ERROR during bootstrap: Unresolved dependency - {e}", exc_info=True)
        print(f"CRITICAL: Unresolved dependency - {e}. Please check registrations.", file=sys.stderr)
        sys.exit(1)
    except ADMFTraderError as e: 
        logger.critical(f"CRITICAL APPLICATION ERROR: {e}", exc_info=True)
        print(f"CRITICAL: Application error - {e}. Please check the logs.", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        logger.critical(f"UNEXPECTED CRITICAL ERROR during bootstrap: {e}", exc_info=True)
        print(f"CRITICAL: Unexpected error during bootstrap - {e}. Please check the logs.", file=sys.stderr)
        sys.exit(1)


def run_application_logic(app_container: Container):
    logger.info("Running main application logic...")
    components_to_manage = []
    try:
        data_handler = app_container.resolve("data_handler")
        strategy = app_container.resolve("strategy") 
        execution_handler = app_container.resolve("execution_handler")
        portfolio_manager = app_container.resolve("portfolio_manager") # Resolve BasicPortfolio
        
        signal_logger = app_container.resolve("signal_consumer")
        order_logger = app_container.resolve("order_consumer")
        
        logger.info("DataHandler, Strategy, ExecutionHandler, PortfolioManager, and Loggers resolved.")

        components_to_manage = [
            data_handler, 
            strategy, 
            execution_handler, 
            portfolio_manager, # Add portfolio to the list
            signal_logger, 
            order_logger   
        ]

        for comp in components_to_manage:
            logger.info(f"--- Setting up {comp.name} ---")
            comp.setup()
            logger.info(f"State of '{comp.name}' after setup: {comp.get_state()}")
            if comp.get_state() == BaseComponent.STATE_FAILED:
                raise ComponentError(f"Component '{comp.name}' failed to setup.")

        for comp in components_to_manage:
            if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                logger.info(f"--- Starting {comp.name} ---")
                comp.start() 
                logger.info(f"State of '{comp.name}' after start: {comp.get_state()}")
                if comp.get_state() == BaseComponent.STATE_FAILED:
                     raise ComponentError(f"Component '{comp.name}' failed to start.")
            else:
                logger.warning(f"Skipping start for component '{comp.name}' as it's not in INITIALIZED state (current: {comp.get_state()}).")
        
        logger.info("All components started. Event flow: BAR -> SIGNAL -> ORDER -> FILL -> PORTFOLIO.")

    except DependencyNotFoundError as e:
        logger.error(f"Dependency not found during application logic: {e}", exc_info=True)
    except ComponentError as e:
        logger.error(f"A component error occurred during application logic: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during application logic: {e}", exc_info=True)
    finally:
        logger.info("--- Initiating application shutdown sequence ---")
        for comp in reversed(components_to_manage):
            if hasattr(comp, 'stop') and callable(comp.stop) and hasattr(comp, 'name'):
                logger.info(f"--- Stopping {comp.name} ---")
                comp.stop()
                logger.info(f"State of '{comp.name}' after stop: {comp.get_state()}")
            elif hasattr(comp, 'name'):
                 logger.warning(f"Component '{comp.name}' may not have a standard stop method or is not a full component.")
            else:
                 logger.warning(f"Attempting to stop an item that may not be a component: {type(comp)}")
    
    logger.info("Main application logic finished.")


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
    main()
