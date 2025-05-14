# main.py
import argparse
import logging
import sys
import datetime # Keep for type hints and potential fallback timestamps
from typing import Optional, Dict, Any # Ensure Dict and Any are imported for type hints

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.exceptions import ConfigurationError, ADMFTraderError, ComponentError, DependencyNotFoundError
from src.core.component import BaseComponent
from src.core.event import Event, EventType # Keep for type hints if used by components directly
from src.core.event_bus import EventBus
from src.core.container import Container

# Application Component imports
from src.data.csv_data_handler import CSVDataHandler
# from src.strategy.ma_strategy import MAStrategy # Original MAStrategy, no longer primary strategy
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy # Import the new ensemble strategy
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.core.dummy_component import DummyComponent
from src.strategy.optimization.basic_optimizer import BasicOptimizer


logger = logging.getLogger(__name__) # Changed from __main__ to __name__ for module context

def main():
    parser = argparse.ArgumentParser(description="ADMF-Trader Application")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--bars", type=int, default=None, help="Number of bars to process. Default is all."
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Run in optimization mode."
    )
    args = parser.parse_args()
    config_path = args.config
    max_bars_to_process = args.bars
    run_optimization_mode = args.optimize

    # --- Basic Setup (Config, Logging, Container, EventBus) ---
    if not logging.getLogger().hasHandlers(): # Ensure root logger setup if not already
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    logger.info(f"Attempting to load configuration from: {config_path}")
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except ConfigurationError as e:
        logger.critical(f"CRITICAL: Config error - {e}. Path: {config_path}", exc_info=True)
        sys.exit(1)

    # Ensure logging is setup using the config loader AFTER it's successfully created
    setup_logging(config_loader) # setup_logging likely uses config_loader.get('logging.level')
    logger.info("Initial configuration loaded and logging system configured.")

    container = Container()
    container.register_instance("config_loader", config_loader)
    event_bus = EventBus()
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container) # Register container itself if components need it

    # --- Component Registration (Common for both modes) ---
    csv_args = {"instance_name": "SPY_CSV_Loader", "config_loader": config_loader,
                "event_bus": event_bus, "component_config_key": "components.data_handler_csv",
                "max_bars": max_bars_to_process }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)

    # Updated Strategy Registration to use EnsembleSignalStrategy
    ensemble_strat_args = {
        "instance_name": "SPY_Ensemble_Strategy", # A unique name for this instance
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": "components.ensemble_strategy" # Points to the new config section
    }
    container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_strat_args)
    logger.info("EnsembleSignalStrategy registered as 'strategy'.")

    portfolio_args = {"instance_name": "BasicPortfolio", "config_loader": config_loader,
                      "event_bus": event_bus, "component_config_key": "components.basic_portfolio"}
    container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    logger.info("BasicPortfolio registered as 'portfolio_manager'.")

    risk_manager_args = {"instance_name": "BasicRiskMan1", "config_loader": config_loader,
                         "event_bus": event_bus, "component_config_key": "components.basic_risk_manager",
                         "container": container, "portfolio_manager_key": "portfolio_manager"}
    container.register_type("risk_manager", BasicRiskManager, True, constructor_kwargs=risk_manager_args)
    logger.info("BasicRiskManager registered as 'risk_manager'.")

    sim_exec_args = {"instance_name": "SimExec_1", "config_loader": config_loader,
                     "event_bus": event_bus, "component_config_key": "components.simulated_execution_handler"}
    container.register_type("execution_handler", SimulatedExecutionHandler, True, constructor_kwargs=sim_exec_args)

    # Dummy signal/order loggers (optional, can be removed if not needed for debugging)
    signal_consumer_args = {"instance_name": "SignalLogger", "config_loader": config_loader,
                            "event_bus": event_bus, "component_config_key": "components.dummy_service",
                            "listen_to_event_type_str": "SIGNAL"}
    container.register_type("signal_consumer", DummyComponent, True, constructor_kwargs=signal_consumer_args)

    order_consumer_args = {"instance_name": "OrderLogger", "config_loader": config_loader,
                           "event_bus": event_bus, "component_config_key": "components.dummy_service",
                           "listen_to_event_type_str": "ORDER"}
    container.register_type("order_consumer", DummyComponent, True, constructor_kwargs=order_consumer_args)


    # --- Mode-Specific Logic ---
    if run_optimization_mode:
        logger.info("OPTIMIZATION MODE DETECTED.")
        if max_bars_to_process is not None:
            logger.info(f"Optimizer will run each parameter set using an initial dataset of up to {max_bars_to_process} bars (before train/test split).")
        else:
            logger.info("Optimizer will run each parameter set on the full dataset (as configured in CSVDataHandler, before train/test split).")

        optimizer_args = {"instance_name": "GridOptimizer", "config_loader": config_loader,
                          "event_bus": event_bus, "component_config_key": "components.optimizer",
                          "container": container}
        container.register_type("optimizer_service", BasicOptimizer, True, constructor_kwargs=optimizer_args)
        logger.info("BasicOptimizer registered as 'optimizer_service'.")

        try:
            optimizer: BasicOptimizer = container.resolve("optimizer_service")
            # The optimizer's setup method should resolve its dependencies internally from the container
            optimizer.setup() 
            if optimizer.get_state() == BaseComponent.STATE_INITIALIZED:
                optimizer.start()
                logger.info("--- Starting Grid Search ---")
                
                optimization_results: Optional[Dict[str, Any]] = optimizer.run_grid_search()
                
                if optimization_results:
                    best_train_params = optimization_results.get("best_parameters_on_train")
                    best_train_metric_val = optimization_results.get("best_training_metric_value")
                    # Ensure metric_name_optimized is fetched correctly from the optimizer instance
                    metric_name_optimized = getattr(optimizer, '_metric_to_optimize', 
                                                    config_loader.get('components.optimizer.metric_to_optimize', 'get_final_portfolio_value'))
                    
                    test_metric_for_best = optimization_results.get("test_set_metric_value_for_best_params")

                    result_log_message = "OPTIMIZATION COMPLETE:\n"
                    if best_train_params is not None and best_train_metric_val is not None:
                        result_log_message += f"  Best Parameters (from Training): {best_train_params}\n"
                        result_log_message += f"  Best Training Metric ('{metric_name_optimized}'): {best_train_metric_val:.4f}\n"
                    else:
                        result_log_message += "  No best parameters found during training phase or training failed to produce a metric.\n"
                    
                    if test_metric_for_best is not None:
                         result_log_message += f"  Test Set Metric for Best Params ('{metric_name_optimized}'): {test_metric_for_best:.4f}"
                    else:
                         result_log_message += f"  Test Set Metric for Best Params ('{metric_name_optimized}'): N/A (No test data or test failed)"
                    logger.info(result_log_message)
                else:
                    logger.info("OPTIMIZATION FAILED: Optimizer run_grid_search() returned no results (None).")
                
                optimizer.stop() # Stop the optimizer after the run
            else:
                logger.error("Failed to initialize optimizer. Cannot run optimization.")

        except (DependencyNotFoundError, ADMFTraderError, ComponentError, Exception) as e: # Added ComponentError
            logger.critical(f"CRITICAL ERROR during optimization setup or run: {e}", exc_info=True)
            sys.exit(1) # Exit on critical error
    else:
        # Normal Backtest Mode
        if max_bars_to_process is not None and max_bars_to_process > 0:
             logger.info(f"Standard backtest: Using first {max_bars_to_process} bars from the dataset.")
        else:
            logger.info("Standard backtest: Processing all available bars from the dataset.")

        system_name = config_loader.get('system.name', 'ADMF-Trader') # Get system name safely
        logger.info(f"App Name: {system_name} - Running Standard Backtest")
        logger.info("Bootstrap complete. Starting application logic...")
        try:
            run_application_logic(container) # Pass the container
        except (DependencyNotFoundError, ADMFTraderError, ComponentError, Exception) as e: # Added ComponentError
            logger.critical(f"CRITICAL ERROR during application logic: {e}", exc_info=True)
            sys.exit(1) # Exit on critical error

    logger.info("ADMF-Trader MVP finished.")


def run_application_logic(app_container: Container): # Added type hint for app_container
    logger.info("Running main application logic (standard backtest)...")
    
    # Resolve all components needed for a standard run
    # Type hints for resolved components for clarity
    data_handler: Optional[CSVDataHandler] = None
    strategy: Optional[EnsembleSignalStrategy] = None # Changed to EnsembleSignalStrategy
    portfolio_manager: Optional[BasicPortfolio] = None
    risk_manager: Optional[BasicRiskManager] = None 
    execution_handler: Optional[SimulatedExecutionHandler] = None
    signal_logger_comp: Optional[DummyComponent] = None # Renamed to avoid conflict with logger module
    order_logger_comp: Optional[DummyComponent] = None   # Renamed

    components_to_manage: List[Optional[BaseComponent]] = [] # List to hold resolved components

    try:
        # Resolve components from the container
        data_handler = app_container.resolve("data_handler")
        strategy = app_container.resolve("strategy") 
        portfolio_manager = app_container.resolve("portfolio_manager")
        risk_manager = app_container.resolve("risk_manager") 
        execution_handler = app_container.resolve("execution_handler")
        signal_logger_comp = app_container.resolve("signal_consumer")
        order_logger_comp = app_container.resolve("order_consumer")
        
        logger.info("DataHandler, Strategy, PortfolioManager, RiskManager, ExecutionHandler, and Loggers resolved.")

        components_to_manage = [
            data_handler, strategy, portfolio_manager, risk_manager, execution_handler, 
            signal_logger_comp, order_logger_comp   
        ]

        # Setup phase
        for comp in components_to_manage:
            if comp is None: # Should ideally not happen if resolve is successful
                logger.error("A core component was not resolved. Aborting setup.")
                raise ComponentError("Core component resolution failed during setup.")
            if isinstance(comp, BaseComponent): # Ensure it's a component before calling methods
                logger.info(f"--- Setting up {comp.name} ---")
                comp.setup()
                logger.info(f"State of '{comp.name}' after setup: {comp.get_state()}")
                if comp.get_state() == BaseComponent.STATE_FAILED:
                    raise ComponentError(f"Component '{comp.name}' failed to setup.")
            else:
                logger.warning(f"Item {type(comp).__name__} is not a BaseComponent, skipping setup for it directly.")


        # Explicitly set active dataset for standard run AFTER data_handler setup
        if data_handler and isinstance(data_handler, CSVDataHandler) and hasattr(data_handler, "set_active_dataset"):
            logger.info("Standard run: Setting active dataset to 'full' in DataHandler (respects --bars if provided).")
            data_handler.set_active_dataset("full") # "full" means use all data loaded (which was limited by --bars)

        # Start phase
        for comp in components_to_manage:
             if isinstance(comp, BaseComponent):
                if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                    logger.info(f"--- Starting {comp.name} ---")
                    comp.start() 
                    logger.info(f"State of '{comp.name}' after start: {comp.get_state()}")
                    if comp.get_state() == BaseComponent.STATE_FAILED:
                        raise ComponentError(f"Component '{comp.name}' failed to start.")
                else:
                    logger.warning(f"Skipping start for component '{comp.name}' as it's not in INITIALIZED state (current: {comp.get_state()}).")

        logger.info("All operational components started. Event flow: BAR -> Strategy -> RiskManager -> ORDER -> ExecutionHandler -> FILL -> Portfolio.")
        # The DataHandler (CSVDataHandler) will start publishing BAR events after its start() method.

    except (DependencyNotFoundError, ComponentError, ADMFTraderError) as e: 
        logger.error(f"Error during application logic: {e}", exc_info=True)
        # No need to re-raise if sys.exit is called in main's except block
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during application logic: {e}", exc_info=True)
    finally:
        logger.info("--- Initiating application shutdown sequence (standard backtest) ---")

        # Close positions before stopping components
        if portfolio_manager and isinstance(portfolio_manager, BasicPortfolio) and \
           portfolio_manager.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED]:
            
            last_timestamp_for_close = None
            if data_handler and isinstance(data_handler, CSVDataHandler) and \
               data_handler.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED] and \
               hasattr(data_handler, 'get_last_timestamp'):
                last_timestamp_for_close = data_handler.get_last_timestamp()

            if not last_timestamp_for_close and hasattr(portfolio_manager, 'get_last_processed_timestamp'):
                 last_timestamp_for_close = portfolio_manager.get_last_processed_timestamp()
            
            if not last_timestamp_for_close: # Absolute fallback
                last_timestamp_for_close = datetime.datetime.now(datetime.timezone.utc)
                logger.warning(f"Using current time as fallback for closing positions: {last_timestamp_for_close}")
            
            logger.info(f"--- Closing all open positions at timestamp: {last_timestamp_for_close} ---")
            portfolio_manager.close_all_open_positions(last_timestamp_for_close)
            logger.info("--- Finished attempt to close open positions. ---")
        else:
            logger.warning("PortfolioManager not available or not in a valid state to close open positions.")

        # Stop components in reverse order of start/setup
        for comp in reversed(components_to_manage):
            if comp and isinstance(comp, BaseComponent) and hasattr(comp, 'stop') and callable(comp.stop):
                # Only stop if it was likely started or setup
                if hasattr(comp, 'get_state') and comp.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_STOPPED, BaseComponent.STATE_FAILED]: 
                    logger.info(f"--- Stopping {comp.name} (Current State: {comp.get_state()}) ---")
                    try:
                        comp.stop()
                        logger.info(f"State of '{comp.name}' after stop: {comp.get_state()}")
                    except Exception as e: # Catch errors during individual component stop
                        logger.error(f"Error stopping component '{comp.name}': {e}", exc_info=True)
                elif hasattr(comp, 'name'): # Log if component was not in a stoppable state
                    logger.debug(f"Component '{comp.name}' was in {comp.get_state()} state; stop action might be minimal or skipped.")
            elif comp and hasattr(comp, 'name'): 
                 logger.warning(f"Item '{comp.name}' ({type(comp).__name__}) may not be a standard BaseComponent or wasn't fully operational to stop.")
            elif comp: 
                 logger.warning(f"Attempting to handle shutdown for an item of type '{type(comp).__name__}' that may not be a BaseComponent.")
    
    logger.info("Main application logic (standard backtest) finished.")

if __name__ == "__main__":
    # Ensure logging is configured at the earliest point possible for __main__
    # This basicConfig is a fallback if setup_logging isn't reached or fails early.
    # setup_logging called within main() will refine this with config.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO, # Default level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)] # Explicitly set handler
        )
    main()
