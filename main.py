# main.py
import argparse
import logging
import sys
import datetime
from typing import Optional, Dict, Any # Ensure Dict and Any are imported for type hints

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging
from src.core.exceptions import ConfigurationError, ADMFTraderError, ComponentError, DependencyNotFoundError
from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.event_bus import EventBus
from src.core.container import Container # Ensure Container is imported for type hints

# Application Component imports
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.ma_strategy import MAStrategy
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio
from src.core.dummy_component import DummyComponent
# Import the BasicOptimizer
from src.strategy.optimization.basic_optimizer import BasicOptimizer


logger = logging.getLogger(__name__)

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
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    logger.info(f"Attempting to load configuration from: {config_path}")
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except ConfigurationError as e:
        logger.critical(f"CRITICAL: Config error - {e}. Path: {config_path}", exc_info=True)
        sys.exit(1)

    setup_logging(config_loader)
    logger.info("Initial configuration loaded and logging system configured.")

    container = Container()
    container.register_instance("config_loader", config_loader)
    event_bus = EventBus()
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)

    # --- Component Registration (Common for both modes) ---
    # max_bars_to_process (from --bars) is passed to CSVDataHandler.
    # CSVDataHandler's setup will use this to limit the initial data load,
    # before any train/test splitting.
    csv_args = {"instance_name": "SPY_CSV_Loader", "config_loader": config_loader,
                "event_bus": event_bus, "component_config_key": "components.data_handler_csv",
                "max_bars": max_bars_to_process }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)

    ma_strat_args = {"instance_name": "SPY_MA_Cross", "config_loader": config_loader,
                     "event_bus": event_bus, "component_config_key": "components.ma_crossover_strategy"}
    container.register_type("strategy", MAStrategy, True, constructor_kwargs=ma_strat_args)

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
            optimizer.setup()
            if optimizer.get_state() == BaseComponent.STATE_INITIALIZED:
                optimizer.start()
                logger.info("--- Starting Grid Search ---")
                
                optimization_results: Optional[Dict[str, Any]] = optimizer.run_grid_search()
                
                if optimization_results:
                    best_train_params = optimization_results.get("best_parameters_on_train")
                    best_train_metric = optimization_results.get("best_training_metric_value")
                    metric_name_optimized = getattr(optimizer, '_metric_to_optimize', 'configured_metric') # Safely get metric name
                    
                    test_metric_for_best = optimization_results.get("test_set_metric_value_for_best_params")

                    if best_train_params is not None and best_train_metric is not None:
                        logger.info(
                            f"OPTIMIZATION COMPLETE:\n"
                            f"  Best Parameters (from Training): {best_train_params}\n"
                            f"  Best Training Metric ('{metric_name_optimized}'): {best_train_metric:.4f}\n"
                            f"  Test Set Metric for Best Params ('{metric_name_optimized}'): {test_metric_for_best if test_metric_for_best is not None else 'N/A (No test data or test failed)'}"
                        )
                    else:
                        logger.info("OPTIMIZATION COMPLETE: No best parameters found during training phase or training failed to produce a metric.")
                else:
                    logger.info("OPTIMIZATION FAILED: Optimizer run_grid_search() returned no results (None).")
                
                optimizer.stop()
            else:
                logger.error("Failed to initialize optimizer. Cannot run optimization.")

        except (DependencyNotFoundError, ADMFTraderError, Exception) as e:
            logger.critical(f"CRITICAL ERROR during optimization setup or run: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Normal Backtest Mode
        if max_bars_to_process is not None and max_bars_to_process > 0:
             logger.info(f"Standard backtest: Using first {max_bars_to_process} bars from the dataset.")
        else:
            logger.info("Standard backtest: Processing all available bars from the dataset.")

        logger.info(f"App Name: {config_loader.get('system.name', 'N/A')} - Running Standard Backtest")
        logger.info("Bootstrap complete. Starting application logic...")
        try:
            run_application_logic(container)
        except (DependencyNotFoundError, ADMFTraderError, Exception) as e:
            logger.critical(f"CRITICAL ERROR during application logic: {e}", exc_info=True)
            sys.exit(1)

    logger.info("ADMF-Trader MVP finished.")


def run_application_logic(app_container: Container):
    logger.info("Running main application logic (standard backtest)...")
    components_to_manage = []
    
    data_handler: Optional[CSVDataHandler] = None
    strategy: Optional[MAStrategy] = None
    portfolio_manager: Optional[BasicPortfolio] = None
    risk_manager: Optional[BasicRiskManager] = None 
    execution_handler: Optional[SimulatedExecutionHandler] = None
    signal_logger: Optional[DummyComponent] = None
    order_logger: Optional[DummyComponent] = None

    try:
        data_handler = app_container.resolve("data_handler")
        strategy = app_container.resolve("strategy") 
        portfolio_manager = app_container.resolve("portfolio_manager")
        risk_manager = app_container.resolve("risk_manager") 
        execution_handler = app_container.resolve("execution_handler")
        signal_logger = app_container.resolve("signal_consumer")
        order_logger = app_container.resolve("order_consumer")
        
        logger.info("DataHandler, Strategy, PortfolioManager, RiskManager, ExecutionHandler, and Loggers resolved.")

        components_to_manage = [
            data_handler, strategy, portfolio_manager, risk_manager, execution_handler, 
            signal_logger, order_logger   
        ]

        for comp in components_to_manage:
            if comp is None: 
                logger.error("A core component was not resolved. Aborting setup.")
                raise ComponentError("Core component resolution failed.")
            logger.info(f"--- Setting up {comp.name} ---")
            comp.setup()
            logger.info(f"State of '{comp.name}' after setup: {comp.get_state()}")
            if comp.get_state() == BaseComponent.STATE_FAILED:
                raise ComponentError(f"Component '{comp.name}' failed to setup.")

        # Explicitly set active dataset for standard run AFTER data_handler setup
        if data_handler and hasattr(data_handler, "set_active_dataset"):
            logger.info("Standard run: Setting active dataset to 'full' in DataHandler (respects --bars).")
            data_handler.set_active_dataset("full") 

        for comp in components_to_manage:
            if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                logger.info(f"--- Starting {comp.name} ---")
                comp.start() 
                logger.info(f"State of '{comp.name}' after start: {comp.get_state()}")
                if comp.get_state() == BaseComponent.STATE_FAILED:
                     raise ComponentError(f"Component '{comp.name}' failed to start.")
            else:
                logger.warning(f"Skipping start for component '{comp.name}' as it's not in INITIALIZED state (current: {comp.get_state()}).")
        
        logger.info("All components started. Event flow: BAR -> SIGNAL -> RiskManager -> ORDER -> ExecutionHandler -> FILL -> Portfolio.")

    except (DependencyNotFoundError, ComponentError) as e: 
        logger.error(f"Error during application logic: {e}", exc_info=True)
        raise 
    except Exception as e:
        logger.error(f"An unexpected error occurred during application logic: {e}", exc_info=True)
        raise 
    finally:
        logger.info("--- Initiating application shutdown sequence (standard backtest) ---")

        if portfolio_manager and data_handler : 
            pm_state_valid = hasattr(portfolio_manager, 'get_state') and portfolio_manager.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED]
            dh_valid_for_timestamp = hasattr(data_handler, 'get_last_timestamp') and \
                                     hasattr(data_handler, 'get_state') and \
                                     data_handler.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED]

            if pm_state_valid and dh_valid_for_timestamp:
                last_bar_timestamp = data_handler.get_last_timestamp()
                
                if last_bar_timestamp:
                    logger.info(f"--- Closing all open positions using last bar timestamp: {last_bar_timestamp} ---")
                    portfolio_manager.close_all_open_positions(last_bar_timestamp)
                else:
                    fallback_ts = portfolio_manager.get_last_processed_timestamp() or datetime.datetime.now(datetime.timezone.utc)
                    logger.warning(
                        f"Could not get last bar timestamp from DataHandler. "
                        f"Using fallback timestamp for closing positions: {fallback_ts}"
                    )
                    portfolio_manager.close_all_open_positions(fallback_ts)
                logger.info("--- Finished attempt to close open positions. Proceeding with component stop. ---")
            elif portfolio_manager and not pm_state_valid:
                 logger.warning("PortfolioManager not in a valid state for closing positions.")
            elif data_handler and not dh_valid_for_timestamp:
                 logger.warning("DataHandler not in a valid state to get last timestamp.")
            elif not portfolio_manager:
                 logger.warning("PortfolioManager instance not available for closing open positions.")
            elif not data_handler:
                 logger.warning("DataHandler instance not available for closing open positions.")

        for comp in reversed(components_to_manage):
            if comp and isinstance(comp, BaseComponent) and hasattr(comp, 'stop') and callable(comp.stop):
                if hasattr(comp, 'get_state') and comp.get_state() != BaseComponent.STATE_CREATED : 
                    logger.info(f"--- Stopping {comp.name} (Current State: {comp.get_state()}) ---")
                    try:
                        comp.stop()
                        logger.info(f"State of '{comp.name}' after stop: {comp.get_state()}")
                    except Exception as e:
                        logger.error(f"Error stopping component '{comp.name}': {e}", exc_info=True)
                elif hasattr(comp, 'name'):
                    logger.warning(f"Component '{comp.name}' was in CREATED state or similar; minimal stop if any.")
            elif comp and hasattr(comp, 'name'): 
                 logger.warning(f"Item '{comp.name}' may not be a standard component or was not fully operational.")
            elif comp: 
                 logger.warning(f"Attempting to stop an item of type '{type(comp).__name__}' that may not be a fully operational component.")
    
    logger.info("Main application logic (standard backtest) finished.")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
    main()
