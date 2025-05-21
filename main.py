# main.py
import argparse
import logging
import sys
import datetime 
from typing import Optional, Dict, Any, List 

# Core imports
from src.core.config import SimpleConfigLoader
from src.core.logging_setup import setup_logging, LOG_LEVEL_STRINGS
from src.core.exceptions import ConfigurationError, ADMFTraderError, ComponentError, DependencyNotFoundError
from src.core.component import BaseComponent
from src.core.event import Event, EventType 
from src.core.event_bus import EventBus
from src.core.container import Container

# Application Component imports
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.strategy.regime_detector import RegimeDetector 
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio # Ensure this is the updated version
from src.core.dummy_component import DummyComponent
from src.strategy.optimization.basic_optimizer import BasicOptimizer
from src.strategy.optimization.enhanced_optimizer import EnhancedOptimizer
from src.strategy.optimization.genetic_optimizer import GeneticOptimizer
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy


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
        "--optimize", action="store_true", help="Run in grid search optimization mode for all parameters."
    )
    parser.add_argument(
        "--optimize-ma", action="store_true", help="Run grid search optimization only for MA rule parameters."
    )
    parser.add_argument(
        "--optimize-rsi", action="store_true", help="Run grid search optimization only for RSI rule parameters."
    )
    parser.add_argument(
        "--optimize-seq", action="store_true", help="Run sequential grid search optimization (each rule in isolation, one after another)."
    )
    parser.add_argument(
        "--optimize-joint", action="store_true", help="Run joint grid search optimization (full Cartesian product of all rule parameters)."
    )
    parser.add_argument(
        "--genetic-optimize", action="store_true", help="Run genetic algorithm to optimize rule weights after grid search."
    )
    parser.add_argument(
        "--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Override the log level defined in the config file."
    )
    parser.add_argument(
        "--debug-log", type=str, default=None, 
        help="Enable detailed DEBUG logging to the specified file. Overwrites file each run."
    )
    args = parser.parse_args()
    config_path = args.config
    max_bars_to_process = args.bars
    
    # Optimization modes
    run_optimization_mode = args.optimize
    run_optimize_ma = args.optimize_ma
    run_optimize_rsi = args.optimize_rsi
    run_optimize_seq = args.optimize_seq
    run_optimize_joint = args.optimize_joint
    run_genetic_optimization = args.genetic_optimize
    
    # If any optimization is enabled, also set general optimization flag
    if run_optimize_ma or run_optimize_rsi or run_optimize_seq or run_optimize_joint:
        run_optimization_mode = True
        
    cmd_log_level = args.log_level
    debug_log_file = args.debug_log

    # Set up minimal logging until we load the config
    if not logging.getLogger().hasHandlers():
        # Use WARNING as default to minimize output until proper config is loaded
        minimal_level = logging.WARNING
        if cmd_log_level:
            minimal_level = LOG_LEVEL_STRINGS.get(cmd_log_level.upper(), logging.WARNING)
        logging.basicConfig(level=minimal_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)

    logger.info(f"Attempting to load configuration from: {config_path}")
    try:
        config_loader = SimpleConfigLoader(config_file_path=config_path)
    except ConfigurationError as e:
        logger.critical(f"CRITICAL: Config error - {e}. Path: {config_path}", exc_info=True)
        sys.exit(1)

    # Pass optimization_mode flag based on command line args
    setup_logging(config_loader, cmd_log_level, optimization_mode=run_optimization_mode, debug_file=debug_log_file)
    logger.debug("Initial configuration loaded and logging system configured.")

    container = Container()
    container.register_instance("config_loader", config_loader)
    event_bus = EventBus()
    container.register_instance("event_bus", event_bus)
    container.register_instance("container", container)

    csv_args = {"instance_name": "SPY_CSV_Loader", "config_loader": config_loader,
                "event_bus": event_bus, "component_config_key": "components.data_handler_csv",
                "max_bars": max_bars_to_process }
    container.register_type("data_handler", CSVDataHandler, True, constructor_kwargs=csv_args)

    # If we're optimizing, use the regular strategy
    if run_optimization_mode:
        ensemble_strat_args = {
            "instance_name": "SPY_Ensemble_Strategy",
            "config_loader": config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.ensemble_strategy",
            "container": container  # Pass container to allow access to RegimeDetector
        }
        container.register_type("strategy", EnsembleSignalStrategy, True, constructor_kwargs=ensemble_strat_args)
        logger.info("EnsembleSignalStrategy registered as 'strategy' for optimization mode.")
    else:
        # For regular runs, use the RegimeAdaptiveStrategy
        adaptive_strat_args = {
            "instance_name": "RegimeAdaptiveStrategy",
            "config_loader": config_loader,
            "event_bus": event_bus,
            "container": container,
            "component_config_key": "components.regime_adaptive_strategy"
        }
        container.register_type("strategy", RegimeAdaptiveStrategy, True, constructor_kwargs=adaptive_strat_args)
        logger.info("RegimeAdaptiveStrategy registered as 'strategy' for regular mode.")

    regime_detector_service_name = "MyPrimaryRegimeDetector"
    regime_detector_instance_name = "MyPrimaryRegimeDetector_Instance"
    regime_detector_config_key = "components.MyPrimaryRegimeDetector"
    regime_detector_constructor_kwargs = {
        "instance_name": regime_detector_instance_name,
        "config_loader": config_loader,
        "event_bus": event_bus,
        "component_config_key": regime_detector_config_key
    }
    container.register_type(
        regime_detector_service_name,
        RegimeDetector,
        True, 
        constructor_kwargs=regime_detector_constructor_kwargs
    )
    logger.info(f"RegimeDetector registered as '{regime_detector_service_name}'.")

    # --- MODIFIED BasicPortfolio REGISTRATION ---
    portfolio_args = {
        "instance_name": "BasicPortfolio", 
        "config_loader": config_loader,
        "event_bus": event_bus, 
        "container": container, # <--- ADDED CONTAINER HERE
        "component_config_key": "components.basic_portfolio"
    }
    container.register_type("portfolio_manager", BasicPortfolio, True, constructor_kwargs=portfolio_args)
    logger.info("BasicPortfolio registered as 'portfolio_manager'.")
    # ------------------------------------------

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

    if run_optimization_mode:
        logger.info("OPTIMIZATION MODE DETECTED.")
        if max_bars_to_process is not None:
            logger.info(f"Optimizer will run each parameter set using an initial dataset of up to {max_bars_to_process} bars (before train/test split).")
        else:
            logger.info("Optimizer will run each parameter set on the full dataset (as configured in CSVDataHandler, before train/test split).")

        # Register the appropriate optimizer
        optimizer_args = {"instance_name": "EnhancedOptimizer", "config_loader": config_loader,
                          "event_bus": event_bus, "component_config_key": "components.optimizer",
                          "container": container}
        container.register_type("optimizer_service", EnhancedOptimizer, True, constructor_kwargs=optimizer_args)
        logger.info("EnhancedOptimizer registered as 'optimizer_service'.")
        
        # Register genetic optimizer if needed
        if run_genetic_optimization:
            genetic_optimizer_args = {"instance_name": "GeneticOptimizer", "config_loader": config_loader,
                                     "event_bus": event_bus, "component_config_key": "components.genetic_optimizer",
                                     "container": container}
            container.register_type("genetic_optimizer_service", GeneticOptimizer, True, constructor_kwargs=genetic_optimizer_args)
            logger.info("GeneticOptimizer registered as 'genetic_optimizer_service'.")

        try:
            # During optimization, specifically set up and start the RegimeDetector first
            try:
                regime_detector = container.resolve("MyPrimaryRegimeDetector")
                logger.info(f"Setting up RegimeDetector '{regime_detector.name}' for optimization")
                regime_detector.setup()
                if regime_detector.get_state() == BaseComponent.STATE_INITIALIZED:
                    regime_detector.start()
                    logger.info(f"RegimeDetector '{regime_detector.name}' started for optimization")
            except Exception as e:
                logger.warning(f"Could not set up RegimeDetector for optimization: {e}")
            
            optimizer: EnhancedOptimizer = container.resolve("optimizer_service")
            optimizer.setup() 
            if optimizer.get_state() == BaseComponent.STATE_INITIALIZED:
                optimizer.start()
                logger.info("--- Starting Grid Search ---")
                
                optimization_results: Optional[Dict[str, Any]] = optimizer.run_grid_search()
                
                if optimization_results:
                    best_train_params = optimization_results.get("best_parameters_on_train")
                    best_train_metric_val = optimization_results.get("best_training_metric_value")
                    metric_name_optimized = getattr(optimizer, '_metric_to_optimize', 
                                                  config_loader.get('components.optimizer.metric_to_optimize', 'get_final_portfolio_value'))
                    
                    test_metric_for_best = optimization_results.get("test_set_metric_value_for_best_params")

                    # Detailed results already logged by optimizer._log_optimization_results, no need to duplicate here
                    logger.debug("Optimization complete. See summary for results.")
                    
                    # If genetic optimization is requested, run per-regime genetic optimization
                    if run_genetic_optimization:
                        logger.info("--- Starting Per-Regime Genetic Weight Optimization ---")
                        print("\n=== PER-REGIME GENETIC OPTIMIZATION ===\n")
                        try:
                            # Run per-regime genetic optimization
                            optimization_results = optimizer.run_per_regime_genetic_optimization(optimization_results)
                            
                            # Log results
                            if "best_weights_per_regime" in optimization_results:
                                regimes_with_weights = list(optimization_results["best_weights_per_regime"].keys())
                                logger.info(f"Genetic optimization complete for {len(regimes_with_weights)} regimes: {regimes_with_weights}")
                                
                                # Print summary
                                print(f"\nOptimized weights for {len(regimes_with_weights)} regimes:")
                                for regime, weight_data in optimization_results["best_weights_per_regime"].items():
                                    weights = weight_data.get("weights", {})
                                    fitness = weight_data.get("fitness", "N/A")
                                    ma_weight = weights.get("ma_rule.weight", 0.5)
                                    rsi_weight = weights.get("rsi_rule.weight", 0.5)
                                    print(f"  {regime}: MA={ma_weight:.3f}, RSI={rsi_weight:.3f} (fitness: {fitness:.2f})")
                            else:
                                logger.warning("No per-regime genetic optimization results found")
                                
                        except Exception as e:
                            logger.error(f"Error during per-regime genetic optimization: {e}", exc_info=True)
                            
                        # Now run adaptive test AFTER genetic optimization
                        logger.info("\n\n!!! RUNNING ADAPTIVE TEST AFTER GENETIC OPTIMIZATION !!!\n\n")
                        try:
                            if hasattr(optimizer, 'run_adaptive_test'):
                                optimizer.run_adaptive_test(optimization_results)
                                logger.info("Completed adaptive test after genetic optimization")
                            else:
                                logger.error("Optimizer does not have run_adaptive_test method")
                        except Exception as e:
                            logger.error(f"Error running adaptive test after genetic optimization: {e}", exc_info=True)
                else:
                    logger.info("OPTIMIZATION FAILED: Optimizer run_grid_search() returned no results (None).")
                
                optimizer.stop()
            else:
                logger.error("Failed to initialize optimizer. Cannot run optimization.")

        except (DependencyNotFoundError, ADMFTraderError, ComponentError, Exception) as e:
            logger.critical(f"CRITICAL ERROR during optimization setup or run: {e}", exc_info=True)
            sys.exit(1)
    else:
        if max_bars_to_process is not None and max_bars_to_process > 0:
             logger.info(f"Standard backtest: Using first {max_bars_to_process} bars from the dataset.")
        else:
            logger.info("Standard backtest: Processing all available bars from the dataset.")

        system_name = config_loader.get('system.name', 'ADMF-Trader')
        logger.info(f"App Name: {system_name} - Running Standard Backtest")
        logger.info("Bootstrap complete. Starting application logic...")
        try:
            run_application_logic(container)
        except (DependencyNotFoundError, ADMFTraderError, ComponentError, Exception) as e:
            logger.critical(f"CRITICAL ERROR during application logic: {e}", exc_info=True)
            sys.exit(1)

    logger.info("ADMF-Trader MVP finished.")


def run_application_logic(app_container: Container):
    logger.info("Running main application logic (standard backtest)...")
    
    data_handler: Optional[CSVDataHandler] = None
    strategy: Optional[EnsembleSignalStrategy] = None
    regime_detector: Optional[RegimeDetector] = None 
    portfolio_manager: Optional[BasicPortfolio] = None
    risk_manager: Optional[BasicRiskManager] = None 
    execution_handler: Optional[SimulatedExecutionHandler] = None
    signal_logger_comp: Optional[DummyComponent] = None
    order_logger_comp: Optional[DummyComponent] = None

    components_to_manage: List[Optional[BaseComponent]] = []

    try:
        data_handler = app_container.resolve("data_handler")
        strategy = app_container.resolve("strategy") 
        regime_detector = app_container.resolve("MyPrimaryRegimeDetector") 
        portfolio_manager = app_container.resolve("portfolio_manager")
        risk_manager = app_container.resolve("risk_manager") 
        execution_handler = app_container.resolve("execution_handler")
        signal_logger_comp = app_container.resolve("signal_consumer")
        order_logger_comp = app_container.resolve("order_consumer")
        
        logger.info("DataHandler, Strategy, RegimeDetector, PortfolioManager, RiskManager, ExecutionHandler, and Loggers resolved.")

        components_to_manage = [
            data_handler, 
            strategy, 
            regime_detector, 
            portfolio_manager, 
            risk_manager, 
            execution_handler, 
            signal_logger_comp, 
            order_logger_comp   
        ]

        for comp in components_to_manage:
            if comp is None:
                logger.error("A core component was not resolved. Aborting setup.")
                raise ComponentError("Core component resolution failed during setup.")
            if isinstance(comp, BaseComponent):
                logger.info(f"--- Setting up {comp.name} ---")
                comp.setup()
                logger.info(f"State of '{comp.name}' after setup: {comp.get_state()}")
                if comp.get_state() == BaseComponent.STATE_FAILED:
                    raise ComponentError(f"Component '{comp.name}' failed to setup.")
            else:
                logger.warning(f"Item {type(comp).__name__} is not a BaseComponent, skipping setup for it directly.")

        if data_handler and isinstance(data_handler, CSVDataHandler) and hasattr(data_handler, "set_active_dataset"):
            logger.info("Standard run: Setting active dataset to 'full' in DataHandler (respects --bars if provided).")
            data_handler.set_active_dataset("full")

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

    except (DependencyNotFoundError, ComponentError, ADMFTraderError) as e: 
        logger.error(f"Error during application logic: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during application logic: {e}", exc_info=True)
    finally:
        logger.info("--- Initiating application shutdown sequence (standard backtest) ---")

        if portfolio_manager and isinstance(portfolio_manager, BasicPortfolio) and \
           portfolio_manager.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED]:
            
            last_timestamp_for_close = None
            if data_handler and isinstance(data_handler, CSVDataHandler) and \
               data_handler.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_FAILED] and \
               hasattr(data_handler, 'get_last_timestamp'):
                last_timestamp_for_close = data_handler.get_last_timestamp()

            if not last_timestamp_for_close and hasattr(portfolio_manager, 'get_last_processed_timestamp'):
                 last_timestamp_for_close = portfolio_manager.get_last_processed_timestamp()
            
            if not last_timestamp_for_close:
                last_timestamp_for_close = datetime.datetime.now(datetime.timezone.utc)
                logger.warning(f"Using current time as fallback for closing positions: {last_timestamp_for_close}")
            
            logger.info(f"--- Closing all open positions at timestamp: {last_timestamp_for_close} ---")
            portfolio_manager.close_all_open_positions(last_timestamp_for_close)
            logger.info("--- Finished attempt to close open positions. ---")
        else:
            logger.warning("PortfolioManager not available or not in a valid state to close open positions.")

        for comp in reversed(components_to_manage):
            if comp and isinstance(comp, BaseComponent) and hasattr(comp, 'stop') and callable(comp.stop):
                if hasattr(comp, 'get_state') and comp.get_state() not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_STOPPED, BaseComponent.STATE_FAILED]: 
                    logger.info(f"--- Stopping {comp.name} (Current State: {comp.get_state()}) ---")
                    try:
                        # Generate summary for RegimeDetector before stopping
                        if comp == regime_detector and hasattr(comp, 'generate_summary'):
                            logger.info(f"Generating summary for RegimeDetector '{comp.name}'...")
                            comp.generate_summary()
                        
                        comp.stop()
                        logger.info(f"State of '{comp.name}' after stop: {comp.get_state()}")
                    except Exception as e:
                        logger.error(f"Error stopping component '{comp.name}': {e}", exc_info=True)
                elif hasattr(comp, 'name'):
                    logger.debug(f"Component '{comp.name}' was in {comp.get_state()} state; stop action might be minimal or skipped.")
            elif comp and hasattr(comp, 'name'): 
                 logger.warning(f"Item '{comp.name}' ({type(comp).__name__}) may not be a standard BaseComponent or wasn't fully operational to stop.")
            elif comp: 
                 logger.warning(f"Attempting to handle shutdown for an item of type '{type(comp).__name__}' that may not be a BaseComponent.")
    
    logger.info("Main application logic (standard backtest) finished.")

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    main()