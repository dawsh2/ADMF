# src/strategy/optimization/basic_optimizer.py
import logging
import itertools
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.core.component import BaseComponent
from src.core.container import Container
from src.core.exceptions import ConfigurationError, ComponentError, DependencyNotFoundError

from src.data.csv_data_handler import CSVDataHandler
from src.strategy.ma_strategy import MAStrategy
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.risk.basic_portfolio import BasicPortfolio


class BasicOptimizer(BaseComponent):
    """
    A basic grid search optimizer for trading strategies.
    Performs optimization on a training dataset and evaluates the best parameters
    on a separate testing dataset. Relies on CSVDataHandler to respect --bars
    before splitting data into train/test.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container: Container):
        super().__init__(instance_name, config_loader, component_config_key)
        
        self._container = container
        self._event_bus = event_bus

        self._strategy_service_name: str = self.get_specific_config("strategy_service_name", "strategy")
        self._portfolio_service_name: str = self.get_specific_config("portfolio_service_name", "portfolio_manager")
        self._data_handler_service_name: str = self.get_specific_config("data_handler_service_name", "data_handler")
        self._risk_manager_service_name: str = self.get_specific_config("risk_manager_service_name", "risk_manager")
        self._exec_handler_service_name: str = self.get_specific_config("execution_handler_service_name", "execution_handler")
        
        self._metric_to_optimize: str = self.get_specific_config("metric_to_optimize", "get_final_portfolio_value")
        self._higher_metric_is_better: bool = self.get_specific_config("higher_metric_is_better", True)

        self.logger.info(
            f"{self.name} initialized. Optimizing strategy '{self._strategy_service_name}' "
            f"using metric '{self._metric_to_optimize}' from '{self._portfolio_service_name}'. "
            f"Higher is better: {self._higher_metric_is_better}."
        )
        self._best_params_from_train: Optional[Dict[str, Any]] = None
        self._best_training_metric_value: Optional[float] = None
        self._test_metric_for_best_params: Optional[float] = None


    def setup(self):
        self.logger.info(f"Setting up {self.name}...")
        try:
            data_handler_check: CSVDataHandler = self._container.resolve(self._data_handler_service_name)
            if not hasattr(data_handler_check, "set_active_dataset"):
                raise ConfigurationError(
                    f"DataHandler '{self._data_handler_service_name}' does not support 'set_active_dataset' method, "
                    "which is required for train/test splitting by BasicOptimizer."
                )
            if not hasattr(data_handler_check, "test_df_exists_and_is_not_empty"):
                 raise ConfigurationError(
                    f"DataHandler '{self._data_handler_service_name}' needs 'test_df_exists_and_is_not_empty' property."
                )

            portfolio_check: BasicPortfolio = self._container.resolve(self._portfolio_service_name)
            metric_method_on_portfolio = getattr(portfolio_check, self._metric_to_optimize, None)
            if not callable(metric_method_on_portfolio):
                raise ConfigurationError(
                    f"PortfolioManager '{self._portfolio_service_name}' does not have callable metric method '{self._metric_to_optimize}'."
                )
            
            strategy_check: MAStrategy = self._container.resolve(self._strategy_service_name)
            if not (hasattr(strategy_check, "get_parameter_space") and
                    hasattr(strategy_check, "set_parameters") and
                    hasattr(strategy_check, "get_parameters")):
                raise ConfigurationError(f"Strategy '{self._strategy_service_name}' does not fully support optimization interfaces.")

        except DependencyNotFoundError as e:
            self.logger.error(f"Optimizer setup check failed to resolve critical component: {e}")
            self.state = BaseComponent.STATE_FAILED
            raise
            
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"{self.name} setup complete.")

    def _generate_parameter_combinations(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not param_space:
            self.logger.warning("Parameter space is empty, will use current/default strategy parameters for one run.")
            strategy_instance: MAStrategy = self._container.resolve(self._strategy_service_name)
            return [strategy_instance.get_parameters()] if hasattr(strategy_instance, "get_parameters") else [{}]
            
        keys = list(param_space.keys())
        value_lists = [param_space[key] for key in keys]
        combinations = [dict(zip(keys, value_combination)) for value_combination in itertools.product(*value_lists)]
        return combinations

    def _perform_single_backtest_run(self, params_to_test: Dict[str, Any], dataset_type: str) -> Optional[float]:
        self.logger.info(
            f"--- Optimizer: Starting '{dataset_type.upper()}' run with parameters: {params_to_test} ---"
        )
        
        data_handler: Optional[CSVDataHandler] = None
        strategy: Optional[MAStrategy] = None 
        portfolio_manager: Optional[BasicPortfolio] = None
        components_for_this_run = []

        try:
            data_handler = self._container.resolve(self._data_handler_service_name)
            strategy = self._container.resolve(self._strategy_service_name)
            portfolio_manager = self._container.resolve(self._portfolio_service_name)
            risk_manager = self._container.resolve(self._risk_manager_service_name)
            execution_handler = self._container.resolve(self._exec_handler_service_name)
            
            if not all([data_handler, strategy, portfolio_manager, risk_manager, execution_handler]):
                self.logger.error("Optimizer: Failed to resolve one or more core components for backtest run.")
                return None

            components_for_this_run = [
                data_handler, strategy, portfolio_manager, risk_manager, execution_handler
            ]
            
            # Reset portfolio state to ensure a clean start
            try:
                self.logger.info(f"Optimizer: Resetting portfolio state before {dataset_type} run with params: {params_to_test}")
                if hasattr(portfolio_manager, 'reset') and callable(portfolio_manager.reset):
                    portfolio_manager.reset()
                else:
                    self.logger.warning("Portfolio does not have a reset method. State may persist between runs.")
            except Exception as e:
                self.logger.error(f"Error resetting portfolio before backtest run: {e}", exc_info=True)
            
            # CRITICAL FIX: Set parameters BEFORE component setup to avoid race conditions
            if not strategy.set_parameters(params_to_test):
                self.logger.error(f"Optimizer: Failed to set parameters {params_to_test} on strategy {strategy.name}. Skipping {dataset_type} run.")
                return None
            
            for comp in components_for_this_run:
                self.logger.debug(f"Optimizer: Setting up component '{comp.name}' for '{dataset_type}' run with correct parameters.")
                comp.setup() 
                if comp.get_state() == BaseComponent.STATE_FAILED:
                    self.logger.error(f"Optimizer: Component '{comp.name}' failed setup. Skipping {dataset_type} run.")
                    return None
            
            data_handler.set_active_dataset(dataset_type)
            
            for comp in components_for_this_run:
                if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                    comp.start()
                    if comp.get_state() == BaseComponent.STATE_FAILED:
                        self.logger.error(f"Optimizer: Component '{comp.name}' failed to start. Skipping {dataset_type} run.")
                        return None
                else:
                     self.logger.error(f"Optimizer: Component '{comp.name}' not INITIALIZED before start (State: {comp.get_state()}). Error in setup?. Skipping {dataset_type} run.")
                     return None
            
            self.logger.debug(f"Optimizer: Data streaming complete for '{dataset_type}' run with {params_to_test}.")

            last_ts = data_handler.get_last_timestamp() or portfolio_manager.get_last_processed_timestamp() or datetime.datetime.now(datetime.timezone.utc)
            self.logger.debug(f"Optimizer: Closing positions for '{dataset_type}' run with {params_to_test} at {last_ts}")
            portfolio_manager.close_all_positions(last_ts)
            
            metric_method = getattr(portfolio_manager, self._metric_to_optimize)
            metric_value = metric_method()
            self.logger.info(f"Optimizer: Params {params_to_test} on '{dataset_type}' data -> Metric '{self._metric_to_optimize}': {metric_value}")
            
            return metric_value

        except (DependencyNotFoundError, ComponentError, ConfigurationError) as e:
            self.logger.error(f"Optimizer: Error during '{dataset_type}' backtest run with params {params_to_test}: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Optimizer: Unexpected error during '{dataset_type}' backtest run with params {params_to_test}: {e}", exc_info=True)
            return None
        finally:
            self.logger.debug(f"Optimizer: Stopping components for '{dataset_type}' run with {params_to_test}")
            for comp in reversed(components_for_this_run): 
                if comp and hasattr(comp, 'stop') and callable(comp.stop):
                    try:
                        comp.stop()
                    except Exception as e:
                        self.logger.error(f"Optimizer: Error stopping component '{comp.name}': {e}")

    def run_grid_search(self) -> Optional[Dict[str, Any]]:
        self.logger.info(f"--- {self.name}: Starting Grid Search Optimization with Train/Test Split ---")
        self.state = BaseComponent.STATE_STARTED
        self._best_params_from_train = None
        self._best_training_metric_value = -float('inf') if self._higher_metric_is_better else float('inf')
        self._test_metric_for_best_params = None 

        results_summary = {
            "best_parameters_on_train": None,
            "best_training_metric_value": None,
            "test_set_metric_value_for_best_params": None,
            "all_training_results": [] 
        }

        try:
            strategy_to_optimize: MAStrategy = self._container.resolve(self._strategy_service_name)
            data_handler_instance: CSVDataHandler = self._container.resolve(self._data_handler_service_name) 
            
            param_space = strategy_to_optimize.get_parameter_space()
            current_strategy_params = strategy_to_optimize.get_parameters() 

            if not param_space:
                self.logger.warning("Optimizer: Parameter space is empty. Running one iteration with current/default strategy parameters for training and testing.")
                param_combinations = [current_strategy_params] if current_strategy_params else [{}]
            else:
                 param_combinations = self._generate_parameter_combinations(param_space)

            if not param_combinations : 
                 self.logger.warning("No parameter combinations to test (parameter space might be empty or produced no combinations).")
                 self.state = BaseComponent.STATE_STOPPED
                 return results_summary

            self.logger.info(f"--- Training Phase: Testing {len(param_combinations)} parameter combinations ---")

            for i, params in enumerate(param_combinations):
                self.logger.info(f"Training: Combination {i+1}/{len(param_combinations)} with params: {params}")
                
                training_metric_value = self._perform_single_backtest_run(params, dataset_type="train")
                results_summary["all_training_results"].append({"parameters": params, "metric_value": training_metric_value})

                if training_metric_value is not None:
                    # Type assertion for mypy, as -float('inf') is a float
                    current_best_metric = self._best_training_metric_value if isinstance(self._best_training_metric_value, float) else (-float('inf') if self._higher_metric_is_better else float('inf'))

                    if self._higher_metric_is_better:
                        if training_metric_value > current_best_metric:
                            self._best_training_metric_value = training_metric_value
                            self._best_params_from_train = params
                            self.logger.info(f"New best training metric: {self._best_training_metric_value:.4f} with params: {self._best_params_from_train}")
                    else: 
                        if training_metric_value < current_best_metric:
                            self._best_training_metric_value = training_metric_value
                            self._best_params_from_train = params
                            self.logger.info(f"New best training metric: {self._best_training_metric_value:.4f} with params: {self._best_params_from_train}")
                else:
                    self.logger.warning(f"Training run failed or returned no metric for params {params}.")

            results_summary["best_parameters_on_train"] = self._best_params_from_train
            results_summary["best_training_metric_value"] = self._best_training_metric_value

            if self._best_params_from_train:
                self.logger.info(
                    f"--- Testing Phase: Evaluating best training parameters {self._best_params_from_train} on test data ---"
                )
                if data_handler_instance.test_df_exists_and_is_not_empty:
                    self._test_metric_for_best_params = self._perform_single_backtest_run(
                        self._best_params_from_train, dataset_type="test"
                    )
                    results_summary["test_set_metric_value_for_best_params"] = self._test_metric_for_best_params
                    # self.logger.info for test result is already inside _perform_single_backtest_run
                else:
                    self.logger.warning("No test data available. Skipping testing phase.")
                    results_summary["test_set_metric_value_for_best_params"] = "N/A (No test data)"
                
                # CORRECTED LOGGING FORMAT
                best_train_metric_str = f"{self._best_training_metric_value:.4f}" if isinstance(self._best_training_metric_value, float) else "N/A"
                test_metric_str = f"{self._test_metric_for_best_params:.4f}" if isinstance(self._test_metric_for_best_params, float) else "N/A"
                if results_summary["test_set_metric_value_for_best_params"] == "N/A (No test data)": # Override if explicitly set to this string
                    test_metric_str = "N/A (No test data)"


                self.logger.info(
                    f"--- Grid Search with Train/Test Complete --- \n"
                    f"  Best Parameters (from Training): {self._best_params_from_train}\n"
                    f"  Best Training Metric ('{self._metric_to_optimize}'): {best_train_metric_str}\n"
                    f"  Test Set Metric for these parameters ('{self._metric_to_optimize}'): {test_metric_str}"
                )
            else:
                self.logger.error("Grid Search Optimization (Training Phase) failed to find any valid best parameters.")
            
            self.state = BaseComponent.STATE_STOPPED
            return results_summary

        except Exception as e:
            self.logger.error(f"Critical error during grid search optimization: {e}", exc_info=True)
            self.state = BaseComponent.STATE_FAILED
            # Ensure results_summary still reflects any partial progress or error state
            results_summary["error"] = str(e) 
            return results_summary # Return partial results with error, or just None
        finally:
            if self.state not in [BaseComponent.STATE_STOPPED, BaseComponent.STATE_FAILED]:
                self.state = BaseComponent.STATE_STOPPED
            self.logger.info(f"--- {self.name} Grid Search with Train/Test Ended. State: {self.state} ---")

    def start(self):
        self.logger.info(f"{self.name} started. Call run_grid_search() to begin train/test optimization.")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping {self.name} (Optimizer)...")
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"{self.name} stopped.")
