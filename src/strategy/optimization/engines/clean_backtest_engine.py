"""
CleanBacktestEngine - Ensures each backtest starts from completely clean state.

This engine creates fresh component instances for each backtest run,
preventing any state leakage between runs. This is essential for:
- Reproducible results
- Accurate optimization
- Matching production behavior
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from src.core.component import BaseComponent
from src.core.container import Container
from src.core.event_bus import EventBus
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.regime_detector import RegimeDetector
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler
from src.core.dummy_component import DummyComponent


class CleanBacktestEngine:
    """
    Backtest engine that ensures complete state isolation between runs.
    
    Each backtest gets:
    - Fresh component instances
    - New event bus
    - Clean container
    - No artifacts from previous runs
    """
    
    def __init__(self, config_loader):
        """
        Initialize the engine with only the config loader.
        Container and event bus are created fresh for each run.
        """
        self.config_loader = config_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        self._debug_mode = False
        
    def run_backtest(
        self,
        parameters: Dict[str, Any],
        dataset_type: str = "full",
        strategy_type: str = "ensemble",
        use_regime_adaptive: bool = False,
        adaptive_params_path: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Execute a backtest with complete state isolation.
        
        Args:
            parameters: Strategy parameters to test
            dataset_type: Which dataset to use ("train", "test", or "full")
            strategy_type: Type of strategy ("ensemble" or "regime_adaptive")
            use_regime_adaptive: Whether to use regime-adaptive parameter switching
            adaptive_params_path: Path to JSON file with regime-specific parameters
            
        Returns:
            Tuple of (metric_value, regime_performance_dict) or (None, None) on error
        """
        self.logger.info(f"Starting clean backtest - dataset: {dataset_type}, strategy: {strategy_type}")
        
        # Create FRESH instances for this backtest
        event_bus = EventBus()
        container = Container()
        
        try:
            # Register all components fresh
            self._register_components(container, event_bus, strategy_type, adaptive_params_path)
            
            # Resolve components
            components = self._resolve_components(container)
            if not components:
                return None, None
                
            # Setup components
            self._setup_components(components)
            
            # Apply parameters to strategy if not using adaptive mode
            if not use_regime_adaptive and parameters:
                self._apply_parameters(components['strategy'], parameters)
            
            # Configure dataset
            self._configure_dataset(components['data_handler'], dataset_type)
            
            # Start components
            self._start_components(components)
            
            # Run backtest
            self._run_backtest_loop(components)
            
            # Collect results
            metric_value, regime_performance = self._collect_results(components)
            
            # Cleanup
            self._cleanup_components(components)
            
            return metric_value, regime_performance
            
        except Exception as e:
            self.logger.error(f"Error in clean backtest: {e}", exc_info=True)
            return None, None
        finally:
            # Ensure cleanup even on error
            self._final_cleanup(container, event_bus)
            
    def _register_components(
        self, 
        container: Container, 
        event_bus: EventBus,
        strategy_type: str,
        adaptive_params_path: Optional[str]
    ) -> None:
        """Register fresh component instances in the container."""
        
        # Store references in container
        container.register_instance("config_loader", self.config_loader)
        container.register_instance("event_bus", event_bus)
        container.register_instance("container", container)
        
        # Data handler
        csv_args = {
            "instance_name": "SPY_CSV_Loader",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.data_handler_csv"
        }
        container.register_type("data_handler", CSVDataHandler, False, constructor_kwargs=csv_args)
        
        # Regime detector
        regime_args = {
            "instance_name": "MyPrimaryRegimeDetector_Instance",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.MyPrimaryRegimeDetector"
        }
        container.register_type("MyPrimaryRegimeDetector", RegimeDetector, False, constructor_kwargs=regime_args)
        
        # Portfolio manager
        portfolio_args = {
            "instance_name": "BasicPortfolio",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "container": container,
            "component_config_key": "components.basic_portfolio"
        }
        container.register_type("portfolio_manager", BasicPortfolio, False, constructor_kwargs=portfolio_args)
        
        # Risk manager
        risk_args = {
            "instance_name": "BasicRiskMan1",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.basic_risk_manager",
            "container": container,
            "portfolio_manager_key": "portfolio_manager"
        }
        container.register_type("risk_manager", BasicRiskManager, False, constructor_kwargs=risk_args)
        
        # Execution handler
        exec_args = {
            "instance_name": "SimExec_1",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.simulated_execution_handler"
        }
        container.register_type("execution_handler", SimulatedExecutionHandler, False, constructor_kwargs=exec_args)
        
        # Strategy - choose based on type
        if strategy_type == "regime_adaptive":
            # Update config with adaptive params path if provided
            if adaptive_params_path:
                self._update_adaptive_config(adaptive_params_path)
                
            strategy_args = {
                "instance_name": "SPY_RegimeAdaptive_Strategy",
                "config_loader": self.config_loader,
                "event_bus": event_bus,
                "container": container,
                "component_config_key": "components.regime_adaptive_strategy"
            }
            container.register_type("strategy", RegimeAdaptiveStrategy, False, constructor_kwargs=strategy_args)
        else:
            strategy_args = {
                "instance_name": "SPY_Ensemble_Strategy",
                "config_loader": self.config_loader,
                "event_bus": event_bus,
                "component_config_key": "components.ensemble_strategy",
                "container": container
            }
            container.register_type("strategy", EnsembleSignalStrategy, False, constructor_kwargs=strategy_args)
            
        # Signal logger
        signal_logger_args = {
            "instance_name": "SignalLogger",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.dummy_service",
            "listen_to_event_type_str": "SIGNAL"
        }
        container.register_type("signal_consumer", DummyComponent, False, constructor_kwargs=signal_logger_args)
        
        # Order logger
        order_logger_args = {
            "instance_name": "OrderLogger",
            "config_loader": self.config_loader,
            "event_bus": event_bus,
            "component_config_key": "components.dummy_service",
            "listen_to_event_type_str": "ORDER"
        }
        container.register_type("order_consumer", DummyComponent, False, constructor_kwargs=order_logger_args)
        
        self.logger.debug("All components registered with transient lifecycle (singleton=False)")
        
    def _update_adaptive_config(self, adaptive_params_path: str) -> None:
        """Temporarily update config with adaptive parameters path."""
        config_key = "components.regime_adaptive_strategy"
        if hasattr(self.config_loader, '_config_data'):
            self.config_loader._config_data.setdefault('components', {}).setdefault(
                'regime_adaptive_strategy', {}
            )['regime_params_file_path'] = adaptive_params_path
            
    def _resolve_components(self, container: Container) -> Optional[Dict[str, Any]]:
        """Resolve all required components from container."""
        try:
            components = {
                'data_handler': container.resolve("data_handler"),
                'regime_detector': container.resolve("MyPrimaryRegimeDetector"),
                'portfolio_manager': container.resolve("portfolio_manager"),
                'risk_manager': container.resolve("risk_manager"),
                'execution_handler': container.resolve("execution_handler"),
                'strategy': container.resolve("strategy"),
                'signal_logger': container.resolve("signal_consumer"),
                'order_logger': container.resolve("order_consumer")
            }
            return components
        except Exception as e:
            self.logger.error(f"Error resolving components: {e}", exc_info=True)
            return None
            
    def _setup_components(self, components: Dict[str, Any]) -> None:
        """Setup all components in correct order."""
        component_order = [
            'regime_detector',
            'execution_handler',
            'risk_manager',
            'strategy',
            'portfolio_manager',
            'data_handler',
            'signal_logger',
            'order_logger'
        ]
        
        for comp_name in component_order:
            comp = components.get(comp_name)
            if comp and isinstance(comp, BaseComponent):
                self.logger.debug(f"Setting up {comp.name}")
                comp.setup()
                if comp.get_state() == BaseComponent.STATE_FAILED:
                    raise RuntimeError(f"Component {comp.name} failed during setup")
                    
    def _apply_parameters(self, strategy: Any, parameters: Dict[str, Any]) -> None:
        """Apply test parameters to the strategy."""
        if hasattr(strategy, 'set_parameters'):
            self.logger.debug(f"Applying parameters to strategy: {parameters}")
            if not strategy.set_parameters(parameters):
                self.logger.warning("Failed to set all parameters on strategy")
        else:
            self.logger.warning("Strategy does not have set_parameters method")
            
    def _configure_dataset(self, data_handler: CSVDataHandler, dataset_type: str) -> None:
        """Configure which dataset to use."""
        if hasattr(data_handler, 'set_active_dataset'):
            self.logger.debug(f"Setting active dataset to: {dataset_type}")
            data_handler.set_active_dataset(dataset_type)
            
    def _start_components(self, components: Dict[str, Any]) -> None:
        """Start all components in correct order."""
        component_order = [
            'regime_detector',
            'execution_handler',
            'risk_manager',
            'strategy',
            'portfolio_manager',
            'data_handler',
            'signal_logger',
            'order_logger'
        ]
        
        for comp_name in component_order:
            comp = components.get(comp_name)
            if comp and isinstance(comp, BaseComponent):
                if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                    self.logger.debug(f"Starting {comp.name}")
                    comp.start()
                    if comp.get_state() == BaseComponent.STATE_FAILED:
                        raise RuntimeError(f"Component {comp.name} failed during start")
                        
    def _run_backtest_loop(self, components: Dict[str, Any]) -> None:
        """Run the backtest by waiting for data handler to complete."""
        data_handler = components.get('data_handler')
        if data_handler:
            import time
            max_wait = 300  # 5 minutes timeout
            start_time = time.time()
            
            while (data_handler.get_state() == BaseComponent.STATE_STARTED and 
                   time.time() - start_time < max_wait):
                time.sleep(0.1)
                
            if data_handler.get_state() == BaseComponent.STATE_STARTED:
                self.logger.warning("Backtest timed out after 5 minutes")
                
    def _collect_results(self, components: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Collect backtest results from portfolio manager."""
        try:
            portfolio = components.get('portfolio_manager')
            if not portfolio:
                return None, None
                
            # Get final portfolio value
            final_value = None
            if hasattr(portfolio, 'get_final_portfolio_value'):
                final_value = portfolio.get_final_portfolio_value()
            elif hasattr(portfolio, 'get_portfolio_value'):
                final_value = portfolio.get_portfolio_value()
                
            # Get regime performance if available
            regime_performance = {}
            if hasattr(portfolio, 'get_performance_by_regime'):
                regime_performance = portfolio.get_performance_by_regime()
                
            return final_value, regime_performance
            
        except Exception as e:
            self.logger.error(f"Error collecting results: {e}", exc_info=True)
            return None, None
            
    def _cleanup_components(self, components: Dict[str, Any]) -> None:
        """Stop all components in reverse order."""
        component_order = [
            'order_logger',
            'signal_logger',
            'data_handler',
            'portfolio_manager',
            'strategy',
            'risk_manager',
            'execution_handler',
            'regime_detector'
        ]
        
        for comp_name in component_order:
            comp = components.get(comp_name)
            if comp and isinstance(comp, BaseComponent):
                if comp.get_state() not in [BaseComponent.STATE_CREATED, 
                                           BaseComponent.STATE_STOPPED, 
                                           BaseComponent.STATE_FAILED]:
                    try:
                        self.logger.debug(f"Stopping {comp.name}")
                        comp.stop()
                    except Exception as e:
                        self.logger.warning(f"Error stopping {comp.name}: {e}")
                        
    def _final_cleanup(self, container: Container, event_bus: EventBus) -> None:
        """Perform final cleanup of container and event bus."""
        # Clear all registrations
        if hasattr(container, '_registry'):
            container._registry.clear()
        if hasattr(container, '_instances'):
            container._instances.clear()
            
        # Clear event bus subscriptions
        if hasattr(event_bus, '_subscribers'):
            event_bus._subscribers.clear()
            
        self.logger.debug("Final cleanup completed - all state cleared")