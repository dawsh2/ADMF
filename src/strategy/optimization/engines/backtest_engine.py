"""
BacktestEngine - Handles single backtest execution for optimization and production runs.

This module extracts the backtest execution logic from EnhancedOptimizer to ensure
consistent behavior between optimization and production runs.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from src.core.component import BaseComponent
from src.strategy.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategy.implementations.ensemble_strategy import EnsembleSignalStrategy
from src.data.csv_data_handler import CSVDataHandler
from src.strategy.regime_detector import RegimeDetector
from src.risk.basic_portfolio import BasicPortfolio
from src.risk.basic_risk_manager import BasicRiskManager
from src.execution.simulated_execution_handler import SimulatedExecutionHandler


class BacktestEngine:
    """
    Handles the execution of a single backtest run with given parameters.
    
    This class encapsulates the logic for setting up components, running a backtest,
    and collecting performance metrics. It ensures consistent behavior whether used
    in optimization or production contexts.
    """
    
    def __init__(self, container, config_loader, event_bus):
        """
        Initialize the BacktestEngine.
        
        Args:
            container: Dependency injection container
            config_loader: Configuration loader instance
            event_bus: Event bus for component communication
        """
        self.container = container
        self.config_loader = config_loader
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self._debug_mode = False  # Will be set from config
        
    def _stop_all_components(self, components: Dict[str, Any]) -> None:
        """Stop all components if they're running."""
        for comp_name, comp in components.items():
            if comp and isinstance(comp, BaseComponent):
                state = comp.get_state()
                if state not in [BaseComponent.STATE_CREATED, BaseComponent.STATE_STOPPED]:
                    try:
                        comp.stop()
                        self.logger.debug(f"Stopped {comp_name} (was in state {state})")
                    except Exception as e:
                        self.logger.warning(f"Error stopping {comp_name}: {e}")
        
    def run_backtest(
        self,
        parameters: Dict[str, Any],
        dataset_type: str = "full",
        strategy_type: str = "ensemble",
        use_regime_adaptive: bool = False,
        adaptive_params_path: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Execute a single backtest with the given parameters.
        
        Args:
            parameters: Strategy parameters to test
            dataset_type: Which dataset to use ("train", "test", or "full")
            strategy_type: Type of strategy to use ("ensemble" or "regime_adaptive")
            use_regime_adaptive: Whether to use regime-adaptive parameter switching
            adaptive_params_path: Path to JSON file with regime-specific parameters
            
        Returns:
            Tuple of (metric_value, regime_performance_dict) or (None, None) on error
        """
        try:
            # Enable debug mode from config
            self._debug_mode = self.config_loader.get("components.regime_detector.debug_mode", False)
            
            if self._debug_mode:
                self.logger.info("[BACKTEST_DEBUG] Starting backtest run with debug logging enabled")
                self.logger.info(f"[BACKTEST_DEBUG] Parameters: {parameters}")
                self.logger.info(f"[BACKTEST_DEBUG] Dataset type: {dataset_type}")
                self.logger.info(f"[BACKTEST_DEBUG] Strategy type: {strategy_type}")
                self.logger.info(f"[BACKTEST_DEBUG] Use regime adaptive: {use_regime_adaptive}")
            
            # Resolve components with fresh instances
            components = self._resolve_components(
                parameters, 
                strategy_type, 
                use_regime_adaptive,
                adaptive_params_path
            )
            
            if not components:
                return None, None
                
            # Stop all components first if they're running
            self._stop_all_components(components)
            
            # Reset components to ensure cold start
            self._reset_components(components)
            
            # Initialize components first (setup phase only)
            self._setup_components(components)
            
            # Then configure dataset
            self._configure_dataset(components['data_handler'], dataset_type)
            
            # Finally start all components
            if self._debug_mode:
                self.logger.info("[BACKTEST_DEBUG] Starting components...")
            self._start_components(components)
            
            if self._debug_mode:
                # Log initial component states
                for name, comp in components.items():
                    if hasattr(comp, 'get_state'):
                        self.logger.info(f"[BACKTEST_DEBUG] Component {name} state: {comp.get_state()}")
            
            # Run the backtest
            self._run_backtest_loop(components)
            
            # Collect results
            metric_value, regime_performance = self._collect_results(components)
            
            # Cleanup
            self._cleanup_components(components)
            
            # Reset components for next run
            self._reset_components(components)
            
            return metric_value, regime_performance
            
        except Exception as e:
            self.logger.error(f"Error in backtest execution: {e}", exc_info=True)
            return None, None
            
    def _resolve_components(
        self, 
        parameters: Dict[str, Any], 
        strategy_type: str,
        use_regime_adaptive: bool,
        adaptive_params_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve fresh component instances for the backtest.
        
        Returns:
            Dictionary of component instances or None on error
        """
        try:
            # Get fresh instances of all required components
            data_handler = self.container.resolve("data_handler")
            regime_detector = self.container.resolve("MyPrimaryRegimeDetector")
            portfolio_manager = self.container.resolve("portfolio_manager")
            risk_manager = self.container.resolve("risk_manager")
            execution_handler = self.container.resolve("execution_handler")
            
            # Get the strategy instance from container
            strategy = self.container.resolve("strategy")
            
            # Apply parameters to the existing strategy
            if not use_regime_adaptive and parameters:
                # Use set_parameters method (not set_parameter)
                if hasattr(strategy, 'set_parameters'):
                    if not strategy.set_parameters(parameters):
                        self.logger.error(f"Failed to set parameters {parameters} on strategy")
                        return None
                    self.logger.debug(f"Applied parameters {parameters} to strategy")
                else:
                    self.logger.warning("Strategy does not have set_parameters method")
                
            return {
                'data_handler': data_handler,
                'strategy': strategy,
                'regime_detector': regime_detector,
                'portfolio_manager': portfolio_manager,
                'risk_manager': risk_manager,
                'execution_handler': execution_handler
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up components: {e}", exc_info=True)
            return None
            
    def _create_ensemble_strategy(self, parameters: Dict[str, Any]) -> Optional[EnsembleSignalStrategy]:
        """Create and configure an ensemble strategy instance."""
        try:
            strategy = EnsembleSignalStrategy(
                instance_name="Backtest_Ensemble_Strategy",
                config_loader=self.config_loader,
                event_bus=self.event_bus,
                component_config_key="components.ensemble_strategy",
                container=self.container
            )
            
            # Apply test parameters
            for param_key, param_value in parameters.items():
                if hasattr(strategy, 'set_parameter'):
                    strategy.set_parameter(param_key, param_value)
                    
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble strategy: {e}", exc_info=True)
            return None
            
    def _create_regime_adaptive_strategy(
        self, 
        parameters: Dict[str, Any],
        adaptive_params_path: Optional[str]
    ) -> Optional[RegimeAdaptiveStrategy]:
        """Create and configure a regime-adaptive strategy instance."""
        try:
            # Create custom config for the strategy if needed
            config_key = "components.regime_adaptive_strategy"
            
            # Temporarily update config with adaptive parameters path if provided
            if adaptive_params_path:
                original_path = self.config_loader.get(f"{config_key}.adaptive_params_file")
                # Access the internal _config_data attribute
                if hasattr(self.config_loader, '_config_data'):
                    self.config_loader._config_data.setdefault('components', {}).setdefault(
                        'regime_adaptive_strategy', {}
                    )['adaptive_params_file'] = adaptive_params_path
                
            strategy = RegimeAdaptiveStrategy(
                instance_name="Backtest_RegimeAdaptive_Strategy",
                config_loader=self.config_loader,
                event_bus=self.event_bus,
                container=self.container,
                component_config_key=config_key
            )
            
            # Restore original config if we modified it
            if adaptive_params_path and 'original_path' in locals() and hasattr(self.config_loader, '_config_data'):
                if original_path:
                    self.config_loader._config_data['components']['regime_adaptive_strategy'][
                        'adaptive_params_file'
                    ] = original_path
                else:
                    self.config_loader._config_data['components']['regime_adaptive_strategy'].pop(
                        'adaptive_params_file', None
                    )
                    
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error creating regime-adaptive strategy: {e}", exc_info=True)
            return None
            
    def _configure_dataset(self, data_handler: CSVDataHandler, dataset_type: str) -> None:
        """Configure which dataset to use for the backtest."""
        # Ensure data handler is set up first
        if data_handler.get_state() == BaseComponent.STATE_CREATED:
            self.logger.debug("Setting up data handler before configuring dataset")
            data_handler.setup()
            
        if hasattr(data_handler, 'set_active_dataset'):
            self.logger.info(f"Setting active dataset to '{dataset_type}'")
            data_handler.set_active_dataset(dataset_type)
            
    def _setup_components(self, components: Dict[str, Any]) -> None:
        """Setup components in the correct order."""
        # Order matches the optimizer's component order for consistency
        component_order = [
            'regime_detector',    # First - process regime changes
            'execution_handler',  # Handles ORDER events
            'risk_manager',
            'strategy',          # After regime detector
            'portfolio_manager',
            'data_handler'       # Last - publishes events
        ]
        
        # Setup phase only
        for comp_name in component_order:
            comp = components.get(comp_name)
            if comp and isinstance(comp, BaseComponent):
                self.logger.debug(f"Setting up {comp.name}")
                comp.setup()
                if comp.get_state() == BaseComponent.STATE_FAILED:
                    raise RuntimeError(f"Component {comp.name} failed during setup")
                    
    def _start_components(self, components: Dict[str, Any]) -> None:
        """Start components in the correct order."""
        # Order matches the optimizer's component order for consistency
        component_order = [
            'regime_detector',    # First - process regime changes
            'execution_handler',  # Handles ORDER events
            'risk_manager',
            'strategy',          # After regime detector
            'portfolio_manager',
            'data_handler'       # Last - publishes events
        ]
        
        # Start phase
        for comp_name in component_order:
            comp = components.get(comp_name)
            if comp and isinstance(comp, BaseComponent):
                if comp.get_state() == BaseComponent.STATE_INITIALIZED:
                    self.logger.debug(f"Starting {comp.name}")
                    comp.start()
                    if comp.get_state() == BaseComponent.STATE_FAILED:
                        raise RuntimeError(f"Component {comp.name} failed during start")
                        
    def _run_backtest_loop(self, components: Dict[str, Any]) -> None:
        """
        Run the main backtest loop.
        
        The data handler will emit BAR events which trigger the strategy
        and other components to process data.
        """
        if self._debug_mode:
            self.logger.info("[BACKTEST_DEBUG] Starting backtest loop...")
            
        # The backtest runs automatically through event processing
        # Data handler emits all bars when started
        data_handler = components.get('data_handler')
        if data_handler:
            # Wait for data handler to finish emitting bars
            while data_handler.get_state() == BaseComponent.STATE_STARTED:
                # Data handler is processing
                pass
        self.logger.debug("Backtest loop completed")
        
    def _collect_results(
        self, 
        components: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Collect performance metrics from the backtest.
        
        Returns:
            Tuple of (metric_value, regime_performance_dict)
        """
        portfolio = components.get('portfolio_manager')
        data_handler = components.get('data_handler')
        
        if not portfolio:
            return None, None
            
        try:
            # Close any open positions before getting final value
            if hasattr(portfolio, 'close_all_open_positions'):
                # Get last timestamp from data handler
                last_timestamp = None
                if data_handler and hasattr(data_handler, 'get_last_timestamp'):
                    last_timestamp = data_handler.get_last_timestamp()
                
                if last_timestamp:
                    self.logger.debug(f"Closing all open positions at {last_timestamp}")
                    portfolio.close_all_open_positions(last_timestamp)
            
            # Get final portfolio value as primary metric
            final_value = portfolio.get_final_portfolio_value()
            self.logger.info(f"Backtest completed. Final portfolio value: {final_value}")
            
            # Collect regime-specific performance if available
            regime_performance = {}
            if hasattr(portfolio, 'get_regime_performance'):
                regime_performance = portfolio.get_regime_performance()
            elif hasattr(portfolio, 'get_performance_by_regime'):
                regime_performance = portfolio.get_performance_by_regime()
            elif hasattr(portfolio, 'context') and hasattr(portfolio.context, 'metrics'):
                # Extract regime performance from metrics
                metrics = portfolio.context.metrics
                if 'regime_performance' in metrics:
                    regime_performance = metrics['regime_performance']
            
            if regime_performance:
                self.logger.debug(f"Collected regime performance for {len(regime_performance)} regimes")
                    
            return final_value, regime_performance
            
        except Exception as e:
            self.logger.error(f"Error collecting results: {e}", exc_info=True)
            return None, None
            
    def _cleanup_components(self, components: Dict[str, Any]) -> None:
        """Stop and cleanup components after backtest."""
        # Stop in reverse order
        component_order = [
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
                        
    def _reset_components(self, components: Dict[str, Any]) -> None:
        """Reset components for next run."""
        # Reset portfolio to initial state
        portfolio = components.get('portfolio_manager')
        if portfolio and hasattr(portfolio, 'reset'):
            portfolio.reset()
            
        # Reset regime detector to ensure cold start
        regime_detector = components.get('regime_detector')
        if regime_detector and hasattr(regime_detector, 'reset'):
            self.logger.debug("Resetting RegimeDetector for cold start")
            regime_detector.reset()
            
        # Reset strategy to ensure cold start
        strategy = components.get('strategy')
        if strategy and hasattr(strategy, 'reset'):
            self.logger.debug("Resetting Strategy for cold start")
            strategy.reset()
        elif portfolio:
            # Manual reset if no reset method
            if hasattr(portfolio, 'cash'):
                portfolio.cash = portfolio._initial_cash
            if hasattr(portfolio, 'holdings'):
                portfolio.holdings.clear()
            if hasattr(portfolio, '_trade_log'):
                portfolio._trade_log.clear()