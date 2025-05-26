"""
Regime-adaptive strategy that dynamically switches parameters based on market regime.
"""

from typing import Dict, Any, Optional, List
import json
import os

from ..base import Strategy, ParameterSpace, Parameter
from ...core.event import Event, EventType


class RegimeAdaptiveStrategy(Strategy):
    """
    Strategy that adapts parameters based on current market regime.
    
    This strategy:
    1. Loads optimal parameters for each regime from optimization results
    2. Subscribes to regime change events
    3. Dynamically switches parameters when regime changes
    4. Delegates actual signal generation to a wrapped strategy
    """
    
    def __init__(self, instance_name: str = "regime_adaptive_strategy", config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
        # Regime-specific parameters
        self._regime_parameters: Dict[str, Dict[str, Any]] = {}
        self._current_regime = "default"
        
        # Wrapped strategy
        self._base_strategy: Optional[Strategy] = None
        self._base_strategy_name: Optional[str] = None
        
    def _initialize(self):
        """Initialize regime adaptive strategy."""
        super()._initialize()
        
        config = self.component_config or {}
        
        # Get base strategy configuration
        self._base_strategy_name = config.get('base_strategy', 'composite_ma_strategy')
        
        # Load regime parameters
        regime_params_file = config.get('regime_parameters_file')
        if regime_params_file and os.path.exists(regime_params_file):
            self._load_regime_parameters(regime_params_file)
        else:
            # Use config-provided parameters
            self._regime_parameters = config.get('regime_parameters', {})
            
        self.logger.info(f"RegimeAdaptiveStrategy initialized with base strategy: {self._base_strategy_name}")
        self.logger.info(f"Loaded parameters for regimes: {list(self._regime_parameters.keys())}")
        
    def _start(self):
        """Start the regime adaptive strategy."""
        super()._start()
        
        # Create base strategy instance
        if self._context and hasattr(self._context, 'container'):
            try:
                # Try to resolve from container
                self._base_strategy = self._context.container.resolve(self._base_strategy_name)
            except:
                # Create new instance if not in container
                # Import dynamically based on strategy name
                if self._base_strategy_name == 'composite_ma_strategy':
                    from .composite_ma_strategy import CompositeMAStrategy
                    self._base_strategy = CompositeMAStrategy(f"{self.instance_name}_base")
                elif self._base_strategy_name == 'strategy':
                    # Use the main strategy class (component-based)
                    from ..base import Strategy as BaseStrategy
                    # Get the class from config if available
                    base_config = self.component_config.get('base_strategy_config', {})
                    strategy_class = base_config.get('class', 'Strategy')
                    if strategy_class == 'Strategy':
                        self._base_strategy = BaseStrategy(f"{self.instance_name}_base")
                else:
                    raise ValueError(f"Unknown base strategy: {self._base_strategy_name}")
                    
                # Initialize the base strategy
                self._base_strategy.initialize(self._context)
                self._base_strategy.start()
        
        # Subscribe to regime changes
        if self.event_bus:
            self.event_bus.subscribe(EventType.CLASSIFICATION, self._handle_regime_change)
            
        # Set initial parameters
        self._update_strategy_parameters()
        
        self.logger.info("RegimeAdaptiveStrategy started")
        
    def _load_regime_parameters(self, filepath: str):
        """Load regime-specific parameters from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract parameters from optimization results format
            if 'regimes' in data:
                # Combined regime results format
                for regime, regime_data in data['regimes'].items():
                    self._regime_parameters[regime] = regime_data.get('best_params', {})
            else:
                # Direct parameter format
                self._regime_parameters = data
                
            self.logger.info(f"Loaded regime parameters from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load regime parameters: {e}")
            
    def _handle_regime_change(self, event: Event):
        """Handle regime classification updates."""
        data = event.data
        
        if not isinstance(data, dict):
            return
            
        new_regime = data.get('classification', 'default')
        
        if new_regime != self._current_regime:
            self.logger.info(f"Regime change detected: {self._current_regime} -> {new_regime}")
            self._current_regime = new_regime
            self._update_strategy_parameters()
            
    def _update_strategy_parameters(self):
        """Update base strategy parameters based on current regime."""
        if not self._base_strategy:
            return
            
        # Get parameters for current regime
        params = self._regime_parameters.get(self._current_regime)
        
        if not params:
            # Fall back to default parameters
            params = self._regime_parameters.get('default', {})
            
        if params:
            self.logger.info(f"Updating strategy parameters for regime '{self._current_regime}': {params}")
            self._base_strategy.set_parameters(params)
        else:
            self.logger.warning(f"No parameters found for regime '{self._current_regime}'")
            
    def process_bar(self, bar_data: Dict[str, Any]) -> None:
        """Process bar data using the base strategy."""
        if not self._base_strategy:
            return
            
        # Delegate to base strategy
        self._base_strategy.process_bar(bar_data)
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space from base strategy."""
        if self._base_strategy and hasattr(self._base_strategy, 'get_parameter_space'):
            return self._base_strategy.get_parameter_space()
        else:
            # Return empty parameter space
            return ParameterSpace("regime_adaptive")
            
    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Set parameters for regime adaptive strategy.
        
        This can either set regime-specific parameters or base strategy parameters.
        """
        # Check if this is regime parameter update
        if 'regime_parameters' in params:
            self._regime_parameters = params['regime_parameters']
            self._update_strategy_parameters()
            return True
            
        # Otherwise pass to base strategy
        if self._base_strategy:
            return self._base_strategy.set_parameters(params)
            
        return False
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        params = {
            'current_regime': self._current_regime,
            'regime_parameters': self._regime_parameters
        }
        
        if self._base_strategy:
            params['current_strategy_params'] = self._base_strategy.get_parameters()
            
        return params