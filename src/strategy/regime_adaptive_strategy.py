# src/strategy/regime_adaptive_strategy.py
import logging
import json
import os
from typing import Dict, Any, Optional, List, Tuple

from src.strategy.ma_strategy import MAStrategy
from src.core.exceptions import ConfigurationError

class RegimeAdaptiveStrategy(MAStrategy):
    """
    A strategy that adapts its parameters based on the detected market regime.
    
    This strategy extends the MAStrategy and loads different parameter sets for different
    market regimes from a configuration file. When the market regime changes, the strategy
    automatically switches to the optimal parameters for the new regime.
    """
    
    def __init__(self, instance_name: str, config_loader, event_bus, container, component_config_key: Optional[str] = None):
        # Safety check for required configuration
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{instance_name}")
        try:
            # Try to get the symbol from the config - needed by MAStrategy
            if component_config_key:
                symbol = config_loader.get_config_value(f"{component_config_key}.symbol", None)
                if not symbol:
                    self.logger.warning(f"No symbol found in config for {component_config_key}")
                else:
                    self.logger.info(f"Found symbol in config: {symbol}")
        except Exception as e:
            self.logger.warning(f"Error checking config for symbol: {e}")
            
        # Initialize with a default symbol if needed
        self._symbol = "SPY"  # Default fallback symbol
        
        # Try to initialize the parent MAStrategy
        try:
            # Create a simple configuration and set it directly before parent initialization
            if component_config_key:
                try:
                    self._symbol = config_loader.get_config_value(f"{component_config_key}.symbol", "SPY")
                except:
                    self.logger.warning(f"Using default symbol {self._symbol}")
                    
            # Now initialize the parent with our prepared configuration
            super().__init__(instance_name, config_loader, event_bus, component_config_key)
            
        except Exception as e:
            self.logger.error(f"Error initializing MAStrategy parent: {e}")
            # Initialize basic properties that would normally be set by the parent
            from src.core.component import Component
            Component.__init__(self, instance_name, config_loader, event_bus, component_config_key)
            # Set up minimal required properties
            self._parameters = {"symbol": self._symbol}
            self.logger.warning(f"Initialized with minimal configuration and symbol: {self._symbol}")
            
        # Additional initialization for regime adaptation
        self._container = container
        self._regime_detector_key: str = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._regime_detector = None
        self._current_regime: str = "default"
        self._is_subscribed_to_classification = False  # Flag to track subscription status
        
        # Load regime-specific parameters
        self._params_file_path: str = self.get_specific_config('regime_params_file_path', "regime_optimized_parameters.json")
        self._fallback_to_overall_best: bool = self.get_specific_config('fallback_to_overall_best', True)
        self._regime_specific_params: Dict[str, Dict[str, Any]] = {}
        self._overall_best_params: Optional[Dict[str, Any]] = None
        
        # Track parameter changes
        self._active_params: Dict[str, Any] = {}
        self._last_applied_regime: Optional[str] = None
        
        # Load parameters from file
        try:
            self._load_parameters_from_file()
        except Exception as e:
            self.logger.error(f"Error loading regime parameters: {e}", exc_info=True)
            # We'll continue with default parameters and try to load the detector during setup
    
    def _load_parameters_from_file(self) -> None:
        """
        Load regime-specific parameters from the specified JSON file.
        """
        if not os.path.isfile(self._params_file_path):
            self.logger.warning(f"Regime parameters file not found: {self._params_file_path}")
            return
            
        try:
            with open(self._params_file_path, 'r') as f:
                data = json.load(f)
                
            # Extract regime-specific parameters
            if 'regime_best_parameters' in data:
                for regime, regime_data in data['regime_best_parameters'].items():
                    if 'parameters' in regime_data:
                        self._regime_specific_params[regime] = regime_data['parameters']
                        self.logger.info(f"Loaded parameters for regime '{regime}': {self._regime_specific_params[regime]}")
            
            # Extract overall best parameters as fallback
            if 'overall_best_parameters' in data:
                self._overall_best_params = data['overall_best_parameters']
                self.logger.info(f"Loaded overall best parameters as fallback: {self._overall_best_params}")
                
            self.logger.info(f"Successfully loaded parameters from {self._params_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error parsing regime parameters file: {e}", exc_info=True)
            raise
    
    def setup(self):
        """
        Overridden to also set up the regime detector connection.
        """
        super().setup()
        
        # Resolve the regime detector
        try:
            if self._container:
                self._regime_detector = self._container.resolve(self._regime_detector_key)
                if hasattr(self._regime_detector, 'get_current_classification') and callable(getattr(self._regime_detector, 'get_current_classification')):
                    initial_regime = self._regime_detector.get_current_classification()
                    self._current_regime = initial_regime if initial_regime else "default"
                    self.logger.info(f"Successfully resolved RegimeDetector. Initial regime: {self._current_regime}")
                    
                    # Apply initial parameters based on regime
                    self._apply_regime_specific_parameters(self._current_regime)
                else:
                    self.logger.warning(f"RegimeDetector '{self._regime_detector_key}' does not have required methods.")
        except Exception as e:
            self.logger.warning(f"Could not resolve RegimeDetector '{self._regime_detector_key}': {e}")
            self.logger.warning("Will continue with default parameters.")
    
    def on_classification_change(self, event):
        """
        Handle regime classification changes by updating parameters.
        """
        self.logger.info(f"'{self.name}' received classification event: {event}")
        
        if not hasattr(event, 'payload'):
            self.logger.warning(f"'{self.name}' received event without payload attribute: {event}")
            return
            
        if not event.payload:
            self.logger.warning(f"'{self.name}' received event with empty payload: {event}")
            return
            
        payload = event.payload
        self.logger.info(f"'{self.name}' classification event payload: {payload}")
        
        if not isinstance(payload, dict):
            self.logger.warning(f"'{self.name}' received non-dict payload: {payload}")
            return
            
        new_regime = payload.get('classification')
        self.logger.info(f"'{self.name}' extracted classification from payload: {new_regime}")
        
        if not new_regime:
            self.logger.warning(f"'{self.name}' missing 'classification' in payload: {payload}")
            return
            
        if new_regime == self._current_regime:
            self.logger.info(f"'{self.name}' regime unchanged: {new_regime}")
            return
            
        self.logger.info(f"'{self.name}' market regime changed from '{self._current_regime}' to '{new_regime}'.")
        self._current_regime = new_regime
        
        # Apply new parameters for the new regime
        self._apply_regime_specific_parameters(new_regime)
    
    def _apply_regime_specific_parameters(self, regime: str) -> None:
        """
        Apply parameters specific to the given regime.
        """
        if regime == self._last_applied_regime:
            # Avoid redundant parameter changes
            return
            
        # Check if we have parameters for this specific regime
        if regime in self._regime_specific_params:
            raw_params = self._regime_specific_params[regime]
            self.logger.info(f"Raw parameters for regime '{regime}': {raw_params}")
            
            # Map dotted parameters to the format expected by MAStrategy
            new_params = self._translate_parameters(raw_params)
            self.logger.info(f"Applying translated parameters for '{regime}': {new_params}")
            
            self.set_parameters(new_params)
            self._last_applied_regime = regime
            
        # Fall back to overall best parameters if configured to do so
        elif self._fallback_to_overall_best and self._overall_best_params:
            raw_params = self._overall_best_params
            self.logger.info(f"Raw overall best parameters: {raw_params}")
            
            # Map dotted parameters to the format expected by MAStrategy
            new_params = self._translate_parameters(raw_params)
            self.logger.info(f"Applying translated overall best parameters for '{regime}': {new_params}")
            
            self.set_parameters(new_params)
            self._last_applied_regime = regime
            
        else:
            self.logger.warning(f"No parameters available for regime '{regime}' and no fallback configured. Keeping current parameters.")
            
    def _translate_parameters(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate dotted parameter names in the config to the format expected by MAStrategy.
        
        For example, 'rsi_indicator.period' would be mapped to MAStrategy-specific parameter names.
        """
        # Parameters directly used by MAStrategy
        translated_params = {
            "short_window": raw_params.get("short_window"),
            "long_window": raw_params.get("long_window")
        }
        
        # Add any other parameters needed by the strategy
        # Check if we have RSI period
        if "rsi_indicator.period" in raw_params:
            translated_params["period"] = raw_params.get("rsi_indicator.period")
            
        # Check if we have RSI thresholds
        if "rsi_rule.oversold_threshold" in raw_params:
            translated_params["oversold_threshold"] = raw_params.get("rsi_rule.oversold_threshold")
        if "rsi_rule.overbought_threshold" in raw_params:
            translated_params["overbought_threshold"] = raw_params.get("rsi_rule.overbought_threshold")
            
        # Check for weights - fix the duplicate weights issue
        if "rsi_rule.weight" in raw_params:
            translated_params["rsi_weight"] = raw_params.get("rsi_rule.weight")
        if "ma_rule.weight" in raw_params:
            translated_params["ma_weight"] = raw_params.get("ma_rule.weight")
            
        # Remove any None values from the translated parameters
        return {k: v for k, v in translated_params.items() if v is not None}
    
    def start(self):
        """
        Overridden to subscribe to classification events.
        """
        super().start()
        
        # Subscribe to classification events to adapt to regime changes
        if self._event_bus and not self._is_subscribed_to_classification:
            from src.core.event import EventType
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self._is_subscribed_to_classification = True
            self.logger.info(f"'{self.name}' subscribed to CLASSIFICATION events.")
    
    def stop(self):
        """
        Overridden to unsubscribe from classification events.
        """
        # Unsubscribe from classification events
        if self._event_bus and self._is_subscribed_to_classification:
            try:
                from src.core.event import EventType
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self._is_subscribed_to_classification = False
                self.logger.info(f"'{self.name}' unsubscribed from CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from CLASSIFICATION events: {e}", exc_info=True)
                
        super().stop()