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
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Additional initialization for regime adaptation
        self._container = container
        self._regime_detector_key: str = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._regime_detector = None
        self._current_regime: str = "default"
        
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
        if not hasattr(event, 'payload') or not event.payload:
            return
            
        payload = event.payload
        if not isinstance(payload, dict):
            return
            
        new_regime = payload.get('classification')
        if new_regime and new_regime != self._current_regime:
            self.logger.info(f"Market regime changed from '{self._current_regime}' to '{new_regime}'.")
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
            new_params = self._regime_specific_params[regime]
            self.logger.info(f"Applying regime-specific parameters for '{regime}': {new_params}")
            self.set_parameters(new_params)
            self._last_applied_regime = regime
            
        # Fall back to overall best parameters if configured to do so
        elif self._fallback_to_overall_best and self._overall_best_params:
            self.logger.info(f"No specific parameters for regime '{regime}'. Falling back to overall best parameters: {self._overall_best_params}")
            self.set_parameters(self._overall_best_params)
            self._last_applied_regime = regime
            
        else:
            self.logger.warning(f"No parameters available for regime '{regime}' and no fallback configured. Keeping current parameters.")
    
    def start(self):
        """
        Overridden to subscribe to classification events.
        """
        super().start()
        
        # Subscribe to classification events to adapt to regime changes
        if self._event_bus:
            from src.core.event import EventType
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.name}' subscribed to CLASSIFICATION events.")
    
    def stop(self):
        """
        Overridden to unsubscribe from classification events.
        """
        # Unsubscribe from classification events
        if self._event_bus:
            try:
                from src.core.event import EventType
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.info(f"'{self.name}' unsubscribed from CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing from CLASSIFICATION events: {e}", exc_info=True)
                
        super().stop()