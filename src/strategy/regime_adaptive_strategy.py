# src/strategy/regime_adaptive_strategy.py
import logging
import json
import os
from typing import Dict, Any, Optional, List, Tuple

from src.strategy.ma_strategy import MAStrategy
from src.core.exceptions import ConfigurationError
from src.core.event import Event, EventType

class RegimeAdaptiveStrategy(MAStrategy):
    """
    A strategy that adapts its parameters based on the detected market regime.
    
    This strategy extends the MAStrategy and loads different parameter sets for different
    market regimes from a configuration file. When the market regime changes, the strategy
    automatically switches to the optimal parameters for the new regime.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        # Initialize parent MAStrategy first
        super().__init__(instance_name, config_key)
        
        # Initialize internal state for regime adaptation
        self._regime_detector_key: str = "MyPrimaryRegimeDetector"
        self._regime_detector = None
        self._current_regime: str = "default"
        
        # Parameters file settings
        self._params_file_path: str = "regime_optimized_parameters.json"
        self._fallback_to_overall_best: bool = True
        self._regime_specific_params: Dict[str, Dict[str, Any]] = {}
        self._overall_best_params: Optional[Dict[str, Any]] = None
        
        # Track parameter changes
        self._active_params: Dict[str, Any] = {}
        self._last_applied_regime: Optional[str] = None
    
    def _load_parameters_from_file(self) -> None:
        """
        Load regime-specific parameters from the specified JSON file.
        """
        if not os.path.isfile(self._params_file_path):
            self.logger.warning(f"Regime parameters file not found: {self._params_file_path}")
            return
            
        try:
            self.logger.info(f"Loading regime parameters from: {self._params_file_path}")
            self.logger.log(35, f"===== LOADING REGIME ADAPTIVE PARAMETERS =====")
            self.logger.log(35, f"Parameter file: {self._params_file_path}")
            with open(self._params_file_path, 'r') as f:
                data = json.load(f)
                
            # Extract regime-specific parameters and weights
            if 'regime_best_parameters' in data:
                for regime, regime_data in data['regime_best_parameters'].items():
                    if 'parameters' in regime_data:
                        # Handle nested parameter structure - extract actual parameters
                        if 'parameters' in regime_data['parameters']:
                            # New nested format: regime_data['parameters']['parameters']
                            base_params = regime_data['parameters']['parameters'].copy()
                        else:
                            # Direct format: regime_data['parameters'] contains the params
                            base_params = regime_data['parameters'].copy()
                        
                        # Add per-regime weights if they exist
                        if 'weights' in regime_data and regime_data['weights']:
                            weights = regime_data['weights']
                            self.logger.info(f"Found optimized weights for regime '{regime}': {weights}")
                            base_params.update(weights)
                        else:
                            self.logger.info(f"No optimized weights found for regime '{regime}', using default weights")
                        
                        self._regime_specific_params[regime] = base_params
                        self.logger.info(f"Loaded complete parameters for regime '{regime}': {self._regime_specific_params[regime]}")
                        self.logger.log(35, f"Regime '{regime}' parameters loaded: {base_params}")
            
            # Extract overall best parameters as fallback
            if 'overall_best_parameters' in data:
                self._overall_best_params = data['overall_best_parameters']
                self.logger.info(f"Loaded overall best parameters as fallback: {self._overall_best_params}")
                
            self.logger.info(f"Successfully loaded parameters from {self._params_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error parsing regime parameters file: {e}", exc_info=True)
            raise
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Call parent's _initialize first
        super()._initialize()
        
        # Load regime-specific configuration
        self._regime_detector_key = self.get_specific_config('regime_detector_service_name', "MyPrimaryRegimeDetector")
        self._params_file_path = self.get_specific_config('regime_params_file_path', "regime_optimized_parameters.json")
        self._fallback_to_overall_best = self.get_specific_config('fallback_to_overall_best', True)
        
        # Load parameters from file
        try:
            self._load_parameters_from_file()
        except Exception as e:
            self.logger.error(f"Error loading regime parameters: {e}", exc_info=True)
            # Continue with default parameters
    
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        # Call parent's event subscriptions first
        super().initialize_event_subscriptions()
        
        # Add our classification subscription
        if self.subscription_manager:
            self.subscription_manager.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.instance_name}' subscribed to CLASSIFICATION events.")
    
    def setup(self):
        """
        Overridden to also set up the regime detector connection.
        """
        super().setup()
        
        # Resolve the regime detector
        try:
            if self.container:
                self._regime_detector = self.container.resolve(self._regime_detector_key)
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
        self.logger.info(f"'{self.instance_name}' received classification event: {event}")
        
        if not hasattr(event, 'payload'):
            self.logger.warning(f"'{self.instance_name}' received event without payload attribute: {event}")
            return
            
        if not event.payload:
            self.logger.warning(f"'{self.instance_name}' received event with empty payload: {event}")
            return
            
        payload = event.payload
        self.logger.info(f"'{self.instance_name}' classification event payload: {payload}")
        
        if not isinstance(payload, dict):
            self.logger.warning(f"'{self.instance_name}' received non-dict payload: {payload}")
            return
            
        new_regime = payload.get('classification')
        self.logger.info(f"'{self.instance_name}' extracted classification from payload: {new_regime}")
        
        if not new_regime:
            self.logger.warning(f"'{self.instance_name}' missing 'classification' in payload: {payload}")
            return
            
        if new_regime == self._current_regime:
            self.logger.info(f"'{self.instance_name}' regime unchanged: {new_regime}")
            return
            
        self.logger.info(f"'{self.instance_name}' market regime changed from '{self._current_regime}' to '{new_regime}'.")
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
            self.logger.log(35, f"REGIME CHANGE: Switching to '{regime}' regime")
            self.logger.log(35, f"Loading regime-specific parameters: {raw_params}")
            
            # Map dotted parameters to the format expected by MAStrategy
            new_params = self._translate_parameters(raw_params)
            self.logger.info(f"Applying translated parameters for '{regime}': {new_params}")
            self.logger.log(35, f"Translated parameters for '{regime}': {new_params}")
            
            self.set_parameters(new_params)
            self._last_applied_regime = regime
            
        # Fall back to overall best parameters if configured to do so
        elif self._fallback_to_overall_best and self._overall_best_params:
            raw_params = self._overall_best_params
            self.logger.info(f"Raw overall best parameters: {raw_params}")
            self.logger.log(35, f"REGIME CHANGE: Switching to '{regime}' regime (using overall best parameters)")
            self.logger.log(35, f"Loading overall best parameters: {raw_params}")
            
            # Map dotted parameters to the format expected by MAStrategy
            new_params = self._translate_parameters(raw_params)
            self.logger.info(f"Applying translated overall best parameters for '{regime}': {new_params}")
            self.logger.log(35, f"Translated overall best parameters for '{regime}': {new_params}")
            
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
        Start the regime adaptive strategy.
        """
        super().start()
        # Event subscriptions are now handled by initialize_event_subscriptions()
        self.logger.info(f"'{self.instance_name}' started with regime adaptation enabled.")
    
    def stop(self):
        """
        Stop the regime adaptive strategy.
        """
        # Event unsubscription is now handled by subscription_manager in teardown()
        super().stop()
        self.logger.info(f"'{self.instance_name}' stopped.")
    
    def teardown(self):
        """
        Clean up resources during component teardown.
        """
        # Clear regime-specific state
        self._regime_detector = None
        self._regime_specific_params.clear()
        self._overall_best_params = None
        self._active_params.clear()
        
        # Call parent teardown
        super().teardown()