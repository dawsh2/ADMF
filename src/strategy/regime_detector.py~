# src/strategy/regime_detector.py
from typing import Any, Dict, Tuple, Optional, Type, TYPE_CHECKING

from .classifier import Classifier
from ..strategy.components.indicators.oscillators import RSIIndicator
from ..strategy.components.indicators.volatility import ATRIndicator
from ..strategy.components.indicators.trend import SimpleMATrendIndicator

# For type hinting BaseIndicator if you create/have one, otherwise remove
if TYPE_CHECKING:
    # Define a common interface or base class your indicators might adhere to,
    # or use a Union of known indicator types for type hinting.
    # For now, we'll assume they have common methods like update, value, ready.
    class BaseIndicatorInterface: # pragma: no cover
        def update(self, data: Dict[str, Any]): ...
        @property
        def value(self) -> Optional[float]: ...
        @property
        def ready(self) -> bool: ...
        def get_parameters(self) -> Dict[str, Any]: ...
        def set_parameters(self, params: Dict[str, Any]) -> bool: ...
else:
    # At runtime, BaseIndicatorInterface might not exist if it's just for type checking
    BaseIndicatorInterface = object


class RegimeDetector(Classifier):
    """
    Detects market regimes based on configurable indicators and thresholds.
    Implements stabilization logic to prevent rapid regime switching.
    """
    
    def __init__(self, 
                instance_name: str, 
                config_loader, 
                event_bus,
                component_config_key: str = None):
        """
        Initialize the RegimeDetector.

        Args:
            instance_name (str): The unique name of this classifier instance.
            config_loader: Configuration loader instance.
            event_bus: Event bus for publishing classification events and subscribing to data.
            component_config_key (str, optional): Configuration key for this component.
        """
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        # Stores instantiated indicator objects
        self._regime_indicators: Dict[str, BaseIndicatorInterface] = {} 
        
        # Configuration for regime thresholds, e.g., {"high_vol": {"volatility_atr": {"min": 0.5}}}
        self._regime_thresholds: Dict[str, Any] = self.get_specific_config("regime_thresholds", {})
        
        # Stabilization parameters
        self._min_regime_duration: int = self.get_specific_config("min_regime_duration", 1)
        self._current_regime_duration: int = 0
        self._pending_regime: Optional[str] = None
        self._pending_duration: int = 0

        self.logger.info(f"RegimeDetector '{self.name}' initialized. Min duration: {self._min_regime_duration}, Thresholds: {self._regime_thresholds}")

    def _get_indicator_class(self, indicator_type_name: str) -> Optional[Type[BaseIndicatorInterface]]:
        """
        Factory method to get indicator class by type name.
        Maps string identifiers (used in config) to actual indicator classes.
        """
        if indicator_type_name == "rsi":
            return RSIIndicator
        elif indicator_type_name == "atr":
            return ATRIndicator
        elif indicator_type_name == "simple_ma_trend":
            return SimpleMATrendIndicator
        else:
            self.logger.error(f"Unknown indicator type specified in config: '{indicator_type_name}' for RegimeDetector '{self.name}'.")
            return None

    def _setup_regime_indicators(self):
        """
        Initialize indicators used for regime detection based on configuration.
        Example config structure for an indicator:
        "indicators": {
            "my_rsi": {"type": "rsi", "parameters": {"period": 14}},
            "my_atr": {"type": "atr", "parameters": {"period": 20}},
            "my_trend": {"type": "simple_ma_trend", "parameters": {"short_period": 10, "long_period": 30}}
        }
        """
        indicator_configs: Dict[str, Any] = self.get_specific_config("indicators", {})
        if not indicator_configs:
            self.logger.warning(f"No indicators configured for RegimeDetector '{self.name}'. It will likely always classify as 'default'.")
            return

        for indicator_instance_name, config_dict in indicator_configs.items():
            indicator_type_str = config_dict.get("type")
            params_from_config = config_dict.get("parameters", {}) # These are passed to indicator's __init__
            
            if not indicator_type_str:
                self.logger.error(f"Indicator type not specified for '{indicator_instance_name}' in RegimeDetector '{self.name}' config.")
                continue

            IndicatorClass = self._get_indicator_class(indicator_type_str)
            if IndicatorClass:
                try:
                    # Pass parameters directly to the indicator's constructor
                    # The indicator's __init__ should handle these parameters.
                    # Also pass other common args if your indicators expect them (like instance_name, config_loader, event_bus)
                    # For simplicity, assuming indicators primarily take 'parameters' dict or specific args like 'period'.
                    # Your indicator __init__ methods take 'parameters' dict.
                    self._regime_indicators[indicator_instance_name] = IndicatorClass(
                        instance_name=f"{self.name}_{indicator_instance_name}", # Give unique name
                        parameters=params_from_config,
                        # config_loader=self._config_loader, # if indicators need these
                        # event_bus=self._event_bus         # if indicators need these
                    )
                    self.logger.info(f"Initialized indicator '{indicator_instance_name}' of type '{indicator_type_str}' for RegimeDetector '{self.name}'.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize indicator '{indicator_instance_name}' of type '{indicator_type_str}': {e}", exc_info=True)
            else:
                # Error already logged by _get_indicator_class
                pass
        
        if not self._regime_indicators:
            self.logger.warning(f"No indicators were successfully initialized for RegimeDetector '{self.name}'. Classification may be unreliable.")

    def setup(self):
        """Initialize indicators and call parent's setup to subscribe to events."""
        super().setup() # This subscribes to BAR events via on_bar in Classifier base
        self._setup_regime_indicators()
        self.logger.info(f"RegimeDetector '{self.name}' setup complete.")
    
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify market data into a regime label.
        This method is called by the parent Classifier's on_bar method.
        
        Args:
            data: Market data (e.g., a bar) to classify.
            
        Returns:
            str: Regime label (e.g., "high_volatility", "trending_up").
        """
        if not self._regime_indicators or not self._regime_thresholds:
            raw_detected_regime = self._current_classification if self._current_classification is not None else "default"
            self.logger.debug(f"RegimeDetector '{self.name}' not fully configured (indicators/thresholds missing). Raw detection: '{raw_detected_regime}'.")
        else:
            # 1. Update all underlying indicators
            for indicator_name, indicator_obj in self._regime_indicators.items():
                try:
                    indicator_obj.update(data)
                except Exception as e:
                    self.logger.error(f"Error updating indicator '{indicator_name}' in {self.name}: {e}", exc_info=True)
            
            # 2. Get current values from indicators
            indicator_values: Dict[str, Optional[float]] = {}
            for name, indicator_obj in self._regime_indicators.items():
                if indicator_obj.ready: # Check if indicator has enough data
                    indicator_values[name] = indicator_obj.value
                else:
                    self.logger.debug(f"Indicator '{name}' not ready in {self.name}. Its value will not be used for classification this bar.")
            
            # 3. Apply regime classification rules
            raw_detected_regime = self._determine_regime_from_indicators(indicator_values)
            self.logger.debug(f"RegimeDetector '{self.name}' raw detection: '{raw_detected_regime}' based on values: {indicator_values}")

        # 4. Apply stabilization logic
        final_regime = self._apply_stabilization(raw_detected_regime)
        # self.logger.debug(f"RegimeDetector '{self.name}' final stabilized regime: '{final_regime}'") # Already logged in _apply_stabilization
        
        return final_regime
    
    def _determine_regime_from_indicators(self, indicator_values: Dict[str, Optional[float]]) -> str:
        """
        Apply rules to classify the current regime based on indicator values.
        Returns "default" if no specific regime matches or if necessary indicators are not ready/present.
        """
        if not indicator_values and self._regime_thresholds: # No ready indicators but thresholds exist
            self.logger.debug(f"No valid indicator values available for regime determination in {self.name}, but thresholds are defined. Returning 'default'.")
            return "default"
        if not self._regime_thresholds: # No thresholds defined
             self.logger.debug(f"No regime thresholds defined for {self.name}. Returning 'default'.")
             return "default"
            
        for regime_name, conditions in self._regime_thresholds.items():
            if regime_name == "default": # Skip explicit "default" in thresholds, it's a fallback
                continue

            matches_all_conditions = True
            if not isinstance(conditions, dict) or not conditions: # Ensure conditions is a non-empty dict
                self.logger.warning(f"Conditions for regime '{regime_name}' in {self.name} are invalid (not a dict or empty). Skipping this regime.")
                continue

            for indicator_instance_name_in_config, threshold_config in conditions.items():
                if indicator_instance_name_in_config not in indicator_values or indicator_values[indicator_instance_name_in_config] is None:
                    matches_all_conditions = False
                    self.logger.debug(f"Regime '{regime_name}': Indicator '{indicator_instance_name_in_config}' missing or not ready. Condition not met.")
                    break 
                
                value = indicator_values[indicator_instance_name_in_config]
                
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")
                
                if min_val is not None and value < float(min_val):
                    matches_all_conditions = False
                    break
                if max_val is not None and value > float(max_val):
                    matches_all_conditions = False
                    break
            
            if matches_all_conditions:
                self.logger.debug(f"Regime '{regime_name}' matched in {self.name}.")
                return regime_name
                
        self.logger.debug(f"No specific regime matched in {self.name} with indicator values {indicator_values}. Returning 'default'.")
        return "default"

    def _apply_stabilization(self, detected_regime: str) -> str:
        """
        Apply stabilization logic to prevent rapid regime switching.
        Uses self._current_classification (from parent Classifier) as the true current regime state.
        """
        true_current_regime = self._current_classification 
        
        if true_current_regime is None: # Initial state, first bar processed by on_bar
            self.logger.debug(f"Stabilization in {self.name}: Initial state. Setting current duration to 1 for detected regime '{detected_regime}'.")
            self._current_regime_duration = 1
            self._pending_regime = None 
            self._pending_duration = 0
            # self._current_classification will be set by parent class after this method returns.
            # So, the first detected_regime becomes the initial true_current_regime.
            return detected_regime 

        if detected_regime == true_current_regime:
            self._current_regime_duration += 1
            self._pending_regime = None 
            self._pending_duration = 0
            self.logger.debug(f"Stabilization in {self.name}: Detected_raw '{detected_regime}' matches current_actual '{true_current_regime}'. Duration: {self._current_regime_duration}.")
            return true_current_regime
        else:
            # Raw detected regime is different from the current actual regime
            if self._pending_regime == detected_regime:
                self._pending_duration += 1
                self.logger.debug(f"Stabilization in {self.name}: Pending_raw '{detected_regime}' confirmed again. Pending_duration: {self._pending_duration}.")
            else:
                self._pending_regime = detected_regime
                self._pending_duration = 1 # Reset counter for new pending regime
                self.logger.debug(f"Stabilization in {self.name}: New pending_raw '{detected_regime}' initiated. Pending_duration: {self._pending_duration}.")

            if self._pending_duration >= self._min_regime_duration:
                self.logger.info(f"Stabilization in {self.name}: Regime changing from '{true_current_regime}' to '{self._pending_regime}' after meeting min_duration {self._min_regime_duration}.")
                self._current_regime_duration = 1 # Reset duration for the new regime
                newly_confirmed_regime = self._pending_regime
                self._pending_regime = None 
                self._pending_duration = 0
                return newly_confirmed_regime
            else:
                self._current_regime_duration +=1 
                self.logger.debug(f"Stabilization in {self.name}: Pending_raw '{self._pending_regime}' (duration {self._pending_duration}/{self._min_regime_duration}) not yet stable. Maintaining current_actual '{true_current_regime}' (duration {self._current_regime_duration}).")
                return true_current_regime
    
    def get_regime_data(self) -> Dict[str, Any]:
        """
        Get additional data about the current regime, including duration and indicator values.
        """
        indicator_values = {}
        for name, indicator_obj in self._regime_indicators.items():
            if indicator_obj.ready:
                indicator_values[name] = indicator_obj.value

        return {
            'regime': self.get_current_classification(), 
            'duration_in_regime': self._current_regime_duration,
            'indicators': indicator_values,
            'pending_regime_info': {
                'pending_label': self._pending_regime,
                'pending_duration_bars': self._pending_duration,
                'min_duration_for_change': self._min_regime_duration
            } if self._pending_regime else None
        }
