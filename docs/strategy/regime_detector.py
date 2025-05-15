# src/strategy/regime_detector.py
from typing import Any, Dict, Tuple, Optional, Type # Added Type for indicator factory

from .classifier import Classifier
# Placeholder for actual indicator imports. 
# You'll need to replace these with your actual indicator paths and classes.
# from ..core.indicator import BaseIndicator # Assuming a base indicator class
# from ..indicators.volatility import VolatilityIndicator # Example
# from ..indicators.trend import TrendStrengthIndicator # Example

# --- Placeholder Indicator Classes (Replace with your actual indicators) ---
# These are minimal placeholders to make RegimeDetector runnable for now.
# You should replace these with your actual, fully implemented indicator classes.
class BaseIndicator:
    def __init__(self, **params):
        self.params = params
        self.current_value = None
        self.is_ready = False
        # self.logger = logging.getLogger(self.__class__.__name__) # If indicators need logging

    def update(self, data: Dict[str, Any]):
        # Subclasses should implement this
        # self.logger.debug(f"Updating with data: {data}")
        pass

    @property
    def value(self) -> Optional[float]: # Assuming indicators return float or None
        return self.current_value

    @property
    def ready(self) -> bool:
        return self.is_ready

class VolatilityIndicator(BaseIndicator): # Example Placeholder
    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period
        self.prices_high = []
        self.prices_low = []
        self.prices_close = []
        # self.logger.info(f"VolatilityIndicator initialized with period {self.period}")

    def update(self, data: Dict[str, Any]):
        # Simplified ATR-like calculation for placeholder
        high = data.get('high')
        low = data.get('low')
        close = data.get('close')
        
        if high is None or low is None or close is None:
            # self.logger.warning("Missing high, low, or close in data for VolatilityIndicator.")
            return

        self.prices_high.append(float(high))
        self.prices_low.append(float(low))
        self.prices_close.append(float(close))

        if len(self.prices_high) > self.period + 1: # Need one previous close
            self.prices_high.pop(0)
            self.prices_low.pop(0)
            self.prices_close.pop(0)

        if len(self.prices_close) >= self.period +1 : # +1 for prev_close calculation
            trs = []
            for i in range(1, len(self.prices_close)): # Start from the second element to use previous close
                prev_close = self.prices_close[i-1]
                tr = max(self.prices_high[i] - self.prices_low[i], 
                         abs(self.prices_high[i] - prev_close), 
                         abs(self.prices_low[i] - prev_close))
                trs.append(tr)
            
            if len(trs) >= self.period:
                self.current_value = sum(trs[-self.period:]) / self.period
                self.is_ready = True
                # self.logger.debug(f"VolatilityIndicator updated. Value: {self.current_value}, Ready: {self.is_ready}")


class TrendStrengthIndicator(BaseIndicator): # Example Placeholder
    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = period
        self.prices = []
        # self.logger.info(f"TrendStrengthIndicator initialized with period {self.period}")

    def update(self, data: Dict[str, Any]):
        close = data.get('close')
        if close is None:
            # self.logger.warning("Missing close in data for TrendStrengthIndicator.")
            return
            
        self.prices.append(float(close))
        if len(self.prices) > self.period:
            self.prices.pop(0)
        
        if len(self.prices) == self.period:
            # Simplified: difference between current price and start of period price
            # A real trend strength indicator (e.g., ADX) would be more complex.
            self.current_value = (self.prices[-1] - self.prices[0]) / self.prices[0] * 100 if self.prices[0] != 0 else 0
            self.is_ready = True
            # self.logger.debug(f"TrendStrengthIndicator updated. Value: {self.current_value}, Ready: {self.is_ready}")
# --- End of Placeholder Indicator Classes ---


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
        self._regime_indicators: Dict[str, BaseIndicator] = {} 
        
        # Configuration for regime thresholds, e.g., {"high_vol": {"volatility_atr": {"min": 0.5}}}
        self._regime_thresholds: Dict[str, Any] = self.get_specific_config("regime_thresholds", {})
        
        # Stabilization parameters
        self._min_regime_duration: int = self.get_specific_config("min_regime_duration", 1) # Default to 1 (no stabilization if not set)
        self._current_regime_duration: int = 0
        self._pending_regime: Optional[str] = None
        self._pending_duration: int = 0

        self.logger.info(f"RegimeDetector '{self.name}' initialized. Min duration: {self._min_regime_duration}, Thresholds: {self._regime_thresholds}")

    def _get_indicator_class(self, indicator_type_name: str) -> Optional[Type[BaseIndicator]]:
        """
        Factory method to get indicator class by type name.
        This should be adapted to your project's indicator loading mechanism.
        """
        if indicator_type_name == "volatility": # Corresponds to your YAML 'type: "volatility"'
            return VolatilityIndicator
        elif indicator_type_name == "trend_strength": # Corresponds to your YAML 'type: "trend_strength"'
            return TrendStrengthIndicator
        # Add other indicator types here
        # elif indicator_type_name == "my_custom_indicator":
        #     from ..indicators.custom import MyCustomIndicator
        #     return MyCustomIndicator
        else:
            self.logger.error(f"Unknown indicator type: {indicator_type_name}")
            return None

    def _setup_regime_indicators(self):
        """
        Initialize indicators used for regime detection based on configuration.
        Example config:
        "indicators": {
            "volatility_atr": {"type": "volatility", "parameters": {"period": 14}},
            "trend_adx": {"type": "trend_strength", "parameters": {"period": 20}}
        }
        """
        indicator_configs: Dict[str, Any] = self.get_specific_config("indicators", {})
        if not indicator_configs:
            self.logger.warning(f"No indicators configured for RegimeDetector '{self.name}'. Will always return 'default' regime.")
            return

        for indicator_name, config_dict in indicator_configs.items():
            indicator_type_str = config_dict.get("type")
            params = config_dict.get("parameters", {})
            
            if not indicator_type_str:
                self.logger.error(f"Indicator type not specified for '{indicator_name}' in RegimeDetector '{self.name}' config.")
                continue

            IndicatorClass = self._get_indicator_class(indicator_type_str)
            if IndicatorClass:
                try:
                    self._regime_indicators[indicator_name] = IndicatorClass(**params)
                    self.logger.info(f"Initialized indicator '{indicator_name}' of type '{indicator_type_str}' for RegimeDetector '{self.name}'.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize indicator '{indicator_name}': {e}", exc_info=True)
            else:
                self.logger.warning(f"Could not find or create indicator class for type '{indicator_type_str}' (indicator name: '{indicator_name}').")
        
        if not self._regime_indicators:
            self.logger.warning(f"No indicators were successfully initialized for RegimeDetector '{self.name}'. Will likely always return 'default' regime.")


    def setup(self):
        """Initialize indicators and call parent's setup to subscribe to events."""
        super().setup() # This subscribes to BAR events via on_bar
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
            # If no indicators or thresholds, cannot determine a specific regime.
            # The stabilization logic will handle this by essentially keeping it 'default'
            # or the last known valid regime if stabilization is configured.
            # To avoid issues, we explicitly return the current classification if unconfigured,
            # which will be 'None' initially, then 'default' after first stabilization pass.
            if self._current_classification is None: # First ever call before any stabilization
                 raw_detected_regime = "default"
            else:
                 raw_detected_regime = self._current_classification # Maintain current if not configured
            self.logger.debug(f"RegimeDetector '{self.name}' not fully configured (indicators/thresholds missing). Raw detection: '{raw_detected_regime}'.")

        else:
            # 1. Update all underlying indicators with new market data
            for indicator_name, indicator_obj in self._regime_indicators.items():
                try:
                    indicator_obj.update(data)
                except Exception as e:
                    self.logger.error(f"Error updating indicator '{indicator_name}' in {self.name}: {e}", exc_info=True)
            
            # 2. Get current values from indicators
            indicator_values: Dict[str, Optional[float]] = {}
            for name, indicator_obj in self._regime_indicators.items():
                if indicator_obj.ready:
                    indicator_values[name] = indicator_obj.value
                else:
                    # If an indicator is not ready, its value is None or it's ignored.
                    # This might affect regime classification.
                    self.logger.debug(f"Indicator '{name}' not ready in {self.name}. Its value will not be used for classification this bar.")
            
            # 3. Apply regime classification rules based on indicator values
            raw_detected_regime = self._determine_regime_from_indicators(indicator_values)
            self.logger.debug(f"RegimeDetector '{self.name}' raw detection: '{raw_detected_regime}' based on values: {indicator_values}")

        # 4. Apply stabilization logic
        final_regime = self._apply_stabilization(raw_detected_regime)
        self.logger.debug(f"RegimeDetector '{self.name}' final stabilized regime: '{final_regime}'")
        
        return final_regime
    
    def _determine_regime_from_indicators(self, indicator_values: Dict[str, Optional[float]]) -> str:
        """
        Apply rules to classify the current regime based on indicator values.
        Returns "default" if no specific regime matches or if indicators are not ready.
        """
        if not indicator_values: # If all indicators are not ready or no indicators
            self.logger.debug(f"No valid indicator values available for regime determination in {self.name}. Returning 'default'.")
            return "default"
            
        # Check each defined regime in the thresholds configuration
        for regime_name, conditions in self._regime_thresholds.items():
            matches_all_conditions = True
            if not isinstance(conditions, dict):
                self.logger.warning(f"Conditions for regime '{regime_name}' in {self.name} are not a dictionary. Skipping this regime.")
                continue

            for indicator_key, threshold_config in conditions.items():
                if indicator_key not in indicator_values or indicator_values[indicator_key] is None:
                    # Required indicator for this regime is missing or not ready
                    matches_all_conditions = False
                    self.logger.debug(f"Regime '{regime_name}': Indicator '{indicator_key}' missing or not ready. Condition not met.")
                    break 
                
                value = indicator_values[indicator_key]
                
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")
                
                # Check min threshold
                if min_val is not None and value < float(min_val):
                    matches_all_conditions = False
                    break
                # Check max threshold
                if max_val is not None and value > float(max_val):
                    matches_all_conditions = False
                    break
            
            if matches_all_conditions:
                self.logger.debug(f"Regime '{regime_name}' matched in {self.name}.")
                return regime_name # First matching regime is returned
                
        self.logger.debug(f"No specific regime matched in {self.name} with indicator values {indicator_values}. Returning 'default'.")
        return "default"  # Default regime if no specific regime matched

    def _apply_stabilization(self, detected_regime: str) -> str:
        """
        Apply stabilization logic to prevent rapid regime switching.
        Uses self._current_classification (from parent) as the true current regime.
        """
        true_current_regime = self._current_classification # This is the state before this bar's classification
        
        if true_current_regime is None: # Initial state, first bar processed
            self.logger.debug(f"Stabilization in {self.name}: Initial state. Setting current duration to 1 for detected regime '{detected_regime}'.")
            self._current_regime_duration = 1
            self._pending_regime = None # No pending regime yet
            self._pending_duration = 0
            return detected_regime # The first detected regime becomes current immediately

        # If the newly detected raw regime is the same as the current actual regime
        if detected_regime == true_current_regime:
            self._current_regime_duration += 1
            self._pending_regime = None # Clear any pending regime
            self._pending_duration = 0
            self.logger.debug(f"Stabilization in {self.name}: Detected regime '{detected_regime}' matches current '{true_current_regime}'. Duration: {self._current_regime_duration}.")
            return true_current_regime
        else:
            # The raw detected regime is different from the current actual regime
            if self._pending_regime == detected_regime:
                # It's the same as the one we were waiting for confirmation on
                self._pending_duration += 1
                self.logger.debug(f"Stabilization in {self.name}: Pending regime '{detected_regime}' confirmed again. Pending duration: {self._pending_duration}.")
            else:
                # A new, different regime is now detected (or first time a different one is seen)
                self._pending_regime = detected_regime
                self._pending_duration = 1
                self.logger.debug(f"Stabilization in {self.name}: New pending regime '{detected_regime}' initiated. Pending duration: {self._pending_duration}.")

            # Check if the pending regime has met the minimum duration
            if self._pending_duration >= self._min_regime_duration:
                self.logger.info(f"Stabilization in {self.name}: Regime changing from '{true_current_regime}' to '{self._pending_regime}' after meeting min duration {self._min_regime_duration}.")
                self._current_regime_duration = 1 # Reset duration for the new regime
                newly_confirmed_regime = self._pending_regime
                self._pending_regime = None # Clear pending state
                self._pending_duration = 0
                return newly_confirmed_regime
            else:
                # Pending regime has not met minimum duration, so stick with the current actual regime
                self._current_regime_duration +=1 # Increment duration of the current (persisting) regime
                self.logger.debug(f"Stabilization in {self.name}: Pending regime '{self._pending_regime}' not yet stable. Maintaining current regime '{true_current_regime}'. Current duration: {self._current_regime_duration}.")
                return true_current_regime
    
    def get_regime_data(self) -> Dict[str, Any]:
        """
        Get additional data about the current regime, including duration and indicator values.
        
        Returns:
            Dict containing current regime label, its duration, and current indicator values.
        """
        indicator_values = {}
        for name, indicator_obj in self._regime_indicators.items():
            if indicator_obj.ready: # Only include values from ready indicators
                indicator_values[name] = indicator_obj.value

        return {
            'regime': self.get_current_classification(), # From parent Classifier class
            'duration_in_regime': self._current_regime_duration,
            'indicators': indicator_values,
            'pending_regime_info': { # For debugging or advanced insight
                'pending_label': self._pending_regime,
                'pending_duration_bars': self._pending_duration,
                'min_duration_for_change': self._min_regime_duration
            } if self._pending_regime else None
        }

