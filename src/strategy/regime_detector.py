# src/strategy/regime_detector.py
from typing import Any, Dict, Tuple, Optional, Type, TYPE_CHECKING

from src.strategy.classifier import Classifier
from src.core.event import EventType
# Assuming these are the correct paths based on your project structure
from ..strategy.components.indicators.oscillators import RSIIndicator
from ..strategy.components.indicators.volatility import ATRIndicator
from ..strategy.components.indicators.trend import SimpleMATrendIndicator

if TYPE_CHECKING:
    class BaseIndicatorInterface: # pragma: no cover
        def update(self, data: Dict[str, Any]): ...
        @property
        def value(self) -> Optional[float]: ...
        @property
        def ready(self) -> bool: ...
        def get_parameters(self) -> Dict[str, Any]: ...
        def set_parameters(self, params: Dict[str, Any]) -> bool: ...
else:
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
        super().__init__(instance_name, config_loader, event_bus, component_config_key)
        
        self._regime_indicators: Dict[str, BaseIndicatorInterface] = {} 
        self._regime_thresholds: Dict[str, Any] = self.get_specific_config("regime_thresholds", {})
        self._min_regime_duration: int = self.get_specific_config("min_regime_duration", 1)
        self._current_regime_duration: int = 0
        self._pending_regime: Optional[str] = None
        self._pending_duration: int = 0

        self.logger.info(f"RegimeDetector '{self.name}' initialized. Min duration: {self._min_regime_duration}, Thresholds: {self._regime_thresholds}")

    def _get_indicator_class(self, indicator_type_name: str) -> Optional[Type[BaseIndicatorInterface]]:
        if indicator_type_name == "rsi":
            return RSIIndicator
        elif indicator_type_name == "atr":
            return ATRIndicator
        elif indicator_type_name == "simple_ma_trend":
            return SimpleMATrendIndicator
        else:
            self.logger.error(f"Unknown indicator type specified: '{indicator_type_name}' for RegimeDetector '{self.name}'.")
            return None

    def _setup_regime_indicators(self):
        indicator_configs: Dict[str, Any] = self.get_specific_config("indicators", {})
        if not indicator_configs:
            self.logger.warning(f"No indicators configured for RegimeDetector '{self.name}'. It will likely always classify as 'default'.")
            return

        for indicator_instance_name, config_dict in indicator_configs.items():
            indicator_type_str = config_dict.get("type")
            params_from_config = config_dict.get("parameters", {}) 
            
            if not indicator_type_str:
                self.logger.error(f"Indicator type not specified for '{indicator_instance_name}' in {self.name} config.")
                continue

            IndicatorClass = self._get_indicator_class(indicator_type_str)
            if IndicatorClass:
                try:
                    self._regime_indicators[indicator_instance_name] = IndicatorClass(
                        instance_name=f"{self.name}_{indicator_instance_name}", 
                        config_loader=self._config_loader,
                        event_bus=self._event_bus,     
                        component_config_key=None, # Indicators get params directly, not separate config block by key
                        parameters=params_from_config
                    )
                    self.logger.info(f"Initialized indicator '{indicator_instance_name}' of type '{indicator_type_str}' for {self.name}.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize indicator '{indicator_instance_name}' for {self.name}: {e}", exc_info=True)
        
        if not self._regime_indicators:
            self.logger.warning(f"No indicators were successfully initialized for {self.name}. Classification may be unreliable.")

    def setup(self):
        super().setup()
        self._setup_regime_indicators()
        self.logger.info(f"RegimeDetector '{self.name}' setup complete.")
    
    def classify(self, data: Dict[str, Any]) -> str:
        """ Classify market data into a regime label. """
        current_bar_timestamp = data.get('timestamp', 'N/A') # Get timestamp for logging

        if not self._regime_indicators or not self._regime_thresholds:
            raw_detected_regime = self._current_classification if self._current_classification is not None else "default"
            # self.logger.debug(f"RegimeDetector '{self.name}' at {current_bar_timestamp}: Not fully configured. Raw: '{raw_detected_regime}'.") # Can be noisy
        else:
            for indicator_name, indicator_obj in self._regime_indicators.items():
                try:
                    if hasattr(indicator_obj, 'update') and callable(getattr(indicator_obj, 'update')):
                        indicator_obj.update(data) # Pass the full bar data
                    else:
                        self.logger.warning(f"Indicator '{indicator_name}' in {self.name} has no callable update method.")
                except Exception as e:
                    self.logger.error(f"Error updating indicator '{indicator_name}' in {self.name} for bar {current_bar_timestamp}: {e}", exc_info=True)
            
            indicator_values: Dict[str, Optional[float]] = {}
            all_necessary_indicators_ready = True # Flag to track if all indicators needed by *any* rule are ready
            
            # Collect values from all configured indicators, note if any are not ready
            for name, indicator_obj in self._regime_indicators.items():
                if hasattr(indicator_obj, 'ready') and hasattr(indicator_obj, 'value'):
                    if indicator_obj.ready:
                        indicator_values[name] = indicator_obj.value
                    else:
                        indicator_values[name] = None # Mark as None if not ready
                        # Check if this non-ready indicator is actually used in any threshold
                        for regime_def in self._regime_thresholds.values():
                            if isinstance(regime_def, dict) and name in regime_def:
                                all_necessary_indicators_ready = False # A needed indicator is not ready
                                self.logger.debug(f"RegimeDetector '{self.name}' at {current_bar_timestamp}: Indicator '{name}' not ready for classification.")
                                break # No need to check other regimes for this specific indicator
                else:
                    self.logger.warning(f"Indicator '{name}' in {self.name} missing 'ready' or 'value' property.")
                    indicator_values[name] = None
                    all_necessary_indicators_ready = False


            # **** ADDED DETAILED LOGGING OF INDICATOR VALUES ****
            self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp} - Values for classification: {indicator_values}")

            if not all_necessary_indicators_ready and any(iv is None for iv_name, iv in indicator_values.items() if any(iv_name in rd_conditions for rd_conditions in self._regime_thresholds.values() if isinstance(rd_conditions, dict))):
                # If any indicator that is actually part of a rule is None (not ready), default.
                self.logger.debug(f"RegimeDetector '{self.name}' at {current_bar_timestamp}: Not all necessary indicators ready, defaulting.")
                raw_detected_regime = "default"
            else:
                raw_detected_regime = self._determine_regime_from_indicators(indicator_values, current_bar_timestamp)
            
            self.logger.debug(f"RegimeDetector '{self.name}' at {current_bar_timestamp}: Raw detection: '{raw_detected_regime}'.")

        final_regime = self._apply_stabilization(raw_detected_regime, current_bar_timestamp)
        return final_regime
    
    def _determine_regime_from_indicators(self, indicator_values: Dict[str, Optional[float]], current_bar_timestamp: Any) -> str:
        if not self._regime_thresholds:
             self.logger.debug(f"RegimeDet '{self.name}' at {current_bar_timestamp}: No regime thresholds defined. Defaulting.")
             return "default"
            
        # Enhanced debugging - log all indicator values at INFO level during optimization
        self.logger.info(f"OPTIMIZATION DEBUG - RegimeDet '{self.name}' at {current_bar_timestamp}: Current indicator values: {indicator_values}")
        self.logger.info(f"OPTIMIZATION DEBUG - Checking against thresholds: {self._regime_thresholds}")
        
        regime_check_results = {}
        for regime_name, conditions in self._regime_thresholds.items():
            if regime_name == "default": 
                continue

            matches_all_conditions = True
            if not isinstance(conditions, dict) or not conditions: 
                self.logger.warning(f"RegimeDet '{self.name}' at {current_bar_timestamp}: Conditions for regime '{regime_name}' are invalid. Skipping.")
                continue

            # Check if all indicators required for *this specific regime* are available
            required_indicator_names = list(conditions.keys())
            all_required_indicators_available_for_this_regime = True
            indicator_check_results = {}
            
            for req_ind_name in required_indicator_names:
                if req_ind_name not in indicator_values or indicator_values[req_ind_name] is None:
                    all_required_indicators_available_for_this_regime = False
                    indicator_check_results[req_ind_name] = "missing or None"
                    self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp}: Regime '{regime_name}' requires '{req_ind_name}', which is not ready/available.")
                    break
            
            if not all_required_indicators_available_for_this_regime:
                matches_all_conditions = False
                regime_check_results[regime_name] = {"result": "missing indicators", "details": indicator_check_results}
                continue # Try next regime definition

            # All required indicators for this regime are available, now check thresholds
            indicator_check_results = {}
            for indicator_instance_name_in_config, threshold_config in conditions.items():
                value = indicator_values[indicator_instance_name_in_config] # Already checked for None above for this regime
                
                min_val = threshold_config.get("min")
                max_val = threshold_config.get("max")
                
                threshold_check = {"value": value}
                
                if min_val is not None:
                    threshold_check["min_check"] = value >= float(min_val)
                    if value < float(min_val):
                        matches_all_conditions = False
                        self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp}: Regime '{regime_name}', Ind '{indicator_instance_name_in_config}' ({value:.4f}) < min ({min_val}). No match.")
                        break
                
                if max_val is not None:
                    threshold_check["max_check"] = value <= float(max_val)
                    if value > float(max_val):
                        matches_all_conditions = False
                        self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp}: Regime '{regime_name}', Ind '{indicator_instance_name_in_config}' ({value:.4f}) > max ({max_val}). No match.")
                        break
                
                indicator_check_results[indicator_instance_name_in_config] = threshold_check
            
            regime_check_results[regime_name] = {"result": "match" if matches_all_conditions else "no match", "details": indicator_check_results}
            
            if matches_all_conditions:
                self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp}: Regime '{regime_name}' MATCHED. Indicator values: {indicator_values}")
                self.logger.info(f"OPTIMIZATION DEBUG - Full regime checks: {regime_check_results}")
                return regime_name
        
        # Log all check results for debugging        
        self.logger.info(f"OPTIMIZATION DEBUG - No specific regime matched. Full regime checks: {regime_check_results}")
        self.logger.info(f"RegimeDet '{self.name}' at {current_bar_timestamp}: No specific regime matched. Defaulting. Indicator values: {indicator_values}")
        return "default"

    def _apply_stabilization(self, detected_regime: str, current_bar_timestamp: Any) -> str:
        true_current_regime = self._current_classification 
        
        if true_current_regime is None: 
            self.logger.debug(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Initial state. Setting duration 1 for detected '{detected_regime}'.")
            self._current_regime_duration = 1
            self._pending_regime = None 
            self._pending_duration = 0
            
            # Publish the initial classification
            self._publish_classification_event(detected_regime, current_bar_timestamp)
            return detected_regime 

        if detected_regime == true_current_regime:
            self._current_regime_duration += 1
            if self._pending_regime is not None: # Clear pending if current regime re-asserts itself
                self.logger.debug(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Detected '{detected_regime}' matches current '{true_current_regime}'. Pending '{self._pending_regime}' cleared.")
                self._pending_regime = None 
                self._pending_duration = 0
            # else: # Already in this regime, just increment duration
            #     self.logger.debug(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Still in '{true_current_regime}'. Duration: {self._current_regime_duration}.")
            return true_current_regime
        else:
            if self._pending_regime == detected_regime:
                self._pending_duration += 1
                # self.logger.debug(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Pending '{detected_regime}' confirmed again. Pending duration: {self._pending_duration}.")
            else:
                self.logger.info(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: New pending regime '{detected_regime}' (was '{true_current_regime}'). Initiating pending duration.")
                self._pending_regime = detected_regime
                self._pending_duration = 1 

            if self._pending_duration >= self._min_regime_duration:
                self.logger.info(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Regime SWITCH from '{true_current_regime}' to '{self._pending_regime}' after meeting min_duration {self._min_regime_duration}.")
                self._current_regime_duration = 1 
                newly_confirmed_regime = self._pending_regime
                self._pending_regime = None 
                self._pending_duration = 0
                
                # Publish regime change event
                self._publish_classification_event(newly_confirmed_regime, current_bar_timestamp)
                return newly_confirmed_regime
            else:
                self._current_regime_duration +=1 
                self.logger.debug(f"RegimeDet '{self.name}' Stabilization at {current_bar_timestamp}: Pending '{self._pending_regime}' (dur {self._pending_duration}/{self._min_regime_duration}) not stable. Maintaining '{true_current_regime}' (dur {self._current_regime_duration}).")
                return true_current_regime
                
    def _publish_classification_event(self, regime: str, timestamp: Any):
        """Explicitly publish a classification event for the given regime."""
        if self._event_bus and hasattr(self._event_bus, 'publish'):
            try:
                from src.core.event import Event, EventType
                classification_payload = {
                    'classification': regime,
                    'timestamp': timestamp,
                    'detector_name': self.name
                }
                # Create an Event object with the classification payload
                classification_event = Event(EventType.CLASSIFICATION, classification_payload)
                self.logger.info(f"RegimeDet '{self.name}' publishing CLASSIFICATION event for regime '{regime}' at {timestamp}")
                self._event_bus.publish(classification_event)
            except Exception as e:
                self.logger.error(f"Error publishing classification event from '{self.name}': {e}", exc_info=True)
    
    def get_regime_data(self) -> Dict[str, Any]:
        indicator_values = {}
        for name, indicator_obj in self._regime_indicators.items():
            if hasattr(indicator_obj, 'ready') and indicator_obj.ready and hasattr(indicator_obj, 'value'):
                indicator_values[name] = indicator_obj.value
            else:
                indicator_values[name] = None

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
