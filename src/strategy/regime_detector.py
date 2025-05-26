# src/strategy/regime_detector.py
from typing import Any, Dict, Tuple, Optional, Type, TYPE_CHECKING

from src.core.component_base import ComponentBase
from src.core.event import Event, EventType
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


class RegimeDetector(ComponentBase):
    """
    Detects market regimes based on configurable indicators and thresholds.
    Implements stabilization logic to prevent rapid regime switching.
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Internal state initialization
        self._regime_indicators: Dict[str, BaseIndicatorInterface] = {} 
        self._regime_thresholds: Dict[str, Any] = {}
        self._min_regime_duration: int = 1
        self._current_regime_duration: int = 0
        self._pending_regime: Optional[str] = None
        self._pending_duration: int = 0
        self._current_classification: Optional[str] = None
        
        # Logging optimization
        self._verbose_logging = False
        self._summary_interval = 100
        
        # Statistics tracking
        self._total_checks = 0
        self._no_match_count = 0
        self._regime_counts = {}
        self._checks_since_last_log = 0
        
        # Enhanced debugging
        self._debug_mode = False
        self._first_100_bars = True
        self._bar_count = 0
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Log instance ID for debugging isolation
        self.logger.debug(f"RegimeDetector '{self.instance_name}' created with ID {id(self)}")
        
        # Load configuration parameters
        self._regime_thresholds = self.get_specific_config("regime_thresholds", {})
        self._min_regime_duration = self.get_specific_config("min_regime_duration", 1)
        self._verbose_logging = self.get_specific_config("verbose_logging", False)
        self._summary_interval = self.get_specific_config("summary_interval", 100)
        self._debug_mode = self.get_specific_config("debug_mode", False)
        
        # Set up regime indicators during initialization
        self._setup_regime_indicators()
        
        self.logger.info(f"RegimeDetector '{self.instance_name}' initialized. Min duration: {self._min_regime_duration}, Thresholds: {self._regime_thresholds}")

    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        if self.subscription_manager:
            self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
            self.logger.info(f"'{self.instance_name}' subscribed to BAR events.")
            # Log instance and event bus IDs for debugging isolation
            event_bus_id = id(self.event_bus) if self.event_bus else 'None'
            self.logger.debug(f"RegimeDetector '{self.instance_name}' (ID {id(self)}) listening to EventBus ID {event_bus_id}")

    def on_bar(self, event):
        """Handle incoming BAR events."""
        if event.event_type != EventType.BAR:
            return
            
        bar_data = event.payload
        if not isinstance(bar_data, dict):
            self.logger.warning(f"Received BAR event with non-dict payload")
            return
            
        # Classify the bar data
        regime = self.classify(bar_data)
        
        # The classify method already handles publishing CLASSIFICATION events

    def _get_indicator_class(self, indicator_type_name: str) -> Optional[Type[BaseIndicatorInterface]]:
        if indicator_type_name == "rsi":
            return RSIIndicator
        elif indicator_type_name == "atr":
            return ATRIndicator
        elif indicator_type_name == "simple_ma_trend":
            return SimpleMATrendIndicator
        else:
            self.logger.error(f"Unknown indicator type specified: '{indicator_type_name}' for RegimeDetector '{self.instance_name}'.")
            return None

    def _setup_regime_indicators(self):
        indicator_configs: Dict[str, Any] = self.get_specific_config("indicators", {})
        if not indicator_configs:
            self.logger.warning(f"No indicators configured for RegimeDetector '{self.instance_name}'. It will likely always classify as 'default'.")
            return

        for indicator_instance_name, config_dict in indicator_configs.items():
            indicator_type_str = config_dict.get("type")
            params_from_config = config_dict.get("parameters", {}) 
            
            if not indicator_type_str:
                self.logger.error(f"Indicator type not specified for '{indicator_instance_name}' in {self.instance_name} config.")
                continue

            IndicatorClass = self._get_indicator_class(indicator_type_str)
            if IndicatorClass:
                try:
                    self._regime_indicators[indicator_instance_name] = IndicatorClass(
                        instance_name=f"{self.instance_name}_{indicator_instance_name}", 
                        config_loader=self.config_loader or self.config,  # Pass config loader or config dict
                        event_bus=self.event_bus,     
                        component_config_key=None, # Indicators get params directly, not separate config block by key
                        parameters=params_from_config
                    )
                    self.logger.info(f"Initialized indicator '{indicator_instance_name}' of type '{indicator_type_str}' for {self.instance_name}.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize indicator '{indicator_instance_name}' for {self.instance_name}: {e}", exc_info=True)
        
        if not self._regime_indicators:
            self.logger.warning(f"No indicators were successfully initialized for {self.instance_name}. Classification may be unreliable.")

    def _start(self):
        """Start the regime detector component."""
        self.logger.info(f"RegimeDetector '{self.instance_name}' started")
        # Indicators are already set up in _initialize()
    
    def classify(self, data: Dict[str, Any]) -> str:
        """ Classify market data into a regime label. """
        current_bar_timestamp = data.get('timestamp', 'N/A') # Get timestamp for logging
        
        # Enhanced debugging for first 100 bars
        if self._debug_mode and self._bar_count < 100:
            self._bar_count += 1
            self.logger.info(f"[REGIME_DEBUG] Bar #{self._bar_count} at {current_bar_timestamp}: Starting classification")

        if not self._regime_indicators or not self._regime_thresholds:
            raw_detected_regime = self._current_classification if self._current_classification is not None else "default"
            # self.logger.debug(f"RegimeDetector '{self.instance_name}' at {current_bar_timestamp}: Not fully configured. Raw: '{raw_detected_regime}'.") # Can be noisy
        else:
            for indicator_name, indicator_obj in self._regime_indicators.items():
                try:
                    if hasattr(indicator_obj, 'update') and callable(getattr(indicator_obj, 'update')):
                        indicator_obj.update(data) # Pass the full bar data
                    else:
                        self.logger.warning(f"Indicator '{indicator_name}' in {self.instance_name} has no callable update method.")
                except Exception as e:
                    self.logger.error(f"Error updating indicator '{indicator_name}' in {self.instance_name} for bar {current_bar_timestamp}: {e}", exc_info=True)
            
            indicator_values: Dict[str, Optional[float]] = {}
            all_necessary_indicators_ready = True # Flag to track if all indicators needed by *any* rule are ready
            
            # Collect values from all configured indicators, note if any are not ready
            for name, indicator_obj in self._regime_indicators.items():
                if hasattr(indicator_obj, 'ready') and hasattr(indicator_obj, 'value'):
                    if indicator_obj.ready:
                        indicator_values[name] = indicator_obj.value
                        if self._debug_mode and self._bar_count <= 100:
                            self.logger.info(f"[REGIME_DEBUG] Bar #{self._bar_count}: {name} = {indicator_obj.value:.4f if indicator_obj.value else 'None'}")
                    else:
                        indicator_values[name] = None # Mark as None if not ready
                        if self._debug_mode and self._bar_count <= 100:
                            self.logger.info(f"[REGIME_DEBUG] Bar #{self._bar_count}: {name} NOT READY")
                        # Check if this non-ready indicator is actually used in any threshold
                        for regime_def in self._regime_thresholds.values():
                            if isinstance(regime_def, dict) and name in regime_def:
                                all_necessary_indicators_ready = False # A needed indicator is not ready
                                self.logger.debug(f"RegimeDetector '{self.instance_name}' at {current_bar_timestamp}: Indicator '{name}' not ready for classification.")
                                break # No need to check other regimes for this specific indicator
                else:
                    self.logger.warning(f"Indicator '{name}' in {self.instance_name} missing 'ready' or 'value' property.")
                    indicator_values[name] = None
                    all_necessary_indicators_ready = False


            # Increment total checks counter
            self._total_checks += 1
            self._checks_since_last_log += 1
            
            # Log indicator values based on verbose setting
            if self._verbose_logging:
                self.logger.info(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp} - Values for classification: {indicator_values}")
            else:
                self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp} - Values for classification: {indicator_values}")

            if not all_necessary_indicators_ready and any(iv is None for iv_name, iv in indicator_values.items() if any(iv_name in rd_conditions for rd_conditions in self._regime_thresholds.values() if isinstance(rd_conditions, dict))):
                # If any indicator that is actually part of a rule is None (not ready), default.
                self.logger.debug(f"RegimeDetector '{self.instance_name}' at {current_bar_timestamp}: Not all necessary indicators ready, defaulting.")
                raw_detected_regime = "default"
            else:
                raw_detected_regime = self._determine_regime_from_indicators(indicator_values, current_bar_timestamp)
            
            self.logger.debug(f"RegimeDetector '{self.instance_name}' at {current_bar_timestamp}: Raw detection: '{raw_detected_regime}'.")

        final_regime = self._apply_stabilization(raw_detected_regime, current_bar_timestamp)
        
        if self._debug_mode and self._bar_count <= 100:
            self.logger.info(f"[REGIME_DEBUG] Bar #{self._bar_count}: raw={raw_detected_regime}, final={final_regime}, duration={self._current_regime_duration}")
        
        return final_regime
    
    def _determine_regime_from_indicators(self, indicator_values: Dict[str, Optional[float]], current_bar_timestamp: Any) -> str:
        if not self._regime_thresholds:
             self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: No regime thresholds defined. Defaulting.")
             return "default"
            
        # Enhanced debugging - log indicator values at DEBUG level
        if self._verbose_logging:
            self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Current indicator values: {indicator_values}")
            self.logger.debug(f"Checking against thresholds: {self._regime_thresholds}")
        
        regime_check_results = {}
        for regime_name, conditions in self._regime_thresholds.items():
            if regime_name == "default": 
                continue

            matches_all_conditions = True
            if not isinstance(conditions, dict) or not conditions: 
                self.logger.warning(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Conditions for regime '{regime_name}' are invalid. Skipping.")
                continue

            # Check if all indicators required for *this specific regime* are available
            required_indicator_names = list(conditions.keys())
            all_required_indicators_available_for_this_regime = True
            indicator_check_results = {}
            
            for req_ind_name in required_indicator_names:
                if req_ind_name not in indicator_values or indicator_values[req_ind_name] is None:
                    all_required_indicators_available_for_this_regime = False
                    indicator_check_results[req_ind_name] = "missing or None"
                    self.logger.info(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Regime '{regime_name}' requires '{req_ind_name}', which is not ready/available.")
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
                        if self._verbose_logging:
                            self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Regime '{regime_name}', Ind '{indicator_instance_name_in_config}' ({value:.4f}) < min ({min_val}). No match.")
                        else:
                            self.logger.debug(f"Regime '{regime_name}' no match: {indicator_instance_name_in_config} below min")
                        break
                
                if max_val is not None:
                    threshold_check["max_check"] = value <= float(max_val)
                    if value > float(max_val):
                        matches_all_conditions = False
                        if self._verbose_logging:
                            self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Regime '{regime_name}', Ind '{indicator_instance_name_in_config}' ({value:.4f}) > max ({max_val}). No match.")
                        else:
                            self.logger.debug(f"Regime '{regime_name}' no match: {indicator_instance_name_in_config} above max")
                        break
                
                indicator_check_results[indicator_instance_name_in_config] = threshold_check
            
            regime_check_results[regime_name] = {"result": "match" if matches_all_conditions else "no match", "details": indicator_check_results}
            
            if matches_all_conditions:
                # Track regime counts for statistics
                self._regime_counts[regime_name] = self._regime_counts.get(regime_name, 0) + 1
                
                # Log classification in a parseable format
                trend_val = indicator_values.get('ma_trend', 'N/A')
                vol_val = indicator_values.get('atr', 'N/A')
                rsi_val = indicator_values.get('rsi', 'N/A')
                self.logger.info(f"Regime classification: trend_strength={trend_val}, volatility={vol_val}, rsi_level={rsi_val} → regime={regime_name}")
                
                self.logger.info(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: Regime '{regime_name}' MATCHED. Indicator values: {indicator_values}")
                if self._verbose_logging:
                    self.logger.debug(f"Full regime checks: {regime_check_results}")
                return regime_name
        
        # Increment no-match counter for statistics
        self._no_match_count += 1
        self._regime_counts["default"] = self._regime_counts.get("default", 0) + 1
        
        # Log detailed results only if verbose
        if self._verbose_logging:
            self.logger.debug(f"No specific regime matched. Full regime checks: {regime_check_results}")
            
        # Log summary periodically to reduce verbosity
        if self._checks_since_last_log >= self._summary_interval:
            match_rate = (self._total_checks - self._no_match_count) / max(1, self._total_checks)
            self.logger.info(f"Regime detection summary: {self._no_match_count}/{self._checks_since_last_log} no matches ({(self._no_match_count/max(1, self._checks_since_last_log))*100:.1f}%), match rate: {match_rate*100:.1f}%")
            # Reset counters for next period
            self._no_match_count = 0
            self._checks_since_last_log = 0
            
        # Log classification in a parseable format for default case
        trend_val = indicator_values.get('ma_trend', 'N/A')
        vol_val = indicator_values.get('atr', 'N/A')
        rsi_val = indicator_values.get('rsi', 'N/A')
        self.logger.info(f"Regime classification: trend_strength={trend_val}, volatility={vol_val}, rsi_level={rsi_val} → regime=default")
        
        self.logger.debug(f"RegimeDet '{self.instance_name}' at {current_bar_timestamp}: No specific regime matched. Defaulting. Indicator values: {indicator_values}")
        return "default"

    def _apply_stabilization(self, detected_regime: str, current_bar_timestamp: Any) -> str:
        true_current_regime = self._current_classification 
        
        if true_current_regime is None: 
            self.logger.debug(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Initial state. Setting duration 1 for detected '{detected_regime}'.")
            self._current_classification = detected_regime
            self._current_regime_duration = 1
            self._pending_regime = None 
            self._pending_duration = 0
            
            # Publish the initial classification
            self._publish_classification_event(detected_regime, current_bar_timestamp)
            return detected_regime 

        if detected_regime == true_current_regime:
            self._current_regime_duration += 1
            if self._pending_regime is not None: # Clear pending if current regime re-asserts itself
                self.logger.debug(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Detected '{detected_regime}' matches current '{true_current_regime}'. Pending '{self._pending_regime}' cleared.")
                self._pending_regime = None 
                self._pending_duration = 0
            # else: # Already in this regime, just increment duration
            #     self.logger.debug(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Still in '{true_current_regime}'. Duration: {self._current_regime_duration}.")
            return true_current_regime
        else:
            if self._pending_regime == detected_regime:
                self._pending_duration += 1
                # self.logger.debug(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Pending '{detected_regime}' confirmed again. Pending duration: {self._pending_duration}.")
            else:
                self.logger.info(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: New pending regime '{detected_regime}' (was '{true_current_regime}'). Initiating pending duration.")
                self._pending_regime = detected_regime
                self._pending_duration = 1 

            if self._pending_duration >= self._min_regime_duration:
                self.logger.info(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Regime SWITCH from '{true_current_regime}' to '{self._pending_regime}' after meeting min_duration {self._min_regime_duration}.")
                self._current_classification = self._pending_regime
                self._current_regime_duration = 1 
                newly_confirmed_regime = self._pending_regime
                self._pending_regime = None 
                self._pending_duration = 0
                
                # Publish regime change event
                self._publish_classification_event(newly_confirmed_regime, current_bar_timestamp)
                return newly_confirmed_regime
            else:
                self._current_regime_duration +=1 
                self.logger.debug(f"RegimeDet '{self.instance_name}' Stabilization at {current_bar_timestamp}: Pending '{self._pending_regime}' (dur {self._pending_duration}/{self._min_regime_duration}) not stable. Maintaining '{true_current_regime}' (dur {self._current_regime_duration}).")
                return true_current_regime
                
    def _publish_classification_event(self, regime: str, timestamp: Any):
        """Explicitly publish a classification event for the given regime."""
        if self.event_bus and hasattr(self.event_bus, 'publish'):
            try:
                from src.core.event import Event, EventType
                classification_payload = {
                    'classification': regime,
                    'timestamp': timestamp,
                    'detector_name': self.instance_name
                }
                # Create an Event object with the classification payload
                classification_event = Event(EventType.CLASSIFICATION, classification_payload)
                self.logger.info(f"RegimeDet '{self.instance_name}' publishing CLASSIFICATION event for regime '{regime}' at {timestamp}")
                
                # Check for any existing subscribers before publishing
                if hasattr(self.event_bus, '_subscribers'):
                    subscribers = getattr(self.event_bus, '_subscribers')
                    classification_subscribers = subscribers.get(EventType.CLASSIFICATION, [])
                    self.logger.info(f"Publishing CLASSIFICATION event with {len(classification_subscribers)} active subscribers")
                    
                    # List all subscribers for debugging
                    for i, subscriber in enumerate(classification_subscribers):
                        subscriber_name = getattr(subscriber, '__name__', 'Unknown')
                        subscriber_module = getattr(subscriber, '__module__', 'Unknown')
                        self.logger.info(f"  Subscriber #{i+1}: {subscriber_name} from {subscriber_module}")
                        
                        # Try to get more information about the subscriber
                        if hasattr(subscriber, '__self__'):
                            subscriber_instance = getattr(subscriber, '__self__')
                            instance_class = subscriber_instance.__class__.__name__
                            instance_name = getattr(subscriber_instance, 'instance_name', 'unnamed')
                            self.logger.info(f"  -> Instance: {instance_class} '{instance_name}'")
                
                # Always publish the event
                self.event_bus.publish(classification_event)
                
                # Log the full event for debugging
                self.logger.debug(f"Published CLASSIFICATION event: {classification_event}")
                
                # Log confirmation
                self.logger.info(f"Successfully published CLASSIFICATION event for regime '{regime}'")
            except Exception as e:
                self.logger.error(f"Error publishing classification event from '{self.instance_name}': {e}", exc_info=True)
                
    def reset(self):
        """Reset the regime detector to initial state for a fresh run."""
        self.logger.info(f"Resetting RegimeDetector '{self.instance_name}' to initial state")
        
        # Reset classification state
        self._current_classification = None
        self._current_regime_duration = 0
        self._pending_regime = None
        self._pending_duration = 0
        
        # Reset statistics tracking
        self._total_checks = 0
        self._no_match_count = 0
        self._regime_counts = {}
        self._checks_since_last_log = 0
        
        # Reset all indicators if they have a reset method
        for name, indicator in self._regime_indicators.items():
            if hasattr(indicator, 'reset') and callable(getattr(indicator, 'reset')):
                try:
                    indicator.reset()
                    self.logger.debug(f"Reset indicator '{name}' in {self.instance_name}")
                except Exception as e:
                    self.logger.warning(f"Error resetting indicator '{name}' in {self.instance_name}: {e}")
        
        self.logger.debug(f"RegimeDetector '{self.instance_name}' reset complete")
    
    def stop(self):
        """Stop the detector and generate a summary report."""
        self.logger.info(f"Stopping component '{self.instance_name}'...")
        
        # Generate summary statistics
        self.generate_summary()
        
        # Call parent stop method
        super().stop()
    
    def teardown(self):
        """Clean up resources during component teardown."""
        # Clear all indicators
        self._regime_indicators.clear()
        
        # Clear state
        self._regime_thresholds.clear()
        self._regime_counts.clear()
        
        # Call parent teardown
        super().teardown()
    
    def get_current_classification(self) -> Optional[str]:
        """Get the current classification/regime."""
        return self._current_classification
    
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
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection for reporting."""
        match_rate = 0
        if self._total_checks > 0:
            match_rate = 1.0 - (self._no_match_count / self._total_checks)
            
        regime_percentages = {}
        for regime, count in self._regime_counts.items():
            if self._total_checks > 0:
                regime_percentages[regime] = (count / self._total_checks) * 100
                
        return {
            'total_checks': self._total_checks,
            'no_match_count': self._no_match_count,
            'match_rate': match_rate,
            'regime_counts': self._regime_counts,
            'regime_percentages': regime_percentages
        }
        
    def generate_summary(self):
        """Generate a summary report of regime detection statistics."""
        # Make sure we have processed at least some bars
        if self._total_checks == 0:
            self.logger.info("=== Regime Detection Summary ===\nNo regime detection checks were performed\n=== End of Summary ===")
            return
            
        stats = self.get_statistics()
        
        self.logger.info("=== Regime Detection Summary ===")
        self.logger.info(f"Total bars checked: {stats['total_checks']}")
        if 'match_rate' in stats:
            self.logger.info(f"Match rate: {stats['match_rate'] * 100:.2f}%")
        
        self.logger.info("Regime distribution:")
        if 'regime_percentages' in stats and stats['regime_percentages']:
            for regime, percentage in stats['regime_percentages'].items():
                self.logger.info(f"  {regime}: {percentage:.2f}%")
        else:
            self.logger.info("  No regime distribution data available")
            
        self.logger.info("=== End of Summary ===")