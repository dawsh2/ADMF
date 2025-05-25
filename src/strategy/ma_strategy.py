# src/strategy/ma_strategy.py
import logging
from collections import deque
from typing import Optional, Dict, Any, List # Added List for parameter space

import pandas as pd # Keep if used
import datetime

from src.core.component_base import ComponentBase
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError

class MAStrategy(ComponentBase):
    """
    A simple Moving Average (MA) Crossover strategy.
    Optimizable for short_window and long_window.
    """

    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Initialize internal state (no external dependencies)
        
        # Configuration parameters (will be set in initialize)
        self._symbol: Optional[str] = None
        self._short_window: int = 10
        self._long_window: int = 20

        # Initialize state
        self._prices: Optional[deque[float]] = None
        self._prev_short_ma: Optional[float] = None
        self._prev_long_ma: Optional[float] = None
        self._current_signal_state: int = 0
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Load configuration
        self._symbol = self.get_specific_config("symbol")
        self._short_window = self.get_specific_config("short_window_default", 10)
        self._long_window = self.get_specific_config("long_window_default", 20)
        
        if not self._symbol:
            raise ConfigurationError(f"Missing 'symbol' in configuration for {self.instance_name}")
        
        self._validate_windows()
        self._initialize_parameter_dependent_state()
        
        self.logger.info(
            f"MAStrategy '{self.instance_name}' configured for symbol '{self._symbol}' "
            f"with initial short_window={self._short_window}, long_window={self._long_window}."
        )
    
    def get_specific_config(self, key: str, default=None):
        """Helper method to get configuration values."""
        if not self.config_loader:
            return default
        config_key = self.config_key or self.instance_name
        config = self.config_loader.get_component_config(config_key)
        return config.get(key, default) if config else default

    def _validate_windows(self):
        if not isinstance(self._short_window, int) or self._short_window <= 0:
            raise ConfigurationError(f"'short_window' must be a positive integer for {self.instance_name}. Got: {self._short_window}")
        if not isinstance(self._long_window, int) or self._long_window <= 0:
            raise ConfigurationError(f"'long_window' must be a positive integer for {self.instance_name}. Got: {self._long_window}")
        if self._short_window >= self._long_window:
            raise ConfigurationError(
                f"'short_window' ({self._short_window}) must be less than 'long_window' ({self._long_window}) for {self.instance_name}."
            )

    def _initialize_parameter_dependent_state(self):
        """Re-initializes state that depends on window parameters."""
        # Preserve existing price history if it exists (for adaptive testing)
        if hasattr(self, '_prices') and len(self._prices) > 0:
            # Keep existing prices but adjust deque size for new long_window
            existing_prices = list(self._prices)
            self._prices = deque(existing_prices, maxlen=self._long_window)
            self.logger.debug(f"Preserving {len(existing_prices)} prices, adjusted maxlen to {self._long_window}")
        else:
            # Fresh start (normal case for training)
            self._prices = deque(maxlen=self._long_window)
            self.logger.debug(f"Fresh start with empty prices, maxlen={self._long_window}")
        
        # Preserve signal calculation state during parameter changes for adaptive trading
        # Only reset if we don't have existing state (fresh start)
        if not hasattr(self, '_prev_short_ma'):
            self._prev_short_ma = None
        if not hasattr(self, '_prev_long_ma'):
            self._prev_long_ma = None
        if not hasattr(self, '_current_signal_state'):
            self._current_signal_state = 0
        
        self.logger.debug(f"Preserved signal state - prev_short_ma: {self._prev_short_ma}, prev_long_ma: {self._prev_long_ma}, signal_state: {self._current_signal_state}")

    def set_parameters(self, params: Dict[str, Any]):
        """
        Sets new parameters for the strategy and re-initializes dependent state.
        Expected params: {"short_window": int, "long_window": int} and other strategy-specific params
        """
        # Log all parameters received
        self.logger.info(f"Setting parameters for '{self.instance_name}': {params}")
        
        # Extract core window parameters
        new_short_window = params.get("short_window", self._short_window)
        new_long_window = params.get("long_window", self._long_window)

        if not (isinstance(new_short_window, int) and isinstance(new_long_window, int)):
            self.logger.error(f"Invalid parameter types for set_parameters. Expected integers. Got: {params}")
            return False # Indicate failure

        old_short = self._short_window
        old_long = self._long_window

        self._short_window = new_short_window
        self._long_window = new_long_window

        try:
            self._validate_windows()
        except ConfigurationError as e:
            # Revert to old parameters if new ones are invalid
            self._short_window = old_short
            self._long_window = old_long
            self.logger.error(f"Invalid parameters in set_parameters: {e}. Parameters reverted.")
            return False # Indicate failure
        
        # Store additional parameters for rules/indicators
        # These will be used later in signal generation
        for key, value in params.items():
            if key not in ["short_window", "long_window"]:
                # Store extended parameters (RSI period, thresholds, weights, etc.)
                self.logger.info(f"'{self.instance_name}' storing extended parameter: {key}={value}")
                setattr(self, f"_{key}", value)
            
        self._initialize_parameter_dependent_state()
        self.logger.info(f"'{self.instance_name}' parameters updated: short_window={self._short_window}, long_window={self._long_window}")
        return True # Indicate success

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the strategy."""
        return {
            "short_window": self._short_window,
            "long_window": self._long_window,
            "symbol": self._symbol # Symbol is fixed per instance for now
        }

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Defines the parameter space for optimization.
        This can be made configurable later (e.g., from YAML).
        """
        # Example space - make this more dynamic or load from config if needed for more flexibility
        return {
            "short_window": [5, 10],  # considering using 'range' (list(range(5, 7)))
            "long_window": [20] # e.g., [20, 30, 40, 50, 60]
        }

    def setup(self):
        """Sets up the component, including re-initializing state based on current parameters."""
        self.logger.info(f"Setting up MAStrategy '{self.instance_name}'...")
        # Ensure state is clean and reflects current parameters (short_window, long_window)
        # _initialize_parameter_dependent_state is called after __init__ and set_parameters
        # so an explicit call here might be redundant if parameters haven't changed since init,
        # but good for ensuring clean state if setup is called multiple times.
        self._initialize_parameter_dependent_state() 
        
        # Clear any existing subscriptions before subscribing again (if setup can be called multiple times)
        # For this simple case, assuming event_bus handles duplicate subscriptions gracefully or we only call setup once.
        # If not, add unsubscribe logic here.
        self.event_bus.subscribe(EventType.BAR, self._on_bar_event)
        self.logger.info(f"'{self.instance_name}' subscribed to BAR events for symbol '{self._symbol}'.")
        self.logger.info(f"MAStrategy '{self.instance_name}' setup complete.")

    # _on_bar_event remains the same as your last working version (using integer signals)
    def _on_bar_event(self, event: Event):
        try:
            self.logger.debug(f"STRATEGY_DEBUG: {self.instance_name} received BAR event")
            if event.event_type != EventType.BAR:
                self.logger.debug(f"STRATEGY_DEBUG: {self.instance_name} ignoring non-BAR event: {event.event_type}")
                return 
            
            bar_data: Dict[str, Any] = event.payload 
            event_symbol = bar_data.get("symbol")
            self.logger.debug(f"STRATEGY_DEBUG: {self.instance_name} checking symbol: event={event_symbol}, expected={self._symbol}")
            if event_symbol != self._symbol:
                self.logger.debug(f"STRATEGY_DEBUG: {self.instance_name} ignoring event for wrong symbol: {event_symbol}")
                return
        except Exception as e:
            self.logger.error(f"STRATEGY_DEBUG: Exception in {self.instance_name} _on_bar_event: {e}", exc_info=True)
            return 

        close_price_val = bar_data.get("close")
        bar_timestamp: Optional[datetime.datetime] = bar_data.get("timestamp")

        if close_price_val is None or bar_timestamp is None:
            self.logger.warning(f"BAR event for '{self._symbol}' missing 'close' price or 'timestamp'. Event: {event.payload}")
            return
        
        try:
            close_price = float(close_price_val)
        except ValueError:
            self.logger.warning(f"Could not convert close price '{close_price_val}' to float for {self._symbol}. Event: {event.payload}")
            return

        self._prices.append(close_price)

        current_short_ma: Optional[float] = None
        current_long_ma: Optional[float] = None

        if len(self._prices) >= self._short_window:
            short_ma_prices = list(self._prices)[-self._short_window:]
            if short_ma_prices:
                current_short_ma = sum(short_ma_prices) / len(short_ma_prices)
        
        if len(self._prices) >= self._long_window:
            if self._prices: # deque is already maxlen=long_window
                 current_long_ma = sum(self._prices) / len(self._prices)
        
        short_ma_display = f"{current_short_ma:.2f}" if current_short_ma is not None else "N/A"
        long_ma_display = f"{current_long_ma:.2f}" if current_long_ma is not None else "N/A"
        
        self.logger.debug(
            f"Symbol: {self._symbol}, Close: {close_price:.2f}, TS: {bar_timestamp}, "
            f"SW={self._short_window}, LW={self._long_window}, " # Log current windows
            f"Short MA: {short_ma_display}, Long MA: {long_ma_display}"
        )

        generated_signal_type_int: Optional[int] = None
        
        if current_short_ma is not None and current_long_ma is not None and \
           self._prev_short_ma is not None and self._prev_long_ma is not None:
            
            # Get the weight for MA signals (default to 0.6 if not set)
            ma_weight = getattr(self, "_ma_weight", 0.6)
            
            # Get RSI thresholds and weight if they exist
            rsi_weight = getattr(self, "_rsi_weight", None)
            oversold_threshold = getattr(self, "_oversold_threshold", None)
            overbought_threshold = getattr(self, "_overbought_threshold", None)
            
            # Apply MA crossing logic for signals
            if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
                if self._current_signal_state != 1:
                    # Log the parameters being used for signal generation
                    self.logger.info(f"'{self.instance_name}' generating BUY signal with MA weight: {ma_weight}, "
                                    f"RSI weight: {rsi_weight}, oversold: {oversold_threshold}, "
                                    f"overbought: {overbought_threshold}")
                    generated_signal_type_int = 1
                    self._current_signal_state = 1
            elif self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
                if self._current_signal_state != -1:
                    # Log the parameters being used for signal generation
                    self.logger.info(f"'{self.instance_name}' generating SELL signal with MA weight: {ma_weight}, "
                                    f"RSI weight: {rsi_weight}, oversold: {oversold_threshold}, "
                                    f"overbought: {overbought_threshold}")
                    generated_signal_type_int = -1
                    self._current_signal_state = -1
            
            if generated_signal_type_int is not None:
                # Collect all parameters for the signal payload
                all_params = {
                    "short_window": self._short_window, 
                    "long_window": self._long_window
                }
                
                # Add extended parameters if they exist
                if hasattr(self, "_ma_weight"):
                    all_params["ma_weight"] = self._ma_weight
                if hasattr(self, "_rsi_weight"):
                    all_params["rsi_weight"] = self._rsi_weight
                if hasattr(self, "_period"):
                    all_params["period"] = self._period
                if hasattr(self, "_oversold_threshold"):
                    all_params["oversold_threshold"] = self._oversold_threshold
                if hasattr(self, "_overbought_threshold"):
                    all_params["overbought_threshold"] = self._overbought_threshold
                
                signal_payload: Dict[str, Any] = {
                    "symbol": self._symbol,
                    "timestamp": bar_timestamp, 
                    "signal_type": generated_signal_type_int,
                    "price_at_signal": close_price,
                    "strategy_id": self.instance_name,
                    "short_ma": current_short_ma,
                    "long_ma": current_long_ma,
                    "params": all_params # Add all current parameters to signal
                }
                signal_event = Event(EventType.SIGNAL, signal_payload)
                self.logger.warning(f"STRATEGY_DEBUG: Publishing SIGNAL event: {generated_signal_type_int}")
                self.event_bus.publish(signal_event)
                # Get current regime for enhanced logging
                current_regime = "unknown"
                try:
                    if hasattr(self, 'container') and self.container:
                        regime_detector = self.container.resolve("MyPrimaryRegimeDetector")
                        current_regime = regime_detector.get_current_classification()
                except:
                    pass
                    
                self.logger.info(
                    f"SIGNAL_GENERATED: Type={generated_signal_type_int}, Symbol={self._symbol}, "
                    f"Price={close_price:.2f}, Regime={current_regime}, Timestamp={bar_timestamp}"
                )

        if current_short_ma is not None:
            self._prev_short_ma = current_short_ma
        if current_long_ma is not None:
            self._prev_long_ma = current_long_ma

    # start() and stop() methods remain largely the same, ensuring state is cleared in stop()
    def start(self):
        """Start the strategy."""
        super().start()
        
        # Parent class handles state checking
        
        # Ensure we're subscribed to BAR events (needed for restarts)
        if self.event_bus:
            self.event_bus.subscribe(EventType.BAR, self._on_bar_event)
            self.logger.debug(f"Re-subscribed to BAR events on start/restart")
        
        self.logger.info(f"MAStrategy '{self.instance_name}' started with SW={self._short_window}, LW={self._long_window}. Listening for BAR events...")
        # Component is now running

    def stop(self):
        """Stop the strategy."""
        self.logger.info(f"Stopping MAStrategy '{self.instance_name}'...")
        if self.event_bus:
            try:
                self.event_bus.unsubscribe(EventType.BAR, self._on_bar_event)
                self.logger.info(f"'{self.instance_name}' attempted to unsubscribe from BAR events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing '{self.instance_name}' from BAR events: {e}")
        
        self._initialize_parameter_dependent_state()  # Clear state on stop
        super().stop()
        self.logger.info(f"MAStrategy '{self.instance_name}' stopped.")
    
    def dispose(self):
        """Clean up resources."""
        super().dispose()
        if self._prices:
            self._prices.clear()
        self._prev_short_ma = None
        self._prev_long_ma = None
        self._current_signal_state = 0
