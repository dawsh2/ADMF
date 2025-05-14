# src/strategy/ma_strategy.py
import logging
from collections import deque
from typing import Optional, Dict, Any, List # Added List for parameter space

import pandas as pd # Keep if used
import datetime

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError

class MAStrategy(BaseComponent):
    """
    A simple Moving Average (MA) Crossover strategy.
    Optimizable for short_window and long_window.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, component_config_key)
        
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required for MAStrategy.")
            raise ConfigurationError("EventBus instance is required for MAStrategy.")

        # Load initial parameters from config - these can be overridden by set_parameters
        self._symbol: str = self.get_specific_config("symbol")
        self._short_window: int = self.get_specific_config("short_window_default", 10) # Provide default if not in config for optimizn
        self._long_window: int = self.get_specific_config("long_window_default", 20)  # Provide default

        if not self._symbol:
            raise ConfigurationError(f"Missing 'symbol' in configuration for {self.name}")
        
        self._validate_windows() # Initial validation

        # Initialize state that depends on parameters
        self._prices: deque[float] = deque(maxlen=self._long_window)
        self._prev_short_ma: Optional[float] = None
        self._prev_long_ma: Optional[float] = None
        self._current_signal_state: int = 0 

        self.logger.info(
            f"MAStrategy '{self.name}' configured for symbol '{self._symbol}' "
            f"with initial short_window={self._short_window}, long_window={self._long_window}."
        )

    def _validate_windows(self):
        if not isinstance(self._short_window, int) or self._short_window <= 0:
            raise ConfigurationError(f"'short_window' must be a positive integer for {self.name}. Got: {self._short_window}")
        if not isinstance(self._long_window, int) or self._long_window <= 0:
            raise ConfigurationError(f"'long_window' must be a positive integer for {self.name}. Got: {self._long_window}")
        if self._short_window >= self._long_window:
            raise ConfigurationError(
                f"'short_window' ({self._short_window}) must be less than 'long_window' ({self._long_window}) for {self.name}."
            )

    def _initialize_parameter_dependent_state(self):
        """Re-initializes state that depends on window parameters."""
        self._prices = deque(maxlen=self._long_window)
        self._prev_short_ma = None
        self._prev_long_ma = None
        self._current_signal_state = 0
        self.logger.debug(f"'{self.name}' state re-initialized for windows: short={self._short_window}, long={self._long_window}")

    def set_parameters(self, params: Dict[str, Any]):
        """
        Sets new parameters for the strategy and re-initializes dependent state.
        Expected params: {"short_window": int, "long_window": int}
        """
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
            
        self._initialize_parameter_dependent_state()
        self.logger.info(f"'{self.name}' parameters updated: short_window={self._short_window}, long_window={self._long_window}")
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
            "short_window": list(range(5, 7)),  # e.g., [5, 10, 15, 20, 25]
            "long_window": list(range(20, 21)) # e.g., [20, 30, 40, 50, 60]
        }

    def setup(self):
        """Sets up the component, including re-initializing state based on current parameters."""
        self.logger.info(f"Setting up MAStrategy '{self.name}'...")
        # Ensure state is clean and reflects current parameters (short_window, long_window)
        # _initialize_parameter_dependent_state is called after __init__ and set_parameters
        # so an explicit call here might be redundant if parameters haven't changed since init,
        # but good for ensuring clean state if setup is called multiple times.
        self._initialize_parameter_dependent_state() 
        
        # Clear any existing subscriptions before subscribing again (if setup can be called multiple times)
        # For this simple case, assuming event_bus handles duplicate subscriptions gracefully or we only call setup once.
        # If not, add unsubscribe logic here.
        self._event_bus.subscribe(EventType.BAR, self._on_bar_event)
        self.logger.info(f"'{self.name}' subscribed to BAR events for symbol '{self._symbol}'.")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"MAStrategy '{self.name}' setup complete. State: {self.state}")

    # _on_bar_event remains the same as your last working version (using integer signals)
    def _on_bar_event(self, event: Event):
        if event.event_type != EventType.BAR:
            return 
        
        bar_data: Dict[str, Any] = event.payload 
        if bar_data.get("symbol") != self._symbol:
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
            
            if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
                if self._current_signal_state != 1:
                    generated_signal_type_int = 1
                    self._current_signal_state = 1
            elif self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
                if self._current_signal_state != -1:
                    generated_signal_type_int = -1
                    self._current_signal_state = -1
            
            if generated_signal_type_int is not None:
                signal_payload: Dict[str, Any] = {
                    "symbol": self._symbol,
                    "timestamp": bar_timestamp, 
                    "signal_type": generated_signal_type_int,
                    "price_at_signal": close_price,
                    "strategy_id": self.name,
                    "short_ma": current_short_ma,
                    "long_ma": current_long_ma,
                    "params": {"short_window": self._short_window, "long_window": self._long_window} # Add current params to signal
                }
                signal_event = Event(EventType.SIGNAL, signal_payload)
                self._event_bus.publish(signal_event)
                self.logger.info(
                    f"Published SIGNAL Event: Type={generated_signal_type_int}, Symbol={self._symbol}, "
                    f"Price={close_price:.2f}, Timestamp={bar_timestamp}, Params=SW:{self._short_window},LW:{self._long_window}"
                )

        if current_short_ma is not None:
            self._prev_short_ma = current_short_ma
        if current_long_ma is not None:
            self._prev_long_ma = current_long_ma

    # start() and stop() methods remain largely the same, ensuring state is cleared in stop()
    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start MAStrategy '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return
        
        self.logger.info(f"MAStrategy '{self.name}' started with SW={self._short_window}, LW={self._long_window}. Listening for BAR events...")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping MAStrategy '{self.name}'...")
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.BAR, self._on_bar_event)
                self.logger.info(f"'{self.name}' attempted to unsubscribe from BAR events.")
            except Exception as e:
                 self.logger.error(f"Error unsubscribing '{self.name}' from BAR events: {e}")
        
        self._initialize_parameter_dependent_state() # Clear state on stop
            
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"MAStrategy '{self.name}' stopped. State: {self.state}")
