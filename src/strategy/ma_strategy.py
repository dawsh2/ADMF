# src/strategy/ma_strategy.py
import logging
from collections import deque
from typing import Optional # Make sure Optional is imported
import pandas as pd

from src.core.component import BaseComponent
from src.core.event import Event, EventType
from src.core.exceptions import ConfigurationError # ComponentError can also be imported if needed

class MAStrategy(BaseComponent):
    """
    A simple Moving Average (MA) Crossover strategy.
    - Subscribes to BAR events for a specific symbol.
    - Calculates short and long period moving averages.
    - Generates SIGNAL events (BUY/SELL) on crossovers.
    """

    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, component_config_key)
        
        self._event_bus = event_bus
        if not self._event_bus:
            self.logger.error("EventBus instance is required for MAStrategy.")
            # Consider raising ConfigurationError or ComponentError for critical missing dependencies
            raise ValueError("EventBus instance is required for MAStrategy.") # Or ConfigurationError

        # Load strategy-specific configuration
        self._symbol: str = self.get_specific_config("symbol")
        self._short_window: int = self.get_specific_config("short_window")
        self._long_window: int = self.get_specific_config("long_window")

        if not self._symbol:
            raise ConfigurationError(f"Missing 'symbol' in configuration for {self.name}")
        if not isinstance(self._short_window, int) or self._short_window <= 0:
            raise ConfigurationError(f"'short_window' must be a positive integer for {self.name}. Got: {self._short_window}")
        if not isinstance(self._long_window, int) or self._long_window <= 0:
            raise ConfigurationError(f"'long_window' must be a positive integer for {self.name}. Got: {self._long_window}")
        if self._short_window >= self._long_window:
            raise ConfigurationError(f"'short_window' ({self._short_window}) must be less than 'long_window' ({self._long_window}) for {self.name}.")

        # Internal state for MA calculation and signal generation
        self._prices: deque[float] = deque(maxlen=self._long_window)
        self._prev_short_ma: Optional[float] = None # To detect crossover
        self._prev_long_ma: Optional[float] = None  # To detect crossover
        
        self._current_position: int = 0 # Simple position state: 0 = flat, 1 = long, -1 = short

        self.logger.info(
            f"MAStrategy '{self.name}' configured for symbol '{self._symbol}' "
            f"with short_window={self._short_window}, long_window={self._long_window}."
        )

    def setup(self):
        self.logger.info(f"Setting up MAStrategy '{self.name}'...")
        self._event_bus.subscribe(EventType.BAR, self._on_bar_event)
        self.logger.info(f"'{self.name}' subscribed to BAR events for symbol '{self._symbol}'.")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"MAStrategy '{self.name}' setup complete. State: {self.state}")

    def _on_bar_event(self, event: Event):
        if event.event_type != EventType.BAR:
            return 
        
        bar_data = event.payload
        if bar_data.get("symbol") != self._symbol:
            return 

        close_price_val = bar_data.get("close")
        if close_price_val is None:
            self.logger.warning(f"BAR event for '{self._symbol}' missing 'close' price. Event: {event}")
            return
        
        try:
            close_price = float(close_price_val)
        except ValueError:
            self.logger.warning(f"Could not convert close price '{close_price_val}' to float for {self._symbol}. Event: {event}")
            return

        self._prices.append(close_price)

        current_short_ma: Optional[float] = None
        current_long_ma: Optional[float] = None

        if len(self._prices) >= self._short_window:
            # Use a slice of the deque for the short MA
            short_ma_prices = list(self._prices)[-self._short_window:]
            if short_ma_prices: # Ensure there are prices to average
                current_short_ma = sum(short_ma_prices) / len(short_ma_prices)
        
        if len(self._prices) >= self._long_window: # deque is already maxlen=long_window
            if self._prices: # Ensure there are prices to average
                 current_long_ma = sum(self._prices) / len(self._prices)
        
        # --- CORRECTED LOGGING ---
        short_ma_display = f"{current_short_ma:.2f}" if current_short_ma is not None else "N/A"
        long_ma_display = f"{current_long_ma:.2f}" if current_long_ma is not None else "N/A"
        
        self.logger.debug(
            f"Symbol: {self._symbol}, Close: {close_price:.2f}, "
            f"Short MA ({self._short_window}): {short_ma_display}, "
            f"Long MA ({self._long_window}): {long_ma_display}"
        )
        # --- END CORRECTED LOGGING ---

        # Check for signals if we have previous MA values AND current MAs are valid
        if current_short_ma is not None and current_long_ma is not None and \
           self._prev_short_ma is not None and self._prev_long_ma is not None:
            
            signal_generated: Optional[str] = None
            
            # Golden Cross (Buy Signal): short MA crosses above long MA
            if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
                if self._current_position <= 0: # Only signal if flat or short, to avoid repeat BUYs
                    signal_generated = "BUY"
                    self._current_position = 1 # Mark position as long
            # Death Cross (Sell Signal): short MA crosses below long MA
            elif self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
                if self._current_position >= 0: # Only signal if flat or long, to avoid repeat SELLs
                    signal_generated = "SELL"
                    self._current_position = -1 # Mark position as short
            
            if signal_generated:
                signal_payload = {
                    "symbol": self._symbol,
                    "timestamp": bar_data.get("timestamp"), 
                    "signal_type": signal_generated,
                    "price_at_signal": close_price,
                    "strategy_id": self.name,
                    "short_ma": current_short_ma,
                    "long_ma": current_long_ma
                }
                signal_event = Event(EventType.SIGNAL, signal_payload)
                self._event_bus.publish(signal_event)
                self.logger.info(
                    f"Published SIGNAL Event: Type={signal_generated}, Symbol={self._symbol}, "
                    f"Price={close_price:.2f}, Timestamp={signal_payload['timestamp']}"
                )

        # Update previous MAs for the next event, only if they were calculated
        if current_short_ma is not None:
            self._prev_short_ma = current_short_ma
        if current_long_ma is not None:
            self._prev_long_ma = current_long_ma

    def start(self):
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start MAStrategy '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return
        
        self.logger.info(f"MAStrategy '{self.name}' started. Listening for BAR events...")
        self.state = BaseComponent.STATE_STARTED

    def stop(self):
        self.logger.info(f"Stopping MAStrategy '{self.name}'...")
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.BAR, self._on_bar_event)
                self.logger.info(f"'{self.name}' attempted to unsubscribe from BAR events.")
            except Exception as e:
                 self.logger.error(f"Error unsubscribing '{self.name}' from BAR events: {e}")
        
        self._prices.clear()
        self._prev_short_ma = None
        self._prev_long_ma = None
        self._current_position = 0
            
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"MAStrategy '{self.name}' stopped. State: {self.state}")
