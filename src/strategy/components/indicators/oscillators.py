# src/strategy/components/indicators/oscillators.py
import logging
from typing import Dict, Any, List, Optional
from collections import deque
from src.core.component import BaseComponent # Using BaseComponent as per your project

class RSIIndicator(BaseComponent):
    """
    Calculates the Relative Strength Index (RSI).
    """
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_loader, component_config_key)
        self.logger = logging.getLogger(f"{__name__}.{instance_name}") # Corrected logger name

        if parameters:
            self.period: int = parameters.get('period', 14)
        else:
            self.period: int = self.get_specific_config('period', 14)

        if self.period <= 1:
            self.logger.error("RSI period must be greater than 1. Defaulting to 14.")
            self.period = 14 # Fallback to a sensible default

        self._prices: deque[float] = deque(maxlen=self.period + 10)
        self._gains: deque[float] = deque(maxlen=self.period)
        self._losses: deque[float] = deque(maxlen=self.period)
        
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._current_value: Optional[float] = None
        self._initialized_smoothing = False
        # Initial log moved to setup after state is properly CREATED
        # self.logger.info(f"RSIIndicator '{self.name}' initialized with period={self.period}.")


    def setup(self):
        """Sets up the component."""
        self.reset_state() # Ensure clean state on setup
        self.logger.info(f"RSIIndicator '{self.name}' configured with period={self.period}.")
        self.state = BaseComponent.STATE_INITIALIZED # Directly set state as per existing pattern
        self.logger.info(f"RSIIndicator '{self.name}' setup complete. State: {self.state}")

    def start(self) -> None:
        """Starts the component's active operations."""
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start RSIIndicator '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return
        self.state = BaseComponent.STATE_STARTED
        self.logger.info(f"RSIIndicator '{self.name}' started. State: {self.state}")

    def stop(self) -> None:
        """Stops the component's active operations and cleans up resources."""
        self.reset_state() # Clean up internal state
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"RSIIndicator '{self.name}' stopped. State: {self.state}")

    def update(self, price: float) -> Optional[float]:
        if not isinstance(price, (int, float)):
            self.logger.warning(f"Invalid price received: {price}")
            return self._current_value

        if not self._prices: 
            self._prices.append(price)
            return None

        change = price - self._prices[-1]
        # self._prices.append(price) # Append after using the last price for change calculation

        if len(self._prices) < self.period: # Need 'period' changes, so 'period+1' prices.
                                            # If len is 1 (first update call after initial price), change is based on it.
            self._prices.append(price) # Append now for next calculation
            if len(self._prices) > 1: # Ensure we have at least one change
                if change > 0:
                    self._gains.append(change)
                    self._losses.append(0.0)
                else:
                    self._gains.append(0.0)
                    self._losses.append(abs(change))
            return None
        
        # Correctly manage deque for gains and losses once period is reached
        if len(self._gains) == self.period:
            self._gains.popleft()
            self._losses.popleft()
        
        current_gain = change if change > 0 else 0.0
        current_loss = abs(change) if change < 0 else 0.0
        self._gains.append(current_gain)
        self._losses.append(current_loss)
        self._prices.append(price) # Append price for the main price deque

        if not self._initialized_smoothing:
            if len(self._gains) == self.period: # Check if we have exactly 'period' gains/losses
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                self._initialized_smoothing = True
            else: # Not enough data yet for initial average
                return None
        else:
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + current_gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + current_loss) / self.period
        
        if self._avg_loss == 0:
            self._current_value = 100.0 if self._avg_gain > 0 else 50.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._current_value = 100.0 - (100.0 / (1.0 + rs))
        
        return self._current_value

    @property
    def value(self) -> Optional[float]:
        return self._current_value

    @property
    def ready(self) -> bool:
        return self._current_value is not None

    def get_parameters(self) -> Dict[str, Any]:
        return {'period': self.period}

    def set_parameters(self, params: Dict[str, Any]) -> None:
        old_period = self.period
        self.period = params.get('period', self.period)
        if self.period <= 1:
            self.logger.warning(f"Invalid RSI period {self.period}. Reverting to {old_period}.")
            self.period = old_period
            return

        if old_period != self.period:
            self.logger.info(f"RSIIndicator '{self.name}' parameters updated: period={self.period}. Resetting state.")
            self.reset_state()

    def reset_state(self):
        self._prices.clear()
        self._gains.clear()
        self._losses.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._current_value = None
        self._initialized_smoothing = False
        # self.logger.debug(f"RSIIndicator '{self.name}' state reset.") # Debug level might be too verbose

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {'period': [9, 14]}
