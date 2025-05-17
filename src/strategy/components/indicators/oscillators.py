# src/strategy/components/indicators/oscillators.py
import logging
from typing import Dict, Any, List, Optional, Union # Added Union
from collections import deque
from src.core.component import BaseComponent # Using BaseComponent as per your project

class RSIIndicator(BaseComponent):
    """
    Calculates the Relative Strength Index (RSI).
    """
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: Optional[str], parameters: Optional[Dict[str, Any]] = None): # component_config_key can be Optional
        super().__init__(instance_name, config_loader, component_config_key)
        # Using component.name for logger, assuming BaseComponent sets self.name
        self.logger = logging.getLogger(f"component.{self.name}")

        # Ensure parameters is a dict if None for safer access
        current_params = parameters if isinstance(parameters, dict) else {}
        
        # Try to get period from parameters, then from specific component config, then default
        self.period: int = current_params.get('period')
        if self.period is None:
            self.period = self.get_specific_config('period', 14)

        if not isinstance(self.period, int) or self.period <= 1:
            self.logger.error(f"Invalid RSI period '{self.period}'. Must be an integer > 1. Defaulting to 14.")
            self.period = 14

        self._prices: deque[float] = deque(maxlen=self.period + 10) 
        self._gains: deque[float] = deque(maxlen=self.period)
        self._losses: deque[float] = deque(maxlen=self.period)
        
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._current_value: Optional[float] = None
        self._initialized_smoothing = False

    def setup(self):
        """Sets up the component."""
        super().setup() # Call BaseComponent's setup
        self.reset_state()
        self.logger.info(f"RSIIndicator '{self.name}' configured with period={self.period}.")
        if self.state != BaseComponent.STATE_INITIALIZED:
             self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"RSIIndicator '{self.name}' setup complete. State: {self.state}")

    def start(self) -> None:
        """Starts the component's active operations."""
        super().start()
        if self.state != BaseComponent.STATE_STARTED: # Check state after super call
            self.state = BaseComponent.STATE_STARTED
        self.logger.info(f"RSIIndicator '{self.name}' started. State: {self.state}")

    def stop(self) -> None:
        """Stops the component's active operations and cleans up resources."""
        super().stop()
        self.reset_state() 
        if self.state != BaseComponent.STATE_STOPPED: # Check state after super call
            self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"RSIIndicator '{self.name}' stopped. State: {self.state}")

    def update(self, data_or_price: Union[Dict[str, Any], float]) -> Optional[float]: # MODIFIED: Accept Union
        price_input: Optional[float] = None

        if isinstance(data_or_price, dict):
            price_input = data_or_price.get('close')
            if not isinstance(price_input, (int, float)) or price_input is None:
                self.logger.warning(f"Invalid or missing 'close' price in data dict for {self.name}: {data_or_price}")
                return self._current_value
        elif isinstance(data_or_price, (int, float)):
            price_input = float(data_or_price)
        else:
            self.logger.warning(f"Invalid data/price type received by {self.name}: {type(data_or_price)}. Expected dict or float.")
            return self._current_value

        # --- The rest of your RSI logic remains the same, using price_input ---
        if not self._prices: 
            self._prices.append(price_input)
            return None

        change = price_input - self._prices[-1]
        
        if len(self._prices) < self.period: 
            self._prices.append(price_input) 
            if len(self._prices) > 1: 
                if change > 0:
                    self._gains.append(change)
                    self._losses.append(0.0)
                else:
                    self._gains.append(0.0)
                    self._losses.append(abs(change))
            return None
        
        if len(self._gains) == self.period: # Should always be true if len(_prices) >= period
            self._gains.popleft()
            self._losses.popleft()
        
        current_gain = change if change > 0 else 0.0
        current_loss = abs(change) if change < 0 else 0.0
        self._gains.append(current_gain)
        self._losses.append(current_loss)
        self._prices.append(price_input) 

        if not self._initialized_smoothing:
            if len(self._gains) == self.period: 
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                self._initialized_smoothing = True
            else: 
                return None # Should not happen if logic above is correct
        else:
            # Ensure _avg_gain and _avg_loss are not None before using them
            if self._avg_gain is None or self._avg_loss is None:
                 # This case implies not enough data for smoothing yet, which should be caught earlier
                 # or indicates a reset without full re-initialization. Re-calculating initial average.
                if len(self._gains) == self.period:
                    self._avg_gain = sum(self._gains) / self.period
                    self._avg_loss = sum(self._losses) / self.period
                    self._initialized_smoothing = True # Ensure it's set
                else:
                    self.logger.error(f"RSI '{self.name}': Smoothing not initialized and insufficient gains/losses. This should not happen.")
                    return None # Cannot proceed

            self._avg_gain = ((self._avg_gain * (self.period - 1)) + current_gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + current_loss) / self.period
        
        if self._avg_loss == 0: # Avoid division by zero
            self._current_value = 100.0 if self._avg_gain > 0 else 50.0 # RSI is 100 if all losses are 0 and gains > 0, 50 if no change
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

    def set_parameters(self, params: Dict[str, Any]) -> bool: 
        old_period = self.period
        new_period_val = params.get('period', self.period) 
        
        if not isinstance(new_period_val, int) or new_period_val <= 1:
            self.logger.warning(f"Invalid RSI period {new_period_val} for {self.name}. Must be an integer > 1. Period not changed from {old_period}.")
            return False

        if old_period != new_period_val:
            self.period = new_period_val
            self.logger.info(f"RSIIndicator '{self.name}' parameters updated: period={self.period}. Resetting state.")
            self.reset_state() # Important: reset internal deques and averages
        return True

    def reset_state(self):
        self._prices.clear()
        self._gains.clear()
        self._losses.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._current_value = None
        self._initialized_smoothing = False
        # self.logger.debug(f"RSIIndicator '{self.name}' state reset.")

    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        return {'period': [9, 14, 21]} # Example
