# src/strategy/components/indicators/volatility.py
from typing import Any, Dict, Optional, List
import logging # It's good practice for components to have logging

# Assuming you have a BaseIndicator or a similar structure.
# If not, this ATRIndicator can stand alone or inherit from a common base if you create one.
# For now, let's make it self-contained but with a structure similar to your RSIIndicator.

class ATRIndicator:
    """
    Average True Range (ATR) Indicator for measuring market volatility.
    """
    def __init__(self, period: int = 14, instance_name: Optional[str] = None, 
                 config_loader=None, event_bus=None, component_config_key: Optional[str]=None,
                 parameters: Optional[Dict[str, Any]] = None): # Match RSIIndicator constructor style
        """
        Initialize the ATRIndicator.

        Args:
            period (int): The lookback period for calculating ATR.
            instance_name (str, optional): Name of the indicator instance.
            config_loader: Configuration loader (if needed, passed from component).
            event_bus: Event bus (if needed, passed from component).
            component_config_key (str, optional): Specific config key (if needed).
            parameters (Dict[str, Any], optional): Parameters dictionary, e.g., {"period": 14}.
                                                   If provided, 'period' from here takes precedence.
        """
        self.instance_name = instance_name or f"ATRIndicator_{period}"
        self.logger = logging.getLogger(f"{__name__}.{self.instance_name}")
        
        # Allow period to be set by parameters dict for consistency with other components
        if parameters and 'period' in parameters:
            self.period = int(parameters['period'])
        else:
            self.period = int(period)

        if self.period <= 0:
            self.logger.error("ATR period must be positive.")
            raise ValueError("ATR period must be positive.")

        self._high_prices: List[float] = []
        self._low_prices: List[float] = []
        self._close_prices: List[float] = []
        self._true_ranges: List[float] = []
        
        self._current_value: Optional[float] = None
        self._is_ready: bool = False
        
        self.logger.info(f"ATRIndicator '{self.instance_name}' initialized with period {self.period}.")

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the indicator with new market data (typically a bar).

        Args:
            data (Dict[str, Any]): A dictionary containing 'high', 'low', and 'close' prices.
                                   It should also contain 'timestamp' for complete bar data.
        """
        high = data.get('high')
        low = data.get('low')
        close = data.get('close')

        if high is None or low is None or close is None:
            self.logger.warning(f"Missing 'high', 'low', or 'close' in data for {self.instance_name}. Skipping update.")
            return

        try:
            current_high = float(high)
            current_low = float(low)
            current_close = float(close)
        except ValueError:
            self.logger.error(f"Invalid price data (not convertible to float) for {self.instance_name}. Skipping update.")
            return

        # Store current prices
        self._high_prices.append(current_high)
        self._low_prices.append(current_low)
        self._close_prices.append(current_close)

        # Calculate True Range
        if len(self._close_prices) > 1: # Need at least one previous close
            prev_close = self._close_prices[-2] # The one before the current `current_close`
            
            tr1 = current_high - current_low
            tr2 = abs(current_high - prev_close)
            tr3 = abs(current_low - prev_close)
            true_range = max(tr1, tr2, tr3)
            self._true_ranges.append(true_range)
        
            # Maintain buffer sizes for TRs (ATR period) and prices (ATR period + 1 for prev_close)
            if len(self._true_ranges) > self.period:
                self._true_ranges.pop(0)
        
        # Trim price buffers once they exceed necessary length
        # Need period + 1 prices to calculate `period` number of TRs
        max_price_len = self.period + 1 
        if len(self._close_prices) > max_price_len:
            self._high_prices.pop(0)
            self._low_prices.pop(0)
            self._close_prices.pop(0)

        # Calculate ATR
        if len(self._true_ranges) >= self.period:
            # For the first ATR, it's a simple average of the TRs.
            # Subsequent ATRs can be calculated using a smoothed moving average (Wilder's smoothing),
            # but for simplicity in a RegimeDetector, a simple moving average of TRs is often sufficient.
            # ATR = (Previous ATR * (n - 1) + Current TR) / n
            # We'll use a simple moving average of TRs for this example.
            self._current_value = sum(self._true_ranges[-self.period:]) / self.period
            self._is_ready = True
            # self.logger.debug(f"{self.instance_name} updated. ATR: {self._current_value:.4f}")
        else:
            self._is_ready = False
            self._current_value = None

    @property
    def value(self) -> Optional[float]:
        """Current value of the ATR indicator."""
        return self._current_value

    @property
    def ready(self) -> bool:
        """Whether the indicator has enough data to provide a value."""
        return self._is_ready

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters of this indicator."""
        return {'period': self.period}

    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """Sets the parameters for this indicator."""
        if 'period' in params:
            new_period = int(params['period'])
            if new_period > 0:
                if new_period != self.period:
                    self.period = new_period
                    # Reset internal state when parameters change
                    self._high_prices.clear()
                    self._low_prices.clear()
                    self._close_prices.clear()
                    self._true_ranges.clear()
                    self._current_value = None
                    self._is_ready = False
                    self.logger.info(f"Parameters updated for {self.instance_name}. New period: {self.period}")
                return True
            else:
                self.logger.warning(f"Attempted to set invalid period {new_period} for {self.instance_name}")
        return False
        
    # If your indicators are expected to be BaseComponents and need these lifecycle methods:
    # def setup(self):
    #     self.logger.info(f"ATRIndicator '{self.instance_name}' setup complete.")
    #     # If it needs to subscribe to events directly, do it here.
    #     # self.state = BaseComponent.STATE_INITIALIZED # If inheriting BaseComponent

    # def start(self):
    #     self.logger.info(f"ATRIndicator '{self.instance_name}' started.")
    #     # self.state = BaseComponent.STATE_STARTED

    # def stop(self):
    #     self.logger.info(f"ATRIndicator '{self.instance_name}' stopped.")
    #     # self.state = BaseComponent.STATE_STOPPED
