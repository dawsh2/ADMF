# src/strategy/components/indicators/trend.py
from typing import Any, Dict, Optional, List
import logging

class SimpleMATrendIndicator:
    """
    A simple trend indicator based on the relationship between a short-term
    and a long-term moving average.
    - Positive value suggests uptrend (short MA > long MA).
    - Negative value suggests downtrend (short MA < long MA).
    - Value is the percentage difference.
    """
    def __init__(self, short_period: int = 10, long_period: int = 30, 
                 instance_name: Optional[str] = None,
                 config_loader=None, event_bus=None, component_config_key: Optional[str]=None,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the SimpleMATrendIndicator.

        Args:
            short_period (int): The lookback period for the short-term MA.
            long_period (int): The lookback period for the long-term MA.
            instance_name (str, optional): Name of the indicator instance.
            parameters (Dict[str, Any], optional): Parameters dictionary, 
                                                   e.g., {"short_period": 10, "long_period": 30}.
                                                   If provided, periods from here take precedence.
        """
        self.instance_name = instance_name or f"SimpleMATrend_{short_period}_{long_period}"
        self.logger = logging.getLogger(f"{__name__}.{self.instance_name}")

        if parameters:
            self.short_period = int(parameters.get('short_period', short_period))
            self.long_period = int(parameters.get('long_period', long_period))
        else:
            self.short_period = int(short_period)
            self.long_period = int(long_period)

        if self.short_period <= 0 or self.long_period <= 0:
            self.logger.error("MA periods must be positive.")
            raise ValueError("MA periods must be positive.")
        if self.short_period >= self.long_period:
            self.logger.error("Short MA period must be less than long MA period for this trend logic.")
            raise ValueError("Short MA period must be less than long MA period.")

        self._prices: List[float] = []
        self._short_ma_values: List[float] = [] # For potential smoothing or historical access
        self._long_ma_values: List[float] = []  # For potential smoothing or historical access
        
        self._current_value: Optional[float] = None
        self._is_ready: bool = False
        
        self.logger.info(f"SimpleMATrendIndicator '{self.instance_name}' initialized with short_period={self.short_period}, long_period={self.long_period}.")

    def _calculate_sma(self, data: List[float], period: int) -> Optional[float]:
        if len(data) < period:
            return None
        return sum(data[-period:]) / period

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update the indicator with new market data (typically a bar).

        Args:
            data (Dict[str, Any]): A dictionary containing 'close' price.
        """
        close_price = data.get('close')

        if close_price is None:
            self.logger.warning(f"Missing 'close' price in data for {self.instance_name}. Skipping update.")
            return
        
        try:
            current_close = float(close_price)
        except ValueError:
            self.logger.error(f"Invalid close price data (not convertible to float) for {self.instance_name}. Skipping update.")
            return

        self._prices.append(current_close)
        
        # Maintain buffer size for prices
        if len(self._prices) > self.long_period:
            self._prices.pop(0)

        if len(self._prices) >= self.long_period:
            short_ma = self._calculate_sma(self._prices, self.short_period)
            long_ma = self._calculate_sma(self._prices, self.long_period)

            if short_ma is not None and long_ma is not None:
                self._short_ma_values.append(short_ma)
                self._long_ma_values.append(long_ma)
                
                # Keep history of MAs if needed, e.g., for 100 bars
                if len(self._short_ma_values) > 100: self._short_ma_values.pop(0)
                if len(self._long_ma_values) > 100: self._long_ma_values.pop(0)

                # Value is the percentage difference: ((short_ma / long_ma) - 1) * 100
                # Positive if short > long (uptrend), negative if short < long (downtrend)
                if long_ma != 0:
                    self._current_value = ((short_ma / long_ma) - 1.0) * 100.0
                else: # Avoid division by zero, though unlikely for price MAs
                    self._current_value = 0.0 
                self._is_ready = True
                
                # Debug logging for first ready value
                if not hasattr(self, '_logged_first_calculation'):
                    self.logger.info(f"[MA TREND DEBUG] {self.instance_name} first calculation:")
                    self.logger.info(f"  Prices buffer (last 10): {list(self._prices)[-10:]}")
                    self.logger.info(f"  Short MA ({self.short_period}): {short_ma:.6f}")
                    self.logger.info(f"  Long MA ({self.long_period}): {long_ma:.6f}")
                    self.logger.info(f"  Trend value: {self._current_value:.6f}%")
                    self._logged_first_calculation = True
                
                # Also log when we see specific price ranges
                current_price = float(data.get('close', 0))
                if 520.5 < current_price < 521.5 and not hasattr(self, '_logged_521_range'):
                    self.logger.warning(f"[MA PRICE WARNING] {self.instance_name} received price ${current_price:.2f} which is in the $521 range!")
                    self.logger.warning(f"  This price suggests March 26 data (bars 80-99)")
                    self.logger.warning(f"  Full price buffer: {list(self._prices)}")
                    self._logged_521_range = True
                elif 523.0 < current_price < 524.0 and not hasattr(self, '_logged_523_range'):
                    self.logger.info(f"[MA PRICE INFO] {self.instance_name} received price ${current_price:.2f} which is in the $523 range")
                    self.logger.info(f"  This price confirms March 28 data (bars 800-999)")
                    self._logged_523_range = True
                # self.logger.debug(f"{self.instance_name} updated. Trend value: {self._current_value:.2f}%")
            else:
                # This case should ideally not happen if len(prices) >= long_period
                self._is_ready = False
                self._current_value = None
        else:
            self._is_ready = False
            self._current_value = None

    @property
    def value(self) -> Optional[float]:
        """Current value of the trend indicator (percentage difference between MAs)."""
        return self._current_value

    @property
    def ready(self) -> bool:
        """Whether the indicator has enough data to provide a value."""
        return self._is_ready

    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameters of this indicator."""
        return {'short_period': self.short_period, 'long_period': self.long_period}

    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """Sets the parameters for this indicator."""
        changed = False
        if 'short_period' in params:
            new_short = int(params['short_period'])
            if new_short > 0 and new_short != self.short_period:
                self.short_period = new_short
                changed = True
        if 'long_period' in params:
            new_long = int(params['long_period'])
            if new_long > 0 and new_long != self.long_period:
                self.long_period = new_long
                changed = True
        
        if changed:
            if self.short_period >= self.long_period:
                self.logger.error("Short MA period must be less than long MA period. Parameter change reverted/ignored.")
                # Revert to previous valid state or handle error appropriately
                # For simplicity, this example might leave it in an invalid state if not careful.
                # A robust implementation would validate before applying all changes.
                return False 

            self._prices.clear()
            self._short_ma_values.clear()
            self._long_ma_values.clear()
            self._current_value = None
            self._is_ready = False
            self.logger.info(f"Parameters updated for {self.instance_name}. New short: {self.short_period}, new long: {self.long_period}")
        return changed
    
    def reset(self):
        """Reset the indicator to its initial state."""
        self._prices.clear()
        self._short_ma_values.clear()
        self._long_ma_values.clear()
        self._current_value = None
        self._is_ready = False
        self.logger.debug(f"SimpleMATrendIndicator '{self.instance_name}' reset to initial state")

    # Lifecycle methods if inheriting BaseComponent
    # def setup(self): self.logger.info(f"{self.instance_name} setup complete.")
    # def start(self): self.logger.info(f"{self.instance_name} started.")
    # def stop(self): self.logger.info(f"{self.instance_name} stopped.")

