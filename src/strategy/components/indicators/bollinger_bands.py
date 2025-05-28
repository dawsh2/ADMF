# src/strategy/components/indicators/bollinger_bands.py
import logging
from typing import Optional, Dict, Any, List
from collections import deque
from src.core.component_base import ComponentBase
from src.strategy.base.parameter import ParameterSpace, Parameter

class BollingerBandsIndicator(ComponentBase):
    """
    Bollinger Bands indicator implementation.
    
    Calculates upper and lower bands based on standard deviation around a moving average.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
        # Parameters
        self.lookback_period: int = 20
        self.num_std_dev: float = 2.0
        
        # State
        self._price_buffer: deque = deque(maxlen=20)
        self._upper_band: Optional[float] = None
        self._lower_band: Optional[float] = None
        self._middle_band: Optional[float] = None
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        self.lookback_period = int(self.get_specific_config('lookback_period', 20))
        self.num_std_dev = float(self.get_specific_config('num_std_dev', 2.0))
        
        # Reinitialize buffer with correct size
        self._price_buffer = deque(maxlen=self.lookback_period)
        
        self.logger.info(f"BollingerBandsIndicator '{self.instance_name}' initialized with "
                        f"period={self.lookback_period}, std_dev={self.num_std_dev}")
        
    def _start(self) -> None:
        """Start the indicator."""
        pass
        
    def _stop(self) -> None:
        """Stop the indicator."""
        pass
        
    def update(self, bar_data: Dict[str, Any]) -> None:
        """
        Update the indicator with new bar data.
        
        Args:
            bar_data: Dictionary containing bar data with 'close' price
        """
        # Extract price from bar data
        if isinstance(bar_data, dict) and 'close' in bar_data:
            price = bar_data['close']
        elif isinstance(bar_data, (int, float)):
            # Handle case where price is passed directly (for backward compatibility)
            price = float(bar_data)
        else:
            return
            
        self._price_buffer.append(price)
        
        if len(self._price_buffer) < self.lookback_period:
            # Not enough data yet
            self._upper_band = None
            self._lower_band = None
            self._middle_band = None
            return
            
        # Calculate middle band (SMA)
        self._middle_band = sum(self._price_buffer) / len(self._price_buffer)
        
        # Calculate standard deviation
        variance = sum((p - self._middle_band) ** 2 for p in self._price_buffer) / len(self._price_buffer)
        std_dev = variance ** 0.5
        
        # Calculate bands
        self._upper_band = self._middle_band + (self.num_std_dev * std_dev)
        self._lower_band = self._middle_band - (self.num_std_dev * std_dev)
        
    @property
    def ready(self) -> bool:
        """Check if the indicator has enough data to produce valid values."""
        return len(self._price_buffer) >= self.lookback_period
        
    @property
    def upper_band(self) -> Optional[float]:
        """Return the current upper band value."""
        return self._upper_band
        
    @property
    def lower_band(self) -> Optional[float]:
        """Return the current lower band value."""
        return self._lower_band
        
    @property
    def middle_band(self) -> Optional[float]:
        """Return the current middle band (SMA) value."""
        return self._middle_band
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the indicator."""
        return {
            'period': self.lookback_period,
            'num_std_dev': self.num_std_dev
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the indicator's parameters."""
        # Support both 'period' and 'lookback_period' for compatibility
        if 'lookback_period' in params:
            self.lookback_period = params['lookback_period']
        elif 'period' in params:
            self.lookback_period = params['period']
            
        self.num_std_dev = params.get('num_std_dev', self.num_std_dev)
        
        # Update buffer size if period changed
        if self._price_buffer.maxlen != self.lookback_period:
            # Create new buffer with updated size, preserving recent data
            old_data = list(self._price_buffer)[-self.lookback_period:]
            self._price_buffer = deque(old_data, maxlen=self.lookback_period)
        
        self.logger.info(
            f"BollingerBandsIndicator '{self.instance_name}' parameters updated: lookback_period={self.lookback_period}, num_std_dev={self.num_std_dev}"
        )
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for this component (ComponentBase interface)."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        space.add_parameter(Parameter(
            name="lookback_period",
            param_type="discrete",
            values=[15, 20, 25],  # Reduced from 5 to 3 values
            default=self.lookback_period
        ))
        
        space.add_parameter(Parameter(
            name="num_std_dev",
            param_type="discrete",
            values=[1.5, 2.0, 2.5],  # Reduced from 4 to 3 values
            default=self.num_std_dev
        ))
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        return {
            'period': [15, 20, 25],  # Matches the reduced parameter space
            'num_std_dev': [1.5, 2.0, 2.5]  # Matches the reduced parameter space
        }
        
    @property
    def band_width(self) -> Optional[float]:
        """Return the current band width."""
        if self._upper_band is not None and self._lower_band is not None:
            return self._upper_band - self._lower_band
        return None
        
    @property
    def band_percentage(self) -> Optional[float]:
        """Return where the last price is within the bands (0=lower, 1=upper)."""
        if (self._upper_band is not None and self._lower_band is not None and 
            len(self._price_buffer) > 0 and self.band_width > 0):
            last_price = self._price_buffer[-1]
            return (last_price - self._lower_band) / self.band_width
        return None
        
    def reset(self) -> None:
        """Reset the indicator state."""
        self._price_buffer.clear()
        self._upper_band = None
        self._lower_band = None
        self._middle_band = None
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the indicator."""
        return {
            'lookback_period': self.lookback_period,
            'num_std_dev': self.num_std_dev,
            'upper_band': self._upper_band,
            'lower_band': self._lower_band,
            'middle_band': self._middle_band,
            'buffer_size': len(self._price_buffer)
        }
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Return current parameter values that can be optimized."""
        return {
            'lookback_period': self.lookback_period,
            'num_std_dev': self.num_std_dev
        }
        
    def apply_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters (ComponentBase interface)."""
        self.set_parameters(params)
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set indicator parameters."""
        if 'lookback_period' in parameters:
            new_period = int(parameters['lookback_period'])
            if new_period != self.lookback_period:
                self.lookback_period = new_period
                # Preserve existing buffer data if possible
                old_data = list(self._price_buffer)
                self._price_buffer = deque(old_data[-new_period:], maxlen=new_period)
                self.logger.info(f"Updated lookback_period to {new_period}, preserved {len(self._price_buffer)} data points")
                
        if 'num_std_dev' in parameters:
            self.num_std_dev = float(parameters['num_std_dev'])
            self.logger.debug(f"Updated num_std_dev to {self.num_std_dev}")