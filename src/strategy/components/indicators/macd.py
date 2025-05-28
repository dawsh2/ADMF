# src/strategy/components/indicators/macd.py
import logging
from typing import Optional, Dict, Any, List
from collections import deque
from src.core.component_base import ComponentBase
from src.strategy.base.parameter import ParameterSpace, Parameter

class MACDIndicator(ComponentBase):
    """
    MACD (Moving Average Convergence Divergence) indicator implementation.
    
    Calculates MACD line, signal line, and histogram.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
        # Parameters
        self.fast_period: int = 12
        self.slow_period: int = 26
        self.signal_period: int = 9
        
        # State - EMA calculations
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._signal_ema: Optional[float] = None
        self._macd_line: Optional[float] = None
        self._histogram: Optional[float] = None
        
        # EMA multipliers
        self._fast_multiplier: float = 2.0 / (self.fast_period + 1)
        self._slow_multiplier: float = 2.0 / (self.slow_period + 1)
        self._signal_multiplier: float = 2.0 / (self.signal_period + 1)
        
        # Price history for initialization
        self._price_history: deque = deque(maxlen=max(self.fast_period, self.slow_period))
        self._macd_history: deque = deque(maxlen=self.signal_period)
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        self.fast_period = int(self.get_specific_config('fast_period', 12))
        self.slow_period = int(self.get_specific_config('slow_period', 26))
        self.signal_period = int(self.get_specific_config('signal_period', 9))
        
        # Recalculate multipliers
        self._fast_multiplier = 2.0 / (self.fast_period + 1)
        self._slow_multiplier = 2.0 / (self.slow_period + 1)
        self._signal_multiplier = 2.0 / (self.signal_period + 1)
        
        # Reinitialize buffers
        self._price_history = deque(maxlen=max(self.fast_period, self.slow_period))
        self._macd_history = deque(maxlen=self.signal_period)
        
        self.logger.info(f"MACDIndicator '{self.instance_name}' initialized with "
                        f"fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
        
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
            
        self._price_history.append(price)
        
        # Initialize EMAs if we have enough data
        if self._fast_ema is None and len(self._price_history) >= self.fast_period:
            self._fast_ema = sum(list(self._price_history)[-self.fast_period:]) / self.fast_period
            
        if self._slow_ema is None and len(self._price_history) >= self.slow_period:
            self._slow_ema = sum(self._price_history) / self.slow_period
            
        # Update EMAs if initialized
        if self._fast_ema is not None:
            self._fast_ema = (price * self._fast_multiplier) + (self._fast_ema * (1 - self._fast_multiplier))
            
        if self._slow_ema is not None:
            self._slow_ema = (price * self._slow_multiplier) + (self._slow_ema * (1 - self._slow_multiplier))
            
        # Calculate MACD line
        if self._fast_ema is not None and self._slow_ema is not None:
            self._macd_line = self._fast_ema - self._slow_ema
            self._macd_history.append(self._macd_line)
            
            # Initialize signal line if we have enough MACD values
            if self._signal_ema is None and len(self._macd_history) >= self.signal_period:
                self._signal_ema = sum(self._macd_history) / self.signal_period
                
            # Update signal line if initialized
            if self._signal_ema is not None:
                self._signal_ema = (self._macd_line * self._signal_multiplier) + (self._signal_ema * (1 - self._signal_multiplier))
                self._histogram = self._macd_line - self._signal_ema
                
    @property
    def ready(self) -> bool:
        """Check if the indicator has enough data to produce valid values."""
        return self._signal_ema is not None
        
    @property
    def macd_line(self) -> Optional[float]:
        """Return the current MACD line value."""
        return self._macd_line
        
    @property
    def signal_line(self) -> Optional[float]:
        """Return the current signal line value."""
        return self._signal_ema
        
    @property
    def histogram(self) -> Optional[float]:
        """Return the current histogram value."""
        return self._histogram
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the indicator."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the indicator's parameters."""
        self.fast_period = params.get('fast_period', self.fast_period)
        self.slow_period = params.get('slow_period', self.slow_period)
        self.signal_period = params.get('signal_period', self.signal_period)
        
        # Recalculate multipliers
        self._fast_multiplier = 2 / (self.fast_period + 1)
        self._slow_multiplier = 2 / (self.slow_period + 1)
        self._signal_multiplier = 2 / (self.signal_period + 1)
        
        # Reset state for recalculation
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._macd_line = None
        self._histogram = None
        
        self.logger.info(
            f"MACDIndicator '{self.instance_name}' parameters updated: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}"
        )
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for this component (ComponentBase interface)."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        space.add_parameter(Parameter(
            name="fast_period",
            param_type="discrete",
            values=[8, 12],
            default=self.fast_period
        ))
        
        space.add_parameter(Parameter(
            name="slow_period",
            param_type="discrete",
            values=[20, 26],
            default=self.slow_period
        ))
        
        space.add_parameter(Parameter(
            name="signal_period",
            param_type="discrete",
            values=[7, 9],
            default=self.signal_period
        ))
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        return {
            'fast_period': [8, 12],
            'slow_period': [20, 26],
            'signal_period': [7, 9]
        }
        
    def reset(self) -> None:
        """Reset the indicator state."""
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None
        self._macd_line = None
        self._histogram = None
        self._price_history.clear()
        self._macd_history.clear()
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the indicator."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'macd_line': self._macd_line,
            'signal_line': self._signal_ema,
            'histogram': self._histogram
        }
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Return current parameter values that can be optimized."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }
        
    def apply_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters (ComponentBase interface)."""
        self.set_parameters(params)
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set indicator parameters."""
        if 'fast_period' in parameters:
            self.fast_period = int(parameters['fast_period'])
            self._fast_multiplier = 2.0 / (self.fast_period + 1)
            
        if 'slow_period' in parameters:
            self.slow_period = int(parameters['slow_period'])
            self._slow_multiplier = 2.0 / (self.slow_period + 1)
            
        if 'signal_period' in parameters:
            self.signal_period = int(parameters['signal_period'])
            self._signal_multiplier = 2.0 / (self.signal_period + 1)
            
        self.logger.debug(f"Updated MACD parameters: {parameters}")