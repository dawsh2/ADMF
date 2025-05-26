"""
Base class for technical indicators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from .strategy import StrategyComponent
from .parameter import ParameterSpace, Parameter


@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    value: float
    values: Optional[Dict[str, float]] = None  # For multi-value indicators
    timestamp: Optional[Any] = None
    ready: bool = True


class IndicatorBase(StrategyComponent, ABC):
    """
    Base class for all technical indicators.
    
    Provides:
    - Buffered calculation with configurable lookback
    - Parameter management
    - State management and reset
    - Ready state tracking
    """
    
    def __init__(self, name: str, lookback_period: int = 1):
        self._name = name
        self._lookback_period = lookback_period
        self._buffer: List[float] = []
        self._value: Optional[float] = None
        self._ready = False
        self._min_periods: int = lookback_period
        self._parameters: Dict[str, Any] = {}
        
    @property
    def name(self) -> str:
        """Unique name for this indicator."""
        return self._name
        
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce valid output."""
        return self._ready
        
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._value
        
    @property
    def lookback_period(self) -> int:
        """Number of periods used in calculation."""
        return self._lookback_period
        
    def update(self, bar_data: Dict[str, Any]) -> IndicatorResult:
        """Update indicator with new bar data."""
        # Extract price from bar data
        price = self._extract_price(bar_data)
        
        # Add to buffer
        self._buffer.append(price)
        
        # Trim buffer to max size
        if len(self._buffer) > self._lookback_period * 2:  # Keep some extra
            self._buffer = self._buffer[-self._lookback_period * 2:]
            
        # Check if we have enough data
        if len(self._buffer) >= self._min_periods:
            self._ready = True
            result = self._calculate(self._buffer[-self._lookback_period:])
            self._value = result.value
            return result
        else:
            self._ready = False
            return IndicatorResult(value=0.0, ready=False)
            
    @abstractmethod
    def _calculate(self, data: List[float]) -> IndicatorResult:
        """Calculate indicator value from price data."""
        pass
        
    def _extract_price(self, bar_data: Dict[str, Any]) -> float:
        """Extract price from bar data. Override for custom price extraction."""
        # Default to close price
        return float(bar_data.get('close', 0.0))
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {
            'lookback_period': self._lookback_period,
            'min_periods': self._min_periods
        }
        params.update(self._parameters)
        return params
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameter values."""
        if 'lookback_period' in params:
            self._lookback_period = params['lookback_period']
        if 'min_periods' in params:
            self._min_periods = params['min_periods']
            
        # Update any additional parameters
        for key, value in params.items():
            if key not in ['lookback_period', 'min_periods']:
                self._parameters[key] = value
                
        # Reset on parameter change
        self.reset()
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = ParameterSpace(f"{self.name}_params")
        
        # Add lookback period parameter
        # For testing: limit to specific values based on indicator name
        if 'fast' in self.name.lower():
            values = [5, 10]
        elif 'slow' in self.name.lower():
            values = [20, 30]
        else:
            values = list(range(5, 50, 5))
            
        space.add_parameter(
            Parameter(
                name='lookback_period',
                param_type='discrete',
                values=values,
                default=self._lookback_period
            )
        )
        
        # Subclasses should override to add specific parameters
        return space
        
    def reset(self) -> None:
        """Reset indicator state."""
        self._buffer.clear()
        self._value = None
        self._ready = False
        

class MovingAverageIndicator(IndicatorBase):
    """Simple Moving Average indicator."""
    
    def __init__(self, name: str = "SMA", lookback_period: int = 20):
        super().__init__(name, lookback_period)
        
    def _calculate(self, data: List[float]) -> IndicatorResult:
        """Calculate simple moving average."""
        if not data:
            return IndicatorResult(value=0.0, ready=False)
            
        sma_value = np.mean(data)
        return IndicatorResult(value=sma_value, ready=True)
        

class ExponentialMovingAverageIndicator(IndicatorBase):
    """Exponential Moving Average indicator."""
    
    def __init__(self, name: str = "EMA", lookback_period: int = 20):
        super().__init__(name, lookback_period)
        self._ema_value: Optional[float] = None
        self._alpha = 2.0 / (lookback_period + 1)
        
    def _calculate(self, data: List[float]) -> IndicatorResult:
        """Calculate exponential moving average."""
        if not data:
            return IndicatorResult(value=0.0, ready=False)
            
        if self._ema_value is None:
            # Initialize with SMA
            self._ema_value = np.mean(data)
        else:
            # Update EMA
            self._ema_value = (data[-1] * self._alpha) + (self._ema_value * (1 - self._alpha))
            
        return IndicatorResult(value=self._ema_value, ready=True)
        
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._ema_value = None
        

class RSIIndicator(IndicatorBase):
    """Relative Strength Index indicator."""
    
    def __init__(self, name: str = "RSI", lookback_period: int = 14):
        super().__init__(name, lookback_period)
        self._gains: List[float] = []
        self._losses: List[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        
    def update(self, bar_data: Dict[str, Any]) -> IndicatorResult:
        """Update RSI with new bar data."""
        price = self._extract_price(bar_data)
        
        # Need at least 2 prices to calculate change
        if len(self._buffer) > 0:
            change = price - self._buffer[-1]
            gain = max(0, change)
            loss = max(0, -change)
            
            self._gains.append(gain)
            self._losses.append(loss)
            
            # Trim history
            if len(self._gains) > self._lookback_period * 2:
                self._gains = self._gains[-self._lookback_period * 2:]
                self._losses = self._losses[-self._lookback_period * 2:]
                
        self._buffer.append(price)
        if len(self._buffer) > self._lookback_period * 2:
            self._buffer = self._buffer[-self._lookback_period * 2:]
            
        # Calculate RSI
        if len(self._gains) >= self._lookback_period:
            self._ready = True
            result = self._calculate(None)  # We use internal buffers
            self._value = result.value
            return result
        else:
            self._ready = False
            return IndicatorResult(value=50.0, ready=False)  # Neutral RSI
            
    def _calculate(self, data: List[float]) -> IndicatorResult:
        """Calculate RSI value."""
        if self._avg_gain is None or self._avg_loss is None:
            # Initialize with SMA
            self._avg_gain = np.mean(self._gains[-self._lookback_period:])
            self._avg_loss = np.mean(self._losses[-self._lookback_period:])
        else:
            # Update with EMA approach
            current_gain = self._gains[-1] if self._gains else 0
            current_loss = self._losses[-1] if self._losses else 0
            
            self._avg_gain = (self._avg_gain * (self._lookback_period - 1) + current_gain) / self._lookback_period
            self._avg_loss = (self._avg_loss * (self._lookback_period - 1) + current_loss) / self._lookback_period
            
        # Calculate RSI
        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
        return IndicatorResult(value=rsi, ready=True)
        
    def reset(self) -> None:
        """Reset indicator state."""
        super().reset()
        self._gains.clear()
        self._losses.clear()
        self._avg_gain = None
        self._avg_loss = None
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = super().get_parameter_space()
        
        # RSI typically uses periods between 5 and 25
        # For now, keep RSI period fixed at 14
        space.update_parameter(
            'lookback_period',
            Parameter(
                name='lookback_period',
                param_type='discrete',
                values=[14],  # Fixed value only
                default=14
            )
        )
        
        return space