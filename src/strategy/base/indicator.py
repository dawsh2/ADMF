"""
Base class for technical indicators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from ...core.component_base import ComponentBase
from .parameter import ParameterSpace, Parameter


@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    value: float
    values: Optional[Dict[str, float]] = None  # For multi-value indicators
    timestamp: Optional[Any] = None
    ready: bool = True


class IndicatorBase(ComponentBase, ABC):
    """
    Base class for all technical indicators.
    
    Provides:
    - Buffered calculation with configurable lookback
    - Parameter management
    - State management and reset
    - Ready state tracking
    - Inherits from ComponentBase for standard lifecycle and optimization
    """
    
    def __init__(self, instance_name: str, lookback_period: int = 1, config_key: Optional[str] = None):
        """Initialize indicator with ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Ensure we have a logger even if not initialized yet
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = logging.getLogger(instance_name)
        
        # Indicator-specific state
        self._lookback_period = lookback_period
        self._buffer: List[float] = []
        self._value: Optional[float] = None
        self._ready = False
        self._min_periods: int = lookback_period
        self._parameters: Dict[str, Any] = {
            'lookback_period': lookback_period
        }
        
        # Will be set based on parameter space to ensure we maintain enough history
        self._max_possible_lookback: Optional[int] = None
    
    def _initialize(self) -> None:
        """Component-specific initialization."""
        # Load configuration if available
        if self.component_config:
            self._lookback_period = self.component_config.get('lookback_period', self._lookback_period)
            self._min_periods = self.component_config.get('min_periods', self._lookback_period)
            self._parameters['lookback_period'] = self._lookback_period
        
        # Determine maximum possible lookback from parameter space
        self._determine_max_lookback()
        
        # Reset state
        self.reset()
    
    def _determine_max_lookback(self) -> None:
        """Determine the maximum lookback period from parameter space."""
        try:
            param_space = self.get_parameter_space()
            for param in param_space.parameters.values():
                if param.name == 'lookback_period' and param.param_type == 'discrete':
                    # Use the maximum value from the discrete values
                    self._max_possible_lookback = max(param.values)
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.info(f"Indicator {self.instance_name} will maintain buffer for max lookback: {self._max_possible_lookback}")
                    return
        except:
            pass
        
        # Default to 2x current lookback if we can't determine from parameter space
        self._max_possible_lookback = max(50, self._lookback_period * 2)
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Indicator {self.instance_name} using default max lookback: {self._max_possible_lookback}")
        
    def _start(self) -> None:
        """Component-specific start logic."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.debug(f"Indicator '{self.instance_name}' started")
        
    def _stop(self) -> None:
        """Component-specific stop logic."""
        self.reset()
        
    @property
    def name(self) -> str:
        """Unique name for this indicator (compatibility property)."""
        return self.instance_name
        
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
        
        # Trim buffer to max possible size we might need
        max_buffer_size = self._max_possible_lookback if self._max_possible_lookback else self._lookback_period * 2
        if len(self._buffer) > max_buffer_size * 2:  # Keep 2x for safety
            self._buffer = self._buffer[-max_buffer_size * 2:]
            
        # Check if we have enough data
        if len(self._buffer) >= self._min_periods:
            if not self._ready and hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Indicator {self.instance_name} now READY with {len(self._buffer)} bars (needed {self._min_periods})")
            self._ready = True
            result = self._calculate(self._buffer[-self._lookback_period:])
            self._value = result.value
            return result
        else:
            if self._ready and hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Indicator {self.instance_name} NOT ready - only {len(self._buffer)} bars, need {self._min_periods}")
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
        old_lookback = self._lookback_period
        
        if 'lookback_period' in params:
            self._lookback_period = params['lookback_period']
        if 'min_periods' in params:
            self._min_periods = params['min_periods']
        else:
            # Update min_periods to match new lookback if not explicitly set
            self._min_periods = self._lookback_period
            
        # Update any additional parameters
        for key, value in params.items():
            if key not in ['lookback_period', 'min_periods']:
                self._parameters[key] = value
                
        # Only reset if we don't have enough data for the new parameters
        # This preserves history when possible
        new_min_periods = self._min_periods
        
        # For lookback period changes, update the min_periods requirement
        if 'lookback_period' in params:
            new_min_periods = max(params['lookback_period'], self._min_periods)
            
        if len(self._buffer) < new_min_periods:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Indicator {self.instance_name} RESET - buffer has {len(self._buffer)} bars but needs {new_min_periods}")
            self.reset()
        else:
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Indicator {self.instance_name} preserving buffer with {len(self._buffer)} bars after parameter change")
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = ParameterSpace(f"{self.instance_name}_params")
        
        # Add lookback period parameter
        # For testing: limit to specific values based on indicator name
        if 'fast' in self.instance_name.lower():
            values = [5, 10]
        elif 'slow' in self.instance_name.lower():
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
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters (can be overridden by subclasses)."""
        if 'lookback_period' in params:
            if params['lookback_period'] < 1:
                return False, "lookback_period must be >= 1"
        return True, None
        
    def apply_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters (ComponentBase interface)."""
        self.set_parameters(params)
        # Ensure lookback period is updated
        if 'lookback_period' in params:
            self._lookback_period = params['lookback_period']
            self._min_periods = params.get('min_periods', self._lookback_period)
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get optimizable parameters (ComponentBase interface)."""
        return self.get_parameters()
    
    def reset(self) -> None:
        """Reset indicator state."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Indicator {self.instance_name} RESET - clearing {len(self._buffer)} bars of history")
        self._buffer.clear()
        self._value = None
        self._ready = False
        

class MovingAverageIndicator(IndicatorBase):
    """Simple Moving Average indicator."""
    
    def __init__(self, name: str = "SMA", lookback_period: int = 20, config_key: Optional[str] = None):
        super().__init__(instance_name=name, lookback_period=lookback_period, config_key=config_key)
        
    def _calculate(self, data: List[float]) -> IndicatorResult:
        """Calculate simple moving average."""
        if not data:
            return IndicatorResult(value=0.0, ready=False)
            
        sma_value = np.mean(data)
        return IndicatorResult(value=sma_value, ready=True)
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get parameter space for optimization."""
        space = ParameterSpace(f"{self.instance_name}_params")
        
        # Add lookback period parameter with values based on indicator type
        if 'fast' in self.instance_name.lower():
            # Fast MA: 5, 10, 15
            values = [5, 10, 15]
        elif 'slow' in self.instance_name.lower():
            # Slow MA: 20, 30, 40
            values = [20, 30, 40]
        else:
            # Generic MA
            values = [5, 10, 15, 20, 30, 40, 50]
            
        space.add_parameter(
            Parameter(
                name='lookback_period',
                param_type='discrete',
                values=values,
                default=self._lookback_period
            )
        )
        
        return space
        

class ExponentialMovingAverageIndicator(IndicatorBase):
    """Exponential Moving Average indicator."""
    
    def __init__(self, name: str = "EMA", lookback_period: int = 20, config_key: Optional[str] = None):
        super().__init__(instance_name=name, lookback_period=lookback_period, config_key=config_key)
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
    
    def __init__(self, name: str = "RSI", lookback_period: int = 14, config_key: Optional[str] = None):
        super().__init__(instance_name=name, lookback_period=lookback_period, config_key=config_key)
        self._gains: List[float] = []
        self._losses: List[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._oversold_threshold = 30.0
        self._overbought_threshold = 70.0
        
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
        # Use common RSI periods for optimization
        space.update_parameter(
            'lookback_period',
            Parameter(
                name='lookback_period',
                param_type='discrete',
                values=[9, 14, 21, 30],  # Common RSI periods
                default=14
            )
        )
        
        return space