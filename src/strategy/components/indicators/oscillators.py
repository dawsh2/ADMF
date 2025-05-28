# src/strategy/components/indicators/oscillators.py
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque
from src.core.component_base import ComponentBase
from src.strategy.base.parameter import Parameter, ParameterSpace


class RSIIndicator(ComponentBase):
    """
    Calculates the Relative Strength Index (RSI).
    
    This indicator is optimizable with the following parameters:
    - period: The number of periods for RSI calculation (default: 14)
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Initialize RSI indicator with minimal setup."""
        super().__init__(instance_name, config_key)
        
        # Default parameter values
        self._default_period = 14
        self.period: int = self._default_period
        
        # Internal state - will be initialized in _initialize
        self._prices: Optional[deque[float]] = None
        self._gains: Optional[deque[float]] = None
        self._losses: Optional[deque[float]] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._current_value: Optional[float] = None
        self._initialized_smoothing = False
        self._first_value_logged = False
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Get period from config or use default
        self.period = self.get_specific_config('period', self._default_period)
        
        # Validate period
        if not isinstance(self.period, int) or self.period <= 1:
            self.logger.error(f"Invalid RSI period '{self.period}'. Must be an integer > 1. Using default {self._default_period}.")
            self.period = self._default_period
            
        # Initialize internal state
        self._reset_state()
        self.logger.info(f"RSIIndicator '{self.instance_name}' initialized with period={self.period}")
        
    def _reset_state(self) -> None:
        """Reset internal state structures."""
        # Use max possible period from parameter space to avoid resets
        max_period = self.period
        try:
            param_space = self.get_parameter_space()
            for param in param_space._parameters.values():
                if param.name == 'period' and param.values:
                    max_period = max(param.values)
        except:
            max_period = max(50, self.period)
            
        # Create buffers that can handle the maximum period
        self._prices = deque(maxlen=max_period + 10)
        self._gains = deque(maxlen=max_period)
        self._losses = deque(maxlen=max_period)
        self._avg_gain = None
        self._avg_loss = None
        self._current_value = None
        self._initialized_smoothing = False
        self._first_value_logged = False
        
    def reset(self) -> None:
        """Reset component state."""
        super().reset()
        self._reset_state()
        self.logger.debug(f"RSIIndicator '{self.instance_name}' reset to initial state")
        
    def update(self, data_or_price: Union[Dict[str, Any], float]) -> Optional[float]:
        """Update RSI with new price data."""
        price_input: Optional[float] = None
        
        if isinstance(data_or_price, dict):
            price_input = data_or_price.get('close')
            if not isinstance(price_input, (int, float)) or price_input is None:
                self.logger.warning(f"Invalid or missing 'close' price in data dict for {self.instance_name}: {data_or_price}")
                return self._current_value
        elif isinstance(data_or_price, (int, float)):
            price_input = float(data_or_price)
        else:
            self.logger.warning(f"Invalid data/price type received by {self.instance_name}: {type(data_or_price)}. Expected dict or float.")
            return self._current_value
            
        # Store the new price and calculate change
        if not self._prices:
            self._prices.append(price_input)
            return None
            
        # Calculate price change
        change = price_input - self._prices[-1]
        self._prices.append(price_input)
        
        # Calculate gain and loss for this change
        current_gain = change if change > 0 else 0.0
        current_loss = abs(change) if change < 0 else 0.0
        
        # Add gain/loss to deques
        self._gains.append(current_gain)
        self._losses.append(current_loss)
        
        # Need at least 'period' gains/losses to start calculating RSI
        if len(self._gains) < self.period:
            return None
            
        if not self._initialized_smoothing:
            if len(self._gains) == self.period:
                self._avg_gain = sum(self._gains) / self.period
                self._avg_loss = sum(self._losses) / self.period
                self._initialized_smoothing = True
            else:
                return None
        else:
            # Smoothed averages
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + current_gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + current_loss) / self.period
            
        # Calculate RSI
        if self._avg_loss == 0:
            self._current_value = 100.0 if self._avg_gain > 0 else 50.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._current_value = 100.0 - (100.0 / (1.0 + rs))
            
        # Log first valid RSI value
        if not self._first_value_logged and self._current_value is not None:
            self._first_value_logged = True
            timestamp = data_or_price.get('timestamp', 'Unknown') if isinstance(data_or_price, dict) else 'Unknown'
            self.logger.info(f"RSI '{self.instance_name}' first valid value: {self._current_value:.2f} at {timestamp} after {len(self._prices)} bars")
            
        return self._current_value
        
    @property
    def value(self) -> Optional[float]:
        """Get current RSI value."""
        return self._current_value
        
    @property
    def ready(self) -> bool:
        """Check if RSI has produced a valid value."""
        return self._current_value is not None
        
    # ===== Optimization Support Methods =====
    
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for RSI optimization."""
        space = ParameterSpace(f"{self.instance_name}_space")
        space.add_parameter(Parameter(
            name="period",
            param_type="discrete",
            values=[7, 9, 14, 21, 30, 40],  # Expanded RSI periods
            default=self._default_period,
            description="Number of periods for RSI calculation"
        ))
        return space
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get current values of optimizable parameters."""
        return {
            "period": self.period
        }
        
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate RSI parameters."""
        if "period" in parameters:
            period = parameters["period"]
            if not isinstance(period, int):
                return False, f"Period must be an integer, got {type(period).__name__}"
            if period <= 1:
                return False, f"Period must be greater than 1, got {period}"
                
        return True, None
        
    def apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply new parameters to RSI."""
        # Validate first
        valid, error = self.validate_parameters(parameters)
        if not valid:
            raise ValueError(f"Invalid parameters for {self.instance_name}: {error}")
            
        # Apply period if provided
        if "period" in parameters:
            old_period = self.period
            self.period = parameters["period"]
            
            # Only reset if we don't have enough historical data
            if old_period != self.period:
                if len(self._prices) < self.period:
                    self.logger.warning(f"RSI '{self.instance_name}' period changed from {old_period} to {self.period}. Resetting due to insufficient data.")
                    self._reset_state()
                else:
                    self.logger.info(f"RSI '{self.instance_name}' period changed from {old_period} to {self.period}. Preserving {len(self._prices)} bars of history.")
                    # Just update the period, don't reset
                    self.period = self.period
            else:
                self.logger.debug(f"RSI '{self.instance_name}' period unchanged at {self.period}")
                
    # ===== Legacy compatibility methods =====
    
    def get_parameters(self) -> Dict[str, Any]:
        """Legacy method for getting parameters."""
        return self.get_optimizable_parameters()
        
    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """Legacy method for setting parameters."""
        try:
            self.apply_parameters(params)
            return True
        except ValueError:
            return False
            
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Legacy property for parameter space."""
        space = self.get_parameter_space()
        # Convert to legacy format
        result = {}
        for name, param in space._parameters.items():
            if param.param_type == 'discrete' and param.values:
                result[name] = param.values
        return result