# src/strategy/components/rules/ma_crossover_rule.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component_base import ComponentBase
from src.strategy.components.indicators.trend import SimpleMovingAverage
from src.strategy.base.parameter import ParameterSpace, Parameter

class MACrossoverRule(ComponentBase):
    """
    Generates trading signals based on moving average crossovers.
    
    This rule generates buy signals when a fast MA crosses above a slow MA,
    and sell signals when a fast MA crosses below a slow MA.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None,
                 fast_ma: Optional[SimpleMovingAverage] = None, 
                 slow_ma: Optional[SimpleMovingAverage] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_key)
        
        # Store dependencies - will be injected during initialize
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self._parameters = parameters or {}
        
        # Rule state
        self._weight: float = 1.0
        self.min_separation: float = 0.0
        self._prev_fast_value: Optional[float] = None
        self._prev_slow_value: Optional[float] = None
        self._current_position: int = 0  # -1: fast below slow, 0: unknown, 1: fast above slow
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self._weight = self._parameters.get('weight', 1.0)
            self.min_separation = self._parameters.get('min_separation', 0.0)
        else:
            self._weight = self.get_specific_config('weight', 1.0)
            self.min_separation = self.get_specific_config('min_separation', 0.0)
            
        self.reset_state()
        
        # Validate dependencies if they were provided
        if self.fast_ma and not hasattr(self.fast_ma, 'value'):
            self.logger.error(f"MACrossoverRule '{self.instance_name}' was not provided with a valid fast MA indicator.")
            return
            
        if self.slow_ma and not hasattr(self.slow_ma, 'value'):
            self.logger.error(f"MACrossoverRule '{self.instance_name}' was not provided with a valid slow MA indicator.")
            return
            
        self.logger.info(f"MACrossoverRule '{self.instance_name}' configured with weight={self._weight}, min_separation={self.min_separation}")
        
    def _start(self) -> None:
        """Component-specific start logic."""
        self.logger.info(f"MACrossoverRule '{self.instance_name}' ready for evaluation")
        
    def _stop(self) -> None:
        """Component-specific stop logic."""
        self.reset_state()
        
    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Optional[str]]:
        """
        Evaluates the MA crossover condition.
        
        Returns:
            Tuple[bool, float, Optional[str]]: (triggered, signal_strength, signal_type)
                - triggered: True if a crossover occurred
                - signal_strength: 1.0 for buy, -1.0 for sell, 0.0 for no signal
                - signal_type: "BUY", "SELL", or None
        """
        if not self.fast_ma or not self.fast_ma.ready or not self.slow_ma or not self.slow_ma.ready:
            return False, 0.0, None
            
        fast_value = self.fast_ma.value
        slow_value = self.slow_ma.value
        
        if fast_value is None or slow_value is None:
            return False, 0.0, None
            
        # Initialize previous values on first call
        if self._prev_fast_value is None or self._prev_slow_value is None:
            self._prev_fast_value = fast_value
            self._prev_slow_value = slow_value
            # Determine initial position
            if fast_value > slow_value:
                self._current_position = 1
            elif fast_value < slow_value:
                self._current_position = -1
            else:
                self._current_position = 0
            return False, 0.0, None
            
        signal_strength = 0.0
        triggered = False
        signal_type_str = None
        
        # Check for crossover
        prev_diff = self._prev_fast_value - self._prev_slow_value
        curr_diff = fast_value - slow_value
        
        # Golden cross: fast MA crosses above slow MA
        if prev_diff <= 0 and curr_diff > 0:
            # Check minimum separation requirement
            if abs(curr_diff) >= self.min_separation:
                signal_strength = 1.0
                triggered = True
                signal_type_str = "BUY"
                self._current_position = 1
                self.logger.debug(f"MA Golden Cross: Fast={fast_value:.2f}, Slow={slow_value:.2f}, Diff={curr_diff:.2f}")
                
        # Death cross: fast MA crosses below slow MA
        elif prev_diff >= 0 and curr_diff < 0:
            # Check minimum separation requirement
            if abs(curr_diff) >= self.min_separation:
                signal_strength = -1.0
                triggered = True
                signal_type_str = "SELL"
                self._current_position = -1
                self.logger.debug(f"MA Death Cross: Fast={fast_value:.2f}, Slow={slow_value:.2f}, Diff={curr_diff:.2f}")
                
        # Update previous values
        self._prev_fast_value = fast_value
        self._prev_slow_value = slow_value
        
        return triggered, signal_strength, signal_type_str
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the rule."""
        return {
            'weight': self._weight,
            'min_separation': self.min_separation
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the rule's parameters."""
        old_weight = self._weight
        old_min_sep = self.min_separation
        
        self._weight = params.get('weight', self._weight)
        self.min_separation = params.get('min_separation', self.min_separation)
        
        self.reset_state()
        self.logger.info(
            f"MACrossoverRule '{self.instance_name}' parameters updated: weight={self._weight}, min_separation={self.min_separation}"
        )
        
    def reset_state(self):
        """Resets the internal state of the rule."""
        self._prev_fast_value = None
        self._prev_slow_value = None
        self._current_position = 0
        
    def reset(self) -> None:
        """Reset component state (required by ComponentBase)."""
        self.reset_state()
        
    @property
    def weight(self) -> float:
        """Expose the weight as a property for easy access."""
        return self._weight
        
    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for this component (ComponentBase interface)."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        space.add_parameter(Parameter(
            name="weight",
            param_type="discrete",
            values=[0.4, 0.6, 0.8, 1.0],
            default=self._weight
        ))
        
        space.add_parameter(Parameter(
            name="min_separation",
            param_type="discrete",
            values=[0.0, 0.001, 0.002],
            default=self.min_separation
        ))
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        return {
            'weight': [0.4, 0.6, 0.8, 1.0],
            'min_separation': [0.0, 0.001, 0.002]
        }