# src/strategy/components/rules/ma_crossover_rule.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component_base import ComponentBase
from src.strategy.base.indicator import MovingAverageIndicator
from src.strategy.base.parameter import ParameterSpace, Parameter

class MACrossoverRule(ComponentBase):
    """
    Generates trading signals based on moving average crossovers.
    
    This rule generates buy signals when a fast MA crosses above a slow MA,
    and sell signals when a fast MA crosses below a slow MA.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None,
                 fast_ma: Optional[MovingAverageIndicator] = None, 
                 slow_ma: Optional[MovingAverageIndicator] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_key)
        
        # Store dependencies - will be injected during initialize
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self._parameters = parameters or {}
        
        # Rule state
        self._weight: float = 1.0
        self.min_separation: float = 0.0
        self.sustain_signal: bool = True  # If True, continue signaling while MAs remain crossed
        self._prev_fast_value: Optional[float] = None
        self._prev_slow_value: Optional[float] = None
        self._current_position: int = 0  # -1: fast below slow, 0: unknown, 1: fast above slow
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self._weight = self._parameters.get('weight', 1.0)
            self.min_separation = self._parameters.get('min_separation', 0.0)
            self.sustain_signal = self._parameters.get('sustain_signal', True)
        else:
            self._weight = self.get_specific_config('weight', 1.0)
            self.min_separation = self.get_specific_config('min_separation', 0.0)
            self.sustain_signal = self.get_specific_config('sustain_signal', True)
            
        self.reset_state()
        
        # Validate dependencies if they were provided
        if self.fast_ma and not hasattr(self.fast_ma, 'value'):
            self.logger.error(f"MACrossoverRule '{self.instance_name}' was not provided with a valid fast MA indicator.")
            return
            
        if self.slow_ma and not hasattr(self.slow_ma, 'value'):
            self.logger.error(f"MACrossoverRule '{self.instance_name}' was not provided with a valid slow MA indicator.")
            return
            
        self.logger.info(f"MACrossoverRule '{self.instance_name}' configured with weight={self._weight}, min_separation={self.min_separation}, sustain_signal={self.sustain_signal}")
        
    def _start(self) -> None:
        """Component-specific start logic."""
        self.logger.info(f"MACrossoverRule '{self.instance_name}' ready for evaluation")
        
    def _stop(self) -> None:
        """Component-specific stop logic."""
        self.reset_state()
        
    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Evaluates the MA crossover condition.
        
        Returns:
            Tuple[float, float]: (signal, strength)
                - signal: 1.0 for buy, -1.0 for sell, 0.0 for no signal
                - strength: Confidence level of the signal (0.0 to 1.0)
        """
        # Count evaluations for debugging
        if not hasattr(self, '_eval_count'):
            self._eval_count = 0
        self._eval_count += 1
        
        if not self.fast_ma or not self.fast_ma.ready or not self.slow_ma or not self.slow_ma.ready:
            if self._eval_count <= 5:  # Log first few attempts
                fast_ready = self.fast_ma.ready if self.fast_ma else "No fast_ma"
                slow_ready = self.slow_ma.ready if self.slow_ma else "No slow_ma" 
                self.logger.debug(f"MA Rule eval #{self._eval_count}: Not ready - Fast: {fast_ready}, Slow: {slow_ready}")
            return 0.0, 0.0
            
        # Validate that fast MA period is less than slow MA period
        if self.fast_ma.lookback_period >= self.slow_ma.lookback_period:
            if self._eval_count <= 3:
                self.logger.debug(f"MA Rule eval #{self._eval_count}: Invalid periods - Fast: {self.fast_ma.lookback_period}, Slow: {self.slow_ma.lookback_period}")
            return 0.0, 0.0
            
        fast_value = self.fast_ma.value
        slow_value = self.slow_ma.value
        
        if fast_value is None or slow_value is None:
            if self._eval_count <= 5:
                self.logger.debug(f"MA Rule eval #{self._eval_count}: Null values - Fast: {fast_value}, Slow: {slow_value}")
            return 0.0, 0.0
            
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
            self.logger.debug(f"MA Rule eval #{self._eval_count}: INITIALIZED - Fast: {fast_value:.4f}, Slow: {slow_value:.4f}, Position: {self._current_position}")
            return 0.0, 0.0
            
        signal = 0.0
        strength = 1.0
        
        # Check for crossover
        prev_diff = self._prev_fast_value - self._prev_slow_value
        curr_diff = fast_value - slow_value
        
        # Check if there's a sustained separation (for continuous signal)
        sustained_signal = abs(curr_diff) >= self.min_separation
        
        # Golden cross: fast MA crosses above slow MA
        if prev_diff <= 0 and curr_diff > 0:
            # Check minimum separation requirement
            if sustained_signal:
                signal = 1.0
                self._current_position = 1
                self.logger.debug(f"MA Golden Cross: Fast={fast_value:.2f}, Slow={slow_value:.2f}, Diff={curr_diff:.2f}")
                
        # Death cross: fast MA crosses below slow MA
        elif prev_diff >= 0 and curr_diff < 0:
            # Check minimum separation requirement
            if sustained_signal:
                signal = -1.0
                self._current_position = -1
                self.logger.debug(f"MA Death Cross: Fast={fast_value:.2f}, Slow={slow_value:.2f}, Diff={curr_diff:.2f}")
                
        # SUSTAINED SIGNAL: Return current position if no crossover but still separated
        elif signal == 0.0 and sustained_signal and self.sustain_signal:
            if curr_diff > 0 and self._current_position == 1:
                # Fast still above slow - sustain BUY signal
                signal = 1.0
            elif curr_diff < 0 and self._current_position == -1:
                # Fast still below slow - sustain SELL signal
                signal = -1.0
                
        # Clear signal if MAs converge below minimum separation
        elif not sustained_signal and self._current_position != 0:
            self._current_position = 0
            # Optionally trigger an exit signal
            # triggered = True
            # signal_strength = 0.0
            # signal_type_str = "EXIT"
                
        # Log evaluation summary every 100 calls
        if self._eval_count % 100 == 0:
            self.logger.debug(f"MA Rule eval #{self._eval_count}: Fast: {fast_value:.4f}, Slow: {slow_value:.4f}, Diff: {curr_diff:.4f}, Signal: {signal}, Min_sep: {self.min_separation}")
        
        # Update previous values
        self._prev_fast_value = fast_value
        self._prev_slow_value = slow_value
        
        return signal, strength
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the rule."""
        return {
            'weight': self._weight,
            'min_separation': self.min_separation,
            'sustain_signal': self.sustain_signal
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the rule's parameters."""
        old_weight = self._weight
        old_min_sep = self.min_separation
        old_sustain = self.sustain_signal
        
        self._weight = params.get('weight', self._weight)
        self.min_separation = params.get('min_separation', self.min_separation)
        self.sustain_signal = params.get('sustain_signal', self.sustain_signal)
        
        # Apply indicator parameters if provided
        fast_period = None
        slow_period = None
        
        if 'fast_ma.lookback_period' in params and self.fast_ma:
            fast_period = params['fast_ma.lookback_period']
            old_fast_period = getattr(self.fast_ma, '_lookback_period', 'N/A')
            self.fast_ma.set_parameters({'lookback_period': fast_period})
            new_fast_period = getattr(self.fast_ma, '_lookback_period', 'N/A')
            self.logger.debug(f"MA Crossover: Updated fast_ma lookback_period from {old_fast_period} to {new_fast_period} (requested: {fast_period})")
            
        if 'slow_ma.lookback_period' in params and self.slow_ma:
            slow_period = params['slow_ma.lookback_period']
            old_slow_period = getattr(self.slow_ma, '_lookback_period', 'N/A')
            self.slow_ma.set_parameters({'lookback_period': slow_period})
            new_slow_period = getattr(self.slow_ma, '_lookback_period', 'N/A')
            self.logger.debug(f"MA Crossover: Updated slow_ma lookback_period from {old_slow_period} to {new_slow_period} (requested: {slow_period})")
        
        # Validate MA periods if both are set
        if fast_period is not None and slow_period is not None:
            if fast_period >= slow_period:
                self.logger.warning(f"Invalid MA configuration: fast_period ({fast_period}) >= slow_period ({slow_period}). Rule will not generate signals.")
        elif self.fast_ma and self.slow_ma:
            # Check existing periods if not being updated
            if self.fast_ma.lookback_period >= self.slow_ma.lookback_period:
                self.logger.warning(f"Invalid MA configuration: fast_period ({self.fast_ma.lookback_period}) >= slow_period ({self.slow_ma.lookback_period}). Rule will not generate signals.")
        
        self.reset_state()
        self.logger.info(
            f"MACrossoverRule '{self.instance_name}' parameters updated: weight={self._weight}, min_separation={self.min_separation}, sustain_signal={self.sustain_signal}"
        )
        
    def apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply parameters to this component (ComponentBase interface)."""
        self.set_parameters(parameters)
    
    def reset_state(self):
        """Resets the internal state of the rule."""
        self._prev_fast_value = None
        self._prev_slow_value = None
        self._current_position = 0
        
    def reset(self) -> None:
        """Reset component state (required by ComponentBase)."""
        self.reset_state()
        
    @property
    def ready(self) -> bool:
        """Check if the rule is ready to generate signals."""
        if not self.fast_ma or not self.slow_ma:
            return False
        return self.fast_ma.ready and self.slow_ma.ready
    
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
        
        # Include MA indicator parameter spaces if available
        if self.fast_ma and hasattr(self.fast_ma, 'get_parameter_space'):
            fast_ma_space = ParameterSpace(f"{self.instance_name}_fast_ma_space")
            fast_ma_space.add_parameter(Parameter(
                name="lookback_period",
                param_type="discrete",
                values=[5, 10, 15],  # Fast MA periods - no overlap with slow
                default=self.fast_ma.lookback_period
            ))
            space.add_subspace('fast_ma', fast_ma_space)
            
        if self.slow_ma and hasattr(self.slow_ma, 'get_parameter_space'):
            slow_ma_space = ParameterSpace(f"{self.instance_name}_slow_ma_space")
            slow_ma_space.add_parameter(Parameter(
                name="lookback_period",
                param_type="discrete",
                values=[20, 30, 40, 50],  # Slow MA periods - no overlap with fast
                default=self.slow_ma.lookback_period
            ))
            space.add_subspace('slow_ma', slow_ma_space)
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        params = {
            'weight': [0.4, 0.6, 0.8, 1.0],
            'min_separation': [0.0, 0.001, 0.002]
        }
        
        # Include MA periods in legacy format
        if self.fast_ma:
            params['fast_ma.lookback_period'] = [5, 10, 15, 20]
        if self.slow_ma:
            params['slow_ma.lookback_period'] = [20, 30, 40, 50]
            
        return params