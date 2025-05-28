# src/strategy/components/rules/macd_rule.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component_base import ComponentBase
from src.strategy.components.indicators.macd import MACDIndicator
from src.strategy.base.parameter import ParameterSpace, Parameter

class MACDRule(ComponentBase):
    """
    Generates trading signals based on MACD crossovers.
    
    Buy signals when MACD line crosses above signal line.
    Sell signals when MACD line crosses below signal line.
    Optional: Can also use histogram zero-line crossovers.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None,
                 macd_indicator: Optional[MACDIndicator] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_key)
        
        # Store dependencies - will be injected during initialize
        self.macd_indicator = macd_indicator
        self._parameters = parameters or {}
        
        # Rule state
        self._weight: float = 1.0
        self.use_histogram: bool = False  # If True, use histogram zero crossovers instead
        self.min_histogram_threshold: float = 0.0  # Minimum histogram value for signal confirmation
        self.sustain_signal: bool = True  # If True, sustain signal until opposite crossover
        self._prev_macd: Optional[float] = None
        self._prev_signal: Optional[float] = None
        self._prev_histogram: Optional[float] = None
        self._current_signal: int = 0  # -1: sell signal, 0: no signal, 1: buy signal
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self._weight = self._parameters.get('weight', 1.0)
            self.use_histogram = self._parameters.get('use_histogram', False)
            
            # Handle min_histogram_threshold - convert tuple to float if needed
            min_hist_value = self._parameters.get('min_histogram_threshold', 0.0)
            if isinstance(min_hist_value, tuple):
                self.min_histogram_threshold = float(min_hist_value[0])
            else:
                self.min_histogram_threshold = float(min_hist_value)
                
            self.sustain_signal = self._parameters.get('sustain_signal', True)
        else:
            self._weight = self.get_specific_config('weight', 1.0)
            self.use_histogram = self.get_specific_config('use_histogram', False)
            
            # Handle min_histogram_threshold - convert tuple to float if needed
            min_hist_value = self.get_specific_config('min_histogram_threshold', 0.0)
            if isinstance(min_hist_value, tuple):
                self.min_histogram_threshold = float(min_hist_value[0])
            else:
                self.min_histogram_threshold = float(min_hist_value)
                
            self.sustain_signal = self.get_specific_config('sustain_signal', True)
            
        self.reset_state()
        
        # Validate dependencies if they were provided
        if self.macd_indicator and not hasattr(self.macd_indicator, 'macd_line'):
            self.logger.error(f"MACDRule '{self.instance_name}' was not provided with a valid MACDIndicator.")
            return
            
        self.logger.info(f"MACDRule '{self.instance_name}' configured with "
                        f"use_histogram={self.use_histogram}, min_threshold={self.min_histogram_threshold}, weight={self._weight}")
        
    def _start(self) -> None:
        """Start the rule component."""
        pass
        
    def _stop(self) -> None:
        """Stop the rule component."""
        pass
        
    def reset_state(self) -> None:
        """Reset the rule state."""
        self._prev_macd = None
        self._prev_signal = None
        self._prev_histogram = None
        self._current_signal = 0
        
    def reset(self) -> None:
        """Reset component state (required by ComponentBase)."""
        self.reset_state()
        
    @property
    def ready(self) -> bool:
        """Check if the rule is ready to generate signals."""
        if not self.macd_indicator:
            return False
        return self.macd_indicator.ready if hasattr(self.macd_indicator, 'ready') else True
        
    @property
    def weight(self) -> float:
        """Return the weight of this rule."""
        return self._weight
    
    @weight.setter
    def weight(self, value: float) -> None:
        """Set the weight of this rule."""
        self._weight = value
        
    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Evaluates the MACD crossover condition.
        
        Returns:
            Tuple[float, float]: (signal, strength)
                - signal: 1.0 for buy, -1.0 for sell, 0.0 for no signal
                - strength: Confidence level of the signal (0.0 to 1.0)
        """
        if not self.macd_indicator or not self.ready:
            return 0.0, 0.0
            
        # Get current MACD values
        macd_line = self.macd_indicator.macd_line
        signal_line = self.macd_indicator.signal_line
        histogram = self.macd_indicator.histogram
        
        # Check if indicators are ready
        if macd_line is None or signal_line is None or histogram is None:
            return 0.0, 0.0
            
        # Initialize previous values on first call
        if self._prev_macd is None or self._prev_signal is None:
            self._prev_macd = macd_line
            self._prev_signal = signal_line
            self._prev_histogram = histogram
            return 0.0, 0.0
            
        signal = 0.0
        strength = 1.0
        
        if self.use_histogram:
            # Use histogram zero-line crossovers
            if self._prev_histogram <= 0 and histogram > 0 and abs(histogram) >= self.min_histogram_threshold:
                # Bullish crossover
                signal = 1.0
                self._current_signal = 1
                self.logger.debug(f"MACD Histogram Buy: Histogram={histogram:.4f}")
            elif self._prev_histogram >= 0 and histogram < 0 and abs(histogram) >= self.min_histogram_threshold:
                # Bearish crossover
                signal = -1.0
                self._current_signal = -1
                self.logger.debug(f"MACD Histogram Sell: Histogram={histogram:.4f}")
            elif self.sustain_signal and self._current_signal != 0:
                # Sustain current signal until opposite crossover
                signal = float(self._current_signal)
        else:
            # Use MACD/Signal line crossovers
            if self._prev_macd <= self._prev_signal and macd_line > signal_line:
                # Bullish crossover
                signal = 1.0
                self._current_signal = 1
                self.logger.debug(f"MACD Buy: MACD={macd_line:.4f} > Signal={signal_line:.4f}")
            elif self._prev_macd >= self._prev_signal and macd_line < signal_line:
                # Bearish crossover
                signal = -1.0
                self._current_signal = -1
                self.logger.debug(f"MACD Sell: MACD={macd_line:.4f} < Signal={signal_line:.4f}")
            elif self.sustain_signal and self._current_signal != 0:
                # Sustain current signal until opposite crossover
                signal = float(self._current_signal)
                
        # Update previous values
        self._prev_macd = macd_line
        self._prev_signal = signal_line
        self._prev_histogram = histogram
        
        return signal, strength
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the rule."""
        return {
            'weight': self._weight,
            'use_histogram': self.use_histogram,
            'min_histogram_threshold': self.min_histogram_threshold,
            'sustain_signal': self.sustain_signal
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the rule's parameters."""
        self._weight = params.get('weight', self._weight)
        self.use_histogram = params.get('use_histogram', self.use_histogram)
        
        # Handle min_histogram_threshold - convert tuple to float if needed
        min_hist_value = params.get('min_histogram_threshold', self.min_histogram_threshold)
        if isinstance(min_hist_value, tuple):
            # If it's a tuple (min, max), use the first value or average
            self.min_histogram_threshold = float(min_hist_value[0])
        else:
            self.min_histogram_threshold = float(min_hist_value)
            
        self.sustain_signal = params.get('sustain_signal', self.sustain_signal)
        
        # Apply indicator parameters if provided
        if self.macd_indicator:
            indicator_params = {}
            for key, value in params.items():
                if key.startswith('macd_indicator.'):
                    param_name = key.replace('macd_indicator.', '')
                    indicator_params[param_name] = value
            if indicator_params:
                self.macd_indicator.set_parameters(indicator_params)
                self.logger.debug(f"Updated MACD indicator parameters: {indicator_params}")
        
        self.reset_state()
        self.logger.info(
            f"MACDRule '{self.instance_name}' parameters updated: weight={self._weight}, "
            f"use_histogram={self.use_histogram}, min_histogram_threshold={self.min_histogram_threshold}, "
            f"sustain_signal={self.sustain_signal}"
        )
        
    def apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply parameters to this component (ComponentBase interface)."""
        self.set_parameters(parameters)
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for this component (ComponentBase interface)."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        space.add_parameter(Parameter(
            name="weight",
            param_type="discrete",
            values=[0.25, 0.5, 0.75, 1.0],
            default=self._weight
        ))
        
        space.add_parameter(Parameter(
            name="min_histogram_threshold",
            param_type="discrete",
            values=[0.0, 0.0001, 0.0002],
            default=self.min_histogram_threshold
        ))
        
        # Include MACD indicator parameter space if available
        if self.macd_indicator and hasattr(self.macd_indicator, 'get_parameter_space'):
            macd_space = self.macd_indicator.get_parameter_space()
            # Add as subspace to maintain namespacing
            space.add_subspace('macd_indicator', macd_space)
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        return {
            'weight': [0.25, 0.5, 0.75, 1.0],
            'min_histogram_threshold': [0.0, 0.0001, 0.0002]
        }
        
    def generate_signal(self, price: float) -> int:
        """
        Generate trading signal based on MACD.
        
        Args:
            price: Current price (not used directly, but required for interface)
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for hold
        """
        if not self.macd_indicator:
            return 0
            
        # Get current MACD values
        macd_line = self.macd_indicator.macd_line
        signal_line = self.macd_indicator.signal_line
        histogram = self.macd_indicator.histogram
        
        # Check if indicators are ready
        if macd_line is None or signal_line is None or histogram is None:
            return 0
            
        signal = 0
        
        if self.use_histogram:
            # Use histogram zero-line crossovers
            if self._prev_histogram is not None:
                # Buy signal: Histogram crosses above zero
                if self._prev_histogram <= 0 and histogram > 0 and abs(histogram) >= self.min_histogram_threshold:
                    signal = 1
                    self.logger.debug(f"MACD Buy signal: histogram crossed above zero ({histogram:.4f})")
                    
                # Sell signal: Histogram crosses below zero
                elif self._prev_histogram >= 0 and histogram < 0 and abs(histogram) >= self.min_histogram_threshold:
                    signal = -1
                    self.logger.debug(f"MACD Sell signal: histogram crossed below zero ({histogram:.4f})")
                    
        else:
            # Use MACD/Signal line crossovers
            if self._prev_macd is not None and self._prev_signal is not None:
                prev_diff = self._prev_macd - self._prev_signal
                curr_diff = macd_line - signal_line
                
                # Buy signal: MACD crosses above signal line
                if prev_diff <= 0 and curr_diff > 0 and abs(histogram) >= self.min_histogram_threshold:
                    signal = 1
                    self._current_position = 1
                    self.logger.debug(f"MACD Buy signal: MACD {macd_line:.4f} crossed above signal {signal_line:.4f}")
                    
                # Sell signal: MACD crosses below signal line
                elif prev_diff >= 0 and curr_diff < 0 and abs(histogram) >= self.min_histogram_threshold:
                    signal = -1
                    self._current_position = -1
                    self.logger.debug(f"MACD Sell signal: MACD {macd_line:.4f} crossed below signal {signal_line:.4f}")
                else:
                    # Update position state
                    self._current_position = 1 if curr_diff > 0 else -1
                    
        # Update previous values
        self._prev_macd = macd_line
        self._prev_signal = signal_line
        self._prev_histogram = histogram
        
        return signal
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the rule."""
        return {
            'weight': self._weight,
            'use_histogram': self.use_histogram,
            'min_histogram_threshold': self.min_histogram_threshold,
            'current_position': self._current_position,
            'prev_histogram': self._prev_histogram
        }
        
    def get_optimizable_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the parameters that can be optimized for this rule.
        
        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        # MACD indicator parameters are optimized separately
        return {
            'min_histogram_threshold': (0.0, 0.001),  # 0 to 0.1% threshold
            'use_histogram': (0, 1)  # Binary: 0 or 1
        }
        
