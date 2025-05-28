# src/strategy/components/rules/bollinger_bands_rule.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component_base import ComponentBase
from src.strategy.components.indicators.bollinger_bands import BollingerBandsIndicator
from src.strategy.base.parameter import ParameterSpace, Parameter

class BollingerBandsRule(ComponentBase):
    """
    Generates trading signals based on Bollinger Bands.
    
    Buy signals when price touches/crosses below the lower band (oversold).
    Sell signals when price touches/crosses above the upper band (overbought).
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None,
                 bb_indicator: Optional[BollingerBandsIndicator] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_key)
        
        # Store dependencies - will be injected during initialize
        self.bb_indicator = bb_indicator
        self._parameters = parameters or {}
        
        # Rule state
        self._weight: float = 1.0
        self.band_width_filter: float = 0.0  # Minimum band width to generate signals
        self.sustain_signal: bool = True  # If True, sustain signal until opposite band touched or price returns to middle
        self.revert_at_middle: bool = False  # If True, clear signal when price crosses middle band
        self._last_price: Optional[float] = None
        self._current_signal_state: int = 0  # -1: sell signal, 0: no signal, 1: buy signal
        
    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self._weight = self._parameters.get('weight', 1.0)
            
            # Handle band_width_filter - convert tuple to float if needed
            bwf_value = self._parameters.get('band_width_filter', 0.0)
            if isinstance(bwf_value, tuple):
                self.band_width_filter = float(bwf_value[0])
            else:
                self.band_width_filter = float(bwf_value)
                
            self.sustain_signal = self._parameters.get('sustain_signal', True)
            self.revert_at_middle = self._parameters.get('revert_at_middle', False)
        else:
            self._weight = self.get_specific_config('weight', 1.0)
            
            # Handle band_width_filter - convert tuple to float if needed
            bwf_value = self.get_specific_config('band_width_filter', 0.0)
            if isinstance(bwf_value, tuple):
                self.band_width_filter = float(bwf_value[0])
            else:
                self.band_width_filter = float(bwf_value)
                
            self.sustain_signal = self.get_specific_config('sustain_signal', True)
            self.revert_at_middle = self.get_specific_config('revert_at_middle', False)
            
        self.reset_state()
        
        # Validate dependencies if they were provided
        if self.bb_indicator and not hasattr(self.bb_indicator, 'upper_band'):
            self.logger.error(f"BollingerBandsRule '{self.instance_name}' was not provided with a valid BollingerBandsIndicator.")
            return
            
        self.logger.info(f"BollingerBandsRule '{self.instance_name}' configured with band_width_filter={self.band_width_filter}, weight={self._weight}")
        
    def _start(self) -> None:
        """Start the rule component."""
        pass
        
    def _stop(self) -> None:
        """Stop the rule component."""
        pass
        
    def reset_state(self) -> None:
        """Reset the rule state."""
        self._last_price = None
        self._current_signal_state = 0
        
    def reset(self) -> None:
        """Reset component state (required by ComponentBase)."""
        self.reset_state()
        
    @property
    def ready(self) -> bool:
        """Check if the rule is ready to generate signals."""
        if not self.bb_indicator:
            return False
        return self.bb_indicator.ready if hasattr(self.bb_indicator, 'ready') else True
        
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
        Evaluates the Bollinger Bands condition.
        
        Returns:
            Tuple[float, float]: (signal, strength)
                - signal: 1.0 for buy, -1.0 for sell, 0.0 for no signal
                - strength: Confidence level of the signal (0.0 to 1.0)
        """
        if not self.bb_indicator or not self.ready:
            return 0.0, 0.0
            
        # Get current price from bar data
        if bar_data and 'close' in bar_data:
            price = bar_data['close']
        else:
            return 0.0, 0.0
            
        # Get current band values
        upper_band = self.bb_indicator.upper_band
        lower_band = self.bb_indicator.lower_band
        middle_band = self.bb_indicator.middle_band
        
        # Check if bands are valid
        if upper_band is None or lower_band is None or middle_band is None:
            return 0.0, 0.0
            
        # Calculate band width
        band_width = upper_band - lower_band
        if band_width < self.band_width_filter:
            return 0.0, 0.0  # Bands too narrow, no signal
            
        signal = 0.0
        strength = 1.0
        
        # Check for new signals or reversals
        if price <= lower_band:
            # Buy signal: Price touches or goes below lower band
            if self._current_signal_state != 1:
                signal = 1.0
                self._current_signal_state = 1
                self.logger.debug(f"BB Buy signal: Price={price:.2f} <= Lower={lower_band:.2f}")
            elif self.sustain_signal:
                signal = 1.0  # Sustain buy signal
                
        elif price >= upper_band:
            # Sell signal: Price touches or goes above upper band  
            if self._current_signal_state != -1:
                signal = -1.0
                self._current_signal_state = -1
                self.logger.debug(f"BB Sell signal: Price={price:.2f} >= Upper={upper_band:.2f}")
            elif self.sustain_signal:
                signal = -1.0  # Sustain sell signal
                
        else:
            # Price within bands - check if we should sustain or clear signal
            if self.sustain_signal and self._current_signal_state != 0:
                # Check if we should clear at middle band
                if self.revert_at_middle:
                    if self._current_signal_state == 1 and price >= middle_band:
                        # Clear buy signal when price crosses above middle
                        self._current_signal_state = 0
                        self.logger.debug(f"BB buy signal cleared: Price={price:.2f} crossed above Middle={middle_band:.2f}")
                    elif self._current_signal_state == -1 and price <= middle_band:
                        # Clear sell signal when price crosses below middle
                        self._current_signal_state = 0
                        self.logger.debug(f"BB sell signal cleared: Price={price:.2f} crossed below Middle={middle_band:.2f}")
                    else:
                        # Sustain current signal
                        signal = float(self._current_signal_state)
                else:
                    # Sustain signal until opposite band touched
                    signal = float(self._current_signal_state)
            else:
                # No sustained signals - clear immediately
                if self._current_signal_state != 0:
                    self._current_signal_state = 0
                    self.logger.debug(f"BB signal cleared: Price={price:.2f} within bands")
                
        self._last_price = price
        return signal, strength
        
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the current parameters of the rule."""
        return {
            'weight': self._weight,
            'band_width_filter': self.band_width_filter,
            'sustain_signal': self.sustain_signal,
            'revert_at_middle': self.revert_at_middle
        }
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Updates the rule's parameters."""
        self._weight = params.get('weight', self._weight)
        
        # Handle band_width_filter - convert tuple to float if needed
        bwf_value = params.get('band_width_filter', self.band_width_filter)
        if isinstance(bwf_value, tuple):
            # If it's a tuple (min, max), use the first value
            self.band_width_filter = float(bwf_value[0])
        else:
            self.band_width_filter = float(bwf_value)
        
        # Validate band_width_filter is non-negative
        if self.band_width_filter < 0:
            self.logger.warning(f"Invalid band_width_filter value: {self.band_width_filter}. Must be >= 0. Setting to 0.")
            self.band_width_filter = 0.0
        
        self.sustain_signal = params.get('sustain_signal', self.sustain_signal)
        self.revert_at_middle = params.get('revert_at_middle', self.revert_at_middle)
        
        # Apply indicator parameters if provided
        if self.bb_indicator:
            indicator_params = {}
            for key, value in params.items():
                if key.startswith('bb_indicator.'):
                    param_name = key.replace('bb_indicator.', '')
                    indicator_params[param_name] = value
            if indicator_params:
                self.bb_indicator.set_parameters(indicator_params)
                self.logger.debug(f"Updated BB indicator parameters: {indicator_params}")
        
        self.reset_state()
        self.logger.info(
            f"BollingerBandsRule '{self.instance_name}' parameters updated: weight={self._weight}, "
            f"band_width_filter={self.band_width_filter}, sustain_signal={self.sustain_signal}, "
            f"revert_at_middle={self.revert_at_middle}"
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
            values=[0.25, 0.5, 1.0],  # Reduced from 4 to 3 values
            default=self._weight
        ))
        
        space.add_parameter(Parameter(
            name="band_width_filter",
            param_type="discrete",
            values=[0.0, 0.01],  # Reduced from 3 to 2 values
            default=self.band_width_filter
        ))
        
        # Include BB indicator parameter space if available
        if self.bb_indicator and hasattr(self.bb_indicator, 'get_parameter_space'):
            bb_space = self.bb_indicator.get_parameter_space()
            # Add as subspace to maintain namespacing
            space.add_subspace('bb_indicator', bb_space)
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Returns the parameter space for optimization (legacy interface)."""
        return {
            'weight': [0.25, 0.5, 1.0],  # Matches the reduced parameter space
            'band_width_filter': [0.0, 0.01]  # Matches the reduced parameter space
        }
        
    def generate_signal(self, price: float) -> int:
        """
        Generate trading signal based on Bollinger Bands.
        
        Args:
            price: Current price
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for hold
        """
        if not self.bb_indicator:
            return 0
            
        # Get current band values
        upper_band = self.bb_indicator.upper_band
        lower_band = self.bb_indicator.lower_band
        middle_band = self.bb_indicator.middle_band
        
        # Check if bands are valid
        if upper_band is None or lower_band is None or middle_band is None:
            return 0
            
        # Calculate band width
        band_width = upper_band - lower_band
        if band_width < self.band_width_filter:
            return 0  # Bands too narrow, no signal
            
        # Determine current position relative to bands
        prev_state = self._current_signal_state
        
        if price <= lower_band:
            self._current_signal_state = -1  # Below lower band
        elif price >= upper_band:
            self._current_signal_state = 1   # Above upper band
        else:
            self._current_signal_state = 0   # Within bands
            
        # Generate signals on band touches/crosses
        signal = 0
        
        # Buy signal: Price touches or goes below lower band
        if self._current_signal_state == -1 and prev_state >= 0:
            signal = 1  # Buy signal
            self.logger.debug(f"BB Buy signal: price {price:.2f} <= lower band {lower_band:.2f}")
            
        # Sell signal: Price touches or goes above upper band
        elif self._current_signal_state == 1 and prev_state <= 0:
            signal = -1  # Sell signal
            self.logger.debug(f"BB Sell signal: price {price:.2f} >= upper band {upper_band:.2f}")
            
        self._last_price = price
        return signal
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the rule."""
        return {
            'weight': self._weight,
            'band_width_filter': self.band_width_filter,
            'current_signal_state': self._current_signal_state,
            'last_price': self._last_price
        }
        
    def get_optimizable_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Return the parameters that can be optimized for this rule.
        
        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        # The BB indicator parameters (period, std_dev) are optimized separately
        # Here we only optimize rule-specific parameters
        return {
            'band_width_filter': (0.0, 0.02),  # 0 to 2% of price
        }
        
