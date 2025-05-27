# src/strategy/components/rules/rsi_rules.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component_base import ComponentBase
from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.base.parameter import ParameterSpace, Parameter

class RSIRule(ComponentBase):
    """
    Generates trading signals based on RSI oversold/overbought levels.
    """
    def __init__(self, instance_name: str, config_key: Optional[str] = None,
                 rsi_indicator: Optional[RSIIndicator] = None, 
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_key)
        
        # Store dependencies - will be injected during initialize
        self.rsi_indicator = rsi_indicator
        self._parameters = parameters or {}
        
        # Rule state
        self.oversold_threshold: float = 30.0
        self.overbought_threshold: float = 70.0
        self._weight: float = 1.0
        self._last_rsi_value: Optional[float] = None
        self._current_signal_state: int = 0


    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self.oversold_threshold = self._parameters.get('oversold_threshold', 30.0)
            self.overbought_threshold = self._parameters.get('overbought_threshold', 70.0)
            self._weight = self._parameters.get('weight', 1.0)
        else:
            self.oversold_threshold = self.get_specific_config('oversold_threshold', 30.0)
            self.overbought_threshold = self.get_specific_config('overbought_threshold', 70.0)
            self._weight = self.get_specific_config('weight', 1.0)
            
        self.reset_state()
        
        # Validate dependencies if they were provided
        if self.rsi_indicator and not hasattr(self.rsi_indicator, 'value'):
            self.logger.error(f"RSIRule '{self.instance_name}' was not provided with a valid RSIIndicator instance.")
            return
            
        self.logger.info(f"RSIRule '{self.instance_name}' configured with OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}")
        
    def _start(self) -> None:
        """Component-specific start logic."""
        self.logger.info(f"RSIRule '{self.instance_name}' ready for evaluation")
        
    def _stop(self) -> None:
        """Component-specific stop logic."""
        self.reset_state()

    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Optional[str]]:
        if not self.rsi_indicator or not self.rsi_indicator.ready:
            return False, 0.0, None

        rsi_value = self.rsi_indicator.value
        if rsi_value is None:
            return False, 0.0, None

        # DEBUG: Log RSI values to understand the range and threshold interaction
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        # Debug logging removed for cleaner output

        signal_strength = 0.0
        triggered = False
        signal_type_str = None 

        # RSI sustained signal logic - maintains signal while RSI stays in extreme zones
        # BUY signal: RSI is oversold (≤ threshold)
        if rsi_value <= self.oversold_threshold:
            if self._current_signal_state != 1:
                signal_strength = 1.0
                triggered = True
                self._current_signal_state = 1
                signal_type_str = "BUY"
                # RSI sustained BUY signal activated
        
        # SELL signal: RSI is overbought (≥ threshold)  
        elif rsi_value >= self.overbought_threshold:
            if self._current_signal_state != -1:
                signal_strength = -1.0
                triggered = True
                self._current_signal_state = -1
                signal_type_str = "SELL"
                # RSI sustained SELL signal activated
        
        # NEUTRAL: RSI is in middle range - clear signal state
        else:
            if self._current_signal_state != 0:
                self._current_signal_state = 0
                self.logger.debug(f"RSI NEUTRAL: RSI={rsi_value:.2f} in normal range")
        
        self._last_rsi_value = rsi_value
        return triggered, signal_strength, signal_type_str

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold,
            'weight': self._weight
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        old_os = self.oversold_threshold
        old_ob = self.overbought_threshold
        old_weight = self._weight
        
        self.oversold_threshold = params.get('oversold_threshold', self.oversold_threshold)
        self.overbought_threshold = params.get('overbought_threshold', self.overbought_threshold)
        self._weight = params.get('weight', self._weight)
        
        
        self.reset_state()
        self.logger.info(
            f"RSIRule '{self.instance_name}' parameters updated: OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}"
        )
        return True
        
    def reset_state(self):
        self._last_rsi_value = None
        self._current_signal_state = 0
        # self.logger.debug(f"RSIRule '{self.instance_name}' state reset.")
        
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
            name="oversold_threshold",
            param_type="discrete",
            values=[20.0, 30.0],
            default=self.oversold_threshold
        ))
        
        space.add_parameter(Parameter(
            name="overbought_threshold",
            param_type="discrete",
            values=[60.0, 70.0],
            default=self.overbought_threshold
        ))
        
        space.add_parameter(Parameter(
            name="weight",
            param_type="discrete",
            values=[0.4, 0.6],
            default=self._weight
        ))
        
        # If we have a dependent RSI indicator, include its parameter space
        if self.rsi_indicator and hasattr(self.rsi_indicator, 'get_parameter_space'):
            rsi_space = self.rsi_indicator.get_parameter_space()
            # Add as subspace to maintain namespacing
            space.add_subspace('rsi_indicator', rsi_space)
        
        return space
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
         return {
             'oversold_threshold': [20.0, 30.0],
             'overbought_threshold': [60.0, 70.0],
            'weight': [0.4, 0.6]
        }
