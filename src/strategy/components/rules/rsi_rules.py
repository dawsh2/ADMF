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
        self.sustain_signal: bool = True  # If True, continue signaling while RSI remains in extreme zones
        self._weight: float = 1.0
        self._last_rsi_value: Optional[float] = None
        self._current_signal_state: int = 0
        
        # TEMP: Log instance creation (can't use logger here - not initialized yet)
        import traceback
        print(f"TEMP RSI RULE CREATED: instance={instance_name}, id={id(self)}, params={parameters}")
        print(f"TEMP Stack: {''.join(traceback.format_stack()[-3:-1])}")


    def _initialize(self) -> None:
        """Component-specific initialization logic."""
        # Load configuration
        if self._parameters:
            self.oversold_threshold = self._parameters.get('oversold_threshold', 30.0)
            self.overbought_threshold = self._parameters.get('overbought_threshold', 70.0)
            self.sustain_signal = self._parameters.get('sustain_signal', True)
            self._weight = self._parameters.get('weight', 1.0)
        else:
            self.oversold_threshold = self.get_specific_config('oversold_threshold', 30.0)
            self.overbought_threshold = self.get_specific_config('overbought_threshold', 70.0)
            self.sustain_signal = self.get_specific_config('sustain_signal', True)
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

    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        if not self.rsi_indicator or not self.rsi_indicator.ready:
            return 0.0, 0.0

        # Validate threshold configuration
        if self.oversold_threshold >= self.overbought_threshold:
            return 0.0, 0.0

        rsi_value = self.rsi_indicator.value
        if rsi_value is None:
            return 0.0, 0.0

        # DEBUG: Log RSI values to understand the range and threshold interaction
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        # Log first 10 RSI values and whenever we hit thresholds
        if self._debug_count <= 10 or rsi_value <= self.oversold_threshold or rsi_value >= self.overbought_threshold:
            self.logger.info(f"TEMP RSI Debug #{self._debug_count}: RSI={rsi_value:.2f}, OS={self.oversold_threshold}, OB={self.overbought_threshold}")

        signal = 0.0
        strength = 1.0 

        # RSI signal logic with sustained signals
        # BUY signal: RSI enters or remains in oversold zone
        if rsi_value <= self.oversold_threshold:
            # Always trigger BUY while oversold (sustained signal)
            signal = 1.0
            if self._current_signal_state != 1:
                self._current_signal_state = 1
                self.logger.debug(f"RSI BUY signal activated: RSI={rsi_value:.2f} <= {self.oversold_threshold}")
        
        # SELL signal: RSI enters or remains in overbought zone
        elif rsi_value >= self.overbought_threshold:
            # Always trigger SELL while overbought (sustained signal)
            signal = -1.0
            if self._current_signal_state != -1:
                self._current_signal_state = -1
                self.logger.debug(f"RSI SELL signal activated: RSI={rsi_value:.2f} >= {self.overbought_threshold}")
        
        # NEUTRAL: RSI returns to middle range - no signal
        else:
            # No signal when in normal range
            signal = 0.0
            if self._current_signal_state != 0:
                self._current_signal_state = 0
                self.logger.debug(f"RSI signal cleared: RSI={rsi_value:.2f} in normal range")
        
        self._last_rsi_value = rsi_value
        return signal, strength

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
        
        # TEMP: Log parameter changes
        self.logger.warning(f"TEMP RSI PARAM UPDATE: OS {old_os}->{self.oversold_threshold}, OB {old_ob}->{self.overbought_threshold}, Weight {old_weight}->{self._weight}")
        
        # Validate threshold relationship
        if self.oversold_threshold >= self.overbought_threshold:
            self.logger.warning(
                f"Invalid RSI configuration: oversold_threshold ({self.oversold_threshold}) >= "
                f"overbought_threshold ({self.overbought_threshold}). Rule will not generate signals."
            )
        
        # Validate threshold ranges
        if not (0 < self.oversold_threshold < 50):
            self.logger.warning(f"Unusual oversold_threshold value: {self.oversold_threshold}. Expected 0 < value < 50")
        if not (50 < self.overbought_threshold < 100):
            self.logger.warning(f"Unusual overbought_threshold value: {self.overbought_threshold}. Expected 50 < value < 100")
        
        # Apply indicator parameters if provided
        if self.rsi_indicator:
            indicator_params = {}
            for key, value in params.items():
                if key.startswith('rsi_indicator.'):
                    param_name = key.replace('rsi_indicator.', '')
                    indicator_params[param_name] = value
            if indicator_params:
                self.rsi_indicator.set_parameters(indicator_params)
                self.logger.warning(f"TEMP Updated RSI indicator parameters: {indicator_params}")
        
        # TEMP: Verify actual thresholds after update
        self.logger.warning(f"TEMP FINAL RSI thresholds: OS={self.oversold_threshold}, OB={self.overbought_threshold}")
        
        self.reset_state()
        self.logger.info(
            f"RSIRule '{self.instance_name}' parameters updated: OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}"
        )
        
    def apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """Apply parameters to this component (ComponentBase interface)."""
        self.set_parameters(parameters)
        
    def reset_state(self):
        self._last_rsi_value = None
        self._current_signal_state = 0
        # self.logger.debug(f"RSIRule '{self.instance_name}' state reset.")
        
    def reset(self) -> None:
        """Reset component state (required by ComponentBase)."""
        self.reset_state()
    
    @property
    def ready(self) -> bool:
        """Check if the rule is ready to generate signals."""
        if not self.rsi_indicator:
            return False
        return self.rsi_indicator.ready
    
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
            values=[20.0, 25.0, 30.0, 35.0],  # Expanded range
            default=self.oversold_threshold
        ))
        
        space.add_parameter(Parameter(
            name="overbought_threshold",
            param_type="discrete",
            values=[60.0, 65.0, 70.0, 75.0, 80.0],  # Expanded range
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
             'oversold_threshold': [20.0, 25.0, 30.0, 35.0],
             'overbought_threshold': [60.0, 65.0, 70.0, 75.0, 80.0],
            'weight': [0.4, 0.6]
        }
