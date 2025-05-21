# src/strategy/components/rules/rsi_rules.py
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.core.component import BaseComponent
from src.strategy.components.indicators.oscillators import RSIIndicator 

class RSIRule(BaseComponent):
    """
    Generates trading signals based on RSI oversold/overbought levels.
    """
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str,
                 rsi_indicator: RSIIndicator, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(instance_name, config_loader, component_config_key)
        self.logger = logging.getLogger(f"{__name__}.{instance_name}") # Corrected logger name
        self.rsi_indicator = rsi_indicator

        if parameters:
            self.oversold_threshold: float = parameters.get('oversold_threshold', 30.0)
            self.overbought_threshold: float = parameters.get('overbought_threshold', 70.0)
            self._weight: float = parameters.get('weight', 1.0)
        else:
            self.oversold_threshold: float = self.get_specific_config('oversold_threshold', 30.0)
            self.overbought_threshold: float = self.get_specific_config('overbought_threshold', 70.0)
            self._weight: float = self.get_specific_config('weight', 1.0)
            
        self._last_rsi_value: Optional[float] = None
        self._current_signal_state: int = 0
        # self.logger.info(f"RSIRule '{self.name}' initialized with OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}")


    def setup(self) -> None:
        """Sets up the component."""
        self.reset_state()
        if not isinstance(self.rsi_indicator, RSIIndicator):
            self.logger.error(f"RSIRule '{self.name}' was not provided with a valid RSIIndicator instance.")
            self.state = BaseComponent.STATE_FAILED
            return
        self.logger.info(f"RSIRule '{self.name}' configured with OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}")
        self.state = BaseComponent.STATE_INITIALIZED
        self.logger.info(f"RSIRule '{self.name}' setup complete. State: {self.state}")

    def start(self) -> None:
        """Starts the component's active operations."""
        if self.state != BaseComponent.STATE_INITIALIZED:
            self.logger.warning(f"Cannot start RSIRule '{self.name}' from state '{self.state}'. Expected INITIALIZED.")
            return
        self.state = BaseComponent.STATE_STARTED
        self.logger.info(f"RSIRule '{self.name}' started. State: {self.state}")

    def stop(self) -> None:
        """Stops the component's active operations and cleans up resources."""
        self.reset_state()
        self.state = BaseComponent.STATE_STOPPED
        self.logger.info(f"RSIRule '{self.name}' stopped. State: {self.state}")

    def evaluate(self, bar_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, Optional[str]]:
        if not self.rsi_indicator or not self.rsi_indicator.ready:
            return False, 0.0, None

        rsi_value = self.rsi_indicator.value
        if rsi_value is None:
            return False, 0.0, None

        signal_strength = 0.0
        triggered = False
        signal_type_str = None 

        # RSI crossing logic
        if self._last_rsi_value is not None:
            # Check oversold crossing (BUY signal)
            oversold_cross = self._last_rsi_value <= self.oversold_threshold and rsi_value > self.oversold_threshold
            # Check overbought crossing (SELL signal)  
            overbought_cross = self._last_rsi_value >= self.overbought_threshold and rsi_value < self.overbought_threshold
            
            if oversold_cross:
                if self._current_signal_state != 1:
                    signal_strength = 1.0 
                    triggered = True
                    self._current_signal_state = 1
                    signal_type_str = "BUY"
            elif overbought_cross:
                if self._current_signal_state != -1:
                    signal_strength = -1.0
                    triggered = True
                    self._current_signal_state = -1
                    signal_type_str = "SELL"
        
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
            f"RSIRule '{self.name}' parameters updated: OS={self.oversold_threshold}, OB={self.overbought_threshold}, W={self._weight}"
        )
        return True
        
    def reset_state(self):
        self._last_rsi_value = None
        self._current_signal_state = 0
        # self.logger.debug(f"RSIRule '{self.name}' state reset.")
    
    @property
    def weight(self) -> float:
        """Expose the weight as a property for easy access."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
         return {
             'oversold_threshold': [20.0, 30.0],
             'overbought_threshold': [60.0, 70.0],
            'weight': [0.4, 0.6]
        }
