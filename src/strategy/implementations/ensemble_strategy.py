# src/strategy/implementations/ensemble_strategy.py
import logging
from typing import Dict, Any, List, Optional
from src.core.component import BaseComponent # <--- ADD THIS LINE

# from src.strategy.base.strategy import Strategy # As per STRATEGY_IMPLEMENTATION.MD
# from src.core.component import BaseComponent # Using BaseComponent if Strategy base class is not ready
from src.strategy.ma_strategy import MAStrategy # For now, let's keep MAStrategy as is and add RSI as a second signal source
                                              # A true ensemble would refactor MAStrategy into Rule/Indicator components
from src.strategy.components.indicators.oscillators import RSIIndicator
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.event import Event, EventType

# This is a simplified example. A full implementation would follow STRATEGY_IMPLEMENTATION.MD
# for the main "Strategy" class that uses add_component.
# For now, let's adapt the existing MAStrategy to also use an RSI rule.
# THIS IS A HYBRID APPROACH FOR QUICKER INTEGRATION. IDEALLY, MAStrategy is also refactored.

class EnsembleSignalStrategy(MAStrategy): # Inheriting MAStrategy for quick demo, not ideal for pure ensemble
    """
    An example strategy that combines MA Crossover signals with RSI signals.
    NOTE: This is a simplified ensemble. Ideally, MACrossover logic would also be a separate Rule.
    """
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str):
        super().__init__(instance_name, config_loader, event_bus, component_config_key) # Initialize MAStrategy part
        self.logger = logging.getLogger(f"{__name__}.{instance_name}")

        # RSI Components - names should be unique if registered in a flat container later
        # Or, the strategy manages them internally if they are not top-level components.
        rsi_indicator_params = {
            'period': self.get_specific_config('rsi_indicator.period', 14)
        }
        self.rsi_indicator = RSIIndicator(
            instance_name=f"{self.name}_RSIIndicator",
            config_loader=config_loader, # Pass through for consistency if BaseComponent needs it
            event_bus=event_bus,         # Pass through
            component_config_key=f"{component_config_key}.rsi_indicator", # For specific config loading
            parameters=rsi_indicator_params
        )

        rsi_rule_params = {
            'oversold_threshold': self.get_specific_config('rsi_rule.oversold_threshold', 30.0),
            'overbought_threshold': self.get_specific_config('rsi_rule.overbought_threshold', 70.0),
            'weight': self.get_specific_config('rsi_rule.weight', 0.5) # Example weight
        }
        self.rsi_rule = RSIRule(
            instance_name=f"{self.name}_RSIRule",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key=f"{component_config_key}.rsi_rule",
            rsi_indicator=self.rsi_indicator,
            parameters=rsi_rule_params
        )
        
        self.ma_weight = self.get_specific_config('ma_rule.weight', 0.5)
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' initialized with MA and RSI components.")

    def setup(self):
        super().setup() # Setup MAStrategy part
        self.rsi_indicator.setup()
        self.rsi_rule.setup()
        # Event subscriptions are handled by MAStrategy and _on_bar will be overridden
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' setup complete.")
        self.state = BaseComponent.STATE_INITIALIZED


    def _on_bar_event(self, event: Event):
        if event.event_type != EventType.BAR or event.payload.get("symbol") != self._symbol:
            return
        
        bar_data: Dict[str, Any] = event.payload
        close_price_val = bar_data.get("close")
        bar_timestamp: Optional[datetime.datetime] = bar_data.get("timestamp")

        if close_price_val is None or bar_timestamp is None:
            return
        close_price = float(close_price_val)

        # 1. Update MA part (from parent MAStrategy) and get its potential signal
        # Parent's _on_bar_event will publish its own signal if conditions are met.
        # We need to capture that or re-evaluate MA logic here to combine.
        # For this quick example, let's re-evaluate MA logic to get its signal state.
        
        self._prices.append(close_price) # MA prices
        current_short_ma, current_long_ma = None, None
        if len(self._prices) >= self._short_window:
            current_short_ma = sum(list(self._prices)[-self._short_window:]) / self._short_window
        if len(self._prices) >= self._long_window:
            current_long_ma = sum(self._prices) / len(self._prices)

        ma_signal_type_int = 0 # Default no signal
        if current_short_ma is not None and current_long_ma is not None and \
           self._prev_short_ma is not None and self._prev_long_ma is not None:
            if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
                if self._current_signal_state != 1: ma_signal_type_int = 1
            elif self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
                if self._current_signal_state != -1: ma_signal_type_int = -1
        
        # Update MA prev values
        if current_short_ma is not None: self._prev_short_ma = current_short_ma
        if current_long_ma is not None: self._prev_long_ma = current_long_ma

        # 2. Update RSI Indicator and Evaluate RSI Rule
        self.rsi_indicator.update(close_price)
        rsi_triggered, rsi_strength, rsi_signal_type_str = self.rsi_rule.evaluate(bar_data)
        
        rsi_signal_type_int = 0
        if rsi_triggered:
            rsi_signal_type_int = int(rsi_strength) # 1 for BUY, -1 for SELL

        # 3. Combine Signals (Simple Weighted Logic Example)
        # This is a placeholder for more sophisticated signal combination
        final_signal_type_int: Optional[int] = None
        
        # Example: If both agree, take signal. If they conflict, could be neutral or prioritize one.
        # Or weighted sum:
        combined_strength = (ma_signal_type_int * self.ma_weight) + \
                              (rsi_signal_type_int * self.rsi_rule.weight)
        
        # self.logger.debug(f"MA Signal: {ma_signal_type_int}, RSI Signal: {rsi_signal_type_int}, Combined: {combined_strength:.2f}")

        current_signal_to_publish = 0
        if combined_strength > 0.5: # Threshold for combined signal (e.g. > 0.5 means BUY)
            current_signal_to_publish = 1
        elif combined_strength < -0.5: # (e.g. < -0.5 means SELL)
            current_signal_to_publish = -1

        if current_signal_to_publish != 0 and self._current_signal_state != current_signal_to_publish:
            final_signal_type_int = current_signal_to_publish
            self._current_signal_state = final_signal_type_int # Update strategy's overall state
            
            signal_payload: Dict[str, Any] = {
                "symbol": self._symbol,
                "timestamp": bar_timestamp,
                "signal_type": final_signal_type_int,
                "price_at_signal": close_price,
                "strategy_id": self.name,
                "reason": f"Ensemble (MA: {ma_signal_type_int}, RSI: {rsi_signal_type_int})"
            }
            signal_event = Event(EventType.SIGNAL, signal_payload)
            self._event_bus.publish(signal_event)
            self.logger.info(
                f"Ensemble Published SIGNAL: Type={final_signal_type_int}, Symbol={self._symbol}, Price={close_price:.2f}"
            )
        elif current_signal_to_publish == 0 and self._current_signal_state != 0:
            # Optional: Generate a FLAT signal if combined strength is neutral and previously in a state
            # self._current_signal_state = 0
            # Or just do nothing, letting existing position ride until next strong signal
            pass


    def set_parameters(self, params: Dict[str, Any]):
        super().set_parameters(params) # For MAStrategy part
        
        # Parameters for RSI components might be prefixed, e.g., "rsi_indicator.period"
        rsi_indicator_params = {k.split('.', 1)[1]: v for k, v in params.items() if k.startswith("rsi_indicator.")}
        if rsi_indicator_params:
            self.rsi_indicator.set_parameters(rsi_indicator_params)

        rsi_rule_params = {k.split('.', 1)[1]: v for k, v in params.items() if k.startswith("rsi_rule.")}
        if rsi_rule_params:
            self.rsi_rule.set_parameters(rsi_rule_params)
            
        self.ma_weight = params.get('ma_rule.weight', self.ma_weight)
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' parameters updated.")

# src/strategy/implementations/ensemble_strategy.py
# Ensure this method is updated in your EnsembleSignalStrategy class:

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        space = super().get_parameter_space() # MAStrategy parameters (this is a method call, which is correct for MAStrategy)
        
        # Access parameter_space as a property (no parentheses)
        if hasattr(self.rsi_indicator, 'parameter_space'):
            # Ensure rsi_indicator is initialized and has the property
            if self.rsi_indicator: # Add a check if it could be None
                rsi_ind_space = self.rsi_indicator.parameter_space # Corrected: access as property
                for key, value in rsi_ind_space.items():
                    space[f"rsi_indicator.{key}"] = value
            else:
                self.logger.error("RSIIndicator object is not initialized in EnsembleSignalStrategy.")
        else:
            self.logger.warning(f"RSIIndicator instance does not have 'parameter_space' property.") # Or log self.rsi_indicator.name if it's initialized
            
        if hasattr(self.rsi_rule, 'parameter_space'):
            # Ensure rsi_rule is initialized and has the property
            if self.rsi_rule: # Add a check
                rsi_rule_space = self.rsi_rule.parameter_space # Corrected: access as property
                for key, value in rsi_rule_space.items():
                    space[f"rsi_rule.{key}"] = value
            else:
                self.logger.error("RSIRule object is not initialized in EnsembleSignalStrategy.")
        else:
            self.logger.warning(f"RSIRule instance does not have 'parameter_space' property.") # Or log self.rsi_rule.name

        # Add any parameters specific to EnsembleSignalStrategy itself or its direct control
        space["ma_rule.weight"] = [0.4, 0.6] # Example: Fixed or optimizable weight for MA part
        # If rsi_rule_space itself contains 'weight', this might be where rsi_rule.weight is defined.
        # If you also want to optimize the rsi_rule's weight from here (overriding its own definition),
        # you could do: space["rsi_rule.weight"] = [0.3, 0.4, 0.5]
        # However, it's cleaner if each component defines its full parameter space.
        # The existing rsi_rule.parameter_space already includes 'weight'.

        return space        

 

    def stop(self):
        super().stop() # Stop MAStrategy part
        self.rsi_rule.stop()
        self.rsi_indicator.stop()
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' stopped.")
