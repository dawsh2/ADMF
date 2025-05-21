# src/strategy/implementations/ensemble_strategy.py
import logging
import datetime
from typing import Dict, Any, List, Optional
from src.core.component import BaseComponent

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
    def __init__(self, instance_name: str, config_loader, event_bus, component_config_key: str, container=None):
        super().__init__(instance_name, config_loader, event_bus, component_config_key) # Initialize MAStrategy part
        self.logger = logging.getLogger(f"{__name__}.{instance_name}")
        
        self._container = container
        self._regime_detector = None
        self._current_regime = "default"
        self.regime_detector_service_name = "MyPrimaryRegimeDetector"
        
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
        
        # Subscribe to CLASSIFICATION events to track regime changes
        if self._event_bus:
            self._event_bus.subscribe(EventType.CLASSIFICATION, self.on_classification_change)
            self.logger.info(f"'{self.name}' subscribed to CLASSIFICATION events.")
            
        # Try to resolve the regime detector if container is available
        if self._container:
            try:
                self._regime_detector = self._container.resolve(self.regime_detector_service_name)
                self.logger.info(f"Successfully resolved RegimeDetector: {self._regime_detector.name}")
                initial_regime = self._regime_detector.get_current_classification()
                if initial_regime:
                    self._current_regime = initial_regime
                    self.logger.info(f"Initial market regime set to: {self._current_regime}")
            except Exception as e:
                self.logger.warning(f"Could not resolve RegimeDetector: {e}. Defaulting to 'default' regime.")
                
        # Event subscriptions are handled by MAStrategy and _on_bar will be overridden
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' setup complete.")
        self.state = BaseComponent.STATE_INITIALIZED
        
    def on_classification_change(self, event: Event):
        """
        Handle regime classification changes and update parameters.
        """
        self.logger.info(f"'{self.name}' received classification event: {event}")
        
        if not hasattr(event, 'payload'):
            self.logger.warning(f"'{self.name}' received event without payload attribute: {event}")
            return
            
        if not event.payload:
            self.logger.warning(f"'{self.name}' received event with empty payload: {event}")
            return
            
        payload = event.payload
        self.logger.info(f"'{self.name}' classification event payload: {payload}")
        
        if not isinstance(payload, dict):
            self.logger.warning(f"'{self.name}' received non-dict payload: {payload}")
            return
            
        new_regime = payload.get('classification')
        self.logger.info(f"'{self.name}' extracted classification from payload: {new_regime}")
        
        if not new_regime:
            self.logger.warning(f"'{self.name}' missing 'classification' in payload: {payload}")
            return
            
        timestamp = payload.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            
        if new_regime == self._current_regime:
            self.logger.info(f"'{self.name}' regime unchanged: {new_regime}")
            return
            
        self.logger.info(f"'{self.name}' market regime changed from '{self._current_regime}' to '{new_regime}' at {timestamp}.")
        
        # Let's force a signal state reset when regime changes to ensure we get new signals
        self._current_signal_state = 0
        self.logger.info(f"Reset signal state due to regime change to ensure new signals can be generated")
        
        self._current_regime = new_regime
        
        # Apply regime-specific parameters if available
        self._apply_regime_specific_parameters(new_regime)
        
    def _apply_regime_specific_parameters(self, regime: str) -> None:
        """
        Apply parameters specific to the given regime.
        
        This method looks for optimized parameters in the standard location
        where EnhancedOptimizer saves them.
        """
        # Default parameters file path
        params_file_path = "regime_optimized_parameters.json"
        
        import os
        import json
        
        if not os.path.isfile(params_file_path):
            self.logger.warning(f"Regime parameters file not found: {params_file_path}")
            return
            
        try:
            with open(params_file_path, 'r') as f:
                data = json.load(f)
                
            # Extract regime-specific parameters
            if 'regime_best_parameters' in data:
                regime_specific_params = {}
                
                # Check if we have parameters for this specific regime
                if regime in data['regime_best_parameters']:
                    regime_data = data['regime_best_parameters'][regime]
                    if 'parameters' in regime_data:
                        regime_specific_params = regime_data['parameters']
                        self.logger.info(f"Found regime-specific parameters for '{regime}': {regime_specific_params}")
                        
                # If not, use overall best parameters as fallback
                elif 'overall_best_parameters' in data:
                    regime_specific_params = data['overall_best_parameters']
                    self.logger.info(f"No specific parameters for regime '{regime}'. Using overall best parameters: {regime_specific_params}")
                
                # Apply the parameters if we found any
                if regime_specific_params:
                    # Translate parameters to format expected by the strategy
                    translated_params = self._translate_parameters(regime_specific_params)
                    self.logger.info(f"Applying translated parameters for '{regime}': {translated_params}")
                    # Make sure we're applying the parameters correctly
                    self.logger.info(f"Before parameter update - strategy params: {self.get_parameters()}")
                    success = self.set_parameters(regime_specific_params)
                    self.logger.info(f"Parameter update success: {success}, after update - strategy params: {self.get_parameters()}")
                    
                    # Force re-setup of child components to ensure parameters take effect
                    self.rsi_indicator.setup()
                    self.rsi_rule.setup()
            else:
                self.logger.warning("No regime-specific parameters found in parameters file")
                
        except Exception as e:
            self.logger.error(f"Error loading or applying regime parameters: {e}", exc_info=True)
    
    def _translate_parameters(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate parameter names if needed.
        
        This is mainly a placeholder for any future parameter name translation needs.
        Currently, the set_parameters method already handles dotted parameter names.
        """
        # Return the parameters as is, since set_parameters already handles the dotted names
        return raw_params


    def _on_bar_event(self, event: Event):
        if event.event_type != EventType.BAR or event.payload.get("symbol") != self._symbol:
            return
        
        bar_data: Dict[str, Any] = event.payload
        close_price_val = bar_data.get("close")
        bar_timestamp: Optional[datetime.datetime] = bar_data.get("timestamp")

        if close_price_val is None or bar_timestamp is None:
            return
        close_price = float(close_price_val)
        
        # Log the current regime during trading for debugging
        self.logger.debug(f"Current market regime during bar processing: {self._current_regime}")

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
        
        self.logger.info(f"MA Signal: {ma_signal_type_int}, RSI Signal: {rsi_signal_type_int}, Combined: {combined_strength:.2f}, Weights: MA={self.ma_weight}, RSI={self.rsi_rule.weight}")
        self.logger.info(f"Current parameters - short_window: {self._short_window}, long_window: {self._long_window}, RSI period: {self.rsi_indicator._period}, RSI thresholds: oversold={self.rsi_rule._oversold_threshold}, overbought={self.rsi_rule._overbought_threshold}")

        # Relax the thresholds to generate more signals during optimization
        current_signal_to_publish = 0
        if combined_strength > 0.2: # Lower threshold to generate more BUY signals
            current_signal_to_publish = 1
        elif combined_strength < -0.2: # Lower threshold to generate more SELL signals
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
                "reason": f"Ensemble (MA: {ma_signal_type_int}, RSI: {rsi_signal_type_int}, Regime: {self._current_regime})"
            }
            signal_event = Event(EventType.SIGNAL, signal_payload)
            self._event_bus.publish(signal_event)
            self.logger.info(
                f"Ensemble Published SIGNAL: Type={final_signal_type_int}, Symbol={self._symbol}, Price={close_price:.2f}, Regime={self._current_regime}"
            )
        elif current_signal_to_publish == 0 and self._current_signal_state != 0:
            # Optional: Generate a FLAT signal if combined strength is neutral and previously in a state
            # self._current_signal_state = 0
            # Or just do nothing, letting existing position ride until next strong signal
            pass
            
    def stop(self):
        # Unsubscribe from CLASSIFICATION events
        if self._event_bus:
            try:
                self._event_bus.unsubscribe(EventType.CLASSIFICATION, self.on_classification_change)
                self.logger.info(f"'{self.name}' unsubscribed from CLASSIFICATION events.")
            except Exception as e:
                self.logger.error(f"Error unsubscribing {self.name} from CLASSIFICATION events: {e}")
                
        # Call parent stop method
        super().stop()
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' stopped.")


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
        return True 

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
