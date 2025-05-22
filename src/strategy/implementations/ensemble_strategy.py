# src/strategy/implementations/ensemble_strategy.py
import logging
import datetime
from logging import DEBUG
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
        
        # For adaptive testing - store regime-specific parameters
        self._regime_best_parameters = {}
        self._adaptive_mode_enabled = False  # Default to disabled - optimizer will enable this
        
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

        # Set default weights (will be optimized by genetic algorithm later)
        # During grid search, we use balanced weights since weights are optimized separately
        self._ma_weight = 0.5  # Equal default weights for grid search
        self._rsi_weight = 0.5
        
        # Track current optimization rule for weight adjustment
        self._current_optimization_rule = None
        
        # Log the default weights - genetic optimizer will override these later
        self.logger.info(f"Using default equal weights for grid search: MA={self._ma_weight}, RSI={self._rsi_weight}")
        
        rsi_rule_params = {
            'oversold_threshold': self.get_specific_config('rsi_rule.oversold_threshold', 30.0),
            'overbought_threshold': self.get_specific_config('rsi_rule.overbought_threshold', 70.0),
            'weight': self._rsi_weight  # Use the default weight
        }
        self.rsi_rule = RSIRule(
            instance_name=f"{self.name}_RSIRule",
            config_loader=config_loader,
            event_bus=event_bus,
            component_config_key=f"{component_config_key}.rsi_rule",
            rsi_indicator=self.rsi_indicator,
            parameters=rsi_rule_params
        )
        
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' initialized with MA weight: {self._ma_weight}, RSI weight: {self._rsi_weight}")

    def setup(self):
        # Check weights before setup
        pre_setup_ma = getattr(self, '_ma_weight', 'not_set')
        pre_setup_rsi = getattr(self, '_rsi_weight', 'not_set')
        self.logger.debug(f"Weights before setup: MA={pre_setup_ma}, RSI={pre_setup_rsi}")
        
        super().setup() # Setup MAStrategy part
        self.rsi_indicator.setup()
        self.rsi_rule.setup()
        
        # Check weights after setup
        post_setup_ma = getattr(self, '_ma_weight', 'not_set')
        post_setup_rsi = getattr(self, '_rsi_weight', 'not_set')
        self.logger.debug(f"Weights after setup: MA={post_setup_ma}, RSI={post_setup_rsi}")
        
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
        # No longer printing full classification event to reduce verbosity
        # Removed verbose event logging to reduce output clutter
        if self.logger.isEnabledFor(DEBUG):
            self.logger.debug(f"Classification event received by {self.name}")
        
        if not hasattr(event, 'payload'):
            self.logger.warning(f"'{self.name}' received event without payload attribute: {event}")
            return
            
        if not event.payload:
            self.logger.warning(f"'{self.name}' received event with empty payload: {event}")
            return
            
        payload = event.payload
        # Simplified logging to reduce verbosity
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Classification payload received for regime: {payload.get('classification', 'unknown')}")
        
        if not isinstance(payload, dict):
            self.logger.warning(f"'{self.name}' received non-dict payload: {payload}")
            return
            
        new_regime = payload.get('classification')
        # More concise logging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Detected regime: {new_regime}")
        
        if not new_regime:
            self.logger.warning(f"'{self.name}' missing 'classification' in payload: {payload}")
            return
            
        timestamp = payload.get('timestamp', datetime.datetime.now(datetime.timezone.utc))
            
        if new_regime == self._current_regime:
            # Only log regime-unchanged in debug mode
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Regime unchanged: {new_regime}")
            return
            
        # Only log regime changes once, not to both logger and stdout
        # Use logger.info level which is visible but less prominent than warnings
        self.logger.info(f"REGIME CHANGED: '{self._current_regime}' → '{new_regime}' at {timestamp}")
        
        # Always preserve signal state during regime changes
        self.logger.info(f"Preserving signal state {self._current_signal_state} during regime change")
        
        self._current_regime = new_regime
        
        # Apply regime-specific parameters if available
        self._apply_regime_specific_parameters(new_regime)
        
    def _apply_regime_specific_parameters(self, regime: str) -> None:
        """
        Apply parameters specific to the given regime.
        
        This method looks for optimized parameters in both:
        1. In-memory parameters (for adaptive test phase during optimization)
        2. Saved parameters file (for normal operation)
        
        IMPORTANT: During optimization TRAINING phase, this is disabled to prevent 
        interference with parameter testing. But during adaptive TEST phase, we WANT 
        parameters to change with regimes.
        """
        # Check if we're in adaptive mode with in-memory parameters
        if self._adaptive_mode_enabled and self._regime_best_parameters:
            # Use explicit print to ensure this shows up regardless of log level
            print(f"\n>>> ADAPTIVE MODE ACTIVE: Checking parameters for regime '{regime}' <<<")
            
            if regime in self._regime_best_parameters:
                params = self._regime_best_parameters[regime]
                # Use stdout print for maximum visibility
                print(f"\n>>> APPLYING REGIME PARAMETERS: Using optimized parameters for '{regime}': {params} <<<")
                self.logger.warning(f">>> ADAPTIVE TEST: Applying regime-specific parameters for '{regime}': {params} <<<")
                
                # Check if params contains weight information
                weights_info = ""
                if 'ma_rule.weight' in params or 'rsi_rule.weight' in params:
                    weights_info = f"Weights in params - MA: {params.get('ma_rule.weight', 'not set')}, RSI: {params.get('rsi_rule.weight', 'not set')}"
                    self.logger.warning(f">>> REGIME WEIGHTS BEFORE APPLICATION: {weights_info} <<<")
                    
                # Apply the parameters
                self.set_parameters(params)
                
                # Log the actual weights after parameters are applied for verification
                self.logger.warning(f">>> REGIME WEIGHTS AFTER APPLICATION: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f} <<<")
                return
            else:
                print(f"\n>>> NO PARAMETERS FOR REGIME: '{regime}' not found, using current parameters <<<")
                self.logger.warning(f">>> ADAPTIVE MODE: No parameters found for regime '{regime}', using current parameters <<<")
                return
                
        # BUGFIX: Skip file-based parameter loading during optimization training phase
        # During optimization's training phase, we're testing specific parameter combinations
        import sys
        if '--optimize' in sys.argv and not self._adaptive_mode_enabled:
            self.logger.debug(f"Skipping regime parameter loading for '{regime}' during optimization training (--optimize mode detected)")
            return
        
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
        self.logger.debug(f"{self.name} received BAR event (state: {self.state})")
        if event.event_type != EventType.BAR or event.payload.get("symbol") != self._symbol:
            event_symbol = event.payload.get("symbol") if hasattr(event, 'payload') else 'N/A'
            self.logger.debug(f"{self.name} ignoring event - type: {event.event_type}, symbol: {event_symbol}, expected: {self._symbol}")
            return
            
        # DEBUG: Add optimization run tracking
        if hasattr(self, '_bar_count'):
            self._bar_count += 1
        else:
            self._bar_count = 1
            
        # Log every 1000th bar during optimization to track progress
        if self._bar_count % 1000 == 0:
            self.logger.warning(f"🔍 OPTIMIZATION DEBUG: Processing bar {self._bar_count}, RSI thresholds: {getattr(self.rsi_rule, 'oversold_threshold', 'N/A')}/{getattr(self.rsi_rule, 'overbought_threshold', 'N/A')}")
        
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
            # Log the MA values for debugging
            self.logger.debug(f"MA values - short: {current_short_ma:.4f}, long: {current_long_ma:.4f}, " +
                             f"prev_short: {self._prev_short_ma:.4f}, prev_long: {self._prev_long_ma:.4f}")
                             
            if self._prev_short_ma <= self._prev_long_ma and current_short_ma > current_long_ma:
                if self._current_signal_state != 1: 
                    ma_signal_type_int = 1
                    rsi_overbought = getattr(self.rsi_rule, '_overbought_threshold', 'N/A')
                    self.logger.info(f"MA BUY signal: short_window={self._short_window}, regime={self._current_regime}, " +
                                   f"price={close_price:.4f}")
            elif self._prev_short_ma >= self._prev_long_ma and current_short_ma < current_long_ma:
                if self._current_signal_state != -1: 
                    ma_signal_type_int = -1
                    rsi_overbought = getattr(self.rsi_rule, '_overbought_threshold', 'N/A')
                    self.logger.info(f"MA SELL signal: short_window={self._short_window}, regime={self._current_regime}, " +
                                   f"price={close_price:.4f}")
            else:
                self.logger.debug(f"No MA signal: Current crossover conditions not met")
        
        # Update MA prev values
        if current_short_ma is not None: self._prev_short_ma = current_short_ma
        if current_long_ma is not None: self._prev_long_ma = current_long_ma

        # 2. Update RSI Indicator and Evaluate RSI Rule
        self.rsi_indicator.update(close_price)
        
        # Get current RSI value for logging
        current_rsi = getattr(self.rsi_indicator, '_current_value', None)
        
        # Get RSI thresholds for logging
        oversold = getattr(self.rsi_rule, 'oversold_threshold', 'N/A')
        overbought = getattr(self.rsi_rule, 'overbought_threshold', 'N/A')
        
        # DEBUG: Log RSI values that approach thresholds to see if they're being hit
        if current_rsi is not None and isinstance(oversold, (int, float)) and isinstance(overbought, (int, float)):
            if current_rsi <= oversold + 5 or current_rsi >= overbought - 5:
                self.logger.info(f"🔍 RSI NEAR THRESHOLD: RSI={current_rsi:.2f}, thresholds=({oversold}, {overbought})")
        
        rsi_triggered, rsi_strength, rsi_signal_type_str = self.rsi_rule.evaluate(bar_data)
        
        # Log RSI signal generation details
        if rsi_triggered:
            self.logger.warning(f"🔍 RSI SIGNAL GENERATED: Bar {self._bar_count}, RSI={current_rsi:.2f}, thresholds=({oversold}, {overbought}), " +
                           f"signal={rsi_signal_type_str}, strength={rsi_strength}, params_hash={hash(str(oversold)+str(overbought))}")
        
        rsi_signal_type_int = 0
        if rsi_triggered:
            rsi_signal_type_int = int(rsi_strength) # 1 for BUY, -1 for SELL

        # 3. Combine Signals Using Voting Weights System
        final_signal_type_int: Optional[int] = None
        
        # Ensure we have valid weights for voting
        ma_influence = self._ma_weight
        rsi_influence = self._rsi_weight
        
        # Make sure weights are not modified by further normalization here
        total = ma_influence + rsi_influence
        if abs(total - 1.0) > 0.01:  # Only renormalize if we're significantly off
            ma_influence = ma_influence / total
            rsi_influence = rsi_influence / total
        
        # Debug log the weights being used for voting
        self.logger.debug(f"Current weights for voting: MA={ma_influence:.4f}, RSI={rsi_influence:.4f} (original weights: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f})")
        
        # Voting-based signal selection
        current_signal_to_publish = 0
        
        # Debug log all the signal inputs
        self.logger.debug(f"Signal inputs - MA: {ma_signal_type_int}, RSI: {rsi_signal_type_int}, " +
                         f"MA_influence: {ma_influence:.4f}, RSI_influence: {rsi_influence:.4f}")
        
        # Both signals agree - always publish
        if ma_signal_type_int != 0 and rsi_signal_type_int != 0 and ma_signal_type_int == rsi_signal_type_int:
            current_signal_to_publish = ma_signal_type_int
            signal_reason = f"Agreement(MA={ma_signal_type_int}, RSI={rsi_signal_type_int})"
        
        # Only one signal active - use the signal based on optimized weights
        elif ma_signal_type_int != 0 and rsi_signal_type_int == 0:
            # MA signal only - always publish (weight optimization handles importance)
            current_signal_to_publish = ma_signal_type_int
            signal_reason = f"MA_only(sig={ma_signal_type_int}, influence={ma_influence:.2f})"
        
        elif rsi_signal_type_int != 0 and ma_signal_type_int == 0:
            # RSI signal only - always publish (weight optimization handles importance)  
            current_signal_to_publish = rsi_signal_type_int
            signal_reason = f"RSI_only(sig={rsi_signal_type_int}, influence={rsi_influence:.2f})"
        
        # Conflicting signals - use weighted priority
        elif ma_signal_type_int != 0 and rsi_signal_type_int != 0 and ma_signal_type_int != rsi_signal_type_int:
            if ma_influence > rsi_influence:
                current_signal_to_publish = ma_signal_type_int
                signal_reason = f"MA_priority(MA={ma_signal_type_int}, RSI={rsi_signal_type_int}, MA_inf={ma_influence:.2f})"
            else:
                current_signal_to_publish = rsi_signal_type_int  
                signal_reason = f"RSI_priority(MA={ma_signal_type_int}, RSI={rsi_signal_type_int}, RSI_inf={rsi_influence:.2f})"
        else:
            signal_reason = "No_signals"
            
        # Log the signal selection result
        self.logger.debug(f"Signal selection result: {current_signal_to_publish} - Reason: {signal_reason}")
            

        if current_signal_to_publish != 0 and self._current_signal_state != current_signal_to_publish:
            final_signal_type_int = current_signal_to_publish
            self._current_signal_state = final_signal_type_int # Update strategy's overall state
            
            
            signal_payload: Dict[str, Any] = {
                "symbol": self._symbol,
                "timestamp": bar_timestamp,
                "signal_type": final_signal_type_int,
                "price_at_signal": close_price,
                "strategy_id": self.name,
                "reason": f"Ensemble_Voting({signal_reason}, Regime: {self._current_regime})"
            }
            signal_event = Event(EventType.SIGNAL, signal_payload)
            self._event_bus.publish(signal_event)
            self.logger.warning(
                f"🔍 ENSEMBLE SIGNAL PUBLISHED: Bar {self._bar_count}, Type={final_signal_type_int}, Symbol={self._symbol}, Price={close_price:.2f}, Regime={self._current_regime}, Reason={signal_reason}"
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


    def _normalize_weights(self):
        """Normalize the MA and RSI weights to ensure they sum to 1.0"""
        total = self._ma_weight + self._rsi_weight
        if total > 0:  # Avoid division by zero
            # Don't fully normalize - maintain original ratio but scale slightly
            # This preserves the optimization intent better
            if abs(total - 1.0) > 0.1:  # Only normalize if significantly different from 1.0
                self._ma_weight = self._ma_weight / total
                self._rsi_weight = self._rsi_weight / total
                self.logger.debug(f"Normalized weights: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f}")
            else:
                self.logger.debug(f"Weights already near 1.0, keeping current: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f}")
        else:
            # If both weights are zero, set to equal weights
            self._ma_weight = 0.5
            self._rsi_weight = 0.5
            self.logger.warning(f"Both weights were zero, reset to equal weights: MA={self._ma_weight}, RSI={self._rsi_weight}")

    @property
    def ma_weight(self):
        return self._ma_weight
        
    @property
    def rsi_rule_weight(self):
        return self._rsi_weight
        
    def set_parameters(self, params: Dict[str, Any]):
        self.logger.debug(f"EnsembleSignalStrategy.set_parameters called with: {params}")
        super().set_parameters(params) # For MAStrategy part (this stores extended params with underscores)
        
        # Detect which rule is being optimized based on parameter presence
        ma_params = any(k.startswith(('short_window', 'long_window')) for k in params.keys())
        rsi_params = any(k.startswith(('rsi_indicator.', 'rsi_rule.')) for k in params.keys())
        
        # Only adjust weights if we're optimizing individual rules (not during genetic optimization)
        weight_params = any(k.endswith('.weight') for k in params.keys())
        
        if not weight_params:  # Only adjust when NOT doing weight optimization (i.e., during grid search)
            if ma_params and not rsi_params:
                self._current_optimization_rule = "MA"
                self._ma_weight = 0.8  # Give more weight to the rule being optimized
                self._rsi_weight = 0.2
                self.logger.info(f"Detected MA rule optimization, adjusting weights: MA={self._ma_weight}, RSI={self._rsi_weight}")
            elif rsi_params and not ma_params:
                self._current_optimization_rule = "RSI"
                self._ma_weight = 0.2  # Give more weight to the rule being optimized  
                self._rsi_weight = 0.8
                self.logger.info(f"Detected RSI rule optimization, adjusting weights: MA={self._ma_weight}, RSI={self._rsi_weight}")
            elif ma_params and rsi_params:
                self._current_optimization_rule = "JOINT"
                self._ma_weight = 0.5  # Use balanced weights for joint optimization
                self._rsi_weight = 0.5
                self.logger.info(f"Detected joint rule optimization, using balanced weights: MA={self._ma_weight}, RSI={self._rsi_weight}")
            else:
                self._current_optimization_rule = None
        
        # CRITICAL FIX: Update ensemble strategy's own parameter copies
        if "short_window" in params:
            self._short_window = params["short_window"]
        if "long_window" in params:
            self._long_window = params["long_window"]
        
        # Track if weights have changed
        weights_changed = False
        
        # BUGFIX: Properly handle ma_rule.weight parameter 
        if 'ma_rule.weight' in params:
            old_ma_weight = self._ma_weight
            new_ma_weight = float(params['ma_rule.weight'])
            # Ensure weight is within reasonable range
            if new_ma_weight <= 0:
                new_ma_weight = 0.1  # Minimum sensible value
                self.logger.warning(f"MA weight was ≤ 0, setting to minimum value: {new_ma_weight}")
            self._ma_weight = new_ma_weight
            self.logger.debug(f"'{self.name}' MA weight changed: {old_ma_weight} -> {self._ma_weight}")
            weights_changed = True
            
        # Handle rsi_rule.weight directly at the strategy level
        if 'rsi_rule.weight' in params:
            old_rsi_weight = self._rsi_weight
            new_rsi_weight = float(params['rsi_rule.weight'])
            # Ensure weight is within reasonable range
            if new_rsi_weight <= 0:
                new_rsi_weight = 0.1  # Minimum sensible value
                self.logger.warning(f"RSI weight was ≤ 0, setting to minimum value: {new_rsi_weight}")
            self._rsi_weight = new_rsi_weight
            self.logger.debug(f"'{self.name}' RSI weight changed: {old_rsi_weight} -> {self._rsi_weight}")
            weights_changed = True
            
        # Force weights to be non-zero to ensure signal generation
        if self._ma_weight <= 0 and self._rsi_weight <= 0:
            self.logger.warning(f"Both weights are zero or negative, resetting to defaults")
            self._ma_weight = 0.5
            self._rsi_weight = 0.5
            weights_changed = True
            
        # If weights changed, normalize them and update the RSI rule
        if weights_changed:
            self._normalize_weights()
            # Update the RSI rule's weight
            if hasattr(self.rsi_rule, 'set_parameters'):
                self.rsi_rule.set_parameters({'weight': self._rsi_weight})
                self.logger.debug(f"Updated RSI rule weight to {self._rsi_weight}")
        
        # Log the current regime and adaptive mode status for debugging
        current_regime = getattr(self, '_current_regime', 'unknown')
        adaptive_mode = getattr(self, '_adaptive_mode_enabled', False)
        self.logger.info(f"Parameters updated for regime: {current_regime}, adaptive_mode: {adaptive_mode}")
        
        # Parameters for RSI components might be prefixed, e.g., "rsi_indicator.period"
        rsi_indicator_params = {k.split('.', 1)[1]: v for k, v in params.items() if k.startswith("rsi_indicator.")}
        if rsi_indicator_params:
            self.logger.debug(f"'{self.name}' updating RSI indicator parameters: {rsi_indicator_params}")
            self.rsi_indicator.set_parameters(rsi_indicator_params)
            # CRITICAL FIX: Reset RSI indicator state when parameters change during optimization
            # This ensures each parameter combination starts with clean RSI calculation state
            if hasattr(self.rsi_indicator, 'reset_state'):
                self.rsi_indicator.reset_state()
                self.logger.debug(f"Reset RSI indicator state after parameter update")

        # Extract RSI rule parameters except for weight (handled directly above)
        rsi_rule_params = {k.split('.', 1)[1]: v for k, v in params.items() 
                           if k.startswith("rsi_rule.") and k != "rsi_rule.weight"}
        if rsi_rule_params:
            self.logger.debug(f"Updating RSI rule parameters: {rsi_rule_params}")
            self.rsi_rule.set_parameters(rsi_rule_params)
            
            # Verify parameters were applied
            applied_oversold = getattr(self.rsi_rule, 'oversold_threshold', 'not_set')
            applied_overbought = getattr(self.rsi_rule, 'overbought_threshold', 'not_set')
            self.logger.debug(f"RSI rule updated: oversold={applied_oversold}, overbought={applied_overbought}")
            # Note: Not calling setup() to preserve rule state during regime changes
            
        # Preserve signal states during all parameter changes, regardless of adaptive mode
        self.logger.info(f"Preserving signal states during parameter changes - current signal: {self._current_signal_state}")
            
        # Reset bar count for each optimization run
        self._bar_count = 0
        
        # Add instance tracking for debugging
        instance_id = id(self)
        self.logger.warning(f"🔍 PARAMS SET: Instance {instance_id} - MA weight: {self._ma_weight}, RSI weight: {self._rsi_weight}, RSI thresholds: {getattr(self.rsi_rule, 'oversold_threshold', 'N/A')}/{getattr(self.rsi_rule, 'overbought_threshold', 'N/A')}")
        return True 
        
    def enable_adaptive_mode(self, regime_parameters: Dict[str, Dict[str, Any]]):
        """
        Enable adaptive regime mode and load regime-specific parameters.
        
        Args:
            regime_parameters: Dictionary mapping regime names to parameter dictionaries
        """
        self._adaptive_mode_enabled = True
        self._regime_best_parameters = regime_parameters
        self.logger.info(f"ADAPTIVE MODE ENABLED: Loaded parameters for {len(regime_parameters)} regimes")
        self.logger.info(f"Available regimes: {list(regime_parameters.keys())}")
        
        # Apply parameters for current regime immediately if available
        if self._current_regime in regime_parameters:
            params = regime_parameters[self._current_regime]
            # Use a very unique and distinct message that will stand out in the logs
            self.logger.warning(f"!!! ADAPTIVE TEST !!! Applying parameters for current regime '{self._current_regime}': {params}")
            self.set_parameters(params)
            
            # Log the actual weights after parameters are applied
            self.logger.warning(f"!!! WEIGHTS AFTER APPLICATION: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f} !!!")
            
    def disable_adaptive_mode(self):
        """
        Disable adaptive regime mode.
        """
        self._adaptive_mode_enabled = False
        self.logger.info("ADAPTIVE MODE DISABLED")
        
    def get_adaptive_mode_status(self):
        """
        Returns the current adaptive mode status and available regimes.
        This is useful for debugging to verify adaptive mode is working.
        """
        status = {
            "adaptive_mode_enabled": self._adaptive_mode_enabled,
            "current_regime": self._current_regime,
            "available_regimes": list(self._regime_best_parameters.keys()) if self._regime_best_parameters else [],
            "parameter_count": len(self._regime_best_parameters) if self._regime_best_parameters else 0
        }
        
        # Print the status for maximum visibility
        print(f"\n>>> ADAPTIVE MODE STATUS: {status['adaptive_mode_enabled']} <<<")
        print(f">>> CURRENT REGIME: {status['current_regime']} <<<")
        print(f">>> AVAILABLE REGIMES: {status['available_regimes']} <<<")
        
        return status
            
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' parameters updated, components re-setup, and signal states reset.")
        return True 

# src/strategy/implementations/ensemble_strategy.py
# Ensure this method is updated in your EnsembleSignalStrategy class:

    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Return the parameter space for grid search optimization.
        
        This is designed to evaluate each rule in isolation:
        1. When optimizing MA parameters, use default RSI parameters
        2. When optimizing RSI parameters, use default MA parameters
        3. When optimize_mode='all', it defaults to MA-only (since MA is the primary strategy)
        
        To optimize both rules in isolation, run separate commands with --optimize-ma and --optimize-rsi
        Weights are always handled by genetic optimization after grid search.
        """
        # Get optimization mode from environment variable or command line args
        import sys
        optimize_mode = self.get_specific_config('optimize_mode', 'rulewise')  # Default to rule-wise optimization
        
        # Check for command line arguments that might indicate optimization strategy
        if '--optimize-ma' in sys.argv:
            optimize_mode = 'ma'
        elif '--optimize-rsi' in sys.argv:
            optimize_mode = 'rsi'
        elif '--optimize-seq' in sys.argv:
            optimize_mode = 'seq'
        elif '--optimize-joint' in sys.argv:
            optimize_mode = 'joint'
        elif '--optimize' in sys.argv and all(flag not in sys.argv for flag in ['--optimize-ma', '--optimize-rsi', '--optimize-seq', '--optimize-joint']):
            # When --optimize is used without specific strategy flags, default to rule-wise
            optimize_mode = 'rulewise'
            self.logger.info("No specific optimization strategy detected. Defaulting to rule-wise optimization (all rules in isolation).")
            self.logger.info("This will run RSI optimization (12 combinations) + MA optimization (2 combinations) = 14 total.")
            self.logger.info("Use --optimize-seq, --optimize-joint, --optimize-ma, or --optimize-rsi for explicit control.")
        
        # Start with an empty parameter space
        space = {}
        rule_name = ""
        
        # Handle different optimization modes
        if optimize_mode == 'ma':
            # MA-only optimization (isolation)
            ma_space = super().get_parameter_space()  # MAStrategy parameters
            rule_name = "MA"
            self.logger.info(f"Optimizing MA rule parameters only: {list(ma_space.keys())}")
            space.update(ma_space)
            
        elif optimize_mode == 'rsi':
            # RSI-only optimization (isolation)
            rule_name = "RSI"
            # RSI Indicator parameters
            if hasattr(self.rsi_indicator, 'parameter_space') and self.rsi_indicator:
                rsi_ind_space = self.rsi_indicator.parameter_space
                for key, value in rsi_ind_space.items():
                    space[f"rsi_indicator.{key}"] = value
                self.logger.info(f"Including RSI indicator parameters in optimization: {list(rsi_ind_space.keys())}")
            else:
                self.logger.warning("RSI indicator parameter space not available")
                
            # RSI Rule parameters
            if hasattr(self.rsi_rule, 'parameter_space') and self.rsi_rule:
                rsi_rule_space = self.rsi_rule.parameter_space
                for key, value in rsi_rule_space.items():
                    if key != 'weight':  # Skip the weight parameter
                        space[f"rsi_rule.{key}"] = value
                self.logger.info(f"Including RSI rule parameters in optimization: {[k for k in rsi_rule_space.keys() if k != 'weight']}")
            else:
                self.logger.warning("RSI rule parameter space not available")
            self.logger.info(f"Optimizing RSI rule parameters only: {list(space.keys())}")
            
        elif optimize_mode == 'joint':
            # Joint optimization (full Cartesian product)
            rule_name = "JOINT"
            self.logger.info("Using JOINT optimization - full Cartesian product of all rule parameters")
            
            # Include MA parameters
            ma_space = super().get_parameter_space()
            space.update(ma_space)
            self.logger.info(f"Including MA parameters: {list(ma_space.keys())}")
            
            # Include RSI parameters  
            if hasattr(self.rsi_indicator, 'parameter_space') and self.rsi_indicator:
                rsi_ind_space = self.rsi_indicator.parameter_space
                for key, value in rsi_ind_space.items():
                    space[f"rsi_indicator.{key}"] = value
                self.logger.info(f"Including RSI indicator parameters: {list(rsi_ind_space.keys())}")
            else:
                self.logger.warning("RSI indicator parameter space not available")
                
            if hasattr(self.rsi_rule, 'parameter_space') and self.rsi_rule:
                rsi_rule_space = self.rsi_rule.parameter_space
                for key, value in rsi_rule_space.items():
                    if key != 'weight':  # Skip the weight parameter  
                        space[f"rsi_rule.{key}"] = value
                self.logger.info(f"Including RSI rule parameters: {[k for k in rsi_rule_space.keys() if k != 'weight']}")
            else:
                self.logger.warning("RSI rule parameter space not available")
                
            # Calculate and warn about total combinations
            total_combinations = 1
            for values in space.values():
                total_combinations *= len(values)
            self.logger.warning(f"JOINT optimization will test {total_combinations} parameter combinations!")
            
        elif optimize_mode == 'rulewise':
            # Rule-wise optimization - handled by the optimizer, not here
            # This code path should not be reached when optimizer detects rule-wise mode
            rule_name = "RULEWISE"
            self.logger.info("Rule-wise optimization mode detected - will be handled at optimizer level")
            self.logger.info("This will run MA parameters (2) + RSI parameters (12) = 14 total combinations")
            
            # Return empty space - optimizer will handle the rule-wise logic
            space = {}
            self.logger.info("Returning empty parameter space - optimizer will run MA and RSI optimizations separately")
            
        elif optimize_mode == 'seq':
            # Sequential optimization - for now, fall back to joint optimization  
            # TODO: Implement true sequential optimization at optimizer level
            rule_name = "SEQUENTIAL (fallback to joint)"
            self.logger.warning("Sequential optimization not yet fully implemented - falling back to joint optimization")
            self.logger.info("Using joint optimization (full Cartesian product) as temporary fallback")
            
            # Include MA parameters
            ma_space = super().get_parameter_space()
            space.update(ma_space)
            
            # Include RSI parameters  
            if hasattr(self.rsi_indicator, 'parameter_space') and self.rsi_indicator:
                rsi_ind_space = self.rsi_indicator.parameter_space
                for key, value in rsi_ind_space.items():
                    space[f"rsi_indicator.{key}"] = value
                    
            if hasattr(self.rsi_rule, 'parameter_space') and self.rsi_rule:
                rsi_rule_space = self.rsi_rule.parameter_space
                for key, value in rsi_rule_space.items():
                    if key != 'weight':  # Skip the weight parameter  
                        space[f"rsi_rule.{key}"] = value
            
            # Calculate total combinations
            total_combinations = 1
            for values in space.values():
                total_combinations *= len(values)
            self.logger.info(f"Fallback joint optimization will test {total_combinations} parameter combinations")

        # Store rule name for use in output messages
        self._current_optimization_rule = rule_name
        
        # Note: Weights are always optimized separately by the genetic optimizer
        self.logger.info(f"Final parameter space for {rule_name} rule grid search: {list(space.keys())}")
        return space

 

    def stop(self):
        super().stop() # Stop MAStrategy part
        self.rsi_rule.stop()
        self.rsi_indicator.stop()
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' stopped.")
