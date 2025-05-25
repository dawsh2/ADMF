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

        # Load weights from config or use defaults
        # Check for weights in config first, fall back to equal weights
        self._ma_weight = self.get_specific_config('ma_rule.weight', 0.5)
        # Get RSI weight from nested config structure
        rsi_config = self.get_specific_config('rsi_rule', {})
        self._rsi_weight = rsi_config.get('weight', 0.5) if isinstance(rsi_config, dict) else 0.5
        
        # Track current optimization rule for weight adjustment
        self._current_optimization_rule = None
        
        # Log the weights being used
        self.logger.info(f"EnsembleSignalStrategy weights: MA={self._ma_weight}, RSI={self._rsi_weight}")
        
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
        
        # Flags to enable/disable rules for true isolation during optimization
        self._ma_enabled = True
        self._rsi_enabled = True

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
                
        # In production mode, automatically enable adaptive mode if regime parameters are available
        import sys
        is_optimization = any(opt in sys.argv for opt in ['--optimize', '--optimize-rsi', '--optimize-ma', '--optimize-seq', '--optimize-joint'])
        if not is_optimization and not self._adaptive_mode_enabled:
            # Try to load regime parameters from file
            params_file_path = "regime_optimized_parameters.json"
            import os
            import json
            
            if os.path.isfile(params_file_path):
                try:
                    with open(params_file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'regime_best_parameters' in data:
                        # Build regime parameters dictionary
                        regime_parameters = {}
                        for regime, regime_data in data['regime_best_parameters'].items():
                            if 'parameters' in regime_data:
                                # Handle double nesting
                                if 'parameters' in regime_data['parameters']:
                                    params = regime_data['parameters']['parameters'].copy()
                                else:
                                    params = regime_data['parameters'].copy()
                            else:
                                params = regime_data.copy()
                            
                            # Let weights come from JSON file or config
                            # Remove hardcoded weight assignments to allow JSON weights to take effect
                            regime_parameters[regime] = params
                        
                        # Enable adaptive mode
                        self.enable_adaptive_mode(regime_parameters)
                        self.logger.warning("!!! PRODUCTION ADAPTIVE MODE ENABLED AUTOMATICALLY !!!")
                        self.logger.warning(f"Loaded parameters for regimes: {list(regime_parameters.keys())}")
                        
                        # Trigger initial classification to match optimization behavior
                        if self._regime_detector and hasattr(self._regime_detector, 'get_current_classification'):
                            initial_regime = self._regime_detector.get_current_classification()
                            if initial_regime:
                                self.logger.info(f"Triggering initial classification for regime: {initial_regime}")
                                # Apply parameters for initial regime immediately
                                self._apply_regime_specific_parameters(initial_regime)
                except Exception as e:
                    self.logger.error(f"Error loading regime parameters for adaptive mode: {e}")
                
        # Event subscriptions are handled by MAStrategy and _on_bar will be overridden
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' setup complete.")
        self.state = BaseComponent.STATE_INITIALIZED
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Override to include weight parameters in addition to MA parameters.
        This is critical for genetic optimization to work properly.
        """
        # Get base MA parameters from parent class
        params = super().get_parameters()
        
        # Add weight parameters
        params.update({
            "ma_rule.weight": getattr(self, '_ma_weight', 0.5),
            "rsi_rule.weight": getattr(self, '_rsi_weight', 0.5)
        })
        
        # Add RSI-specific parameters if they exist
        if hasattr(self, 'rsi_rule') and self.rsi_rule:
            if hasattr(self.rsi_rule, 'oversold_threshold'):
                params["rsi_rule.oversold_threshold"] = self.rsi_rule.oversold_threshold
            if hasattr(self.rsi_rule, 'overbought_threshold'):
                params["rsi_rule.overbought_threshold"] = self.rsi_rule.overbought_threshold
                
        # Add RSI indicator parameters
        if hasattr(self, 'rsi_indicator') and self.rsi_indicator:
            if hasattr(self.rsi_indicator, 'period'):
                params["rsi_indicator.period"] = self.rsi_indicator.period
        
        return params
        
    def set_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Override to handle weight parameters in addition to MA parameters.
        This is critical for genetic optimization to work properly.
        """
        # Log the parameters we're setting
        self.logger.info(f"EnsembleStrategy '{self.name}' setting parameters: {params}")
        
        # Extract and handle weight parameters
        if "ma_rule.weight" in params:
            new_ma_weight = params["ma_rule.weight"]
            try:
                self._ma_weight = float(new_ma_weight)
                self.logger.info(f"Updated MA weight to: {self._ma_weight}")
            except (ValueError, TypeError):
                self.logger.error(f"Invalid MA weight value: {new_ma_weight}")
                return False
                
        if "rsi_rule.weight" in params:
            new_rsi_weight = params["rsi_rule.weight"]
            try:
                self._rsi_weight = float(new_rsi_weight)
                self.logger.info(f"Updated RSI weight to: {self._rsi_weight}")
                # Also update the RSI rule's weight if it exists
                if hasattr(self, 'rsi_rule') and self.rsi_rule:
                    if hasattr(self.rsi_rule, 'weight'):
                        self.rsi_rule.weight = self._rsi_weight
                    elif hasattr(self.rsi_rule, '_weight'):
                        self.rsi_rule._weight = self._rsi_weight
            except (ValueError, TypeError):
                self.logger.error(f"Invalid RSI weight value: {new_rsi_weight}")
                return False
        
        # Handle RSI rule parameters
        if "rsi_rule.oversold_threshold" in params:
            new_oversold = params["rsi_rule.oversold_threshold"]
            if hasattr(self, 'rsi_rule') and self.rsi_rule:
                if hasattr(self.rsi_rule, 'oversold_threshold'):
                    self.rsi_rule.oversold_threshold = float(new_oversold)
                elif hasattr(self.rsi_rule, '_oversold_threshold'):
                    self.rsi_rule._oversold_threshold = float(new_oversold)
                self.logger.info(f"Updated RSI oversold threshold to: {new_oversold}")
        
        if "rsi_rule.overbought_threshold" in params:
            new_overbought = params["rsi_rule.overbought_threshold"]
            if hasattr(self, 'rsi_rule') and self.rsi_rule:
                if hasattr(self.rsi_rule, 'overbought_threshold'):
                    self.rsi_rule.overbought_threshold = float(new_overbought)
                elif hasattr(self.rsi_rule, '_overbought_threshold'):
                    self.rsi_rule._overbought_threshold = float(new_overbought)
                self.logger.info(f"Updated RSI overbought threshold to: {new_overbought}")
        
        # Handle RSI indicator parameters
        if "rsi_indicator.period" in params:
            new_period = params["rsi_indicator.period"]
            if hasattr(self, 'rsi_indicator') and self.rsi_indicator:
                if hasattr(self.rsi_indicator, 'period'):
                    self.rsi_indicator.period = int(new_period)
                elif hasattr(self.rsi_indicator, '_period'):
                    self.rsi_indicator._period = int(new_period)
                self.logger.info(f"Updated RSI period to: {new_period}")
        
        # Call parent set_parameters for MA window parameters
        success = super().set_parameters(params)
        
        # Log final weights after parameter update
        self.logger.info(f"Final weights after parameter update: MA={self._ma_weight}, RSI={self._rsi_weight}")
        
        return success
        
    def reset(self):
        """
        Reset the strategy state for fresh evaluation.
        This is critical for genetic algorithm optimization where the same strategy instance
        is used to evaluate multiple parameter combinations.
        """
        # Reset MA strategy state from parent class
        self._prices.clear()
        self._prev_short_ma = None
        self._prev_long_ma = None
        self._current_signal_state = 0
        
        # Reset RSI indicator state
        if hasattr(self, 'rsi_indicator') and self.rsi_indicator:
            if hasattr(self.rsi_indicator, '_prices'):
                self.rsi_indicator._prices.clear()
            if hasattr(self.rsi_indicator, '_current_value'):
                self.rsi_indicator._current_value = None
            if hasattr(self.rsi_indicator, '_gains'):
                self.rsi_indicator._gains.clear()
            if hasattr(self.rsi_indicator, '_losses'):
                self.rsi_indicator._losses.clear()
            if hasattr(self.rsi_indicator, '_avg_gain'):
                self.rsi_indicator._avg_gain = None
            if hasattr(self.rsi_indicator, '_avg_loss'):
                self.rsi_indicator._avg_loss = None
        
        # Reset any evaluation counters
        if hasattr(self, '_bar_count'):
            self._bar_count = 0
            
        # Reset regime tracking state (but preserve regime parameters)
        self._current_regime = "default"
        
        self.logger.debug(f"Strategy '{self.name}' state reset for fresh evaluation")
        
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
            # Only log regime changes, not every parameter check
            if regime != getattr(self, '_last_logged_regime', None):
                self.logger.debug(f"ADAPTIVE MODE: Checking parameters for regime '{regime}'")
                self._last_logged_regime = regime
            
            if regime in self._regime_best_parameters:
                params = self._regime_best_parameters[regime]
                # Only log when regime actually changes, not on every bar
                if regime != getattr(self, '_last_applied_regime', None):
                    self.logger.info(f"🔄 REGIME PARAMETER UPDATE: '{regime}' applying: {params}")
                    self._last_applied_regime = regime
                
                # Apply the parameters and log the before/after state
                old_weights = (getattr(self, '_ma_weight', None), getattr(self, '_rsi_weight', None))
                old_thresholds = (getattr(self.rsi_rule, 'oversold_threshold', None), getattr(self.rsi_rule, 'overbought_threshold', None))
                
                self.set_parameters(params)
                
                new_weights = (getattr(self, '_ma_weight', None), getattr(self, '_rsi_weight', None))
                new_thresholds = (getattr(self.rsi_rule, 'oversold_threshold', None), getattr(self.rsi_rule, 'overbought_threshold', None))
                
                self.logger.info(f"🔄 PARAMETER CHANGE APPLIED: Weights {old_weights} → {new_weights}, RSI_thresholds {old_thresholds} → {new_thresholds}")
                
                # Log all parameters after applying regime-specific parameters
                current_params = self.get_parameters()
                self.logger.debug(f"📊 WEIGHTS AFTER REGIME PARAMETER APPLICATION: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f}")
                self.logger.debug(f"📋 ALL PARAMETERS AFTER APPLICATION: {current_params}")
                return
            else:
                # Only log this once per regime
                if regime != getattr(self, '_last_missing_regime', None):
                    self.logger.debug(f"ADAPTIVE MODE: No parameters found for regime '{regime}', using current parameters")
                    self._last_missing_regime = regime
                return
                
        # Only skip file-based parameter loading during optimization training phase
        # During optimization's training phase, we're testing specific parameter combinations
        # But in production (no optimization flags), we WANT to load from file
        import sys
        if any(opt in sys.argv for opt in ['--optimize', '--optimize-rsi', '--optimize-ma', '--optimize-seq', '--optimize-joint']) and not self._adaptive_mode_enabled:
            self.logger.debug(f"Skipping regime parameter loading for '{regime}' during optimization training")
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
                        # Handle double nesting in JSON structure
                        if 'parameters' in regime_data['parameters']:
                            regime_specific_params = regime_data['parameters']['parameters'].copy()
                        else:
                            regime_specific_params = regime_data['parameters'].copy()
                        
                        # CRITICAL FIX: Also load weights from the regime data
                        if 'weights' in regime_data and regime_data['weights']:
                            regime_specific_params.update(regime_data['weights'])
                            self.logger.info(f"Added regime weights to parameters: {regime_data['weights']}")
                        
                        self.logger.info(f"Found regime-specific parameters for '{regime}': {regime_specific_params}")
                        
                # If not, use overall best parameters as fallback
                elif 'overall_best_parameters' in data:
                    regime_specific_params = data['overall_best_parameters']
                    self.logger.info(f"No specific parameters for regime '{regime}'. Using overall best parameters: {regime_specific_params}")
                
                # Apply the parameters if we found any
                if regime_specific_params:
                    # Translate parameters to format expected by the strategy
                    translated_params = self._translate_parameters(regime_specific_params)
                    # Let weights come from JSON file - no hardcoded overrides
                    # The optimized weights are now saved in the JSON file
                    
                    self.logger.info(f"🔄 FILE-BASED REGIME PARAMETER UPDATE: '{regime}' applying: {regime_specific_params}")
                    
                    # Log before/after state for file-based parameters too
                    old_weights = (getattr(self, '_ma_weight', None), getattr(self, '_rsi_weight', None))
                    old_thresholds = (getattr(self.rsi_rule, 'oversold_threshold', None), getattr(self.rsi_rule, 'overbought_threshold', None))
                    
                    # Apply the parameters
                    success = self.set_parameters(regime_specific_params)
                    
                    new_weights = (getattr(self, '_ma_weight', None), getattr(self, '_rsi_weight', None))
                    new_thresholds = (getattr(self.rsi_rule, 'oversold_threshold', None), getattr(self.rsi_rule, 'overbought_threshold', None))
                    
                    self.logger.info(f"🔄 FILE-BASED PARAMETER CHANGE: Weights {old_weights} → {new_weights}, RSI_thresholds {old_thresholds} → {new_thresholds}")
                    if success:
                        self.logger.info(f"Successfully applied regime parameters for '{regime}'")
                        # Log current weights to verify they were updated
                        self.logger.info(f"Current weights after update: MA={self._ma_weight}, RSI={self._rsi_weight}")
                    else:
                        self.logger.error(f"Failed to apply parameters for regime '{regime}'")
                    
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
            
        # Progress tracking removed for cleaner output
        
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
        if self._ma_enabled and current_short_ma is not None and current_long_ma is not None and \
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
        if self._rsi_enabled:
            self.rsi_indicator.update(close_price)
        
        # Get current RSI value for logging
        current_rsi = getattr(self.rsi_indicator, '_current_value', None) if self._rsi_enabled else None
        
        # Get RSI thresholds for logging
        oversold = getattr(self.rsi_rule, 'oversold_threshold', 'N/A')
        overbought = getattr(self.rsi_rule, 'overbought_threshold', 'N/A')
        
        # COMPREHENSIVE INDICATOR LOGGING FOR DEBUGGING SIGNAL DIFFERENCES
        # Log every bar's indicator values to compare optimizer vs standalone
        if hasattr(self, '_bar_count') and self._bar_count <= 50:  # First 50 bars only
            ma_short_str = f"{current_short_ma:.4f}" if current_short_ma is not None else "N/A"
            ma_long_str = f"{current_long_ma:.4f}" if current_long_ma is not None else "N/A"
            rsi_str = f"{current_rsi:.2f}" if current_rsi is not None else "N/A"
            
            self.logger.info(f"📊 BAR_{self._bar_count:03d} [{bar_timestamp}] INDICATORS: " +
                           f"Price={close_price:.4f}, " +
                           f"MA_short={ma_short_str}, " +
                           f"MA_long={ma_long_str}, " +
                           f"RSI={rsi_str}, " +
                           f"RSI_thresholds=({oversold},{overbought}), " +
                           f"Regime={self._current_regime}, " +
                           f"Weights=(MA:{self._ma_weight:.3f},RSI:{self._rsi_weight:.3f})")
        
        # DEBUG: Log RSI values that approach thresholds to see if they're being hit
        if current_rsi is not None and isinstance(oversold, (int, float)) and isinstance(overbought, (int, float)):
            if current_rsi <= oversold + 5 or current_rsi >= overbought - 5:
                self.logger.info(f"🔍 RSI NEAR THRESHOLD: RSI={current_rsi:.2f}, thresholds=({oversold}, {overbought})")
        
        if self._rsi_enabled:
            rsi_triggered, rsi_strength, rsi_signal_type_str = self.rsi_rule.evaluate(bar_data)
        else:
            rsi_triggered, rsi_strength, rsi_signal_type_str = False, 0.0, 'HOLD'
        
        # RSI signal logging removed for cleaner output
        
        rsi_signal_type_int = 0
        if rsi_triggered:
            rsi_signal_type_int = int(rsi_strength) # 1 for BUY, -1 for SELL

        # 3. Combine Signals Using Weighted Signal Strength System
        final_signal_type_int: Optional[int] = None
        
        # Ensure we have valid weights for voting
        # Set weight to 0 for disabled rules
        ma_influence = self._ma_weight if self._ma_enabled else 0.0
        rsi_influence = self._rsi_weight if self._rsi_enabled else 0.0
        
        # Make sure weights are not modified by further normalization here
        total = ma_influence + rsi_influence
        if total > 0 and abs(total - 1.0) > 0.01:  # Only renormalize if we're significantly off
            ma_influence = ma_influence / total
            rsi_influence = rsi_influence / total
        
        # Debug log the weights being used for voting
        self.logger.debug(f"Current weights for voting: MA={ma_influence:.4f}, RSI={rsi_influence:.4f} (original weights: MA={self._ma_weight:.4f}, RSI={self._rsi_weight:.4f})")
        
        # CONTINUOUS SIGNAL SYSTEM: Make weights affect ALL decisions continuously
        current_signal_to_publish = 0
        
        # Calculate weighted signal strength (continuous between -2.0 and +2.0)
        ma_weighted_signal = ma_signal_type_int * ma_influence
        rsi_weighted_signal = rsi_signal_type_int * rsi_influence  
        combined_signal_strength = ma_weighted_signal + rsi_weighted_signal
        
        # CONTINUOUS SIGNAL GENERATION: Multiple thresholds for different signal strengths
        # This creates a truly continuous fitness landscape
        
        # Strong signals (above 0.6): Full position
        # Medium signals (0.3-0.6): Half position  
        # Weak signals (0.1-0.3): Quarter position
        # Very weak signals (below 0.1): No position
        
        abs_strength = abs(combined_signal_strength)
        signal_strength_multiplier = 0.0
        
        if abs_strength >= 0.6:
            current_signal_to_publish = 1 if combined_signal_strength > 0 else -1
            signal_strength_multiplier = 1.0  # Full position
            signal_reason = f"StrongSignal(strength={combined_signal_strength:.3f})"
        elif abs_strength >= 0.3:
            current_signal_to_publish = 1 if combined_signal_strength > 0 else -1
            signal_strength_multiplier = 0.5  # Half position
            signal_reason = f"MediumSignal(strength={combined_signal_strength:.3f})"
        elif abs_strength >= 0.1:
            current_signal_to_publish = 1 if combined_signal_strength > 0 else -1
            signal_strength_multiplier = 0.25  # Quarter position
            signal_reason = f"WeakSignal(strength={combined_signal_strength:.3f})"
        else:
            signal_reason = f"NoSignal(strength={combined_signal_strength:.3f})"
            
        # Add continuous weight influence to signal strength
        # This ensures even tiny weight differences affect the outcome
        weight_influence_bonus = (ma_influence - 0.5) * 0.1  # Small continuous bonus
        signal_strength_multiplier += weight_influence_bonus
        signal_strength_multiplier = max(0.01, min(signal_strength_multiplier, 1.0))  # Keep in valid range
        
        # Debug log all the signal inputs
        self.logger.debug(f"Signal inputs - MA: {ma_signal_type_int} (weighted: {ma_weighted_signal:.3f}), " +
                         f"RSI: {rsi_signal_type_int} (weighted: {rsi_weighted_signal:.3f}), " +
                         f"Combined strength: {combined_signal_strength:.3f}, Final multiplier: {signal_strength_multiplier:.3f}")
            
        # Log the signal selection result
        self.logger.debug(f"Signal selection result: {current_signal_to_publish} - Reason: {signal_reason}")
            

        if current_signal_to_publish != 0 and self._current_signal_state != current_signal_to_publish:
            final_signal_type_int = current_signal_to_publish
            self._current_signal_state = final_signal_type_int # Update strategy's overall state
            
            # DETAILED SIGNAL LOGGING FOR DEBUGGING
            self.logger.info(f"🚨 SIGNAL GENERATED #{self._bar_count}: " +
                           f"Type={final_signal_type_int}, Price={close_price:.4f}, " +
                           f"Regime={self._current_regime}, " +
                           f"MA_signal={ma_signal_type_int}(w={ma_influence:.3f}), " +
                           f"RSI_signal={rsi_signal_type_int}(w={rsi_influence:.3f}), " +
                           f"Combined_strength={combined_signal_strength:.3f}, " +
                           f"Final_multiplier={signal_strength_multiplier:.3f}")
            
            signal_payload: Dict[str, Any] = {
                "symbol": self._symbol,
                "timestamp": bar_timestamp,
                "signal_type": final_signal_type_int,
                "price_at_signal": close_price,
                "strategy_id": self.name,
                "reason": f"Ensemble_Voting({signal_reason}, Regime: {self._current_regime})",
                "signal_strength": signal_strength_multiplier  # Weight-influenced position sizing
            }
            signal_event = Event(EventType.SIGNAL, signal_payload)
            self._event_bus.publish(signal_event)
            self.logger.info(f"Signal: {final_signal_type_int} at {close_price:.2f} ({signal_reason})")
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
        
        # BUGFIX: Handle nested parameter format from enhanced optimizer
        if 'parameters' in params and isinstance(params['parameters'], dict):
            actual_params = params['parameters']
        else:
            actual_params = params
            
        super().set_parameters(actual_params) # For MAStrategy part (this stores extended params with underscores)
        
        # Detect which rule is being optimized based on parameter presence
        ma_params = any(k.startswith(('short_window', 'long_window')) for k in actual_params.keys())
        rsi_params = any(k.startswith(('rsi_indicator.', 'rsi_rule.')) for k in actual_params.keys())
        
        # Only adjust weights during grid search optimization (not during genetic optimization or adaptive test)
        weight_params = any(k.endswith('.weight') for k in actual_params.keys())
        
        # CRITICAL FIX: Skip weight adjustment during adaptive test mode and genetic optimization
        if not weight_params and not self._adaptive_mode_enabled:  # Only adjust during grid search, not adaptive test
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
        if "short_window" in actual_params:
            self._short_window = actual_params["short_window"]
        if "long_window" in actual_params:
            self._long_window = actual_params["long_window"]
        
        # Track if weights have changed
        weights_changed = False
        
        # BUGFIX: Properly handle ma_rule.weight parameter 
        if 'ma_rule.weight' in actual_params:
            old_ma_weight = self._ma_weight
            new_ma_weight = float(actual_params['ma_rule.weight'])
            # Ensure weight is within reasonable range
            if new_ma_weight <= 0:
                new_ma_weight = 0.1  # Minimum sensible value
                self.logger.warning(f"MA weight was ≤ 0, setting to minimum value: {new_ma_weight}")
            self._ma_weight = new_ma_weight
            self.logger.debug(f"'{self.name}' MA weight changed: {old_ma_weight} -> {self._ma_weight}")
            weights_changed = True
            
        # Handle rsi_rule.weight directly at the strategy level
        if 'rsi_rule.weight' in actual_params:
            old_rsi_weight = self._rsi_weight
            new_rsi_weight = float(actual_params['rsi_rule.weight'])
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
        rsi_indicator_params = {k.split('.', 1)[1]: v for k, v in actual_params.items() if k.startswith("rsi_indicator.")}
        if rsi_indicator_params:
            self.logger.debug(f"Applying RSI indicator parameters: {rsi_indicator_params}")
            self.rsi_indicator.set_parameters(rsi_indicator_params)
            # CRITICAL FIX: Reset RSI indicator state when parameters change during optimization
            # This ensures each parameter combination starts with clean RSI calculation state
            if hasattr(self.rsi_indicator, 'reset_state'):
                self.rsi_indicator.reset_state()
                self.logger.debug(f"RSI indicator state reset after parameter update")

        # Extract RSI rule parameters except for weight (handled directly above)
        rsi_rule_params = {k.split('.', 1)[1]: v for k, v in actual_params.items() 
                           if k.startswith("rsi_rule.") and k != "rsi_rule.weight"}
        if rsi_rule_params:
            self.logger.debug(f"Applying RSI rule parameters: {rsi_rule_params}")
            self.rsi_rule.set_parameters(rsi_rule_params)
            
            # Verify parameters were applied
            applied_oversold = getattr(self.rsi_rule, 'oversold_threshold', 'not_set')
            applied_overbought = getattr(self.rsi_rule, 'overbought_threshold', 'not_set')
            self.logger.debug(f"RSI rule verified: oversold={applied_oversold}, overbought={applied_overbought}")
            # Note: Not calling setup() to preserve rule state during regime changes
            
        # Preserve signal states during all parameter changes, regardless of adaptive mode
        self.logger.info(f"Preserving signal states during parameter changes - current signal: {self._current_signal_state}")
            
        # Reset bar count for each optimization run
        self._bar_count = 0
        
        # Parameter tracking removed for cleaner output
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

    def set_rule_isolation_mode(self, mode: str):
        """
        Set rule isolation mode for optimization.
        
        Args:
            mode: 'ma' to enable only MA, 'rsi' to enable only RSI, 'all' to enable both
        """
        if mode == 'ma':
            self._ma_enabled = True
            self._rsi_enabled = False
            self.logger.info("Rule isolation: MA enabled, RSI disabled")
        elif mode == 'rsi':
            self._ma_enabled = False
            self._rsi_enabled = True
            self.logger.info("Rule isolation: RSI enabled, MA disabled")
        else:  # 'all' or any other value
            self._ma_enabled = True
            self._rsi_enabled = True
            self.logger.info("Rule isolation: Both MA and RSI enabled")
    
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
        self.logger.debug(f"Parameter space for {rule_name}: {space}")
        self.logger.info(f"Final parameter space for {rule_name} rule grid search: {list(space.keys())}")
        return space

 

    def stop(self):
        super().stop() # Stop MAStrategy part
        self.rsi_rule.stop()
        self.rsi_indicator.stop()
        self.logger.info(f"EnsembleSignalStrategy '{self.name}' stopped.")
