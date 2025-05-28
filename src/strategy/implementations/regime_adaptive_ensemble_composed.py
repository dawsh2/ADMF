# src/strategy/regime_adaptive_ensemble_composed.py
"""
Regime Adaptive Ensemble Strategy - Properly Composed Version

This strategy follows the documented component composition pattern from STRATEGY.MD,
using add_indicator(), add_rule(), etc. rather than direct implementation.
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List, Tuple

from src.strategy.base import Strategy
from src.strategy.base.indicator import MovingAverageIndicator, RSIIndicator
from src.strategy.components.rules.ma_crossover_rule import MACrossoverRule
from src.strategy.components.rules.rsi_rules import RSIRule
from src.core.event import Event, EventType
from src.strategy.base.parameter import ParameterSet, Parameter, ParameterSpace


class RegimeAdaptiveEnsembleComposed(Strategy):
    """
    A properly composed ensemble strategy with regime adaptation.
    
    This strategy:
    - Uses component composition as per the documentation
    - Combines MA crossover and RSI signals
    - Adapts parameters based on detected market regime
    - Supports full optimization of all parameters and weights
    """
    
    def _initialize(self):
        """Initialize the strategy with components."""
        # Call parent initialization first
        super()._initialize()
        
        # Get configuration
        config = self.component_config or {}
        
        # Symbol for trading
        self._symbol = config.get('symbol', 'SPY')
        
        # Regime adaptation settings
        self._regime_params_file = config.get('regime_params_file_path', 'regime_optimized_parameters.json')
        self._fallback_to_overall_best = config.get('fallback_to_overall_best', True)
        self._regime_specific_params = {}
        self._overall_best_params = None
        
        # Disable regime switching during optimization
        self._enable_regime_switching = config.get('enable_regime_switching', True)
        
        # Check if we're in optimization mode via CLI args
        if hasattr(self, '_context') and isinstance(self._context, dict) and 'metadata' in self._context:
            metadata = self._context['metadata']
            if metadata:
                cli_args = metadata.get('cli_args', {})
                if cli_args.get('optimize', False):
                    self._enable_regime_switching = False
                    self.logger.info("Regime switching DISABLED - optimization mode detected via CLI")
                
                # Also check for explicit optimization_mode flag
                if metadata.get('optimization_mode', False):
                    self._enable_regime_switching = False
                    self.logger.info("Regime switching DISABLED - optimization mode detected")
        
        # Initialize components
        self._create_components(config)
        
        # Load regime parameters if available
        self._load_regime_parameters()
        
        self.logger.info(
            f"{self.instance_name} initialized with {len(self._indicators)} indicators, "
            f"{len(self._rules)} rules"
        )
        self.logger.log(35, f"===== ENSEMBLE INITIALIZATION COMPLETE =====")
        self.logger.log(35, f"Indicators: {list(self._indicators.keys())}")
        self.logger.log(35, f"Rules: {list(self._rules.keys())}")
        self.logger.log(35, f"Initial weights: {self._component_weights}")
        
    def _create_components(self, config: Dict[str, Any]):
        """Create and add strategy components."""
        # Create MA indicator
        ma_config = config.get('ma_indicator', {})
        
        # Create fast and slow MA indicators for crossover rule
        fast_ma = MovingAverageIndicator(
            name=f"{self.instance_name}_fast_ma",
            lookback_period=ma_config.get('short_window', 10)
        )
        self.add_indicator('fast_ma', fast_ma)
        
        slow_ma = MovingAverageIndicator(
            name=f"{self.instance_name}_slow_ma",
            lookback_period=ma_config.get('long_window', 20)
        )
        self.add_indicator('slow_ma', slow_ma)
        
        # Create RSI indicator
        rsi_config = config.get('rsi_indicator', {})
        rsi_indicator = RSIIndicator(
            name=f"{self.instance_name}_rsi",
            lookback_period=rsi_config.get('period', 14)
        )
        self.add_indicator('rsi', rsi_indicator)
        
        # Create Bollinger Bands indicator (if configured)
        bb_config = config.get('bb_indicator', {})
        if bb_config.get('enabled', True):  # Enable by default
            from src.strategy.components.indicators.bollinger_bands import BollingerBandsIndicator
            bb_indicator = BollingerBandsIndicator(instance_name=f"{self.instance_name}_bb")
            bb_indicator.lookback_period = bb_config.get('period', 20)
            bb_indicator.num_std_dev = bb_config.get('std_dev', 2.0)
            self.add_indicator('bb', bb_indicator)
        else:
            bb_indicator = None
            
        # Create MACD indicator (if configured)
        macd_config = config.get('macd_indicator', {})
        if macd_config.get('enabled', True):  # Enable by default
            from src.strategy.components.indicators.macd import MACDIndicator
            macd_indicator = MACDIndicator(instance_name=f"{self.instance_name}_macd")
            macd_indicator.fast_period = macd_config.get('fast_period', 12)
            macd_indicator.slow_period = macd_config.get('slow_period', 26)
            macd_indicator.signal_period = macd_config.get('signal_period', 9)
            self.add_indicator('macd', macd_indicator)
        else:
            macd_indicator = None
        
        # Create MA crossover rule
        ma_rule_config = config.get('ma_rule', {})
        ma_rule = MACrossoverRule(
            instance_name=f"{self.instance_name}_ma_crossover",
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            parameters={
                'min_separation': ma_rule_config.get('min_separation', 0.0)
            }
        )
        ma_weight = ma_rule_config.get('weight', 0.25)  # Adjust default weight for more rules
        self.add_rule('ma_crossover', ma_rule, weight=ma_weight)
        
        # Create RSI rule
        rsi_rule_config = config.get('rsi_rule', {})
        rsi_rule = RSIRule(
            instance_name=f"{self.instance_name}_rsi_rule",
            rsi_indicator=rsi_indicator,
            parameters={
                'oversold_threshold': rsi_rule_config.get('oversold_threshold', 30.0),
                'overbought_threshold': rsi_rule_config.get('overbought_threshold', 70.0)
            }
        )
        rsi_weight = rsi_rule_config.get('weight', 0.25)  # Adjust default weight
        self.add_rule('rsi', rsi_rule, weight=rsi_weight)
        
        # Create Bollinger Bands rule (if indicator exists)
        if bb_indicator:
            bb_rule_config = config.get('bb_rule', {})
            from src.strategy.components.rules.bollinger_bands_rule import BollingerBandsRule
            bb_rule = BollingerBandsRule(
                instance_name=f"{self.instance_name}_bb_rule",
                bb_indicator=bb_indicator,
                parameters={
                    'band_width_filter': bb_rule_config.get('band_width_filter', 0.0)
                }
            )
            bb_weight = bb_rule_config.get('weight', 0.25)
            self.add_rule('bb', bb_rule, weight=bb_weight)
            
        # Create MACD rule (if indicator exists)
        if macd_indicator:
            macd_rule_config = config.get('macd_rule', {})
            from src.strategy.components.rules.macd_rule import MACDRule
            macd_rule = MACDRule(
                instance_name=f"{self.instance_name}_macd_rule",
                macd_indicator=macd_indicator,
                parameters={
                    'use_histogram': macd_rule_config.get('use_histogram', False),
                    'min_histogram_threshold': macd_rule_config.get('min_histogram_threshold', 0.0)
                }
            )
            macd_weight = macd_rule_config.get('weight', 0.25)
            self.add_rule('macd', macd_rule, weight=macd_weight)
        
        # Normalize weights
        self._normalize_weights()
        
    def _normalize_weights(self):
        """Normalize rule weights to sum to 1."""
        total = sum(self._component_weights.values())
        if total > 0:
            for name in self._component_weights:
                self._component_weights[name] /= total
                
    def _load_regime_parameters(self):
        """Load regime-specific parameters from file."""
        # Try absolute path first, then relative to optimization_results
        file_paths = [
            self._regime_params_file,
            os.path.join('optimization_results', self._regime_params_file)
        ]
        
        loaded = False
        for file_path in file_paths:
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle new format with 'regimes' key
                    if 'regimes' in data:
                        self.logger.info(f"Loading regime parameters from {file_path}")
                        self.logger.log(35, f"===== LOADING REGIME ADAPTIVE ENSEMBLE PARAMETERS =====")
                        self.logger.log(35, f"Parameter file: {file_path}")
                        for regime, params in data['regimes'].items():
                            self._regime_specific_params[regime] = params
                            self.logger.info(f"Loaded parameters for regime '{regime}': {len(params)} parameters")
                            self.logger.log(35, f"Regime '{regime}' parameters: {params}")
                        loaded = True
                        
                    # Handle old format with 'regime_best_parameters'
                    elif 'regime_best_parameters' in data:
                        for regime, regime_data in data['regime_best_parameters'].items():
                            params = self._extract_parameters(regime_data)
                            if params:
                                self._regime_specific_params[regime] = params
                                self.logger.info(f"Loaded parameters for regime '{regime}'")
                        loaded = True
                        
                    # Extract overall best parameters if available
                    if 'overall_best_parameters' in data:
                        self._overall_best_params = data['overall_best_parameters']
                        self.logger.info("Loaded overall best parameters as fallback")
                    
                    if loaded:
                        self.logger.info(f"Successfully loaded regime parameters for {len(self._regime_specific_params)} regimes")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error loading regime parameters from {file_path}: {e}")
        
        if not loaded:
            self.logger.warning(f"No regime parameters file found at {self._regime_params_file} or optimization_results/")
            
    def _extract_parameters(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from regime data structure."""
        params = {}
        
        # Handle nested parameter structure
        if 'parameters' in regime_data:
            if 'parameters' in regime_data['parameters']:
                params = regime_data['parameters']['parameters'].copy()
            else:
                params = regime_data['parameters'].copy()
                
        # Add weights if available
        if 'weights' in regime_data:
            params.update(regime_data['weights'])
            
        return params
        
    def _on_regime_change(self, old_regime: str, new_regime: str):
        """Handle regime change by updating parameters."""
        # Skip regime switching if disabled (e.g., during optimization)
        if not self._enable_regime_switching:
            self.logger.debug(f"Regime change {old_regime} -> {new_regime} ignored - regime switching disabled")
            return
            
        self.logger.info(f"{'='*60}")
        self.logger.info(f"REGIME CHANGE DETECTED: {old_regime} -> {new_regime}")
        self.logger.info(f"{'='*60}")
        self.logger.log(35, f"===== REGIME CHANGE: {old_regime} -> {new_regime} =====")
        
        # Get parameters for new regime
        params = None
        if new_regime in self._regime_specific_params:
            params = self._regime_specific_params[new_regime]
            self.logger.info(f"Found regime-specific parameters for '{new_regime}'")
            # Log the parameters being loaded
            self.logger.info("Loading parameters:")
            self.logger.log(35, f"Loading regime-specific parameters:")
            for key, value in params.items():
                self.logger.info(f"  {key}: {value}")
                self.logger.log(35, f"  {key}: {value}")
        elif self._fallback_to_overall_best and self._overall_best_params:
            params = self._overall_best_params
            self.logger.info(f"No specific parameters for '{new_regime}', using overall best parameters")
            self.logger.info("Loading fallback parameters:")
            for key, value in params.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.warning(f"No parameters available for regime '{new_regime}'")
            
        if params:
            # Log current state before applying
            self.logger.info("Current parameters before update:")
            if 'fast_ma' in self._indicators:
                self.logger.info(f"  fast_ma.lookback_period: {self._indicators['fast_ma'].lookback_period}")
            if 'slow_ma' in self._indicators:
                self.logger.info(f"  slow_ma.lookback_period: {self._indicators['slow_ma'].lookback_period}")
            if 'rsi' in self._indicators:
                self.logger.info(f"  rsi.lookback_period: {self._indicators['rsi'].lookback_period}")
            if 'bb' in self._indicators:
                self.logger.info(f"  bb.lookback_period: {self._indicators['bb'].lookback_period}")
                self.logger.info(f"  bb.num_std_dev: {self._indicators['bb'].num_std_dev}")
            if 'macd' in self._indicators:
                self.logger.info(f"  macd.fast_period: {self._indicators['macd'].fast_period}")
                self.logger.info(f"  macd.slow_period: {self._indicators['macd'].slow_period}")
                self.logger.info(f"  macd.signal_period: {self._indicators['macd'].signal_period}")
            weights_str = ", ".join([f"{rule}={weight:.2f}" for rule, weight in self._component_weights.items()])
            self.logger.info(f"  weights: {weights_str}")
            
            self._apply_regime_parameters(params)
            
            # Log confirmation of applied parameters
            # Give indicators a moment to update their internal state
            self.logger.info("Parameters after update:")
            if 'fast_ma' in self._indicators:
                actual_period = self._indicators['fast_ma'].lookback_period
                self.logger.info(f"  fast_ma.lookback_period: {actual_period}")
            if 'slow_ma' in self._indicators:
                actual_period = self._indicators['slow_ma'].lookback_period
                self.logger.info(f"  slow_ma.lookback_period: {actual_period}")
            if 'rsi' in self._indicators:
                actual_period = self._indicators['rsi'].lookback_period
                self.logger.info(f"  rsi.lookback_period: {actual_period}")
            if 'bb' in self._indicators:
                self.logger.info(f"  bb.lookback_period: {self._indicators['bb'].lookback_period}")
                self.logger.info(f"  bb.num_std_dev: {self._indicators['bb'].num_std_dev}")
            if 'macd' in self._indicators:
                self.logger.info(f"  macd.fast_period: {self._indicators['macd'].fast_period}")
                self.logger.info(f"  macd.slow_period: {self._indicators['macd'].slow_period}")
                self.logger.info(f"  macd.signal_period: {self._indicators['macd'].signal_period}")
                
            # Log actual weights
            weights_str = ", ".join([f"{rule}={weight:.2f}" for rule, weight in self._component_weights.items()])
            self.logger.info(f"  weights: {weights_str}")
            
            # Also verify rule parameters were set
            self.logger.log(35, "Rule parameters after update:")
            if 'ma_crossover' in self._rules and hasattr(self._rules['ma_crossover'], 'min_separation'):
                self.logger.log(35, f"  MA Crossover - min_separation: {self._rules['ma_crossover'].min_separation}")
            if 'rsi' in self._rules:
                rsi_rule = self._rules['rsi']
                if hasattr(rsi_rule, 'oversold_threshold') and hasattr(rsi_rule, 'overbought_threshold'):
                    self.logger.log(35, f"  RSI - oversold: {rsi_rule.oversold_threshold}, overbought: {rsi_rule.overbought_threshold}")
            if 'bb' in self._rules and hasattr(self._rules['bb'], 'band_width_filter'):
                self.logger.log(35, f"  BB - band_width_filter: {self._rules['bb'].band_width_filter}")
            if 'macd' in self._rules and hasattr(self._rules['macd'], 'min_histogram_threshold'):
                self.logger.log(35, f"  MACD - min_histogram_threshold: {self._rules['macd'].min_histogram_threshold}")
        
        self.logger.info(f"{'='*60}")
            
    def _apply_regime_parameters(self, params: Dict[str, Any]):
        """Apply parameters to components dynamically."""
        
        # Parse and apply parameters dynamically
        for param_key, param_value in params.items():
            # Parse the parameter key to understand its structure
            parts = param_key.split('.')
            
            # Handle weight parameters (e.g., "ma_crossover.weight", "rsi.weight")
            if len(parts) >= 2 and parts[-1] == 'weight':
                rule_name = parts[0] if len(parts) == 2 else parts[1]
                # Map common variations to standard names
                rule_mapping = {
                    'ma_rule': 'ma_crossover',
                    'rsi_rule': 'rsi',
                    'bb_rule': 'bb',
                    'macd_rule': 'macd'
                }
                rule_name = rule_mapping.get(rule_name, rule_name)
                
                if rule_name in self._rules:
                    old_val = self._component_weights.get(rule_name, 0)
                    new_val = float(param_value)
                    self.logger.debug(f"Setting {rule_name} weight: {old_val} -> {new_val}")
                    self._component_weights[rule_name] = new_val
            
            # Handle rule parameters
            elif param_key.startswith('strategy_') and ('_rule.' in param_key or 'ma_crossover.' in param_key):
                # Extract rule name and parameter name
                # Format: "strategy_<rule_name>_rule.<param_name>"
                prefix_parts = param_key.split('.')
                strategy_part = prefix_parts[0]  # e.g., "strategy_rsi_rule"
                param_name = '.'.join(prefix_parts[1:])  # e.g., "oversold_threshold"
                
                # Extract rule name from strategy_XXX_rule or strategy_ma_crossover pattern
                rule_name = None
                if strategy_part.startswith('strategy_'):
                    if strategy_part == 'strategy_ma_crossover':
                        rule_name = 'ma_crossover'
                    elif strategy_part.endswith('_rule'):
                        rule_base = strategy_part.replace('strategy_', '').replace('_rule', '')
                        # Map to actual rule names in self._rules
                        rule_mapping = {
                            'ma_crossover': 'ma_crossover',
                            'rsi': 'rsi',
                            'bb': 'bb',
                            'macd': 'macd'
                        }
                        rule_name = rule_mapping.get(rule_base, rule_base)
                
                # Apply to rule if found
                if rule_name and rule_name in self._rules:
                    rule = self._rules[rule_name]
                    rule_params = {param_name: param_value}
                    
                    # Convert parameter value to appropriate type
                    if param_name in ['oversold_threshold', 'overbought_threshold', 'band_width_filter', 
                                     'min_histogram_threshold', 'min_separation']:
                        rule_params[param_name] = float(param_value)
                    
                    self.logger.debug(f"Setting {rule_name}.{param_name}: {param_value}")
                    
                    # Apply parameters
                    if hasattr(rule, 'set_parameters'):
                        rule.set_parameters(rule_params)
                        self.logger.log(35, f"Applied rule parameter: {rule_name}.{param_name} = {param_value}")
                    else:
                        self.logger.warning(f"Rule {rule_name} does not have set_parameters method")
                else:
                    self.logger.log(35, f"WARNING: Rule '{rule_name}' not found in self._rules")
            
            # Handle indicator parameters dynamically
            else:
                # Extract indicator name and parameter name from various formats
                indicator_name = None
                param_name = None
                
                # Format: "strategy_<indicator>.<param>" or "<indicator>.<param>"
                if '.' in param_key:
                    prefix_parts = param_key.split('.')
                    
                    # Handle "strategy_xxx.yyy" format
                    if prefix_parts[0].startswith('strategy_'):
                        # Handle complex names like strategy_ma_crossover.fast_ma.lookback_period
                        if len(prefix_parts) >= 3:
                            # Extract potential indicator name from the middle part
                            potential_indicator = prefix_parts[1]
                            
                            # Special handling for MA crossover parameters
                            if 'ma_crossover' in prefix_parts[0] and potential_indicator in ['fast_ma', 'slow_ma']:
                                indicator_name = potential_indicator.replace('_ma', '_ma')
                                param_name = '.'.join(prefix_parts[2:])
                            # Handle RSI parameters
                            elif 'rsi' in prefix_parts[0] and 'rsi_indicator' in prefix_parts[1]:
                                indicator_name = 'rsi'
                                param_name = '.'.join(prefix_parts[2:])
                            # Handle BB parameters  
                            elif 'bb' in prefix_parts[0] and 'bb_indicator' in prefix_parts[1]:
                                indicator_name = 'bb'
                                param_name = '.'.join(prefix_parts[2:])
                            # Handle MACD parameters
                            elif 'macd' in prefix_parts[0] and 'macd_indicator' in prefix_parts[1]:
                                indicator_name = 'macd'
                                param_name = '.'.join(prefix_parts[2:])
                    else:
                        # Direct format like "rsi.lookback_period" or "bb.num_std_dev"
                        indicator_name = prefix_parts[0]
                        param_name = '.'.join(prefix_parts[1:])
                
                # Apply to indicator if found
                if indicator_name and indicator_name in self._indicators:
                    indicator = self._indicators[indicator_name]
                    
                    # Build parameter dict for the indicator
                    indicator_params = {}
                    
                    # Convert parameter value to appropriate type
                    if hasattr(indicator, param_name):
                        current_val = getattr(indicator, param_name)
                        if isinstance(current_val, int):
                            new_val = int(param_value)
                        elif isinstance(current_val, float):
                            new_val = float(param_value)
                        else:
                            new_val = param_value
                            
                        indicator_params[param_name] = new_val
                        self.logger.debug(f"Setting {indicator_name}.{param_name}: {current_val} -> {new_val}")
                        
                        # Apply parameters
                        indicator.set_parameters(indicator_params)
                else:
                    # Log unmatched parameters (only if not already handled)
                    if not param_key.endswith('.weight') and not ('_rule.' in param_key):
                        self.logger.log(35, f"WARNING: Parameter '{param_key}' could not be matched to any indicator")
            
        # Normalize weights after update
        self._normalize_weights()
        
        # Log summary of applied parameters
        fast_ma_period = self._indicators['fast_ma'].lookback_period if 'fast_ma' in self._indicators else 'N/A'
        slow_ma_period = self._indicators['slow_ma'].lookback_period if 'slow_ma' in self._indicators else 'N/A'
        rsi_period = self._indicators['rsi'].lookback_period if 'rsi' in self._indicators else 'N/A'
        bb_period = self._indicators['bb'].lookback_period if 'bb' in self._indicators else 'N/A'
        bb_std = self._indicators['bb'].num_std_dev if 'bb' in self._indicators else 'N/A'
        
        if 'macd' in self._indicators:
            macd_info = f"MACD: {self._indicators['macd'].fast_period}/{self._indicators['macd'].slow_period}/{self._indicators['macd'].signal_period}"
        else:
            macd_info = "MACD: N/A"
        
        # Log all rule weights
        weights_str = ", ".join([f"{rule}={weight:.2f}" for rule, weight in self._component_weights.items()])
        self.logger.info(
            f"Applied parameters - MA: {fast_ma_period}/{slow_ma_period}, "
            f"RSI: {rsi_period}, BB: {bb_period}/{bb_std}, {macd_info}, "
            f"Weights: {weights_str}"
        )
        
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for optimization."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        # MA parameters
        space.add_parameter(Parameter(
            name="short_window",
            param_type="discrete",
            values=[5, 10, 15],
            default=10
        ))
        
        space.add_parameter(Parameter(
            name="long_window",
            param_type="discrete",
            values=[20, 30, 40],
            default=20
        ))
        
        # RSI parameters
        space.add_parameter(Parameter(
            name="rsi_indicator.period",
            param_type="discrete",
            values=[9, 14, 21],
            default=14
        ))
        
        space.add_parameter(Parameter(
            name="rsi_rule.oversold_threshold",
            param_type="discrete",
            values=[20, 25, 30],
            default=30
        ))
        
        space.add_parameter(Parameter(
            name="rsi_rule.overbought_threshold",
            param_type="discrete",
            values=[70, 75, 80],
            default=70
        ))
        
        # Weight parameters
        space.add_parameter(Parameter(
            name="ma_rule.weight",
            param_type="discrete",
            values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            default=0.5
        ))
        
        space.add_parameter(Parameter(
            name="rsi_rule.weight",
            param_type="discrete",
            values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            default=0.5
        ))
        
        return space
        
    def get_optimizable_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {}
        
        # Get indicator parameters
        if 'fast_ma' in self._indicators:
            params['short_window'] = self._indicators['fast_ma'].lookback_period
            
        if 'slow_ma' in self._indicators:
            params['long_window'] = self._indicators['slow_ma'].lookback_period
            
        if 'rsi' in self._indicators:
            params['rsi_indicator.period'] = self._indicators['rsi'].lookback_period
            
        # Get rule parameters
        if 'rsi' in self._rules:
            rsi_params = self._rules['rsi'].get_parameters()
            params['rsi_rule.oversold_threshold'] = rsi_params.get('buy_threshold', 30.0)
            params['rsi_rule.overbought_threshold'] = rsi_params.get('sell_threshold', 70.0)
            
        # Get weights
        params['ma_rule.weight'] = self._component_weights.get('ma_crossover', 0.5)
        params['rsi_rule.weight'] = self._component_weights.get('rsi', 0.5)
        
        return params
        
    def apply_parameters(self, params: Dict[str, Any]):
        """Apply parameters for optimization."""
        self._apply_regime_parameters(params)
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        # Check MA window constraint
        short = params.get('short_window', 10)
        long = params.get('long_window', 20)
        
        if short >= long:
            return False, "Short window must be less than long window"
            
        # Check weight constraint (should sum to approximately 1)
        ma_weight = params.get('ma_rule.weight', 0.5)
        rsi_weight = params.get('rsi_rule.weight', 0.5)
        
        if abs((ma_weight + rsi_weight) - 1.0) > 0.1:
            return False, "Weights should sum to approximately 1.0"
            
        return True, None