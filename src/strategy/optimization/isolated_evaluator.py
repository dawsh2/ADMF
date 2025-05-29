"""
Isolated Component Evaluator for granular optimization.

This evaluator creates minimal strategy wrappers to test individual
components in isolation, enabling efficient rule-by-rule optimization.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from src.core.component_base import ComponentBase
from src.strategy.base.strategy import Strategy
from src.strategy.base.indicator import IndicatorBase
from src.strategy.base.rule import RuleBase
from src.core.event import Event, EventType


logger = logging.getLogger(__name__)


class IsolatedStrategy(Strategy):
    """
    Minimal strategy wrapper for testing individual components in isolation.
    
    This strategy contains only a single rule or indicator and generates
    signals based solely on that component's output.
    """
    
    def __init__(self, instance_name: str, component: ComponentBase, 
                 component_type: str = "rule"):
        """
        Initialize isolated strategy.
        
        Args:
            instance_name: Name for this strategy instance
            component: The rule or indicator to test in isolation
            component_type: 'rule' or 'indicator'
        """
        super().__init__(instance_name=instance_name)
        self._isolated_component = component
        self._component_type = component_type
        self._symbol = "TEST"  # Will be set during initialization
        
    def _initialize(self):
        """Initialize the isolated strategy."""
        # Don't call super()._initialize() yet - we need to add components first
        
        # Get symbol from config if available
        self._symbol = self.component_config.get('symbol', 'SPY') if self.component_config else 'SPY'
        
        # Add the isolated component
        if self._component_type == "rule":
            if isinstance(self._isolated_component, RuleBase):
                # Extract and add fresh indicator dependencies directly from the component
                self._add_fresh_indicator_dependencies()
                
                self.add_rule('isolated_rule', self._isolated_component, weight=1.0)
            else:
                # For components that need wrapping
                self._setup_rule_wrapper()
        elif self._component_type == "indicator":
            self._setup_indicator_wrapper()
        
        # Now call parent initialization which will log the correct counts
        super()._initialize()
        
    def _add_fresh_indicator_dependencies(self):
        """Add fresh indicator dependencies from the isolated component to the strategy."""
        component = self._isolated_component
        
        # Check for common indicator attributes and add them to the strategy
        if hasattr(component, 'rsi_indicator') and component.rsi_indicator:
            self.add_indicator('rsi', component.rsi_indicator)
            self.logger.debug(f"Added fresh RSI indicator {component.rsi_indicator.instance_name} to isolated strategy")
            
        if hasattr(component, 'bb_indicator') and component.bb_indicator:
            self.add_indicator('bb', component.bb_indicator)
            self.logger.debug(f"Added fresh BB indicator {component.bb_indicator.instance_name} to isolated strategy")
            
        if hasattr(component, 'macd_indicator') and component.macd_indicator:
            self.add_indicator('macd', component.macd_indicator)
            self.logger.debug(f"Added fresh MACD indicator {component.macd_indicator.instance_name} to isolated strategy")
            
        if hasattr(component, 'fast_ma') and component.fast_ma:
            self.add_indicator('fast_ma', component.fast_ma)
            self.logger.debug(f"Added fresh fast_ma indicator {component.fast_ma.instance_name} to isolated strategy")
            
        if hasattr(component, 'slow_ma') and component.slow_ma:
            self.add_indicator('slow_ma', component.slow_ma)
            self.logger.debug(f"Added fresh slow_ma indicator {component.slow_ma.instance_name} to isolated strategy")
            
        self.logger.info(f"Isolated strategy initialized with {self._component_type}: "
                        f"{self._isolated_component.instance_name}")
        
        # Log the actual rules and indicators we have
        self.logger.info(f"  Rules: {list(self._rules.keys())}")
        self.logger.info(f"  Indicators: {list(self._indicators.keys())}")
    
    def _setup_rule_wrapper(self):
        """Create a rule wrapper for non-RuleBase components."""
        # Handle MACrossoverRule and similar components
        if hasattr(self._isolated_component, 'fast_ma') and hasattr(self._isolated_component, 'slow_ma'):
            # MA Crossover rule - add its indicator dependencies
            if self._isolated_component.fast_ma:
                self.add_indicator('fast_ma', self._isolated_component.fast_ma)
                self.logger.debug("Added fast_ma indicator to isolated strategy")
            if self._isolated_component.slow_ma:
                self.add_indicator('slow_ma', self._isolated_component.slow_ma)
                self.logger.debug("Added slow_ma indicator to isolated strategy")
                
        elif hasattr(self._isolated_component, 'rsi_indicator'):
            # RSI rule - add its indicator dependency
            if self._isolated_component.rsi_indicator:
                self.add_indicator('rsi', self._isolated_component.rsi_indicator)
                self.logger.debug("Added rsi indicator to isolated strategy")
                
        # Add the rule itself
        self.add_rule('isolated_rule', self._isolated_component, weight=1.0)
        self.logger.info(f"Added rule {self._isolated_component.instance_name} with its dependencies")
        
    def _setup_indicator_wrapper(self):
        """Create signal generation logic for indicators."""
        # Add the indicator
        if isinstance(self._isolated_component, IndicatorBase):
            self.add_indicator('isolated_indicator', self._isolated_component)
        
        # Create a simple threshold rule for the indicator
        # This is a simplified approach - in practice might need custom logic
        self.logger.info("Using simple threshold rule for indicator evaluation")
    
    def process_market_data(self, bar_data: Dict[str, Any]):
        """
        Process market data in isolation mode.
        
        For isolated testing, we generate signals based purely on the
        single component's output.
        """
        # Update indicators first
        for indicator in self._indicators.values():
            indicator.update(bar_data)
        
        # For rule-based isolation, use standard signal generation
        if self._component_type == "rule" and self._rules:
            signal_strength = 0.0
            signal_count = 0
            
            for rule_name, rule in self._rules.items():
                signal, strength = rule.evaluate(bar_data)
                
                # Log first few evaluations to verify rule is being called
                if not hasattr(self, '_debug_eval_count'):
                    self._debug_eval_count = 0
                self._debug_eval_count += 1
                
                if self._debug_eval_count <= 10:
                    self.logger.warning(f"ISOLATED STRATEGY DEBUG #{self._debug_eval_count}: Rule {rule_name} ({rule.__class__.__name__}) returned signal={signal}, strength={strength}")
                    # Also log rule object ID to verify we're using the fresh component
                    if hasattr(rule, 'fast_ma') and hasattr(rule, 'slow_ma'):
                        self.logger.warning(f"ISOLATED STRATEGY DEBUG #{self._debug_eval_count}: Rule {rule_name} fast_ma.period={rule.fast_ma.lookback_period}, slow_ma.period={rule.slow_ma.lookback_period}, fast_ma_id={id(rule.fast_ma)}, slow_ma_id={id(rule.slow_ma)}")
                
                # Debug logging for rule evaluation
                if hasattr(self, '_signal_count'):
                    self._signal_count += 1
                else:
                    self._signal_count = 1
                    
                # Log every 100th evaluation to avoid spam
                if self._signal_count % 100 == 0:
                    self.logger.debug(f"Rule {rule_name} evaluation #{self._signal_count}: signal={signal}, strength={strength}")
                    if hasattr(rule, 'get_optimizable_parameters'):
                        params = rule.get_optimizable_parameters()
                        self.logger.debug(f"  Current parameters: {params}")
                    # Log MA indicator values for crossover rule
                    if hasattr(rule, 'fast_ma') and hasattr(rule, 'slow_ma'):
                        fast_val = rule.fast_ma.get_value() if hasattr(rule.fast_ma, 'get_value') else 'N/A'
                        slow_val = rule.slow_ma.get_value() if hasattr(rule.slow_ma, 'get_value') else 'N/A'
                        fast_period = getattr(rule.fast_ma, '_lookback_period', 'N/A')
                        slow_period = getattr(rule.slow_ma, '_lookback_period', 'N/A')
                        self.logger.warning(f"MA Debug #{self._signal_count}: Fast MA({fast_period})={fast_val}, Slow MA({slow_period})={slow_val}, Signal={signal}")
                
                if signal != 0:
                    signal_strength += signal * strength
                    signal_count += 1
                    # Log all non-zero signals
                    self.logger.debug(f"Non-zero signal from {rule_name}: signal={signal}, strength={strength}")
            
            # Generate signal event
            if signal_count > 0:
                avg_strength = signal_strength / signal_count
                signal_type = "BUY" if avg_strength > 0 else "SELL"
                
                event = Event(
                    event_type=EventType.SIGNAL,
                    data={
                        'symbol': self._symbol,
                        'signal': signal_type,
                        'strength': abs(avg_strength),
                        'source': self.instance_name,
                        'component': self._isolated_component.instance_name,
                        'timestamp': bar_data.get('timestamp')
                    }
                )
                self.publish_event(event)
                self.logger.debug(f"Published {signal_type} signal with strength {abs(avg_strength)}")
                
        # For indicator-based isolation, use threshold logic
        elif self._component_type == "indicator":
            self._process_indicator_signal(bar_data)
    
    def _process_indicator_signal(self, bar_data: Dict[str, Any]):
        """Generate signals from isolated indicator."""
        # Get the current price
        price = float(bar_data.get('close', 0))
        
        # Check if we have a Bollinger Bands indicator
        if hasattr(self._isolated_component, 'upper_band') and hasattr(self._isolated_component, 'lower_band'):
            upper = self._isolated_component.upper_band
            lower = self._isolated_component.lower_band
            
            if upper is not None and lower is not None:
                # Generate signals based on band touches
                signal = 0
                strength = 1.0
                
                if price <= lower:
                    signal = 1  # Buy signal at lower band
                    strength = min(1.0, (lower - price) / lower * 10)  # Stronger signal further below band
                elif price >= upper:
                    signal = -1  # Sell signal at upper band  
                    strength = min(1.0, (price - upper) / upper * 10)  # Stronger signal further above band
                
                if signal != 0:
                    event = Event(
                        event_type=EventType.SIGNAL,
                        data={
                            'symbol': self._symbol,
                            'signal': "BUY" if signal > 0 else "SELL",
                            'strength': abs(strength),
                            'source': self.instance_name,
                            'component': self._isolated_component.instance_name,
                            'timestamp': bar_data.get('timestamp'),
                            'price': price,
                            'upper_band': upper,
                            'lower_band': lower
                        }
                    )
                    self.publish_event(event)
                    
                    # Log signal generation
                    if not hasattr(self, '_bb_signal_count'):
                        self._bb_signal_count = 0
                    self._bb_signal_count += 1
                    
                    if self._bb_signal_count <= 5 or self._bb_signal_count % 50 == 0:
                        self.logger.debug(f"BB indicator signal #{self._bb_signal_count}: "
                                        f"price={price:.2f}, upper={upper:.2f}, lower={lower:.2f}, "
                                        f"signal={'BUY' if signal > 0 else 'SELL'}, strength={strength:.3f}")
        
        # Add similar logic for other indicator types (RSI, MACD, etc.) as needed


class IsolatedComponentEvaluator:
    """
    Evaluates components in isolation by creating minimal strategy wrappers.
    
    This evaluator enables testing of individual rules and indicators without
    interference from other components, significantly reducing optimization time.
    """
    
    def __init__(self, backtest_runner, data_handler, portfolio, risk_manager, 
                 execution_handler):
        """
        Initialize the evaluator with required components.
        
        Args:
            backtest_runner: BacktestRunner instance
            data_handler: Data handler for market data
            portfolio: Portfolio manager
            risk_manager: Risk management component
            execution_handler: Execution handler
        """
        self.backtest_runner = backtest_runner
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _create_isolated_container(self, component: ComponentBase) -> 'Container':
        """
        Create a fresh container with isolated instances of all required components.
        
        This ensures complete isolation between evaluations with no state leakage.
        """
        from src.core.container import Container
        from src.core.event_bus import EventBus
        
        # Create new container and event bus
        isolated_container = Container()
        isolated_event_bus = EventBus()
        
        # Create new context (it's just a dictionary)
        isolated_context = {
            'container': isolated_container,
            'event_bus': isolated_event_bus,
            'config_loader': component._context.get('config_loader') if hasattr(component, '_context') else None,
            'logger': logging.getLogger('isolated_evaluation'),
            # CRITICAL: Include metadata so CLI args are available to isolated components
            'metadata': component._context.get('metadata') if hasattr(component, '_context') else None,
            # Include config as well for components that need it
            'config': component._context.get('config') if hasattr(component, '_context') else None
        }
        
        # Register core services
        isolated_container.register('event_bus', isolated_event_bus)
        isolated_container.register('context', isolated_context)
        
        # Create fresh instances of all required components
        # 1. Portfolio - MUST be fresh to avoid P&L contamination
        fresh_portfolio = self.portfolio.__class__(
            instance_name='isolated_portfolio',
            config_key=self.portfolio.config_key
        )
        fresh_portfolio.initialize(isolated_context)
        isolated_container.register('portfolio_manager', fresh_portfolio)
        
        # 2. Risk Manager - fresh instance
        fresh_risk_manager = self.risk_manager.__class__(
            instance_name='isolated_risk_manager',
            config_key=self.risk_manager.config_key
        )
        fresh_risk_manager.initialize(isolated_context)
        isolated_container.register('risk_manager', fresh_risk_manager)
        
        # 3. Execution Handler - fresh instance
        fresh_execution_handler = self.execution_handler.__class__(
            instance_name='isolated_execution_handler',
            config_key=self.execution_handler.config_key
        )
        fresh_execution_handler.initialize(isolated_context)
        isolated_container.register('execution_handler', fresh_execution_handler)
        
        # 4. Data Handler - fresh instance but same data source
        # CRITICAL: We must set component_config BEFORE creating the instance
        # because some components might use it in __init__
        
        # Get the config from the original data handler
        data_config = {}
        if hasattr(self.data_handler, 'component_config') and self.data_handler.component_config:
            data_config = self.data_handler.component_config.copy()
        
        # Ensure we have the required fields
        if 'symbol' not in data_config and hasattr(self.data_handler, '_symbol'):
            data_config['symbol'] = self.data_handler._symbol
        if 'csv_file_path' not in data_config and hasattr(self.data_handler, '_csv_file_path'):
            data_config['csv_file_path'] = self.data_handler._csv_file_path
            
        # If still missing, use defaults
        if 'symbol' not in data_config:
            data_config['symbol'] = 'SPY'
        if 'csv_file_path' not in data_config:
            data_config['csv_file_path'] = 'data/SPY.csv'
        
        fresh_data_handler = self.data_handler.__class__(
            instance_name='isolated_data_handler',
            config_key=self.data_handler.config_key
        )
        
        # CRITICAL: Set the bootstrap override config to ensure it's used
        # This bypasses the normal config loading in initialize()
        fresh_data_handler._bootstrap_override_config = data_config
                
        # Log what config we're using for debugging
        self.logger.debug(f"Initializing isolated data handler with override config: {data_config}")
            
        fresh_data_handler.initialize(isolated_context)
        
        # Copy all data from the original data handler
        # The original should already be started and have data loaded
        self.logger.debug(f"[ISOLATED DATA DEBUG] Copying data from original handler:")
        if hasattr(self.data_handler, '_data_for_run') and self.data_handler._data_for_run is not None:
            fresh_data_handler._data_for_run = self.data_handler._data_for_run.copy()
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _data_for_run: {len(fresh_data_handler._data_for_run)} bars")
        else:
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _data_for_run: NOT FOUND")
            
        if hasattr(self.data_handler, '_train_df') and self.data_handler._train_df is not None:
            fresh_data_handler._train_df = self.data_handler._train_df.copy()
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _train_df: {len(fresh_data_handler._train_df)} bars")
        else:
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _train_df: NOT FOUND")
            
        if hasattr(self.data_handler, '_test_df') and self.data_handler._test_df is not None:
            fresh_data_handler._test_df = self.data_handler._test_df.copy()
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _test_df: {len(fresh_data_handler._test_df)} bars")
        else:
            self.logger.debug(f"[ISOLATED DATA DEBUG]   _test_df: NOT FOUND")
        
        # Copy all the necessary attributes
        if hasattr(self.data_handler, '_df'):
            fresh_data_handler._df = self.data_handler._df
        if hasattr(self.data_handler, '_symbol'):
            fresh_data_handler._symbol = self.data_handler._symbol
        if hasattr(self.data_handler, '_csv_file_path'):
            fresh_data_handler._csv_file_path = self.data_handler._csv_file_path
        if hasattr(self.data_handler, '_cli_max_bars'):
            fresh_data_handler._cli_max_bars = self.data_handler._cli_max_bars
            self.logger.debug(f"[ISOLATED DATA DEBUG] Copied _cli_max_bars: {self.data_handler._cli_max_bars}")
            
        # Set the active dataset and _active_df
        if hasattr(self.data_handler, '_active_dataset'):
            fresh_data_handler._active_dataset = self.data_handler._active_dataset
            # Use the set_active_dataset method to properly set _active_df
            fresh_data_handler.set_active_dataset(self.data_handler._active_dataset)
            self.logger.debug(f"[ISOLATED DATA DEBUG] Set active dataset to: {self.data_handler._active_dataset}")
        elif fresh_data_handler._train_df is not None:
            # Default to train if no active dataset specified
            fresh_data_handler.set_active_dataset('train')
            self.logger.debug(f"[ISOLATED DATA DEBUG] No active dataset found, defaulting to 'train'")
        else:
            # Fall back to full data
            fresh_data_handler._active_df = fresh_data_handler._data_for_run
            self.logger.debug(f"[ISOLATED DATA DEBUG] No train_df found, using full _data_for_run")
            
        # Log final active_df size
        if hasattr(fresh_data_handler, '_active_df') and fresh_data_handler._active_df is not None:
            self.logger.debug(f"[ISOLATED DATA DEBUG] Final _active_df size: {len(fresh_data_handler._active_df)} bars")
        else:
            self.logger.error(f"[ISOLATED DATA DEBUG] ERROR: _active_df is None or not set!")
            
        # Mark as started since we've copied the data
        fresh_data_handler.running = True
        isolated_container.register('data_handler', fresh_data_handler)
        
        # 5. Backtest Runner - fresh instance
        fresh_backtest_runner = self.backtest_runner.__class__(
            instance_name='isolated_backtest_runner',
            config_key=self.backtest_runner.config_key
        )
        fresh_backtest_runner.initialize(isolated_context)
        fresh_backtest_runner.container = isolated_container
        isolated_container.register('backtest_runner', fresh_backtest_runner)
        
        # 6. Regime detector - copy from original context for regime analysis
        try:
            if hasattr(component, '_context') and component._context and 'container' in component._context:
                original_container = component._context['container']
                original_regime_detector = original_container.resolve('regime_detector')
                if original_regime_detector:
                    # Create a fresh regime detector instance with copied configuration
                    self.logger.debug(f"[REGIME SETUP DEBUG] Original regime detector config_key: {original_regime_detector.config_key}")
                    fresh_regime_detector = original_regime_detector.__class__(
                        instance_name='isolated_regime_detector',
                        config_key=original_regime_detector.config_key
                    )
                    fresh_regime_detector.initialize(isolated_context)
                    
                    # CRITICAL FIX: Copy configuration directly from original
                    if hasattr(original_regime_detector, '_regime_thresholds'):
                        fresh_regime_detector._regime_thresholds = original_regime_detector._regime_thresholds.copy()
                        self.logger.debug(f"[REGIME SETUP DEBUG] Copied {len(fresh_regime_detector._regime_thresholds)} thresholds to isolated detector")
                    
                    if hasattr(original_regime_detector, '_min_regime_duration'):
                        fresh_regime_detector._min_regime_duration = original_regime_detector._min_regime_duration
                    
                    # CRITICAL FIX: Copy indicator configuration and recreate indicators
                    try:
                        # Get the original indicator configuration from the original regime detector
                        original_indicator_config = original_regime_detector.get_specific_config("indicators", {})
                        if original_indicator_config:
                            # Override the get_specific_config method to return the copied config for indicators
                            original_get_specific_config = fresh_regime_detector.get_specific_config
                            def patched_get_specific_config(key, default=None):
                                if key == "indicators":
                                    return original_indicator_config
                                return original_get_specific_config(key, default)
                            fresh_regime_detector.get_specific_config = patched_get_specific_config
                            
                            # Now recreate indicators with the copied configuration
                            fresh_regime_detector._setup_regime_indicators()
                            self.logger.debug(f"[REGIME SETUP DEBUG] Copied and recreated {len(original_indicator_config)} indicators for isolated detector")
                        else:
                            self.logger.debug(f"[REGIME SETUP DEBUG] Original regime detector has no indicator configuration")
                    except Exception as e:
                        self.logger.debug(f"[REGIME SETUP DEBUG] Failed to copy indicator configuration: {e}")
                    
                    isolated_container.register('regime_detector', fresh_regime_detector)
                    self.logger.debug(f"[REGIME SETUP] Added regime detector to isolated container for {component.instance_name}")
                else:
                    self.logger.debug(f"[REGIME SETUP] No regime detector found in original container")
            else:
                self.logger.debug(f"[REGIME SETUP] No original container available in component context")
        except Exception as e:
            self.logger.debug(f"[REGIME SETUP] Could not set up regime detector: {e}")
        
        # 7. Config loader if needed
        if hasattr(component, '_context') and isinstance(component._context, dict) and 'config_loader' in component._context:
            isolated_container.register('config_loader', component._context['config_loader'])
        
        self.logger.info(f"Created isolated container with fresh instances for {component.instance_name}")
        return isolated_container
    
    def _create_component_copy(self, component: ComponentBase, isolated_context: Dict[str, Any]) -> ComponentBase:
        """
        Create a fresh copy of a component with isolated context and dependencies.
        """
        # Get parameters to apply
        # Check if parameters were passed from component optimizer
        current_params = {}
        if hasattr(component, '_pending_params'):
            current_params = component._pending_params
            self.logger.debug(f"Got pending params for {component.__class__.__name__}: {current_params}")
        elif hasattr(component, 'get_optimizable_parameters'):
            # Fallback: get current parameters if no pending params
            current_params = component.get_optimizable_parameters()
            self.logger.debug(f"Got current params from {component.__class__.__name__}: {current_params}")
        else:
            self.logger.debug(f"{component.__class__.__name__} has no parameters to apply")
        
        # Create fresh instance of the component
        fresh_component = component.__class__(
            instance_name=f"{component.instance_name}_isolated",
            config_key=component.config_key
        )
        
        # Handle components with indicator dependencies
        if hasattr(component, 'rsi_indicator') and component.rsi_indicator:
            # Create fresh RSI indicator
            fresh_indicator = component.rsi_indicator.__class__(
                instance_name=f"{component.rsi_indicator.instance_name}_isolated",
                config_key=component.rsi_indicator.config_key
            )
            fresh_indicator.initialize(isolated_context)
            fresh_component.rsi_indicator = fresh_indicator
            
        elif hasattr(component, 'bb_indicator') and component.bb_indicator:
            # Create fresh BB indicator
            fresh_indicator = component.bb_indicator.__class__(
                instance_name=f"{component.bb_indicator.instance_name}_isolated",
                config_key=component.bb_indicator.config_key
            )
            fresh_indicator.initialize(isolated_context)
            fresh_component.bb_indicator = fresh_indicator
            
        elif hasattr(component, 'macd_indicator') and component.macd_indicator:
            # Create fresh MACD indicator
            fresh_indicator = component.macd_indicator.__class__(
                instance_name=f"{component.macd_indicator.instance_name}_isolated",
                config_key=component.macd_indicator.config_key
            )
            fresh_indicator.initialize(isolated_context)
            fresh_component.macd_indicator = fresh_indicator
            
        elif hasattr(component, 'fast_ma') and hasattr(component, 'slow_ma'):
            # MA Crossover - create fresh MA indicators with default parameters
            # The apply_parameters call later will set the correct test parameters
            fresh_fast_ma = component.fast_ma.__class__(
                name=f"{component.fast_ma.instance_name}_isolated",
                config_key=component.fast_ma.config_key
            )
            fresh_slow_ma = component.slow_ma.__class__(
                name=f"{component.slow_ma.instance_name}_isolated", 
                config_key=component.slow_ma.config_key
            )
            fresh_fast_ma.initialize(isolated_context)
            fresh_slow_ma.initialize(isolated_context)
            fresh_component.fast_ma = fresh_fast_ma
            fresh_component.slow_ma = fresh_slow_ma
        
        # Initialize the fresh component with isolated context
        fresh_component.initialize(isolated_context)
        
        # Apply current parameters to the fresh component
        if current_params:
            self.logger.debug(f"PARAM APPLICATION DEBUG - Applying parameters to fresh {fresh_component.__class__.__name__}: {current_params}")
            self.logger.debug(f"PARAM APPLICATION DEBUG - Component object ID BEFORE: {id(fresh_component)}")
            if hasattr(fresh_component, 'fast_ma') and fresh_component.fast_ma:
                self.logger.debug(f"PARAM APPLICATION DEBUG - Fast MA object ID BEFORE: {id(fresh_component.fast_ma)}, period: {fresh_component.fast_ma.lookback_period}")
            if hasattr(fresh_component, 'slow_ma') and fresh_component.slow_ma:
                self.logger.debug(f"PARAM APPLICATION DEBUG - Slow MA object ID BEFORE: {id(fresh_component.slow_ma)}, period: {fresh_component.slow_ma.lookback_period}")
                
            fresh_component.apply_parameters(current_params)
            
            if hasattr(fresh_component, 'fast_ma') and fresh_component.fast_ma:
                self.logger.debug(f"PARAM APPLICATION DEBUG - Fast MA object ID AFTER: {id(fresh_component.fast_ma)}, period: {fresh_component.fast_ma.lookback_period}")
            if hasattr(fresh_component, 'slow_ma') and fresh_component.slow_ma:
                self.logger.debug(f"PARAM APPLICATION DEBUG - Slow MA object ID AFTER: {id(fresh_component.slow_ma)}, period: {fresh_component.slow_ma.lookback_period}")
            self.logger.debug(f"PARAM APPLICATION DEBUG - Successfully applied parameters to fresh component")
        
        return fresh_component
        
    def evaluate_component_simple(self, component: ComponentBase, 
                         metric: str = "sharpe_ratio",
                         component_type: Optional[str] = None) -> float:
        """
        Simpler evaluation approach that creates truly isolated portfolio instances.
        
        Each evaluation gets its own fresh portfolio to prevent state leakage.
        """
        # Auto-detect component type if not specified
        if component_type is None:
            if hasattr(component, 'evaluate') and callable(getattr(component, 'evaluate')):
                component_type = 'rule'
            elif hasattr(component, 'update') and callable(getattr(component, 'update')):
                component_type = 'indicator'
            else:
                raise ValueError(f"Cannot determine component type for {component.instance_name}")
        
        self.logger.info(f"Evaluating {component_type} component {component.instance_name} "
                        f"in isolation using metric: {metric}")
        
        # Create isolated strategy
        isolated_strategy = IsolatedStrategy(
            instance_name=f"isolated_strategy_{component.instance_name}",
            component=component,
            component_type=component_type
        )
        
        # Initialize the isolated strategy with the same context
        isolated_strategy.initialize(component._context)
        
        # Ensure the component itself is initialized if it hasn't been
        if hasattr(component, 'initialized') and not component.initialized:
            component.initialize(component._context)
        
        # CRITICAL: Create a fresh portfolio instance for this evaluation
        # This ensures complete isolation - no state leakage between evaluations
        fresh_portfolio = self.portfolio.__class__(
            instance_name=f'isolated_portfolio_{component.instance_name}',
            config_key=self.portfolio.config_key
        )
        fresh_portfolio.initialize(component._context)
        
        # DEBUG: Log component parameters before evaluation
        if hasattr(component, 'get_optimizable_parameters'):
            current_params = component.get_optimizable_parameters()
            self.logger.warning(f"ISOLATION DEBUG - Component {component.instance_name} current parameters: {current_params}")
            
            # Check RSI indicator parameters specifically
            if hasattr(component, 'rsi_indicator') and component.rsi_indicator:
                rsi_params = component.rsi_indicator.get_optimizable_parameters()
                self.logger.warning(f"ISOLATION DEBUG - RSI indicator {component.rsi_indicator.instance_name} parameters: {rsi_params}")
                
            # Check rule thresholds specifically for RSI rules
            if hasattr(component, 'oversold_threshold'):
                self.logger.warning(f"ISOLATION DEBUG - RSI rule thresholds: OS={component.oversold_threshold}, OB={component.overbought_threshold}, Weight={component._weight}")
            
            # Log the actual object ID to verify we have unique instances
            self.logger.warning(f"ISOLATION DEBUG - Component object ID: {id(component)}")
            if hasattr(component, 'rsi_indicator') and component.rsi_indicator:
                self.logger.warning(f"ISOLATION DEBUG - RSI indicator object ID: {id(component.rsi_indicator)}")
        
        # Get container from context
        container = self.backtest_runner.container if hasattr(self.backtest_runner, 'container') else None
        if not container and hasattr(component, '_context') and component._context.get('container'):
            container = component._context['container']
            self.backtest_runner.container = container
        
        # Store the original portfolio and strategy
        original_portfolio = None
        original_strategy = None
        if container:
            try:
                original_portfolio = container.resolve('portfolio_manager')
            except:
                pass
            try:
                original_strategy = container.resolve('strategy')
            except:
                pass
        
        try:
            # Register the fresh portfolio and isolated strategy
            if container:
                container.register('portfolio_manager', fresh_portfolio)
                container.register('strategy', isolated_strategy)
            
            # Setup the fresh portfolio with event subscriptions
            fresh_portfolio.setup()
            
            # Start the fresh portfolio
            if hasattr(fresh_portfolio, 'running') and not fresh_portfolio.running:
                fresh_portfolio.start()
            
            self.logger.info(f"Created fresh portfolio for {component.instance_name}: "
                           f"Cash={fresh_portfolio.current_cash}, "
                           f"Realized P&L={fresh_portfolio.realized_pnl}, "
                           f"Total Value={fresh_portfolio.current_total_value}")
            
            # Reset the data handler to ensure it streams data fresh
            if hasattr(self.data_handler, 'reset'):
                self.data_handler.reset()
                self.logger.debug(f"Reset data handler before isolated evaluation")
                
            # Ensure isolated strategy is started
            if not isolated_strategy.running:
                isolated_strategy.start()
            
            # Ensure backtest runner is started
            if not self.backtest_runner.running:
                self.backtest_runner.start()
            
            # Log data handler state before running backtest
            if hasattr(self.data_handler, '_active_df') and self.data_handler._active_df is not None:
                df = self.data_handler._active_df
                self.logger.info(f"[ISOLATED EVAL DATA CHECK] Running backtest for {component.instance_name}")
                self.logger.info(f"  Active dataset size: {len(df)} bars")
                if len(df) > 0:
                    first_row = df.iloc[0]
                    last_row = df.iloc[-1]
                    self.logger.info(f"  First bar: {first_row.name} close=${first_row.get('close', 'N/A')}")
                    self.logger.info(f"  Last bar: {last_row.name} close=${last_row.get('close', 'N/A')}")
                    # Check for specific price ranges
                    if len(df) >= 5:
                        for i in range(min(5, len(df))):
                            row = df.iloc[i]
                            close = row.get('close', 0)
                            if 520.5 < close < 521.5:
                                self.logger.warning(f"  [WARNING] Bar {i} has close=${close:.2f} in $521 range (March 26 data)!")
            
            # Run the backtest using execute()
            results = self.backtest_runner.execute()
            
            # Extract the requested metric from fresh portfolio
            performance_metrics = results.get('performance_metrics', {})
            
            # Log portfolio state after evaluation
            self.logger.info(f"Portfolio state after evaluation for {component.instance_name}: "
                           f"Cash={fresh_portfolio.current_cash}, "
                           f"Realized P&L={fresh_portfolio.realized_pnl}, "
                           f"Total Value={fresh_portfolio.current_total_value}, "
                           f"Trades={len(fresh_portfolio._trade_log) if hasattr(fresh_portfolio, '_trade_log') else 'N/A'}")
            
            # DEBUG: Check if component parameters changed during evaluation
            if hasattr(component, 'get_optimizable_parameters'):
                final_params = component.get_optimizable_parameters()
                self.logger.warning(f"ISOLATION DEBUG - Component {component.instance_name} final parameters: {final_params}")
                
                # Check if parameters are different after evaluation
                if hasattr(component, 'rsi_indicator') and component.rsi_indicator:
                    final_rsi_params = component.rsi_indicator.get_optimizable_parameters()
                    self.logger.warning(f"ISOLATION DEBUG - RSI indicator final parameters: {final_rsi_params}")
            
            # Debug logging
            self.logger.info(f"Isolated evaluation complete for {component.instance_name}:")
            self.logger.info(f"  Total trades: {performance_metrics.get('total_trades', 0)}")
            self.logger.info(f"  Win rate: {performance_metrics.get('win_rate', 0):.2%}")
            self.logger.info(f"  Sharpe ratio: {performance_metrics.get('sharpe_ratio', 'N/A')}")
            self.logger.info(f"  Total return: {performance_metrics.get('total_return', 'N/A')}")
            
            # Log current component parameters for debugging identical scores
            if hasattr(component, 'get_optimizable_parameters'):
                current_params = component.get_optimizable_parameters()
                self.logger.warning(f"IDENTICAL SCORE DEBUG - Component {component.instance_name} params: {current_params}")
                # Also log final portfolio value and detailed metrics for comparison
                final_value = getattr(fresh_portfolio, 'current_total_value', 'N/A')
                initial_value = getattr(fresh_portfolio, 'initial_cash', 'N/A')
                self.logger.warning(f"IDENTICAL SCORE DEBUG - Portfolio: Initial={initial_value}, Final={final_value}")
                self.logger.warning(f"IDENTICAL SCORE DEBUG - Performance metrics: {performance_metrics}")
                # Log the actual results object to see if it's being reused
                self.logger.warning(f"IDENTICAL SCORE DEBUG - Results object ID: {id(results)}")
            
            # Extract score based on metric
            score = 0.0
            if metric == "sharpe_ratio":
                sharpe = performance_metrics.get('sharpe_ratio', performance_metrics.get('portfolio_sharpe_ratio', 0.0))
                score = sharpe if sharpe is not None else 0.0
            elif metric == "total_return":
                score = performance_metrics.get('total_return', 0.0)
            elif metric == "win_rate":
                score = performance_metrics.get('win_rate', 0.0)
            elif metric == "max_drawdown":
                score = -abs(performance_metrics.get('max_drawdown', 0.0))
            else:
                self.logger.warning(f"Unknown metric {metric}, using sharpe_ratio")
                score = performance_metrics.get('sharpe_ratio', 0.0)
            
            # Return both score and full results for regime analysis
            return score, results
                
        except Exception as e:
            self.logger.error(f"Error evaluating component {component.instance_name}: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return float('-inf'), {}
        finally:
            # Restore original components and clean up
            try:
                if container:
                    if original_portfolio:
                        container.register('portfolio_manager', original_portfolio)
                    if original_strategy:
                        container.register('strategy', original_strategy)
                
                # Stop and teardown the fresh portfolio
                if fresh_portfolio and fresh_portfolio.running:
                    fresh_portfolio.stop()
                if fresh_portfolio:
                    fresh_portfolio.teardown()
                    
                # Clean up the isolated strategy
                if isolated_strategy and isolated_strategy.running:
                    isolated_strategy.stop()
                if isolated_strategy:
                    isolated_strategy.teardown()
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup: {cleanup_error}")
    
    def evaluate_component(self, component: ComponentBase, 
                         metric: str = "sharpe_ratio",
                         component_type: Optional[str] = None) -> float:
        """
        Evaluate a single component in isolation.
        
        Args:
            component: The component to evaluate
            metric: Performance metric to optimize
            component_type: Type of component ('rule' or 'indicator')
            
        Returns:
            Performance score for the given metric
        """
        # Auto-detect component type if not specified
        if component_type is None:
            if hasattr(component, 'evaluate') and callable(getattr(component, 'evaluate')):
                component_type = 'rule'
            elif hasattr(component, 'update') and callable(getattr(component, 'update')):
                # Check for various indicator types
                if (hasattr(component, 'value') or 
                    hasattr(component, 'upper_band') or  # BB indicator
                    hasattr(component, 'macd_line') or   # MACD indicator
                    hasattr(component, 'get_parameter_space')):  # Generic indicator with parameter space
                    component_type = 'indicator'
                else:
                    # If it has update method and parameter space, it's likely an indicator
                    component_type = 'indicator'
            else:
                raise ValueError(f"Cannot determine component type for {component.instance_name}")
        
        self.logger.info(f"Evaluating {component_type} component {component.instance_name} "
                        f"in isolation using metric: {metric}")
        
        # Create completely isolated container for this evaluation
        isolated_container = self._create_isolated_container(component)
        isolated_context = isolated_container.resolve('context')
        
        # Get fresh instances from isolated container
        isolated_portfolio = isolated_container.resolve('portfolio_manager')
        isolated_risk_manager = isolated_container.resolve('risk_manager')
        isolated_execution = isolated_container.resolve('execution_handler')
        isolated_data_handler = isolated_container.resolve('data_handler')
        isolated_backtest_runner = isolated_container.resolve('backtest_runner')
        
        # Create isolated strategy
        isolated_strategy = IsolatedStrategy(
            instance_name=f"isolated_strategy_{component.instance_name}",
            component=component,
            component_type=component_type
        )
        
        # Initialize the isolated strategy with isolated context
        isolated_strategy.initialize(isolated_context)
        isolated_container.register('strategy', isolated_strategy)
        
        self.logger.debug(f"Registered isolated strategy '{isolated_strategy.instance_name}' in isolated container")
        
        # Create a fresh copy of the component for isolated evaluation
        # This ensures the component and its dependencies use the isolated container
        fresh_component = self._create_component_copy(component, isolated_context)
        
        # Update the isolated strategy to use the fresh component
        isolated_strategy._isolated_component = fresh_component
        
        # CRITICAL FIX: Update the strategy's indicators AND rules to use the fresh component
        # Clear old indicators and rules first
        isolated_strategy._indicators.clear()
        isolated_strategy._rules.clear()
        
        # Add fresh indicators from the updated component
        if hasattr(fresh_component, 'fast_ma') and fresh_component.fast_ma:
            isolated_strategy.add_indicator('fast_ma', fresh_component.fast_ma)
            self.logger.debug(f"FRESH INDICATOR UPDATE: Added fast_ma with period {fresh_component.fast_ma.lookback_period}")
            
        if hasattr(fresh_component, 'slow_ma') and fresh_component.slow_ma:
            isolated_strategy.add_indicator('slow_ma', fresh_component.slow_ma)
            self.logger.debug(f"FRESH INDICATOR UPDATE: Added slow_ma with period {fresh_component.slow_ma.lookback_period}")
            
        if hasattr(fresh_component, 'rsi_indicator') and fresh_component.rsi_indicator:
            isolated_strategy.add_indicator('rsi', fresh_component.rsi_indicator)
            self.logger.debug(f"FRESH INDICATOR UPDATE: Added rsi_indicator")
            
        if hasattr(fresh_component, 'bb_indicator') and fresh_component.bb_indicator:
            isolated_strategy.add_indicator('bb', fresh_component.bb_indicator)
            self.logger.debug(f"FRESH INDICATOR UPDATE: Added bb_indicator")
            
        if hasattr(fresh_component, 'macd_indicator') and fresh_component.macd_indicator:
            isolated_strategy.add_indicator('macd', fresh_component.macd_indicator)
            self.logger.debug(f"FRESH INDICATOR UPDATE: Added macd_indicator")
        
        # CRITICAL: Also update the rule to use the fresh component
        isolated_strategy.add_rule('isolated_rule', fresh_component, weight=1.0)
        self.logger.debug(f"FRESH RULE UPDATE: Added fresh rule with updated indicators")
        
        # Pass pending params if they exist
        if hasattr(component, '_pending_params'):
            fresh_component._pending_params = component._pending_params
        
        try:
            # Start all components in proper order
            components_to_start = [
                isolated_portfolio,
                isolated_risk_manager,
                isolated_execution,
                isolated_data_handler,
                isolated_strategy
            ]
            
            # Add regime detector if it exists in the container
            try:
                isolated_regime_detector = isolated_container.resolve('regime_detector')
                if isolated_regime_detector:
                    components_to_start.append(isolated_regime_detector)
                    self.logger.debug(f"[REGIME STARTUP] Added regime detector to startup sequence")
            except:
                self.logger.debug(f"[REGIME STARTUP] No regime detector found in isolated container")
            
            for comp in components_to_start:
                if hasattr(comp, 'start') and not comp.running:
                    comp.start()
                    self.logger.debug(f"Started {comp.instance_name}")
            
            # Debug log to ensure we're evaluating the right component
            self.logger.debug(f"Starting isolated evaluation of {component.instance_name}")
            if hasattr(component, 'get_optimizable_parameters'):
                current_params = component.get_optimizable_parameters()
                self.logger.debug(f"Component current parameters: {current_params}")
            
            # Verify clean portfolio state
            self.logger.debug(f"Isolated portfolio initial state: Cash={isolated_portfolio.current_cash}, "
                            f"Realized P&L={isolated_portfolio.realized_pnl}, "
                            f"Total Value={isolated_portfolio.current_total_value}")
            
            # Run backtest with completely isolated components
            if not isolated_backtest_runner.running:
                isolated_backtest_runner.start()
            
            # Run the backtest using execute()
            results = isolated_backtest_runner.execute()
            
            # Debug the results
            self.logger.debug(f"[ISOLATED BACKTEST DEBUG] {component.instance_name} results: {type(results)} {results is None}")
            if results is not None:
                self.logger.debug(f"[ISOLATED BACKTEST DEBUG] {component.instance_name} keys: {list(results.keys()) if isinstance(results, dict) else 'not dict'}")
            
            # Extract the requested metric
            performance_metrics = results.get('performance_metrics', {}) if results else {}
            
            # Count signals generated during the backtest
            signal_count = 0
            if hasattr(isolated_strategy, '_signal_count'):
                signal_count = isolated_strategy._signal_count
            
            # Debug logging
            self.logger.info(f"Isolated evaluation complete for {component.instance_name}:")
            self.logger.info(f"  Total trades: {performance_metrics.get('total_trades', 0)}")
            self.logger.info(f"  Win rate: {performance_metrics.get('win_rate', 0):.2%}")
            self.logger.info(f"  Sharpe ratio: {performance_metrics.get('sharpe_ratio', 'N/A')}")
            self.logger.info(f"  Total return: {performance_metrics.get('total_return', 'N/A')}")
            self.logger.info(f"  Rule evaluations: {signal_count}")
            
            # Check if we had any signals
            if 'signal_count' in results:
                self.logger.info(f"  Signals generated: {results['signal_count']}")
            
            # Log current component parameters
            if hasattr(component, 'get_optimizable_parameters'):
                current_params = component.get_optimizable_parameters()
                self.logger.debug(f"  Component parameters: {current_params}")
            
            # Extract score based on metric
            score = 0.0
            if metric == "sharpe_ratio":
                # Try both keys for sharpe ratio
                sharpe = performance_metrics.get('sharpe_ratio', performance_metrics.get('portfolio_sharpe_ratio', 0.0))
                score = sharpe if sharpe is not None else 0.0
            elif metric == "total_return":
                score = performance_metrics.get('total_return', 0.0)
            elif metric == "win_rate":
                score = performance_metrics.get('win_rate', 0.0)
            elif metric == "max_drawdown":
                # For drawdown, lower is better so we return negative
                score = -abs(performance_metrics.get('max_drawdown', 0.0))
            else:
                self.logger.warning(f"Unknown metric {metric}, using sharpe_ratio")
                score = performance_metrics.get('sharpe_ratio', 0.0)
            
            # Return both score and full results for regime analysis
            return score, results
                
        except Exception as e:
            self.logger.error(f"Error evaluating component {component.instance_name}: {e}")
            return float('-inf'), {}  # Return worst possible score and empty results on error
        finally:
            # CRITICAL: Clean up the entire isolated container to prevent any leakage
            try:
                # Stop all components in reverse order
                components_to_stop = [
                    isolated_strategy,
                    isolated_data_handler,
                    isolated_execution,
                    isolated_risk_manager,
                    isolated_portfolio,
                    isolated_backtest_runner
                ]
                
                for comp in components_to_stop:
                    if comp and hasattr(comp, 'stop') and comp.running:
                        comp.stop()
                        self.logger.debug(f"Stopped {comp.instance_name}")
                
                # Teardown all components
                for comp in components_to_stop:
                    if comp and hasattr(comp, 'teardown'):
                        comp.teardown()
                        self.logger.debug(f"Torn down {comp.instance_name}")
                
                # Clear the isolated container
                isolated_container.reset()
                self.logger.debug(f"Reset isolated container for {component.instance_name}")
                
                # Verify portfolio was clean after evaluation
                self.logger.debug(f"Isolated portfolio final state: Cash={isolated_portfolio.current_cash}, "
                                f"Realized P&L={isolated_portfolio.realized_pnl}, "
                                f"Total Value={isolated_portfolio.current_total_value}")
                
            except Exception as cleanup_error:
                self.logger.error(f"Error during isolated container cleanup: {cleanup_error}")
    
    def create_evaluator_function(self, metric: str = "sharpe_ratio", 
                                component_type: Optional[str] = None) -> Callable:
        """
        Create an evaluator function compatible with ComponentOptimizer.
        
        Args:
            metric: Performance metric to optimize
            component_type: Type of component being evaluated
            
        Returns:
            Evaluator function that takes a component and returns a score
        """
        def evaluator(component: ComponentBase):
            # Use the full isolation approach with isolated containers
            # This ensures complete isolation between evaluations
            score, results = self.evaluate_component(component, metric, component_type)
            return score, results
        
        return evaluator