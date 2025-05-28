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
                # Ensure the rule's dependencies are also added to the strategy
                if hasattr(self._isolated_component, '_dependencies'):
                    for dep_name, dep in self._isolated_component._dependencies.items():
                        if hasattr(dep, 'instance_name'):
                            # Add indicators as dependencies
                            self.add_indicator(dep_name, dep)
                            self.logger.debug(f"Added dependency {dep_name} to isolated strategy")
                
                self.add_rule('isolated_rule', self._isolated_component, weight=1.0)
            else:
                # For components that need wrapping
                self._setup_rule_wrapper()
        elif self._component_type == "indicator":
            self._setup_indicator_wrapper()
        
        # Now call parent initialization which will log the correct counts
        super()._initialize()
            
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
        # This is a placeholder - real implementation would depend on indicator type
        # For example, RSI might generate signals at oversold/overbought levels
        pass


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
            'logger': logging.getLogger('isolated_evaluation')
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
        fresh_data_handler = self.data_handler.__class__(
            instance_name='isolated_data_handler',
            config_key=self.data_handler.config_key
        )
        
        # Copy essential configuration from original data handler before initialization
        # First, copy the component_config which should have all necessary settings
        if hasattr(self.data_handler, 'component_config') and self.data_handler.component_config:
            fresh_data_handler.component_config = self.data_handler.component_config.copy()
        else:
            # Build component config from data handler attributes
            fresh_data_handler.component_config = {}
            if hasattr(self.data_handler, '_symbol'):
                fresh_data_handler.component_config['symbol'] = self.data_handler._symbol
            if hasattr(self.data_handler, '_csv_file_path'):
                fresh_data_handler.component_config['csv_file_path'] = self.data_handler._csv_file_path
                
        # Also set as attributes to be sure
        if hasattr(self.data_handler, '_symbol'):
            fresh_data_handler._symbol = self.data_handler._symbol
        if hasattr(self.data_handler, '_csv_file_path'):
            fresh_data_handler._csv_file_path = self.data_handler._csv_file_path
            
        fresh_data_handler.initialize(isolated_context)
        
        # Copy data configuration but ensure fresh iterator
        if hasattr(self.data_handler, '_data_for_run'):
            fresh_data_handler._data_for_run = self.data_handler._data_for_run.copy()
        if hasattr(self.data_handler, '_active_df'):
            fresh_data_handler._active_df = self.data_handler._active_df.copy()
        # Set the active dataset type
        if hasattr(self.data_handler, '_active_dataset'):
            fresh_data_handler.set_active_dataset(self.data_handler._active_dataset)
        isolated_container.register('data_handler', fresh_data_handler)
        
        # 5. Backtest Runner - fresh instance
        fresh_backtest_runner = self.backtest_runner.__class__(
            instance_name='isolated_backtest_runner',
            config_key=self.backtest_runner.config_key
        )
        fresh_backtest_runner.initialize(isolated_context)
        fresh_backtest_runner.container = isolated_container
        isolated_container.register('backtest_runner', fresh_backtest_runner)
        
        # 6. Config loader if needed
        if hasattr(component, '_context') and isinstance(component._context, dict) and 'config_loader' in component._context:
            isolated_container.register('config_loader', component._context['config_loader'])
        
        self.logger.info(f"Created isolated container with fresh instances for {component.instance_name}")
        return isolated_container
    
    def _create_component_copy(self, component: ComponentBase, isolated_context: Dict[str, Any]) -> ComponentBase:
        """
        Create a fresh copy of a component with isolated context and dependencies.
        """
        # Get current parameters before creating copy
        current_params = {}
        if hasattr(component, 'get_optimizable_parameters'):
            current_params = component.get_optimizable_parameters()
        
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
            # MA Crossover - create fresh MA indicators
            fresh_fast_ma = component.fast_ma.__class__(
                name=f"{component.fast_ma.instance_name}_isolated",
                lookback_period=component.fast_ma._lookback_period,
                config_key=component.fast_ma.config_key
            )
            fresh_slow_ma = component.slow_ma.__class__(
                name=f"{component.slow_ma.instance_name}_isolated",
                lookback_period=component.slow_ma._lookback_period,
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
            fresh_component.apply_parameters(current_params)
            self.logger.debug(f"Applied parameters to fresh component: {current_params}")
        
        return fresh_component
        
    def evaluate_component_simple(self, component: ComponentBase, 
                         metric: str = "sharpe_ratio",
                         component_type: Optional[str] = None) -> float:
        """
        Simpler evaluation approach that focuses on portfolio reset.
        
        This is a temporary solution until full containerization is working.
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
        
        try:
            # CRITICAL: Reset portfolio and verify it's clean
            self.portfolio.reset()
            
            # Verify portfolio is actually reset
            if self.portfolio.realized_pnl != 0.0:
                self.logger.error(f"Portfolio not properly reset! Realized P&L = {self.portfolio.realized_pnl}")
                # Force a harder reset
                self.portfolio.realized_pnl = 0.0
                self.portfolio.current_cash = self.portfolio.initial_cash
                self.portfolio.current_total_value = self.portfolio.initial_cash
                self.portfolio.open_positions = {}
                self.portfolio._trade_log = []
            
            self.logger.debug(f"Portfolio state before evaluation: Cash={self.portfolio.current_cash}, "
                            f"Realized P&L={self.portfolio.realized_pnl}, "
                            f"Total Value={self.portfolio.current_total_value}")
            
            # Ensure portfolio is started to resubscribe to events
            if hasattr(self.portfolio, 'running') and not self.portfolio.running:
                self.portfolio.start()
            
            # Reset the data handler to ensure it streams data fresh
            if hasattr(self.data_handler, 'reset'):
                self.data_handler.reset()
                self.logger.debug(f"Reset data handler before isolated evaluation")
            
            # Get container from context
            container = self.backtest_runner.container if hasattr(self.backtest_runner, 'container') else None
            if not container and hasattr(component, '_context') and component._context.get('container'):
                container = component._context['container']
                self.backtest_runner.container = container
                
            # Store the original strategy if there is one
            original_strategy = None
            if container:
                try:
                    original_strategy = container.resolve('strategy')
                except:
                    pass
                    
            # Register our isolated strategy temporarily
            if container:
                container.register('strategy', isolated_strategy)
                
            # Ensure isolated strategy is started
            if not isolated_strategy.running:
                isolated_strategy.start()
            
            # Ensure backtest runner is started
            if not self.backtest_runner.running:
                self.backtest_runner.start()
            
            # Run the backtest using execute()
            results = self.backtest_runner.execute()
            
            # Restore original strategy
            if original_strategy and container:
                container.register('strategy', original_strategy)
            
            # Extract the requested metric
            performance_metrics = results.get('performance_metrics', {})
            
            # Log portfolio state after evaluation
            self.logger.debug(f"Portfolio state after evaluation: Cash={self.portfolio.current_cash}, "
                            f"Realized P&L={self.portfolio.realized_pnl}, "
                            f"Total Value={self.portfolio.current_total_value}")
            
            # Debug logging
            self.logger.info(f"Isolated evaluation complete for {component.instance_name}:")
            self.logger.info(f"  Total trades: {performance_metrics.get('total_trades', 0)}")
            self.logger.info(f"  Win rate: {performance_metrics.get('win_rate', 0):.2%}")
            self.logger.info(f"  Sharpe ratio: {performance_metrics.get('sharpe_ratio', 'N/A')}")
            self.logger.info(f"  Total return: {performance_metrics.get('total_return', 'N/A')}")
            
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
            # Clean up the isolated strategy
            try:
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
        
        # Create a fresh copy of the component for isolated evaluation
        # This ensures the component and its dependencies use the isolated container
        fresh_component = self._create_component_copy(component, isolated_context)
        
        # Update the isolated strategy to use the fresh component
        isolated_strategy._isolated_component = fresh_component
        
        try:
            # Start all components in proper order
            components_to_start = [
                isolated_portfolio,
                isolated_risk_manager,
                isolated_execution,
                isolated_data_handler,
                isolated_strategy
            ]
            
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
            
            # Extract the requested metric
            performance_metrics = results.get('performance_metrics', {})
            
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
        def evaluator(component: ComponentBase) -> float:
            return self.evaluate_component(component, metric, component_type)
        
        return evaluator