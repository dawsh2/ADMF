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
    
    def _setup_rule_wrapper(self):
        """Create a rule wrapper for non-RuleBase components."""
        # This would wrap components like RSIRule or MACrossoverRule
        # that don't inherit from RuleBase but still generate signals
        self.logger.warning("Rule wrapping for non-RuleBase components not yet implemented")
        
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
                if signal != 0:
                    signal_strength += signal * strength
                    signal_count += 1
            
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
            elif hasattr(component, 'update') and hasattr(component, 'value'):
                component_type = 'indicator'
            else:
                raise ValueError(f"Cannot determine component type for {component.instance_name}")
        
        self.logger.info(f"Evaluating {component_type} component {component.instance_name} "
                        f"in isolation using metric: {metric}")
        
        # Create isolated strategy
        isolated_strategy = IsolatedStrategy(
            instance_name=f"isolated_{component.instance_name}",
            component=component,
            component_type=component_type
        )
        
        # Initialize the isolated strategy with the same context
        isolated_strategy.initialize(component._context)
        
        # Ensure the component itself is initialized if it hasn't been
        if hasattr(component, 'initialized') and not component.initialized:
            component.initialize(component._context)
        
        try:
            # Reset portfolio for clean evaluation
            self.portfolio.reset()
            
            # Ensure portfolio is started to resubscribe to events
            if hasattr(self.portfolio, 'running') and not self.portfolio.running:
                self.portfolio.start()
            
            # Run backtest with isolated strategy
            # The backtest runner needs the strategy to be registered
            # Get container from context if backtest runner doesn't have one
            container = self.backtest_runner.container if hasattr(self.backtest_runner, 'container') else None
            if not container and hasattr(component, '_context') and hasattr(component._context, 'container'):
                container = component._context.container
                # Set the container on the backtest runner
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
            if original_strategy and self.backtest_runner.container:
                self.backtest_runner.container.register('strategy', original_strategy)
            
            # Extract the requested metric
            performance_metrics = results.get('performance_metrics', {})
            
            # Debug logging
            self.logger.debug(f"Backtest results for {component.instance_name}: {results}")
            self.logger.debug(f"Performance metrics: {performance_metrics}")
            
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