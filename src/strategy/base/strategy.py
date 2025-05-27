"""
Base Strategy class implementing component-based architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

from ...core.component_base import ComponentBase
from ...core.event import Event, EventType
from .parameter import ParameterSet, ParameterSpace, Parameter


class StrategyComponent(ABC):
    """Base interface for all strategy components (indicators, rules, features)."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this component."""
        pass
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether the component has enough data to produce valid output."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameter values."""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> ParameterSpace:
        """Get the parameter space for optimization."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset component state."""
        pass


class Strategy(ComponentBase):
    """
    Base Strategy class with component composition.
    
    This class provides:
    - Component management (indicators, rules, features)
    - Parameter namespacing and management
    - Signal generation and aggregation
    - Event handling (BAR, CLASSIFICATION)
    - Optimization support
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        
        # Component storage
        self._indicators: Dict[str, StrategyComponent] = {}
        self._rules: Dict[str, StrategyComponent] = {}
        self._features: Dict[str, StrategyComponent] = {}
        
        # Signal aggregation
        self._aggregation_method = 'weighted'  # weighted, majority, unanimous
        self._component_weights: Dict[str, float] = {}
        
        # State tracking
        self._current_regime = 'default'
        self._bars_processed = 0
        self._last_signal = None
        
        # Parameters
        self._parameter_set = ParameterSet(f"{instance_name}_params")
        
    def _initialize(self):
        """Initialize strategy components and configuration."""
        # Load configuration
        config = self.component_config or {}
        
        # Set aggregation method
        self._aggregation_method = config.get('aggregation_method', 'weighted')
        
        # Initialize components from config
        self._initialize_components_from_config(config)
        
        # Set up parameter management
        self._setup_parameters()
        
        self.logger.info(f"Strategy '{self.instance_name}' initialized with "
                        f"{len(self._indicators)} indicators, "
                        f"{len(self._rules)} rules, "
                        f"{len(self._features)} features")
        
    def _start(self):
        """Start the strategy and subscribe to events."""
        # Subscribe to events
        if self.event_bus:
            self.event_bus.subscribe(EventType.BAR, self._on_bar)
            
            # Only subscribe to classification events if we're not in optimization mode
            # or if regime switching is explicitly enabled
            should_subscribe_classification = True
            
            # Check if we're in optimization mode
            if hasattr(self._context, 'metadata') and self._context.metadata.get('cli_args', {}).get('optimize', False):
                # We're in optimization mode - check if this is the isolated strategy
                if 'isolated' not in self.instance_name.lower():
                    # This is the main strategy during optimization - don't subscribe
                    should_subscribe_classification = False
                    self.logger.info(f"Strategy '{self.instance_name}' NOT subscribing to classification events - optimization mode")
            
            if should_subscribe_classification:
                self.event_bus.subscribe(EventType.CLASSIFICATION, self._on_classification)
            
        self.logger.info(f"Strategy '{self.instance_name}' started")
        
    def _stop(self):
        """Stop the strategy and unsubscribe from events."""
        if self.event_bus:
            self.event_bus.unsubscribe(EventType.BAR, self._on_bar)
            
            # Only unsubscribe from classification if we subscribed
            try:
                self.event_bus.unsubscribe(EventType.CLASSIFICATION, self._on_classification)
            except:
                # We might not have subscribed during optimization
                pass
            
        self.logger.info(f"Strategy '{self.instance_name}' stopped")
        
    # Component Management
    
    def add_indicator(self, name: str, indicator: StrategyComponent) -> None:
        """Add an indicator component."""
        self._indicators[name] = indicator
        self._update_parameter_namespace(f"indicators.{name}", indicator)
        
    def add_rule(self, name: str, rule: StrategyComponent, weight: float = 1.0) -> None:
        """Add a rule component with optional weight."""
        self._rules[name] = rule
        self._component_weights[name] = weight
        self._update_parameter_namespace(f"rules.{name}", rule)
        
    def add_feature(self, name: str, feature: StrategyComponent) -> None:
        """Add a feature component."""
        self._features[name] = feature
        self._update_parameter_namespace(f"features.{name}", feature)
        
    # Event Handlers
    
    def _on_bar(self, event: Event) -> None:
        """Handle incoming bar data."""
        if event.event_type != EventType.BAR:
            return
            
        bar_data = event.payload
        self._bars_processed += 1
        
        # Update all indicators with new data
        self._update_indicators(bar_data)
        
        # Update features (which may depend on indicators)
        self._update_features(bar_data)
        
        # Evaluate rules and generate signal
        signal = self._evaluate_rules(bar_data)
        
        if signal != 0:  # Non-zero signal
            self._publish_signal(signal, bar_data)
            
    def _on_classification(self, event: Event) -> None:
        """Handle regime classification events."""
        if event.event_type != EventType.CLASSIFICATION:
            return
            
        classification = event.payload
        new_regime = classification.get('classification', classification.get('regime', 'default'))
        
        if new_regime != self._current_regime:
            old_regime = self._current_regime
            self.logger.info(f"Strategy '{self.instance_name}' switching regime: "
                           f"{old_regime} -> {new_regime}")
            self._current_regime = new_regime
            self._on_regime_change(old_regime, new_regime)
            
    # Signal Generation
    
    def _evaluate_rules(self, bar_data: Dict[str, Any]) -> int:
        """Evaluate all rules and aggregate signals."""
        if not self._rules:
            return 0
            
        rule_signals = []
        rule_weights = []
        
        for name, rule in self._rules.items():
            if hasattr(rule, 'evaluate') and rule.ready:
                signal, strength = rule.evaluate(bar_data)
                if signal != 0:
                    rule_signals.append(signal * strength)
                    rule_weights.append(self._component_weights.get(name, 1.0))
                    
        if not rule_signals:
            return 0
            
        # Aggregate signals based on method
        return self._aggregate_signals(rule_signals, rule_weights)
        
    def _aggregate_signals(self, signals: List[float], weights: List[float]) -> int:
        """Aggregate multiple signals into final signal."""
        if self._aggregation_method == 'weighted':
            # Weighted average
            weighted_sum = sum(s * w for s, w in zip(signals, weights))
            weight_sum = sum(weights)
            if weight_sum > 0:
                avg_signal = weighted_sum / weight_sum
                return 1 if avg_signal > 0 else -1 if avg_signal < 0 else 0
                
        elif self._aggregation_method == 'majority':
            # Majority vote
            positive = sum(1 for s in signals if s > 0)
            negative = sum(1 for s in signals if s < 0)
            return 1 if positive > negative else -1 if negative > positive else 0
            
        elif self._aggregation_method == 'unanimous':
            # All must agree
            if all(s > 0 for s in signals):
                return 1
            elif all(s < 0 for s in signals):
                return -1
            else:
                return 0
                
        return 0
        
    def _publish_signal(self, signal: int, bar_data: Dict[str, Any]) -> None:
        """Publish trading signal to event bus."""
        signal_data = {
            'signal_type': signal,  # Changed from 'signal' to match risk manager
            'symbol': bar_data.get('symbol'),
            'timestamp': bar_data.get('timestamp'),
            'price_at_signal': bar_data.get('close'),  # Changed from 'price' to match risk manager
            'strategy_id': self.instance_name,  # Changed from 'strategy' to match risk manager
            'regime': self._current_regime,
            'bars_processed': self._bars_processed
        }
        
        event = Event(EventType.SIGNAL, signal_data)
        if self.event_bus:
            self.event_bus.publish(event)
            
        self._last_signal = signal
        self.logger.info(f"Strategy '{self.instance_name}' generated signal: {signal}")
        
    # Parameter Management
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all strategy parameters with proper namespacing."""
        return self._parameter_set.to_dict()
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set strategy parameters with namespace handling."""
        self._parameter_set.update(params)
        
        # Propagate to components
        for namespace, component_params in self._parameter_set.get_namespaced_params().items():
            component = self._get_component_by_namespace(namespace)
            if component and hasattr(component, 'set_parameters'):
                component.set_parameters(component_params)
                
    def get_parameter_space(self) -> ParameterSpace:
        """Get combined parameter space for optimization."""
        space = ParameterSpace(f"{self.instance_name}_space")
        
        # Add component parameter spaces
        for name, indicator in self._indicators.items():
            if hasattr(indicator, 'get_parameter_space'):
                space.add_subspace(f"indicators.{name}", indicator.get_parameter_space())
                
        for name, rule in self._rules.items():
            if hasattr(rule, 'get_parameter_space'):
                space.add_subspace(f"rules.{name}", rule.get_parameter_space())
                
        for name, feature in self._features.items():
            if hasattr(feature, 'get_parameter_space'):
                space.add_subspace(f"features.{name}", feature.get_parameter_space())
                
        # Add strategy-level parameters
        if self._aggregation_method == 'weighted':
            for name in self._rules:
                space.add_parameter(
                    Parameter(
                        name=f"weights.{name}",
                        param_type='continuous',
                        min_value=0.0,
                        max_value=1.0,
                        default=1.0
                    )
                )
                                  
        return space
        
    # Lifecycle Methods
    
    def reset(self) -> None:
        """Reset strategy state."""
        # Reset components
        for component in self._get_all_components():
            if hasattr(component, 'reset'):
                component.reset()
                
        # Reset state
        self._bars_processed = 0
        self._last_signal = None
        self._current_regime = 'default'
        
        self.logger.info(f"Strategy '{self.instance_name}' reset")
        
    # Helper Methods
    
    def _initialize_components_from_config(self, config: Dict[str, Any]) -> None:
        """Initialize components based on configuration."""
        # Initialize indicators
        indicators_config = config.get('indicators', {})
        for name, indicator_config in indicators_config.items():
            indicator_class = indicator_config.get('class', 'MovingAverageIndicator')
            
            # Create indicator based on class name
            if indicator_class == 'MovingAverageIndicator':
                from .indicator import MovingAverageIndicator
                indicator = MovingAverageIndicator(
                    name=name,
                    lookback_period=indicator_config.get('lookback_period', 20)
                )
                self.add_indicator(name, indicator)
            elif indicator_class == 'RSIIndicator':
                from .indicator import RSIIndicator
                indicator = RSIIndicator(
                    name=name,
                    lookback_period=indicator_config.get('lookback_period', 14)
                )
                self.add_indicator(name, indicator)
                
        # Initialize rules
        rules_config = config.get('rules', {})
        for name, rule_config in rules_config.items():
            rule_class = rule_config.get('class', 'CrossoverRule')
            dependencies = rule_config.get('dependencies', [])
            
            # Get dependency components
            dep_components = []
            for dep_name in dependencies:
                if dep_name in self._indicators:
                    dep_components.append(self._indicators[dep_name])
                    
            # Create rule based on class name
            if rule_class == 'CrossoverRule':
                from .rule import CrossoverRule
                rule = CrossoverRule(
                    name=name,
                    dependencies=dep_components,
                    generate_exit_signals=rule_config.get('generate_exit_signals', False)
                )
                weight = config.get('weights', {}).get(name, 1.0)
                self.add_rule(name, rule, weight)
            elif rule_class == 'TrueCrossoverRule':
                from .rules.crossover import TrueCrossoverRule
                rule = TrueCrossoverRule(name=name)
                # Set dependencies
                for i, dep in enumerate(dep_components):
                    if i == 0:
                        rule.add_dependency('fast_ma', dep)
                    elif i == 1:
                        rule.add_dependency('slow_ma', dep)
                # Set parameters
                rule.set_parameters({
                    'generate_exit_signals': rule_config.get('generate_exit_signals', False),
                    'min_separation': rule_config.get('min_separation', 0.0)
                })
                weight = config.get('weights', {}).get(name, 1.0)
                self.add_rule(name, rule, weight)
                
        # Set signal aggregation method
        self._aggregation_method = config.get('signal_aggregation_method', 'weighted')
        
    def _setup_parameters(self) -> None:
        """Set up parameter management for all components."""
        # Collect parameters from all components
        for name, indicator in self._indicators.items():
            self._update_parameter_namespace(f"indicators.{name}", indicator)
            
        for name, rule in self._rules.items():
            self._update_parameter_namespace(f"rules.{name}", rule)
            
        for name, feature in self._features.items():
            self._update_parameter_namespace(f"features.{name}", feature)
            
    def _update_parameter_namespace(self, namespace: str, component: StrategyComponent) -> None:
        """Update parameter set with component parameters."""
        if hasattr(component, 'get_parameters'):
            params = component.get_parameters()
            for key, value in params.items():
                self._parameter_set.set(f"{namespace}.{key}", value)
                
    def _update_indicators(self, bar_data: Dict[str, Any]) -> None:
        """Update all indicators with new bar data."""
        for indicator in self._indicators.values():
            if hasattr(indicator, 'update'):
                indicator.update(bar_data)
                
    def _update_features(self, bar_data: Dict[str, Any]) -> None:
        """Update all features with new bar data."""
        for feature in self._features.values():
            if hasattr(feature, 'update'):
                feature.update(bar_data)
                
    def _get_all_components(self) -> List[StrategyComponent]:
        """Get all components."""
        components = []
        components.extend(self._indicators.values())
        components.extend(self._rules.values())
        components.extend(self._features.values())
        return components
        
    def _get_component_by_namespace(self, namespace: str) -> Optional[StrategyComponent]:
        """Get component by namespace path."""
        parts = namespace.split('.')
        if len(parts) >= 2:
            component_type = parts[0]
            component_name = parts[1]
            
            if component_type == 'indicators':
                return self._indicators.get(component_name)
            elif component_type == 'rules':
                return self._rules.get(component_name)
            elif component_type == 'features':
                return self._features.get(component_name)
                
        return None
        
    def _on_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Handle regime changes - can be overridden by subclasses."""
        pass