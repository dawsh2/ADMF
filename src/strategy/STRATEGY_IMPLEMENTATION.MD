# Strategy Module Implementation Guide

## Overview

The Strategy module defines the trading logic that analyzes market data and generates trading signals in the ADMF-Trader system. It provides a comprehensive framework for implementing, composing, and optimizing trading strategies with a focus on modularity, extensibility, and maintainability. This module forms the decision-making core of the trading system, translating market data into actionable trading signals through configurable, composable strategy components.

## Directory Structure

```
src/strategy/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── strategy.py           # Unified strategy base class
│   └── parameter_set.py      # Parameter management
├── components/
│   ├── __init__.py
│   ├── indicators/           # Technical calculations (MA, RSI, etc.)
│   │   ├── __init__.py
│   │   ├── moving_averages.py
│   │   ├── oscillators.py
│   │   └── volatility.py
│   ├── features/             # Higher-level market characteristics
│   │   ├── __init__.py
│   │   ├── trend.py
│   │   ├── momentum.py
│   │   └── regime.py
│   └── rules/                # Trading decision logic
│       ├── __init__.py
│       ├── trend_rules.py
│       ├── breakout_rules.py
│       └── volatility_rules.py
├── implementations/          # Strategy implementations
│   ├── __init__.py
│   ├── ma_crossover.py       # Moving average crossover strategy
│   ├── mean_reversion.py     # Mean reversion strategy
│   ├── breakout.py           # Breakout strategy
│   └── multi_timeframe.py    # Multi-timeframe strategy
├── factory/
│   ├── __init__.py
│   └── strategy_factory.py   # Strategy discovery and configuration
├── lifecycle/
│   ├── __init__.py
│   ├── parameter_store.py    # Parameter version storage
│   ├── versioned_parameters.py # Versioned parameter set
│   └── deployment_manager.py # Deployment management
└── optimization/
    ├── __init__.py
    ├── parameter_space.py    # Parameter space definition
    ├── objective.py          # Objective function framework
    └── grid_search.py        # Grid search optimization
```

## Core Interfaces

### StrategyBase Interface

The `StrategyBase` interface defines the contract that all strategies must implement, based on the system's interface-based design:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from core.base import ComponentBase

class StrategyBase(ComponentBase, ABC):
    """Base interface for trading strategies."""
    
    @abstractmethod
    def on_bar(self, event: Dict[str, Any]) -> None:
        """
        Process bar event.
        
        Args:
            event: Bar event data
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate signals based on bar data.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            List of signal dictionaries
        """
        pass
    
    @abstractmethod
    def emit_signal(self, symbol: str, direction: str, quantity: float, 
                   price: float, **kwargs) -> bool:
        """
        Emit a signal event.
        
        Args:
            symbol: Instrument symbol
            direction: Signal direction ('BUY' or 'SELL')
            quantity: Signal quantity
            price: Signal price
            **kwargs: Additional signal parameters
            
        Returns:
            bool: Whether signal was successfully emitted
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
    
    @property
    @abstractmethod
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Get strategy parameter space for optimization."""
        pass
```

### Component Interfaces

The component interfaces define the contracts for strategy components:

```python
class IndicatorBase(ComponentBase, ABC):
    """Base interface for technical indicators."""
    
    @abstractmethod
    def calculate(self, data) -> Any:
        """
        Calculate indicator value(s).
        
        Args:
            data: Price/volume data to calculate indicator from
            
        Returns:
            Calculated indicator values
        """
        pass
    
    @abstractmethod
    def update(self, value: float) -> float:
        """
        Update indicator with a new value.
        
        Args:
            value: New data point
            
        Returns:
            Updated indicator value
        """
        pass
    
    @property
    @abstractmethod
    def value(self) -> float:
        """Get current indicator value."""
        pass
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether indicator has enough data to provide valid values."""
        pass
```

```python
class RuleBase(ComponentBase, ABC):
    """Base interface for trading rules."""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate rule on data.
        
        Args:
            data: Data dictionary with market/indicator values
            
        Returns:
            Tuple of (rule_triggered, signal_strength)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get rule parameters.
        
        Returns:
            Dict of parameter names to values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set rule parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        pass
    
    @property
    @abstractmethod
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Get rule parameter space for optimization."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Get rule weight for composite strategies."""
        pass
    
    @weight.setter
    @abstractmethod
    def weight(self, value: float) -> None:
        """Set rule weight."""
        pass
```

```python
class FeatureBase(ComponentBase, ABC):
    """Base interface for higher-level market features."""
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> Any:
        """
        Calculate feature value.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Calculated feature value
        """
        pass
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> Any:
        """
        Update feature with new data.
        
        Args:
            data: New market data
            
        Returns:
            Updated feature value
        """
        pass
    
    @property
    @abstractmethod
    def value(self) -> Any:
        """Get current feature value."""
        pass
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether feature has enough data to provide valid values."""
        pass
```

## Unified Strategy Implementation

The `Strategy` class provides a composable base implementation that can be used for all strategy types:

```python
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from core.base import Component

class Strategy(Component):
    """
    Base class for all trading strategies.
    
    Provides a composable framework for building strategies from components.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        self.components = []  # Can hold rules, features, indicators or other strategies
        self.indicators = {}
        self.signals = []
        
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Get dependencies
        self.data_handler = self._get_dependency(context, 'data_handler')
        
        # Subscribe to bar events
        if self.event_bus:
            self.initialize_event_subscriptions()
            
        # Initialize all components
        self._initialize_components(context)
            
    def initialize_event_subscriptions(self) -> None:
        """Set up event subscriptions."""
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def _initialize_components(self, context: Dict[str, Any]) -> None:
        """Initialize all components."""
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, 'initialize') and not component.initialized:
                component.initialize(context)
        
    def add_component(self, component, weight: float = 1.0) -> 'Strategy':
        """
        Add a sub-component (rule, feature, indicator or strategy) with weight.
        
        Args:
            component: Component to add
            weight: Component weight (default: 1.0)
            
        Returns:
            self: For method chaining
        """
        self.components.append({
            'component': component,
            'weight': weight
        })
        
        # Initialize component if this strategy is already initialized
        if self.initialized and hasattr(component, 'initialize') and not component.initialized:
            context = {
                'event_bus': self.event_bus,
                'logger': self.logger,
                'config': self.config,
                'data_handler': getattr(self, 'data_handler', None)
            }
            component.initialize(context)
            
        return self
        
    def on_bar(self, event: Dict[str, Any]) -> None:
        """
        Process bar event and generate signals.
        
        Args:
            event: Bar event
        """
        # Extract bar data
        bar_data = event.get_data()
        
        # Update indicators
        self._update_indicators(bar_data)
        
        # Calculate signals
        signals = self.calculate_signals(bar_data)
        
        # Emit signal events
        for signal in signals:
            self.emit_signal(
                signal['symbol'], 
                signal['direction'], 
                signal['quantity'], 
                signal['price'],
                **{k: v for k, v in signal.items() 
                   if k not in ['symbol', 'direction', 'quantity', 'price']}
            )
            
    def _update_indicators(self, bar_data: Dict[str, Any]) -> None:
        """
        Update strategy indicators with new bar data.
        
        Args:
            bar_data: Bar data dictionary
        """
        # Update own indicators
        self._update_own_indicators(bar_data)
        
        # Update component indicators
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, '_update_indicators'):
                component._update_indicators(bar_data)
            elif hasattr(component, 'update'):
                component.update(bar_data)
                
    def _update_own_indicators(self, bar_data: Dict[str, Any]) -> None:
        """
        Update strategy's own indicators (to be overridden).
        
        Args:
            bar_data: Bar data dictionary
        """
        pass
        
    def calculate_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate trading signals based on current indicators.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            list: Signal dictionaries
        """
        # Calculate own signals
        signals = self._calculate_own_signals(bar_data)
        
        # Collect component signals
        for component_info in self.components:
            component = component_info['component']
            weight = component_info['weight']
            
            # Get signals from components
            if hasattr(component, 'calculate_signals'):
                component_signals = component.calculate_signals(bar_data)
                for signal in component_signals:
                    signal['weight'] = weight
                    signals.append(signal)
            elif hasattr(component, 'evaluate'):
                # Handle rule components
                triggered, strength = component.evaluate(bar_data)
                if triggered:
                    symbol = bar_data.get('symbol')
                    signals.append({
                        'symbol': symbol,
                        'direction': self._determine_direction(component, strength),
                        'quantity': self._determine_quantity(component, strength),
                        'price': bar_data.get('close'),
                        'weight': weight * abs(strength),
                        'timestamp': bar_data.get('timestamp', datetime.now()),
                        'reason': f"rule_{component.name}"
                    })
                
        # Optionally aggregate signals if needed
        if self.parameters.get('aggregate_signals', False) and signals:
            signals = self._aggregate_signals(signals, bar_data)
            
        return signals
        
    def _calculate_own_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate strategy's own signals (to be overridden).
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            list: Signal dictionaries
        """
        return []
        
    def _determine_direction(self, rule, strength: float) -> str:
        """
        Determine signal direction based on rule and strength.
        
        Args:
            rule: Rule component
            strength: Signal strength
            
        Returns:
            Direction string ('BUY' or 'SELL')
        """
        return 'BUY' if strength > 0 else 'SELL'
        
    def _determine_quantity(self, rule, strength: float) -> float:
        """
        Determine signal quantity based on rule and strength.
        
        Args:
            rule: Rule component
            strength: Signal strength
            
        Returns:
            Quantity value
        """
        base_size = self.parameters.get('position_size', 100)
        return base_size
        
    def _aggregate_signals(self, signals: List[Dict[str, Any]], 
                          bar_data: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate signals using configured method.
        
        Args:
            signals: List of signals to aggregate
            bar_data: Bar data dictionary
            
        Returns:
            list: Aggregated signals
        """
        method = self.parameters.get('aggregation_method', 'weighted')
        
        if method == 'weighted':
            return self._aggregate_weighted(signals, bar_data)
        elif method == 'majority':
            return self._aggregate_majority(signals, bar_data)
        elif method == 'unanimous':
            return self._aggregate_unanimous(signals, bar_data)
        else:
            return signals
        
    def _aggregate_weighted(self, signals: List[Dict[str, Any]], 
                          bar_data: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate signals using weighted voting.
        
        Args:
            signals: List of signals to aggregate
            bar_data: Bar data dictionary
            
        Returns:
            list: Aggregated signals
        """
        if not signals:
            return []
            
        # Group signals by symbol and direction
        signal_groups = {}
        for signal in signals:
            symbol = signal['symbol']
            direction = signal['direction']
            key = f"{symbol}_{direction}"
            weight = signal.get('weight', 1.0)
            
            if key not in signal_groups:
                signal_groups[key] = {
                    'signals': [],
                    'total_weight': 0
                }
                
            signal_groups[key]['signals'].append(signal)
            signal_groups[key]['total_weight'] += weight
            
        # Find weighted majority for each symbol
        aggregated_signals = []
        total_weight = sum(info['weight'] for info in self.components) or 1.0
        threshold = self.parameters.get('weight_threshold', 0.5)
        
        for key, group in signal_groups.items():
            # Check if this group has majority weight
            if group['total_weight'] > total_weight * threshold:
                # Use the first signal as template
                base_signal = group['signals'][0].copy()
                
                # Update with aggregated information
                base_signal['strategy'] = self.name
                base_signal['reason'] = 'weighted_vote'
                base_signal['weight'] = group['total_weight'] / total_weight
                
                # Combine quantities proportionally
                quantity = sum(signal.get('quantity', 0) * signal.get('weight', 1.0) 
                              for signal in group['signals']) / group['total_weight']
                base_signal['quantity'] = quantity
                
                aggregated_signals.append(base_signal)
                
        return aggregated_signals
        
    def emit_signal(self, symbol: str, direction: str, quantity: float, 
                   price: float, **kwargs) -> bool:
        """
        Emit a signal event.
        
        Args:
            symbol: Instrument symbol
            direction: Signal direction ('BUY' or 'SELL')
            quantity: Signal quantity
            price: Signal price
            **kwargs: Additional signal parameters
            
        Returns:
            bool: Whether signal was successfully emitted
        """
        if not self.event_bus:
            return False
            
        # Create signal data
        signal_data = {
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'timestamp': kwargs.get('timestamp', datetime.now()),
            'strategy': self.name,
            'reason': kwargs.get('reason', 'unspecified')
        }
        
        # Add additional data
        for key, value in kwargs.items():
            if key not in signal_data:
                signal_data[key] = value
                
        # Create and emit event
        event = Event(EventType.SIGNAL, signal_data)
        return self.event_bus.publish(event)
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dict of parameter names to values
        """
        return self.parameters.copy()
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        # Update parameters
        self.parameters.update(parameters)
        
        # Update components if parameters are for them
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, 'set_parameters'):
                component_name = getattr(component, 'name', None)
                if component_name:
                    # Find parameters for this component
                    component_params = {}
                    prefix = f"{component_name}."
                    for key, value in parameters.items():
                        if key.startswith(prefix):
                            component_params[key[len(prefix):]] = value
                    
                    # Update component parameters if any found
                    if component_params:
                        component.set_parameters(component_params)
                
        # Reset state
        self.reset()
        
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get strategy parameter space for optimization.
        
        Returns:
            Dict of parameter names to possible values
        """
        # Get own parameter space
        space = self._get_own_parameter_space()
        
        # Add component parameter spaces
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, 'parameter_space'):
                component_name = getattr(component, 'name', f"component_{len(space)}")
                component_space = component.parameter_space
                
                # Add with component name prefix
                for param_name, param_values in component_space.items():
                    space[f"{component_name}.{param_name}"] = param_values
                    
        return space
        
    def _get_own_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get strategy's own parameter space (to be overridden).
        
        Returns:
            Dict of parameter names to possible values
        """
        return {}
        
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        
        # Clear indicators and signals
        self.indicators = {}
        self.signals = []
        
        # Reset all components
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, 'reset'):
                component.reset()
        
        # Unsubscribe from events
        if hasattr(self, 'subscription_manager'):
            self.subscription_manager.unsubscribe_all()
            
    def teardown(self) -> None:
        """Clean up resources and finalize strategy state."""
        # Unsubscribe from events
        if hasattr(self, 'subscription_manager'):
            self.subscription_manager.unsubscribe_all()
            
        # Teardown all components
        for component_info in self.components:
            component = component_info['component']
            if hasattr(component, 'teardown'):
                component.teardown()
                
        # Call parent teardown
        super().teardown()
```

## Component Implementations

### 1. Indicator Components

Indicators are the base building blocks for technical analysis:

```python
class MovingAverage(IndicatorBase):
    """
    Moving average indicator.
    
    Calculates simple moving average over a specified window.
    """
    
    def __init__(self, window: int, name: str = None):
        """
        Initialize with window size.
        
        Args:
            window: Window size
            name: Optional name override
        """
        super().__init__(name or f"ma_{window}")
        self.window = window
        self.values = []
        self._current_value = None
        
    def calculate(self, data) -> float:
        """
        Calculate MA from data.
        
        Args:
            data: List of price values
            
        Returns:
            float: Moving average value
        """
        if len(data) < self.window:
            return None
            
        return sum(data[-self.window:]) / self.window
        
    def update(self, value: float) -> float:
        """
        Update indicator with new value.
        
        Args:
            value: New price value
            
        Returns:
            float: Updated MA value
        """
        self.values.append(value)
        
        # Limit history size
        if len(self.values) > self.window + 10:  # Keep extra values for efficiency
            self.values = self.values[-(self.window + 10):]
            
        # Calculate new value if we have enough data
        if len(self.values) >= self.window:
            self._current_value = sum(self.values[-self.window:]) / self.window
            
        return self._current_value
        
    @property
    def value(self) -> float:
        """Get current MA value."""
        return self._current_value
        
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return len(self.values) >= self.window and self._current_value is not None
```

### 2. Feature Components

Features are higher-level abstractions built on indicators:

```python
class TrendStrength(FeatureBase):
    """
    Trend strength feature.
    
    Calculates trend strength based on multiple indicators.
    """
    
    def __init__(self, short_ma: MovingAverage, long_ma: MovingAverage, name: str = None):
        """
        Initialize with indicators.
        
        Args:
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            name: Optional name override
        """
        super().__init__(name or "trend_strength")
        self.short_ma = short_ma
        self.long_ma = long_ma
        self._current_value = None
        
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate trend strength.
        
        Args:
            data: Market data dictionary
            
        Returns:
            float: Trend strength value (-1.0 to 1.0)
        """
        # Make sure we have indicators calculated
        if not self.short_ma.ready or not self.long_ma.ready:
            return None
            
        # Calculate trend strength as normalized difference between MAs
        short_value = self.short_ma.value
        long_value = self.long_ma.value
        
        if long_value == 0:
            return 0
            
        # Calculate normalized difference
        diff = (short_value - long_value) / long_value
        
        # Normalize to -1.0 to 1.0 range
        strength = max(min(diff * 10, 1.0), -1.0)
        
        self._current_value = strength
        return strength
        
    def update(self, data: Dict[str, Any]) -> float:
        """
        Update feature with new data.
        
        Args:
            data: New market data
            
        Returns:
            float: Updated trend strength
        """
        # Update indicators
        price = data.get('close')
        if price is None:
            return self._current_value
            
        self.short_ma.update(price)
        self.long_ma.update(price)
        
        # Calculate new trend strength
        return self.calculate(data)
        
    @property
    def value(self) -> float:
        """Get current trend strength."""
        return self._current_value
        
    @property
    def ready(self) -> bool:
        """Whether feature has enough data."""
        return self.short_ma.ready and self.long_ma.ready and self._current_value is not None
```

### 3. Rule Components

Rules define trading decision logic:

```python
class TrendFollowingRule(RuleBase):
    """
    Trend following rule.
    
    Generates signals based on trend direction and strength.
    """
    
    def __init__(self, trend_feature: TrendStrength, threshold: float = 0.2, name: str = None):
        """
        Initialize with trend feature.
        
        Args:
            trend_feature: Trend strength feature
            threshold: Signal threshold (0.0 to 1.0)
            name: Optional name override
        """
        super().__init__(name or "trend_following")
        self.trend_feature = trend_feature
        self.threshold = threshold
        self._weight = 1.0
        self._parameter_space = {
            'threshold': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate rule on data.
        
        Args:
            data: Data dictionary with market values
            
        Returns:
            Tuple of (rule_triggered, signal_strength)
        """
        if not self.trend_feature.ready:
            return False, 0.0
            
        trend_strength = self.trend_feature.value
        
        # Check if trend strength exceeds threshold
        if abs(trend_strength) >= self.threshold:
            return True, trend_strength
            
        return False, 0.0
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get rule parameters.
        
        Returns:
            Dict of parameter names to values
        """
        return {
            'threshold': self.threshold
        }
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set rule parameters.
        
        Args:
            parameters: Dict of parameter names to values
        """
        if 'threshold' in parameters:
            self.threshold = parameters['threshold']
            
    @property
    def parameter_space(self) -> Dict[str, List[Any]]:
        """Get rule parameter space for optimization."""
        return self._parameter_space
        
    @property
    def weight(self) -> float:
        """Get rule weight."""
        return self._weight
        
    @weight.setter
    def weight(self, value: float) -> None:
        """Set rule weight."""
        self._weight = value
```

## Strategy Implementation Example

Below is an example of a strategy implementation using the new composable architecture:

```python
class MACrossoverStrategy(Strategy):
    """
    Moving average crossover strategy implementation.
    
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(self, name: str = "ma_crossover", parameters: Dict[str, Any] = None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Get parameters
        self.fast_window = self.parameters.get('fast_window', 10)
        self.slow_window = self.parameters.get('slow_window', 30)
        self.position_size = self.parameters.get('position_size', 100)
        
        # Create indicators
        self.fast_ma = MovingAverage(self.fast_window, "fast_ma")
        self.slow_ma = MovingAverage(self.slow_window, "slow_ma")
        
        # Create feature
        self.trend = TrendStrength(self.fast_ma, self.slow_ma, "ma_trend")
        
        # Create rule
        self.trend_rule = TrendFollowingRule(self.trend, 0.0, "ma_crossover_rule")
        
        # Add components
        self.add_component(self.fast_ma)
        self.add_component(self.slow_ma)
        self.add_component(self.trend)
        self.add_component(self.trend_rule)
        
        # Initialize state
        self.current_position = {}
        
    def _update_own_indicators(self, bar_data: Dict[str, Any]) -> None:
        """Update strategy's own indicators."""
        # Indicator updates are handled by components
        pass
        
    def _calculate_own_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate strategy's own signals."""
        # Most signal generation is handled by the rule components
        # This method can be used for custom signal logic
        return []
        
    def calculate_signals(self, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate trading signals based on current indicators.
        
        This overrides the parent method to provide additional position tracking
        logic specific to this strategy.
        """
        # Get signals from parent method (including rule signals)
        signals = super().calculate_signals(bar_data)
        
        # Apply position tracking
        symbol = bar_data.get('symbol')
        if not symbol:
            return signals
            
        # Track current position
        for signal in signals:
            if signal['symbol'] == symbol:
                direction = signal['direction']
                quantity = signal['quantity']
                
                if direction == 'BUY':
                    self.current_position[symbol] = quantity
                else:  # SELL
                    self.current_position[symbol] = -quantity
                    
        return signals
        
    def _get_own_parameter_space(self) -> Dict[str, List[Any]]:
        """Get strategy's own parameter space."""
        return {
            'fast_window': list(range(5, 21)),
            'slow_window': list(range(20, 51, 5)),
            'position_size': [50, 100, 200, 500]
        }
```

This example shows how to build a strategy using the composable architecture:
1. Create indicators (MovingAverage)
2. Build features on top of indicators (TrendStrength)
3. Define rules using features (TrendFollowingRule)
4. Compose all components into a strategy
5. Add custom logic as needed

## Multi-Component Strategy Example

Here's an example of a more complex strategy that combines multiple indicators, features, and rules:

```python
class MultiComponentStrategy(Strategy):
    """
    Multi-component strategy combining several technical indicators.
    
    Uses trend following, momentum, and volatility components.
    """
    
    def __init__(self, name: str = "multi_component", parameters: Dict[str, Any] = None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Create indicators
        self.short_ma = MovingAverage(self.parameters.get('short_window', 10))
        self.long_ma = MovingAverage(self.parameters.get('long_window', 30))
        self.rsi = RSI(self.parameters.get('rsi_window', 14))
        self.atr = ATR(self.parameters.get('atr_window', 14))
        
        # Create features
        self.trend = TrendStrength(self.short_ma, self.long_ma)
        self.momentum = Momentum(self.rsi)
        self.volatility = Volatility(self.atr)
        
        # Create rules
        self.trend_rule = TrendFollowingRule(self.trend, 
                                           self.parameters.get('trend_threshold', 0.2))
        self.momentum_rule = MomentumRule(self.momentum, 
                                        self.parameters.get('momentum_threshold', 0.3))
        self.volatility_rule = VolatilityFilter(self.volatility, 
                                              self.parameters.get('volatility_max', 0.03))
        
        # Set rule weights
        self.trend_rule.weight = self.parameters.get('trend_weight', 0.6)
        self.momentum_rule.weight = self.parameters.get('momentum_weight', 0.4)
        self.volatility_rule.weight = self.parameters.get('volatility_weight', 0.2)
        
        # Add all components
        for component in [self.short_ma, self.long_ma, self.rsi, self.atr,
                         self.trend, self.momentum, self.volatility,
                         self.trend_rule, self.momentum_rule, self.volatility_rule]:
            self.add_component(component)
            
        # Set signal aggregation
        self.parameters['aggregate_signals'] = True
        self.parameters['aggregation_method'] = 'weighted'
        self.parameters['weight_threshold'] = 0.5
```

## Parameter Versioning and Lifecycle Management

The framework includes a comprehensive parameter versioning system for managing strategy lifecycle:

```python
class VersionedParameterSet:
    """Immutable, versioned set of strategy parameters."""
    
    def __init__(self, strategy_id: str, parameters: Dict[str, Any], 
                 version: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a versioned parameter set.
        
        Args:
            strategy_id: Identifier of the strategy
            parameters: Strategy parameters
            version: Version identifier (auto-generated if None)
            metadata: Parameter metadata
        """
        self.strategy_id = strategy_id
        self.parameters = dict(parameters)  # Make a copy to ensure immutability
        self.version = version or self._generate_version()
        self.metadata = metadata or {}
        self.creation_timestamp = datetime.now()
        
    def _generate_version(self) -> str:
        """Generate a unique version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.strategy_id}_{timestamp}"
        
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name, default)
        
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get a copy of all parameters."""
        return dict(self.parameters)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get parameter metadata."""
        return dict(self.metadata)
        
    def with_updated_metadata(self, metadata_updates: Dict[str, Any]) -> 'VersionedParameterSet':
        """Create a new instance with updated metadata."""
        new_metadata = {**self.metadata, **metadata_updates}
        return VersionedParameterSet(
            self.strategy_id,
            self.parameters,
            self.version,
            new_metadata
        )
```

### Parameter Repository

```python
class ParameterRepository:
    """Repository for versioned parameter sets."""
    
    def __init__(self, storage_adapter):
        """
        Initialize repository.
        
        Args:
            storage_adapter: Storage backend adapter
        """
        self.storage = storage_adapter
        
    def save(self, parameter_set: VersionedParameterSet) -> str:
        """Save a parameter set."""
        return self.storage.save(parameter_set)
        
    def get_by_version(self, strategy_id: str, version: str) -> Optional[VersionedParameterSet]:
        """Get parameter set by version."""
        return self.storage.get_by_version(strategy_id, version)
        
    def get_latest(self, strategy_id: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Optional[VersionedParameterSet]:
        """Get latest parameter set, optionally filtered by metadata."""
        return self.storage.get_latest(strategy_id, filter_metadata)
        
    def get_version_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get version history for a strategy."""
        return self.storage.get_version_history(strategy_id)
        
    def update_metadata(self, strategy_id: str, version: str, metadata_updates: Dict[str, Any]) -> bool:
        """Update metadata for a parameter set."""
        return self.storage.update_metadata(strategy_id, version, metadata_updates)
```

### Optimization Workflow

```python
class OptimizationWorkflow:
    """Workflow for strategy optimization and validation."""
    
    def __init__(self, parameter_repository, optimizer, validator):
        """
        Initialize workflow.
        
        Args:
            parameter_repository: ParameterRepository instance
            optimizer: Strategy optimizer
            validator: Strategy validator
        """
        self.repository = parameter_repository
        self.optimizer = optimizer
        self.validator = validator
        
    def run_optimization(self, strategy_id, config, train_data, test_data):
        """Run optimization workflow."""
        # Step 1: Run optimization on training data
        optimization_results = self.optimizer.optimize(
            strategy_id=strategy_id,
            config=config,
            data=train_data
        )
        
        # Step 2: Extract best parameters
        best_parameters = optimization_results['best_parameters']
        train_metrics = optimization_results['metrics']
        
        # Step 3: Validate on test data
        validation_results = self.validator.validate(
            strategy_id=strategy_id,
            parameters=best_parameters,
            data=test_data
        )
        
        test_metrics = validation_results['metrics']
        
        # Step 4: Create and store parameter set with metadata
        metadata = {
            ParameterMetadata.STRATEGY_VERSION: config.get('strategy_version', '1.0.0'),
            ParameterMetadata.OPTIMIZATION_DATE: datetime.now().isoformat(),
            ParameterMetadata.OPTIMIZATION_OBJECTIVE: config.get('objective', 'sharpe_ratio'),
            ParameterMetadata.OPTIMIZATION_METHOD: config.get('method', 'grid_search'),
            ParameterMetadata.TRAIN_DATA_START: train_data.get_start_date().isoformat(),
            ParameterMetadata.TRAIN_DATA_END: train_data.get_end_date().isoformat(),
            ParameterMetadata.TEST_DATA_START: test_data.get_start_date().isoformat(),
            ParameterMetadata.TEST_DATA_END: test_data.get_end_date().isoformat(),
            ParameterMetadata.SYMBOLS: ','.join(train_data.get_symbols()),
            ParameterMetadata.TRAIN_SHARPE_RATIO: train_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TRAIN_RETURNS: train_metrics.get('returns', 0),
            ParameterMetadata.TRAIN_DRAWDOWN: train_metrics.get('max_drawdown', 0),
            ParameterMetadata.TEST_SHARPE_RATIO: test_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TEST_RETURNS: test_metrics.get('returns', 0),
            ParameterMetadata.TEST_DRAWDOWN: test_metrics.get('max_drawdown', 0),
            ParameterMetadata.DEPLOYMENT_STATUS: 'pending_approval'
        }
        
        parameter_set = VersionedParameterSet(
            strategy_id=strategy_id,
            parameters=best_parameters,
            metadata=metadata
        )
        
        # Step 5: Save parameter set
        version = self.repository.save(parameter_set)
        
        return {
            'version': version,
            'parameters': best_parameters,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'validation_passed': validation_results['passed']
        }
```

## Factory Pattern

The `StrategyFactory` creates and configures strategy instances:

```python
class StrategyFactory:
    """
    Factory for creating and configuring strategy instances.
    
    Handles strategy discovery, instantiation, and dependency injection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.strategy_classes = {}
        self.discover_strategies()
        
    def discover_strategies(self) -> None:
        """Discover available strategy implementations."""
        # Import strategies module
        import implementations
        
        # Find strategy classes
        for name in dir(implementations):
            item = getattr(implementations, name)
            
            # Check if item is a class that inherits from Strategy
            if (isinstance(item, type) and 
                issubclass(item, Strategy) and 
                item != Strategy):
                self.strategy_classes[name] = item
                
    def create_strategy(self, name: str, config: Optional[Dict[str, Any]] = None) -> Strategy:
        """
        Create a strategy instance.
        
        Args:
            name: Strategy name or class name
            config: Optional configuration
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        # Get strategy class
        strategy_class = self.get_strategy_class(name)
        
        if not strategy_class:
            raise ValueError(f"Strategy '{name}' not found")
            
        # Create instance
        strategy_config = config or self.config.get('strategies', {}).get(name, {})
        instance = strategy_class(name, strategy_config.get('parameters'))
        
        # Build component tree
        if 'components' in strategy_config:
            for component_config in strategy_config['components']:
                component_type = component_config.get('type')
                component_name = component_config.get('name')
                component_params = component_config.get('parameters', {})
                component_weight = component_config.get('weight', 1.0)
                
                # Create component
                component = self._create_component(
                    component_type, 
                    component_name, 
                    component_params
                )
                
                # Add to strategy
                instance.add_component(component, component_weight)
                
        return instance
        
    def _create_component(self, component_type: str, 
                         name: str, parameters: Dict[str, Any]) -> Any:
        """Create a strategy component."""
        if component_type == 'indicator':
            return self._create_indicator(name, parameters)
        elif component_type == 'feature':
            return self._create_feature(name, parameters)
        elif component_type == 'rule':
            return self._create_rule(name, parameters)
        elif component_type == 'strategy':
            return self.create_strategy(name, {'parameters': parameters})
        else:
            raise ValueError(f"Unknown component type: {component_type}")
```

## Best Practices

### 1. Component Composition

Use the composable architecture to build strategies from components:

```python
# Create strategy with components
strategy = Strategy("my_strategy")

# Add indicators
strategy.add_component(MovingAverage(10))
strategy.add_component(MovingAverage(30))
strategy.add_component(RSI(14))

# Add features
strategy.add_component(TrendStrength(...))
strategy.add_component(Momentum(...))

# Add rules with weights
strategy.add_component(TrendFollowingRule(...), weight=0.6)
strategy.add_component(MomentumRule(...), weight=0.4)
```

### 2. Parameter Management

Define a consistent parameter schema for your components:

```python
self._parameter_space = {
    'fast_window': list(range(5, 21)),
    'slow_window': list(range(20, 51, 5)),
    'position_size': [50, 100, 200, 500],
    'trend_threshold': [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

### 3. Memory Optimization

Use efficient data structures and limit history:

```python
# Limit price history to what's needed
max_window = max(self.fast_window, self.slow_window)
if len(self.prices[symbol]) > max_window + 10:  # Keep a few extra for efficiency
    self.prices[symbol] = self.prices[symbol][-max_window-10:]

# Use numpy for vector operations when possible
import numpy as np
prices = np.array(self.prices[symbol])
self.fast_ma[symbol] = np.mean(prices[-fast_window:])
```

### 4. Preventing Overfitting

Use proper train/test splitting and validation:

```python
# Train parameters
data_handler.set_active_split('train')
train_results = optimizer.optimize(strategy, parameter_space, objective_function)

# Validate on test set
data_handler.set_active_split('test')
test_results = optimizer.evaluate(strategy, train_results['best_parameters'])

# Check validation threshold
validation_threshold = 0.7  # 70% of training performance
relative_performance = test_results['sharpe_ratio'] / train_results['best_score']
if relative_performance < validation_threshold:
    print(f"Warning: Potential overfitting - test performance is {relative_performance*100:.1f}% of training performance")
```

### 5. Compatibility with Optimization

Ensure all components support the optimization interface:

```python
# Every component should implement:
def get_parameters(self) -> Dict[str, Any]:
    """Get component parameters."""
    pass
    
def set_parameters(self, parameters: Dict[str, Any]) -> None:
    """Set component parameters."""
    pass
    
@property
def parameter_space(self) -> Dict[str, List[Any]]:
    """Get component parameter space for optimization."""
    pass
```

## Implementation Timeline

### Phase 1: Core Framework (Week 1)
   - Implement unified Strategy base class
   - Create component interfaces (IndicatorBase, FeatureBase, RuleBase)
   - Develop event handling system
   - Implement component architecture

### Phase 2: Component Library (Week 2)
   - Implement indicator components
   - Create feature components
   - Develop rule components
   - Create unit tests for components

### Phase 3: Strategy Implementations (Week 3)
   - Implement example strategies using component architecture
   - Create multi-component strategies
   - Develop strategy factory
   - Write integration tests for strategies

### Phase 4: Optimization Framework (Week 4)
   - Implement parameter space definition
   - Create optimization targets for components
   - Develop optimization methods
   - Implement optimization metrics

### Phase 5: Parameter Versioning (Week 5)
   - Implement versioned parameter sets
   - Create parameter repository
   - Develop optimization workflow
   - Implement deployment management system

### Phase 6: Testing & Integration (Week 6)
   - Create unit tests for all components
   - Integrate with data and execution modules
   - Benchmark performance
   - Develop documentation and examples