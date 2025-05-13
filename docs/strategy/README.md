# Strategy Module

The Strategy module defines the trading logic that analyzes market data and generates trading signals.

## Strategy Base Class

```
StrategyBase (Abstract)
  └── initialize(context)
  └── initialize_event_subscriptions()
  └── on_bar(event)
  └── calculate_signals(bar)
  └── emit_signal(symbol, direction, price, quantity)
  └── reset()
  └── teardown()
```

The StrategyBase class provides the framework for trading strategies. It handles event subscription management, parameter management, and the core strategy lifecycle.

## Component Architecture

```
Component (Base)
  └── StrategyBase (Abstract)
      └── CompositeStrategy
      └── MultipleTimeframeStrategy
      └── ConcreteStrategies (e.g., SimpleMACrossoverStrategy)
```

Strategies can be composed hierarchically to create complex trading systems with multiple timeframes and instrument combinations.

## Strategy Factory

The StrategyFactory manages strategy discovery, instantiation, and configuration:

- Discovers available strategy implementations
- Creates strategy instances with proper configuration
- Manages strategy dependencies and parameters

## Parameter Management

All strategies support the following parameter management features:

- Default parameters with overrides from configuration
- Runtime parameter validation
- Parameter optimization support
- Hierarchical parameter inheritance (for composed strategies)

## Strategy Implementation Example

```python
class SimpleMACrossoverStrategy(StrategyBase):
    def __init__(self, name, config=None):
        super().__init__(name, config)
        # Get parameters with defaults
        self.fast_window = self.get_parameter('fast_window', 10)
        self.slow_window = self.get_parameter('slow_window', 30)
        self.position_size = self.get_parameter('position_size', 100)
        
        # Initialize state
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
    
    def initialize(self, context):
        super().initialize(context)
        # Get dependencies
        self.data_handler = self._get_dependency(context, 'data_handler')
        
    def initialize_event_subscriptions(self):
        # Subscribe to BAR events
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event):
        # Extract bar data
        bar_data = event.get_data()
        symbol = bar_data['symbol']
        
        # Update price history
        if symbol not in self.prices:
            self.prices[symbol] = []
        
        self.prices[symbol].append(bar_data['close'])
        
        # Calculate moving averages when we have enough data
        if len(self.prices[symbol]) >= self.slow_window:
            # Calculate fast MA
            fast_ma = sum(self.prices[symbol][-self.fast_window:]) / self.fast_window
            self.fast_ma[symbol] = fast_ma
            
            # Calculate slow MA
            slow_ma = sum(self.prices[symbol][-self.slow_window:]) / self.slow_window
            self.slow_ma[symbol] = slow_ma
            
            # Generate signals
            self.calculate_signals(symbol, bar_data)
            
    def calculate_signals(self, symbol, bar_data):
        # Skip if we don't have both MAs
        if symbol not in self.fast_ma or symbol not in self.slow_ma:
            return
            
        # Get current position
        position = self.current_position.get(symbol, 0)
        
        # Check for crossover
        if self.fast_ma[symbol] > self.slow_ma[symbol] and position <= 0:
            # Bullish crossover - emit buy signal
            self.emit_signal(symbol, 'BUY', bar_data['close'], self.position_size)
            self.current_position[symbol] = self.position_size
            
        elif self.fast_ma[symbol] < self.slow_ma[symbol] and position >= 0:
            # Bearish crossover - emit sell signal
            self.emit_signal(symbol, 'SELL', bar_data['close'], self.position_size)
            self.current_position[symbol] = -self.position_size
            
    def reset(self):
        """Reset strategy state."""
        super().reset()
        # Clear state
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
```