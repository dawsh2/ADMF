# Strategy Module Implementation Guide

## Overview

The Strategy module defines the trading logic that analyzes market data and generates trading signals. It provides a framework for implementing, composing, and optimizing trading strategies in the ADMF-Trader system.

## Key Components

1. **Strategy Base Class**
   - Abstract base class for all strategies
   - Standard event handling and parameter management
   - Signal generation and emission

2. **Strategy Components**
   - Reusable trading rule implementations
   - Indicator calculation and management
   - Composable strategy elements

3. **Composite Strategies**
   - Combining multiple strategy components
   - Rule weighting and aggregation
   - Ensemble strategy implementation

4. **Parameter Management**
   - Standardized parameter access and validation
   - Default parameter handling
   - Parameter space definition for optimization

5. **Optimization Interface**
   - Parameter space definition
   - Objective function implementation
   - Strategy evaluation metrics

## Implementation Structure

```
src/strategy/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── strategy.py           # Strategy base class
│   └── parameter_set.py      # Parameter management
├── components/
│   ├── __init__.py
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── moving_averages.py
│   │   ├── oscillators.py
│   │   └── volatility.py
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── trend_rules.py
│   │   ├── breakout_rules.py
│   │   └── volatility_rules.py
│   └── filters/
│       ├── __init__.py
│       └── signal_filters.py
├── strategies/
│   ├── __init__.py
│   ├── ma_crossover.py       # Moving average crossover strategy
│   ├── mean_reversion.py     # Mean reversion strategy
│   └── breakout.py           # Breakout strategy
├── composite/
│   ├── __init__.py
│   ├── composite_strategy.py # Strategy composition framework
│   └── ensemble_strategy.py  # Ensemble strategy implementation
└── optimization/
    ├── __init__.py
    ├── parameter_space.py    # Parameter space definition
    ├── objective.py          # Objective function framework
    └── grid_search.py        # Grid search optimization
```

## Component Specifications

### 1. Strategy Base Class

The Strategy base class provides the foundation for all strategy implementations:

```python
class Strategy(Component):
    """
    Base class for all trading strategies.
    
    Defines the interface and common functionality for strategy components.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        self.indicators = {}
        self.signals = []
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Subscribe to bar events
        if self.event_bus:
            self.initialize_event_subscriptions()
            
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event):
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
        signals = self._calculate_signals(bar_data)
        
        # Emit signal events
        for signal in signals:
            self._emit_signal(signal)
            
    def _update_indicators(self, bar_data):
        """
        Update strategy indicators with new bar data.
        
        Args:
            bar_data: Bar data dictionary
        """
        raise NotImplementedError
        
    def _calculate_signals(self, bar_data):
        """
        Calculate trading signals based on current indicators.
        
        Args:
            bar_data: Bar data dictionary
            
        Returns:
            list: Signal dictionaries
        """
        raise NotImplementedError
        
    def _emit_signal(self, signal_data):
        """
        Emit signal event.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            bool: Success or failure
        """
        # Add strategy name to signal
        signal_data['strategy'] = self.name
        
        # Add timestamp if not present
        if 'timestamp' not in signal_data:
            signal_data['timestamp'] = datetime.now()
            
        # Create and publish event
        event = Event(EventType.SIGNAL, signal_data)
        return self.event_bus.publish(event)
        
    def reset(self):
        """Reset strategy state."""
        super().reset()
        
        # Clear indicators and signals
        self.indicators = {}
        self.signals = []
        
        # Unsubscribe from events
        if hasattr(self, 'subscription_manager'):
            self.subscription_manager.unsubscribe_all()
```

### 2. Parameter Management

The ParameterSet class provides structured parameter management:

```python
class ParameterSet:
    """
    Parameter management for strategies.
    
    Handles parameter validation, type conversion, and access.
    """
    
    def __init__(self, schema, values=None):
        """
        Initialize with schema and values.
        
        Args:
            schema: Parameter schema dictionary
            values: Parameter values dictionary
        """
        self.schema = schema
        self.values = {}
        
        # Set default values
        for name, spec in schema.items():
            self.values[name] = spec.get('default')
            
        # Override with provided values
        if values:
            self.update(values)
            
    def update(self, values):
        """
        Update parameter values.
        
        Args:
            values: New parameter values
            
        Returns:
            dict: Updated parameter values
        """
        for name, value in values.items():
            if name in self.schema:
                self.set(name, value)
                
        return self.values
        
    def set(self, name, value):
        """
        Set a parameter value with validation.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            bool: Success or failure
        """
        # Check if parameter exists in schema
        if name not in self.schema:
            return False
            
        # Get parameter specification
        spec = self.schema[name]
        
        # Validate value
        try:
            converted_value = self._validate_and_convert(name, value, spec)
            self.values[name] = converted_value
            return True
        except ValueError:
            return False
            
    def get(self, name, default=None):
        """
        Get a parameter value.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.values.get(name, default)
        
    def _validate_and_convert(self, name, value, spec):
        """
        Validate and convert parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            spec: Parameter specification
            
        Returns:
            Converted value
            
        Raises:
            ValueError: If validation fails
        """
        # Check type
        expected_type = spec.get('type')
        if expected_type and not isinstance(value, expected_type):
            # Try to convert
            try:
                if expected_type == int:
                    value = int(value)
                elif expected_type == float:
                    value = float(value)
                elif expected_type == bool and isinstance(value, str):
                    value = value.lower() in ('true', 'yes', 'y', '1')
                elif expected_type == str:
                    value = str(value)
                else:
                    raise ValueError(f"Cannot convert {value} to {expected_type}")
            except (ValueError, TypeError):
                raise ValueError(f"Parameter '{name}' must be of type {expected_type}")
                
        # Check min/max constraints
        if 'min' in spec and value < spec['min']:
            raise ValueError(f"Parameter '{name}' must be >= {spec['min']}")
            
        if 'max' in spec and value > spec['max']:
            raise ValueError(f"Parameter '{name}' must be <= {spec['max']}")
            
        # Check allowed values
        if 'allowed_values' in spec and value not in spec['allowed_values']:
            allowed = ', '.join(str(v) for v in spec['allowed_values'])
            raise ValueError(f"Parameter '{name}' must be one of: {allowed}")
            
        return value
```

### 3. Moving Average Crossover Strategy

A concrete strategy implementation example:

```python
class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving average crossover strategy implementation.
    
    Generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Define parameter schema
        self.parameter_schema = {
            'fast_window': {
                'type': int,
                'default': 10,
                'min': 2,
                'max': 200
            },
            'slow_window': {
                'type': int,
                'default': 30,
                'min': 5,
                'max': 500
            },
            'position_size': {
                'type': int,
                'default': 100,
                'min': 1
            }
        }
        
        # Create parameter set
        self.parameters = ParameterSet(self.parameter_schema, parameters)
        
        # Initialize state containers
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
        
    def _update_indicators(self, bar_data):
        """Update indicators with new bar data."""
        symbol = bar_data['symbol']
        close = bar_data['close']
        
        # Initialize price history if needed
        if symbol not in self.prices:
            self.prices[symbol] = []
            
        # Add price to history
        self.prices[symbol].append(close)
        
        # Get MA windows
        fast_window = self.parameters.get('fast_window')
        slow_window = self.parameters.get('slow_window')
        
        # Calculate MAs if we have enough data
        if len(self.prices[symbol]) >= slow_window:
            self.fast_ma[symbol] = sum(self.prices[symbol][-fast_window:]) / fast_window
            self.slow_ma[symbol] = sum(self.prices[symbol][-slow_window:]) / slow_window
            
    def _calculate_signals(self, bar_data):
        """Calculate trading signals based on MA crossover."""
        symbol = bar_data['symbol']
        signals = []
        
        # Skip if we don't have both MAs yet
        if symbol not in self.fast_ma or symbol not in self.slow_ma:
            return signals
            
        # Get current position
        current_position = self.current_position.get(symbol, 0)
        
        # Get MA values
        fast_ma = self.fast_ma[symbol]
        slow_ma = self.slow_ma[symbol]
        
        # Get previous MA values (if available)
        prev_fast_ma = None
        prev_slow_ma = None
        
        if symbol in self.indicators and 'prev_fast_ma' in self.indicators[symbol]:
            prev_fast_ma = self.indicators[symbol]['prev_fast_ma']
            prev_slow_ma = self.indicators[symbol]['prev_slow_ma']
            
        # Store current MAs for next bar
        if symbol not in self.indicators:
            self.indicators[symbol] = {}
            
        self.indicators[symbol]['prev_fast_ma'] = fast_ma
        self.indicators[symbol]['prev_slow_ma'] = slow_ma
        
        # Check for crossover
        if prev_fast_ma is not None and prev_slow_ma is not None:
            # Buy signal - fast MA crosses above slow MA
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and current_position <= 0:
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY',
                    'quantity': self.parameters.get('position_size'),
                    'price': bar_data['close'],
                    'timestamp': bar_data['timestamp'],
                    'reason': 'ma_crossover_up'
                })
                
                # Update position tracking
                self.current_position[symbol] = self.parameters.get('position_size')
                
            # Sell signal - fast MA crosses below slow MA
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and current_position >= 0:
                signals.append({
                    'symbol': symbol,
                    'direction': 'SELL',
                    'quantity': self.parameters.get('position_size'),
                    'price': bar_data['close'],
                    'timestamp': bar_data['timestamp'],
                    'reason': 'ma_crossover_down'
                })
                
                # Update position tracking
                self.current_position[symbol] = -self.parameters.get('position_size')
                
        return signals
        
    def reset(self):
        """Reset strategy state."""
        super().reset()
        
        # Clear state containers
        self.prices = {}
        self.fast_ma = {}
        self.slow_ma = {}
        self.current_position = {}
```

### 4. Composite Strategy

The CompositeStrategy enables combining multiple strategies:

```python
class CompositeStrategy(Strategy):
    """
    Composite strategy that combines multiple sub-strategies.
    
    Allows for combining and weighting signals from multiple strategies.
    """
    
    def __init__(self, name, parameters=None):
        """Initialize with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize strategies list
        self.strategies = []
        
        # Initialize signal aggregation method
        self.aggregation_method = self.parameters.get('aggregation_method', 'majority')
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Initialize sub-strategies
        for strategy in self.strategies:
            strategy.initialize(context)
            
    def add_strategy(self, strategy, weight=1.0):
        """
        Add a sub-strategy to the composite.
        
        Args:
            strategy: Strategy instance
            weight: Strategy weight (default: 1.0)
            
        Returns:
            bool: Success or failure
        """
        # Check if strategy is already initialized
        if self.initialized and not strategy.initialized:
            # Initialize strategy with same context
            context = {
                'event_bus': self.event_bus,
                'logger': self.logger,
                'config': self.config
            }
            strategy.initialize(context)
            
        # Add strategy and weight
        self.strategies.append({
            'strategy': strategy,
            'weight': weight
        })
        
        return True
        
    def _update_indicators(self, bar_data):
        """Update indicators for all sub-strategies."""
        for strategy_info in self.strategies:
            strategy = strategy_info['strategy']
            strategy._update_indicators(bar_data)
            
    def _calculate_signals(self, bar_data):
        """Calculate signals by aggregating sub-strategy signals."""
        all_signals = []
        
        # Collect signals from all sub-strategies
        for strategy_info in self.strategies:
            strategy = strategy_info['strategy']
            weight = strategy_info['weight']
            
            # Get signals from this strategy
            signals = strategy._calculate_signals(bar_data)
            
            # Add weight information
            for signal in signals:
                signal['weight'] = weight
                all_signals.append(signal)
                
        # Aggregate signals
        if self.aggregation_method == 'majority':
            return self._aggregate_majority(all_signals, bar_data)
        elif self.aggregation_method == 'weighted':
            return self._aggregate_weighted(all_signals, bar_data)
        else:
            return all_signals
            
    def _aggregate_majority(self, signals, bar_data):
        """Aggregate signals using majority vote."""
        if not signals:
            return []
            
        # Group signals by symbol and direction
        signal_groups = {}
        for signal in signals:
            symbol = signal['symbol']
            direction = signal['direction']
            key = f"{symbol}_{direction}"
            
            if key not in signal_groups:
                signal_groups[key] = []
                
            signal_groups[key].append(signal)
            
        # Find majority signal for each symbol
        aggregated_signals = []
        for key, signal_group in signal_groups.items():
            if len(signal_group) > len(self.strategies) / 2:
                # Majority agrees on this signal
                # Use the first signal as template
                base_signal = signal_group[0].copy()
                
                # Update with aggregated information
                base_signal['strategy'] = self.name
                base_signal['reason'] = 'majority_vote'
                
                aggregated_signals.append(base_signal)
                
        return aggregated_signals
        
    def _aggregate_weighted(self, signals, bar_data):
        """Aggregate signals using weighted vote."""
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
        total_weight = sum(info['weight'] for info in self.strategies)
        
        for key, group in signal_groups.items():
            # Check if this group has majority weight
            if group['total_weight'] > total_weight / 2:
                # Use the first signal as template
                base_signal = group['signals'][0].copy()
                
                # Update with aggregated information
                base_signal['strategy'] = self.name
                base_signal['reason'] = 'weighted_vote'
                base_signal['weight'] = group['total_weight'] / total_weight
                
                aggregated_signals.append(base_signal)
                
        return aggregated_signals
        
    def start(self):
        """Start the strategy and sub-strategies."""
        super().start()
        
        # Start all sub-strategies
        for strategy_info in self.strategies:
            strategy = strategy_info['strategy']
            strategy.start()
            
    def stop(self):
        """Stop the strategy and sub-strategies."""
        super().stop()
        
        # Stop all sub-strategies
        for strategy_info in self.strategies:
            strategy = strategy_info['strategy']
            strategy.stop()
            
    def reset(self):
        """Reset the strategy and sub-strategies."""
        super().reset()
        
        # Reset all sub-strategies
        for strategy_info in self.strategies:
            strategy = strategy_info['strategy']
            strategy.reset()
```

### 5. Parameter Space for Optimization

The ParameterSpace class defines the optimization search space:

```python
class ParameterSpace:
    """
    Parameter space definition for optimization.
    
    Defines the search space for strategy parameter optimization.
    """
    
    def __init__(self):
        """Initialize parameter space."""
        self.parameters = {}
        
    def add_parameter(self, name, min_value, max_value, step=1, param_type=int):
        """
        Add a parameter to the space.
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            step: Step size
            param_type: Parameter type (int, float)
            
        Returns:
            self: For method chaining
        """
        self.parameters[name] = {
            'min': min_value,
            'max': max_value,
            'step': step,
            'type': param_type
        }
        
        return self
        
    def add_categorical_parameter(self, name, values):
        """
        Add a categorical parameter to the space.
        
        Args:
            name: Parameter name
            values: List of possible values
            
        Returns:
            self: For method chaining
        """
        self.parameters[name] = {
            'type': 'categorical',
            'values': values
        }
        
        return self
        
    def get_combinations(self):
        """
        Get all parameter combinations for grid search.
        
        Returns:
            list: Parameter combinations
        """
        param_values = {}
        
        # Generate values for each parameter
        for name, spec in self.parameters.items():
            if spec['type'] == 'categorical':
                param_values[name] = spec['values']
            else:
                values = []
                current = spec['min']
                
                while current <= spec['max']:
                    values.append(current)
                    
                    if spec['type'] == int:
                        current += spec['step']
                    else:
                        current = round(current + spec['step'], 10)
                        
                param_values[name] = values
                
        # Generate all combinations
        keys = list(param_values.keys())
        combinations = []
        
        # Base case
        if not keys:
            return [{}]
            
        # Generate combinations recursively
        def generate_combinations(index, current_params):
            if index == len(keys):
                combinations.append(current_params.copy())
                return
                
            key = keys[index]
            for value in param_values[key]:
                current_params[key] = value
                generate_combinations(index + 1, current_params)
                
        generate_combinations(0, {})
        return combinations
```

## Best Practices

### Strategy Implementation

When implementing strategies, follow these best practices:

1. **State Management**: Initialize all state containers in the constructor
   ```python
   def __init__(self, name, parameters=None):
       super().__init__(name, parameters)
       self.prices = {}
       self.indicators = {}
       self.current_position = {}
   ```

2. **Parameter Validation**: Define a parameter schema with validation rules
   ```python
   self.parameter_schema = {
       'window_size': {
           'type': int,
           'default': 20,
           'min': 2,
           'max': 200
       }
   }
   ```

3. **Proper Reset**: Clear all state in the reset method
   ```python
   def reset(self):
       super().reset()
       self.prices = {}
       self.indicators = {}
       self.current_position = {}
   ```

4. **Signal Creation**: Use a standard format for signal dictionaries
   ```python
   signal = {
       'symbol': symbol,
       'direction': 'BUY',  # or 'SELL'
       'quantity': 100,
       'price': price,
       'timestamp': timestamp,
       'reason': 'ma_crossover'
   }
   ```

### Event Handling

Use these patterns for event handling:

```python
def initialize_event_subscriptions(self):
    """Set up event subscriptions."""
    self.subscription_manager = SubscriptionManager(self.event_bus)
    self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
    
def on_bar(self, event):
    """Handle bar event."""
    bar_data = event.get_data()
    
    # Update indicators
    self._update_indicators(bar_data)
    
    # Calculate signals
    signals = self._calculate_signals(bar_data)
    
    # Emit signals
    for signal in signals:
        self._emit_signal(signal)
```

### Component Composition

Use the Composite pattern for strategy composition:

```python
# Create composite strategy
composite = CompositeStrategy("ensemble")

# Add sub-strategies with weights
composite.add_strategy(MovingAverageCrossoverStrategy("ma_cross"), weight=0.6)
composite.add_strategy(BreakoutStrategy("breakout"), weight=0.4)

# Initialize and use
composite.initialize(context)
```

## Implementation Considerations

### 1. Indicator Management

Consider these approaches for indicator management:

1. **Efficient Calculation**: Use rolling windows for better performance
   ```python
   # Instead of recalculating the full MA each time
   self.prices[symbol] = self.prices[symbol][-self.max_window:]
   
   # Consider using numpy for vectorized calculations
   import numpy as np
   prices = np.array(self.prices[symbol])
   self.fast_ma[symbol] = np.mean(prices[-fast_window:])
   ```

2. **Indicator Libraries**: Consider using established libraries
   ```python
   # Using TA-Lib or similar library
   import talib
   self.rsi[symbol] = talib.RSI(np.array(self.prices[symbol]), timeperiod=14)[-1]
   ```

3. **Indicator Caching**: Cache calculated indicators to avoid duplication
   ```python
   def _get_indicator(self, symbol, name, func, *args):
       if symbol not in self.indicators:
           self.indicators[symbol] = {}
           
       if name not in self.indicators[symbol]:
           self.indicators[symbol][name] = func(*args)
           
       return self.indicators[symbol][name]
   ```

### 2. Signal Management

For complex signal logic, consider:

1. **Rule-Based Approach**: Split logic into individual rules
   ```python
   # Define rules
   def ma_crossover_rule(self, symbol, data):
       # MA crossover logic
       return signal_or_none
       
   def oversold_rule(self, symbol, data):
       # Oversold condition logic
       return signal_or_none
       
   # Apply rules
   def _calculate_signals(self, bar_data):
       signals = []
       symbol = bar_data['symbol']
       
       # Apply each rule
       for rule in self.rules:
           signal = rule(symbol, bar_data)
           if signal:
               signals.append(signal)
               
       return signals
   ```

2. **Signal Filtering**: Implement signal filtering pipeline
   ```python
   def _filter_signals(self, signals):
       # Apply filters in sequence
       for filter_func in self.filters:
           signals = filter_func(signals)
           
       return signals
   ```

### 3. Optimization Considerations

For strategy optimization, consider:

1. **Parameter Bounds**: Define reasonable parameter bounds
   ```python
   # For a moving average, very large windows rarely make sense
   parameter_space.add_parameter('fast_window', 2, 50, 1, int)
   parameter_space.add_parameter('slow_window', 10, 200, 5, int)
   ```

2. **Objective Function**: Define appropriate performance metrics
   ```python
   def sharpe_ratio_objective(backtest_result):
       # Calculate Sharpe ratio from equity curve
       returns = calculate_returns(backtest_result['equity_curve'])
       sharpe = calculate_sharpe_ratio(returns)
       return sharpe
   ```

3. **Prevent Overfitting**: Use train/test splitting
   ```python
   # Train parameters
   data_handler.set_active_split('train')
   train_results = optimizer.optimize(strategy, parameter_space, objective_function)
   
   # Validate on test set
   data_handler.set_active_split('test')
   test_results = optimizer.evaluate(strategy, train_results['best_parameters'])
   ```

By following these guidelines, you'll create a robust Strategy module that provides a flexible framework for implementing and optimizing trading strategies in the ADMF-Trader system.