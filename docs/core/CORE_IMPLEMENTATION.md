# Core Module Implementation Guide

## Overview

The Core module is the foundation of the ADMF-Trader system. It provides essential infrastructure services that all other modules depend on, implementing critical patterns for component lifecycle, configuration, dependency injection, event management, and performance analytics.

## Key Components

1. **Component Lifecycle System**
   - Base class for all system components
   - Standard lifecycle methods (initialize, start, stop, reset, teardown)
   - Dependency injection through context
   - Event subscription management

2. **Dependency Injection Container**
   - Component registration and resolution
   - Singleton and transient component management
   - Circular dependency detection
   - Factory support for dynamic component creation

3. **Configuration System**
   - YAML/JSON configuration loading
   - Environment variable support
   - Hierarchical parameter access
   - Schema validation and type conversion

4. **Event System**
   - Standardized event types
   - Publish/subscribe mechanism
   - Event context isolation
   - Thread-safe event handling

5. **Bootstrap System**
   - System initialization and orchestration
   - Configuration loading and validation
   - Component discovery and registration
   - Lifecycle management

6. **Analytics Submodule**
   - Performance measurement and reporting
   - Trade statistics calculation
   - Risk and return metrics
   - Visualization and reporting

## Implementation Structure

```
src/core/
├── __init__.py
├── bootstrap/
│   ├── __init__.py
│   └── system_bootstrap.py        # Main bootstrap implementation
├── component/
│   ├── __init__.py
│   └── component.py               # Base component class
├── config/
│   ├── __init__.py
│   ├── config.py                  # Configuration system
│   └── parameter_store.py         # Parameter storage and retrieval
├── container/
│   ├── __init__.py
│   └── container.py               # DI container
├── events/
│   ├── __init__.py
│   ├── event.py                   # Event base class
│   ├── event_bus.py               # Event bus implementation
│   ├── event_types.py             # Standard event type definitions
│   └── event_context.py           # Event isolation context
├── logging/
│   ├── __init__.py
│   └── logger.py                  # Logging implementation
├── analytics/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── returns.py             # Return-based metrics
│   │   ├── risk.py                # Risk metrics
│   │   └── trading.py             # Trading metrics
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── html_report.py         # HTML report generation
│   │   └── text_report.py         # Text report generation
│   └── visualization/
│       ├── __init__.py
│       └── charts.py              # Chart generation
└── utils/
    ├── __init__.py
    ├── validation.py              # Validation utilities
    └── thread_safe_collections.py # Thread-safe collections
```

## Component Specifications

### 1. Component Base Class

The Component base class serves as the foundation for all system components, enforcing a consistent lifecycle pattern:

```python
class Component:
    """Base class for all system components."""
    
    def __init__(self, name, parameters=None):
        """Initialize component with name and parameters."""
        self.name = name
        self.parameters = parameters or {}
        self.initialized = False
        self.running = False
        
    def initialize(self, context):
        """Set up component with dependencies from context."""
        self.event_bus = context.get('event_bus')
        self.logger = context.get('logger')
        self.config = context.get('config')
        
        # Initialize event subscriptions
        if self.event_bus:
            self.initialize_event_subscriptions()
            
        self.initialized = True
        
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        pass
        
    def start(self):
        """Begin component operation."""
        if not self.initialized:
            raise ComponentError("Component must be initialized before starting")
            
        self.running = True
        
    def stop(self):
        """End component operation."""
        self.running = False
        
    def reset(self):
        """Clear component state for a new run."""
        pass
        
    def teardown(self):
        """Release resources."""
        # Unsubscribe from events
        if self.event_bus:
            self.event_bus.unsubscribe_all(self)
            
        self.initialized = False
        self.running = False
```

### 2. Event System

The event system enables loosely coupled communication between components:

```python
class EventType(Enum):
    """Standard event types for the system."""
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO = "PORTFOLIO"
    BACKTEST_START = "BACKTEST_START"
    BACKTEST_END = "BACKTEST_END"
```

```python
class Event:
    """Event message passed between components."""
    
    def __init__(self, event_type, data=None, timestamp=None, context=None):
        """Initialize event with type, data, and metadata."""
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.context = context
        
    def get_type(self):
        """Get event type."""
        return self.event_type
        
    def get_data(self):
        """Get event data."""
        return self.data
```

```python
class EventBus:
    """Central event distribution system."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers = {}
        self.context_subscribers = {}
        self._lock = threading.RLock()
        
    def publish(self, event):
        """Publish an event to subscribers."""
        event_type = event.get_type()
        
        with self._lock:
            # Get handlers
            handlers = []
            
            # Add global handlers
            if event_type in self.subscribers:
                handlers.extend(self.subscribers[event_type])
                
            # Add context-specific handlers
            context = event.get_context()
            if context and context.name in self.context_subscribers:
                if event_type in self.context_subscribers[context.name]:
                    handlers.extend(self.context_subscribers[context.name][event_type])
                    
            # Notify handlers
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but continue
                    print(f"Error in event handler: {e}")
                    
        return len(handlers) > 0
        
    def subscribe(self, event_type, handler, context=None):
        """Subscribe to events of a specific type."""
        with self._lock:
            if context:
                # Context-specific subscription
                if context.name not in self.context_subscribers:
                    self.context_subscribers[context.name] = {}
                    
                if event_type not in self.context_subscribers[context.name]:
                    self.context_subscribers[context.name][event_type] = []
                    
                self.context_subscribers[context.name][event_type].append(handler)
            else:
                # Global subscription
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                    
                self.subscribers[event_type].append(handler)
```

### 3. Analytics Framework

The Analytics submodule provides performance measurement and reporting:

```python
class PerformanceAnalytics(Component):
    """
    Performance analytics component for calculating and reporting trading performance metrics.
    Part of the Core module's Analytics submodule.
    """
    
    def __init__(self, name="performance_analytics", parameters=None):
        super().__init__(name=name, parameters=parameters or {})
        self.metrics = {}
        
    def calculate_returns(self, equity_curve):
        """Calculate return metrics from equity curve."""
        if not equity_curve or len(equity_curve) < 2:
            return {}
            
        # Extract values and dates
        values = [point['portfolio_value'] for point in equity_curve]
        dates = [point['timestamp'] for point in equity_curve]
        
        # Calculate returns
        returns = []
        for i in range(1, len(values)):
            ret = (values[i] / values[i-1]) - 1
            returns.append(ret)
            
        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1
        annualized_return = self._calculate_annualized_return(total_return, dates)
        volatility = np.std(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Store and return metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return self.metrics
        
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio."""
        if not returns:
            return 0
            
        excess_returns = [r - risk_free_rate for r in returns]
        avg_excess_return = sum(excess_returns) / len(excess_returns)
        
        # Avoid division by zero
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0
            
        # Annualize (assuming daily returns)
        sharpe = avg_excess_return / std_dev * np.sqrt(252)
        
        return sharpe
        
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown."""
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def generate_report(self, equity_curve, trades):
        """Generate performance report."""
        # Calculate metrics
        self.calculate_returns(equity_curve)
        
        # Calculate trade statistics
        win_count = sum(1 for trade in trades if trade.get('realized_pnl', 0) > 0)
        loss_count = sum(1 for trade in trades if trade.get('realized_pnl', 0) < 0)
        
        if win_count + loss_count > 0:
            win_rate = win_count / (win_count + loss_count)
        else:
            win_rate = 0
            
        # Format report
        report = {
            'metrics': self.metrics,
            'trade_stats': {
                'total_trades': len(trades),
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate
            }
        }
        
        return report
```

### 4. Thread-Safe Collections

Thread-safe collections for shared state management:

```python
class ThreadSafeDict:
    """Thread-safe dictionary implementation."""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
        
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
            
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
            
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
            
    def __contains__(self, key):
        with self._lock:
            return key in self._dict
            
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
            
    def items(self):
        with self._lock:
            return list(self._dict.items())
            
    def keys(self):
        with self._lock:
            return list(self._dict.keys())
```

```python
class ThreadSafeList:
    """Thread-safe list implementation."""
    
    def __init__(self):
        self._list = []
        self._lock = threading.RLock()
        
    def __getitem__(self, index):
        with self._lock:
            return self._list[index]
            
    def __setitem__(self, index, value):
        with self._lock:
            self._list[index] = value
            
    def __len__(self):
        with self._lock:
            return len(self._list)
            
    def append(self, item):
        with self._lock:
            self._list.append(item)
            
    def extend(self, items):
        with self._lock:
            self._list.extend(items)
            
    def pop(self, index=-1):
        with self._lock:
            return self._list.pop(index)
```

## Best Practices

### Component Lifecycle

All components should follow this lifecycle:

1. **Construction**: Initialize with name and parameters
   ```python
   def __init__(self, name, parameters=None):
       super().__init__(name=name, parameters=parameters or {})
   ```

2. **Initialization**: Set up dependencies
   ```python
   def initialize(self, context):
       super().initialize(context)
       # Extract dependencies
   ```

3. **Start**: Begin operation
   ```python
   def start(self):
       super().start()
       # Start operations
   ```

4. **Stop**: Pause operation
   ```python
   def stop(self):
       super().stop()
       # Stop operations
   ```

5. **Reset**: Clear state for new run
   ```python
   def reset(self):
       super().reset()
       # Clear state
   ```

6. **Teardown**: Clean up resources
   ```python
   def teardown(self):
       super().teardown()
       # Release resources
   ```

### Event Pattern

Components should use this pattern for event handling:

```python
def initialize_event_subscriptions(self):
    """Set up event subscriptions."""
    self.subscription_manager = SubscriptionManager(self.event_bus)
    self.subscription_manager.subscribe(EventType.SIGNAL, self.on_signal)
    
def on_signal(self, event):
    """Handle signal event."""
    # Extract data
    signal_data = event.get_data()
    
    # Process event
    result = self._process_signal(signal_data)
    
    # Publish new event if needed
    if result:
        self.event_bus.publish(Event(EventType.ORDER, result))
```

### Parameter Handling

Components should use this pattern for accessing parameters:

```python
def initialize(self, context):
    super().initialize(context)
    
    # Get parameters with defaults
    self.max_items = self.parameters.get('max_items', 1000)
    self.log_level = self.parameters.get('log_level', 'INFO')
    
    # Validate parameters
    if self.max_items <= 0:
        raise ValueError("max_items must be positive")
```

### Thread Safety

Use these patterns for thread-safe operations:

```python
def update_position(self, symbol, quantity):
    """Thread-safe position update."""
    with self._lock:
        if symbol not in self.positions:
            self.positions[symbol] = 0
            
        self.positions[symbol] += quantity
        
    return self.positions[symbol]
```

## Implementation Considerations

1. **Memory Management**:
   - Use bounded collections for history
   - Implement pruning for time-series data
   - Use weak references for event handlers when appropriate

2. **Error Handling**:
   - Use structured logging for errors
   - Implement proper exception hierarchy
   - Add context to exceptions for easier debugging

3. **Performance**:
   - Use profiling to identify bottlenecks
   - Implement caching for expensive calculations
   - Use batch processing for event handling when appropriate

4. **Testing**:
   - Create mock implementations of interfaces
   - Use dependency injection for testability
   - Implement test contexts for isolation

## Event Isolation Between Optimization Runs

Event isolation is critical for reliable optimization, ensuring that events from one run don't leak into another. Here's a detailed explanation of the isolation mechanism:

### Event Context Implementation

The EventContext class provides the foundation for event isolation:

```python
class EventContext:
    """Context for event isolation between runs."""
    
    def __init__(self, name):
        """Initialize event context with name."""
        self.name = name
        
    def __enter__(self):
        """Enter context and activate it."""
        # Set as current context
        _current_context.set(self)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and deactivate it."""
        # Clear current context
        _current_context.set(None)
```

### Proper Setup for Train/Test Isolation

Follow this pattern for clean train/test isolation:

```python
def optimize_strategy(parameter_space, data_handler):
    """Run optimization with proper train/test isolation."""
    best_params = None
    best_train_result = None
    best_test_result = None
    
    # Set up data splitter
    data_handler.setup_train_test_split(method='ratio', train_ratio=0.7)
    
    # Iterate through parameter combinations
    for params in parameter_space.get_combinations():
        # First run on training data
        data_handler.set_active_split('train')
        with EventContext("train") as train_context:
            train_result = run_backtest(params, context=train_context)
        
        # Then validate on test data
        data_handler.set_active_split('test')
        with EventContext("test") as test_context:
            test_result = run_backtest(params, context=test_context)
        
        # Evaluate results with focus on test performance
        if is_better_result(test_result, best_test_result):
            best_params = params
            best_train_result = train_result
            best_test_result = test_result
    
    return best_params, best_train_result, best_test_result
```

### Preventing Data Leakage

To prevent data leakage between contexts:

1. **Event Bus Implementation**:
   ```python
   def publish(self, event):
       """Publish event to appropriate handlers based on context."""
       # Get current context
       current_context = get_current_context()
       
       # Set event context if not already set
       if not hasattr(event, 'context') or event.context is None:
           event.context = current_context
           
       # Only deliver to handlers in same context
       handlers = self._get_handlers_for_context(event.get_type(), event.context)
       
       # Notify handlers
       for handler in handlers:
           handler(event)
   ```

2. **Component Reset**:
   ```python
   def reset(self):
       """Reset component state completely between runs."""
       # Clear all internal collections
       self.data = {}
       self.signals = []
       self.metrics = {}
       
       # Reset derived components
       for component in self.child_components:
           component.reset()
   ```

### Event Flow in Isolated Contexts

Events flow through the system within isolated context boundaries:

```
┌─────────────────────────────────────────────────────┐
│                  Context Boundary                   │
│                                                     │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐        │
│  │ Bar │────▶│Signal│────▶│Order│────▶│Fill │        │
│  └─────┘     └─────┘     └─────┘     └─────┘        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Each optimization run operates inside its own isolated context boundary, preventing event leakage between runs.

## Enhanced Event Context Validation

To strengthen event isolation and prevent leakage between contexts, we need to implement enhanced validation mechanisms. This section describes the design for adding robust event context validation.

### EventContextValidator

The EventContextValidator class will ensure event context integrity:

```python
class EventContextValidator:
    """Ensures events respect context boundaries."""
    
    @staticmethod
    def validate_event_context(event, current_context=None):
        """
        Validate that an event's context matches the current context.
        
        Args:
            event: Event to validate
            current_context: Current context (defaults to get_current_context())
            
        Returns:
            bool: Whether the event context is valid
            
        Raises:
            EventContextError: If validation fails and strict mode is enabled
        """
        if current_context is None:
            current_context = get_current_context()
            
        # If no contexts are set, event is valid
        if current_context is None and event.context is None:
            return True
            
        # If event has no context but current context exists, set it
        if event.context is None:
            event.context = current_context
            return True
            
        # If contexts don't match, we have a problem
        if event.context != current_context:
            error_msg = f"Event context '{event.context.name}' does not match current context '{current_context.name}'"
            strict_mode = GlobalConfig.get('event.strict_context_validation', False)
            
            if strict_mode:
                raise EventContextError(error_msg)
            else:
                Logger.warning(f"Event context mismatch: {error_msg}")
                return False
                
        return True
    
    @staticmethod
    def verify_context_boundary(context_name, event_log):
        """
        Verify that no events leaked across the context boundary.
        
        Args:
            context_name: Name of context to verify
            event_log: List of (event, context) tuples from EventTracer
            
        Returns:
            bool: Whether boundary is intact
            
        Raises:
            EventContextError: If boundary verification fails
        """
        # Check if any events with this context were processed outside it
        leaked_events = [
            (event, processed_in) 
            for event, processed_in in event_log
            if event.context and event.context.name == context_name and processed_in != context_name
        ]
        
        if leaked_events:
            error_msg = f"Context boundary violation: {len(leaked_events)} events from '{context_name}' leaked to other contexts"
            if GlobalConfig.get('event.strict_boundary_verification', False):
                raise EventContextBoundaryError(error_msg, leaked_events)
            else:
                Logger.warning(error_msg)
                return False
                
        return True
```

### EventTracer

To track event flow for debugging and verification:

```python
class EventTracer:
    """Traces event flow through the system."""
    
    def __init__(self, max_events=10000):
        """Initialize event tracer with max event limit."""
        self.events = []
        self.max_events = max_events
        self._enabled = False
        self._lock = threading.RLock()
        
    def enable(self):
        """Enable event tracing."""
        self._enabled = True
        
    def disable(self):
        """Disable event tracing."""
        self._enabled = False
        
    def clear(self):
        """Clear event trace."""
        with self._lock:
            self.events = []
            
    def trace_event(self, event, current_context=None):
        """
        Trace an event through the system.
        
        Args:
            event: Event being processed
            current_context: Context in which event is being processed
        """
        if not self._enabled:
            return
            
        with self._lock:
            if current_context is None:
                current_context = get_current_context()
                
            context_name = current_context.name if current_context else None
            self.events.append((event, context_name))
            
            # Prune if necessary
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
```

### Enhanced EventBus

The EventBus needs to be updated to include validation:

```python
class EventBus:
    """Central event distribution system with enhanced context validation."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers = {}
        self.context_subscribers = {}
        self._lock = threading.RLock()
        self.event_tracer = EventTracer()
        
    def enable_tracing(self, enabled=True):
        """Enable or disable event tracing."""
        if enabled:
            self.event_tracer.enable()
        else:
            self.event_tracer.disable()
            
    def publish(self, event):
        """Publish an event to subscribers with context validation."""
        event_type = event.get_type()
        
        # Get current context
        current_context = get_current_context()
        
        # Validate event context
        if not EventContextValidator.validate_event_context(event, current_context):
            # If validation fails and we're not in strict mode, we'll continue but log it
            pass
            
        # Trace event
        self.event_tracer.trace_event(event, current_context)
        
        with self._lock:
            # Get handlers
            handlers = []
            
            # Add global handlers
            if event_type in self.subscribers:
                handlers.extend(self.subscribers[event_type])
                
            # Add context-specific handlers
            context = event.context
            if context and context.name in self.context_subscribers:
                if event_type in self.context_subscribers[context.name]:
                    handlers.extend(self.context_subscribers[context.name][event_type])
                    
            # Notify handlers
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    Logger.error(f"Error in event handler: {e}")
                    
        return len(handlers) > 0
```

### Enhanced EventContext

The EventContext needs to include boundary verification:

```python
class EventContext:
    """Context for event isolation between runs with boundary verification."""
    
    def __init__(self, name):
        """Initialize event context with name."""
        self.name = name
        self.event_tracer = None
        
    def __enter__(self):
        """Enter context and activate it."""
        # Set as current context
        previous_context = _current_context.get()
        _current_context.set(self)
        
        # Get event bus to enable tracing
        event_bus = get_global_event_bus()
        if event_bus:
            self.event_tracer = event_bus.event_tracer
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and deactivate it."""
        # Verify context boundary if tracing enabled
        if GlobalConfig.get('event.verify_context_boundaries', False) and self.event_tracer:
            EventContextValidator.verify_context_boundary(self.name, self.event_tracer.events)
            
        # Clear current context
        _current_context.set(None)
```

### Exception Types

New exception types for event context validation:

```python
class EventContextError(Exception):
    """Exception raised when event context validation fails."""
    pass
    
class EventContextBoundaryError(EventContextError):
    """Exception raised when event context boundary is violated."""
    
    def __init__(self, message, leaked_events=None):
        super().__init__(message)
        self.leaked_events = leaked_events or []
```

### Usage Example

```python
# Configure event context validation
GlobalConfig.set('event.strict_context_validation', True)
GlobalConfig.set('event.verify_context_boundaries', True)

# Run optimization with context validation
def optimize_strategy(parameter_space, data_handler):
    """Run optimization with validated context isolation."""
    best_params = None
    
    # Set up data splitter
    data_handler.setup_train_test_split(method='ratio', train_ratio=0.7)
    
    # Iterate through parameter combinations
    for params in parameter_space.get_combinations():
        # Run with training data
        data_handler.set_active_split('train')
        with EventContext("train") as train_context:
            # This will validate that all events remain in train context
            train_result = run_backtest(params, context=train_context)
        
        # Run with test data
        data_handler.set_active_split('test')
        with EventContext("test") as test_context:
            # This will validate that all events remain in test context
            test_result = run_backtest(params, context=test_context)
        
        # Update best parameters
        if is_better_result(test_result, best_result):
            best_params = params
            
    return best_params
```

By implementing these enhancements, you ensure complete isolation between optimization runs, preventing data leakage and enabling reliable performance evaluation.

## State Reset Verification Framework

Ensuring that components properly reset their state between optimization runs is critical for reliable results. The following design describes a framework for verifying that state reset is complete and correct.

### StateSnapshot

The StateSnapshot class captures the state of a component:

```python
class StateSnapshot:
    """Captures the state of a component for comparison."""
    
    def __init__(self, component):
        """
        Create a snapshot of component state.
        
        Args:
            component: Component to snapshot
        """
        self.component_id = id(component)
        self.component_name = component.name
        self.state_dict = self._capture_state(component)
        self.timestamp = datetime.now()
        
    def _capture_state(self, component):
        """
        Capture component state details.
        
        Args:
            component: Component to capture
            
        Returns:
            dict: State representation
        """
        # Create a dictionary to represent state
        state = {}
        
        # Capture explicit state attributes
        if hasattr(component, '__state_attributes__'):
            for attr in component.__state_attributes__:
                if hasattr(component, attr):
                    value = getattr(component, attr)
                    state[attr] = self._get_state_representation(value)
        
        # Capture common collections
        for attr in dir(component):
            value = getattr(component, attr)
            if not attr.startswith('_') and not callable(value):
                if isinstance(value, (dict, list, set)):
                    state[attr] = self._get_state_representation(value)
                    
        # Track memory info
        state['_memory_size'] = self._estimate_size(component)
        state['_object_count'] = self._count_objects(component)
        
        return state
        
    def _get_state_representation(self, value):
        """
        Get a representation of a value for state comparison.
        
        Args:
            value: Value to represent
            
        Returns:
            object: State representation
        """
        if isinstance(value, dict):
            return {k: self._get_state_representation(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._get_state_representation(v) for v in value]
        elif isinstance(value, set):
            return {self._get_state_representation(v) for v in value}
        elif hasattr(value, '__dict__'):
            # Avoid circular references and only capture basic info for objects
            return f"{type(value).__name__}:{id(value)}"
        else:
            return value
            
    def _estimate_size(self, obj):
        """Estimate memory size of object in bytes."""
        # Implementation depends on available tools
        # For example, using sys.getsizeof recursively
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return -1
            
    def _count_objects(self, obj):
        """Count number of child objects."""
        # Implementation specific to component structure
        return 0
        
    def compare_with(self, other):
        """
        Compare this snapshot with another.
        
        Args:
            other: Another StateSnapshot
            
        Returns:
            tuple: (is_same, differences)
        """
        if not isinstance(other, StateSnapshot):
            return False, {"error": "Not a StateSnapshot"}
            
        if self.component_id != other.component_id:
            return False, {"error": "Different components"}
            
        # Compare state dictionaries
        differences = {}
        for key in set(self.state_dict) | set(other.state_dict):
            if key not in self.state_dict:
                differences[key] = {"error": "Missing in original", "value": other.state_dict[key]}
            elif key not in other.state_dict:
                differences[key] = {"error": "Missing in other", "value": self.state_dict[key]}
            elif self.state_dict[key] != other.state_dict[key]:
                differences[key] = {
                    "original": self.state_dict[key],
                    "other": other.state_dict[key]
                }
                
        return len(differences) == 0, differences
```

### StateVerifier

The StateVerifier manages verification of component state resets:

```python
class StateVerifier:
    """Verifies that component state is properly reset."""
    
    def __init__(self):
        """Initialize state verifier."""
        self.snapshots = {}
        self.enabled = True
        
    def enable(self):
        """Enable state verification."""
        self.enabled = True
        
    def disable(self):
        """Disable state verification."""
        self.enabled = False
        
    def take_snapshot(self, component, key=None):
        """
        Take a snapshot of component state.
        
        Args:
            component: Component to snapshot
            key: Optional key to identify snapshot (defaults to component name)
            
        Returns:
            StateSnapshot: The created snapshot
        """
        if not self.enabled:
            return None
            
        key = key or f"{component.name}_{id(component)}"
        snapshot = StateSnapshot(component)
        self.snapshots[key] = snapshot
        return snapshot
        
    def verify_reset(self, component, original_key=None, reset_key=None):
        """
        Verify that a component has been properly reset.
        
        Args:
            component: Component to verify
            original_key: Key for original snapshot (default: component_name_id + "_original")
            reset_key: Key for reset snapshot (default: component_name_id + "_reset")
            
        Returns:
            tuple: (is_reset, differences)
            
        Raises:
            StateVerificationError: If verification fails in strict mode
        """
        if not self.enabled:
            return True, {}
            
        base_key = f"{component.name}_{id(component)}"
        original_key = original_key or f"{base_key}_original"
        reset_key = reset_key or f"{base_key}_reset"
        
        # Check if we have the original snapshot
        if original_key not in self.snapshots:
            return False, {"error": f"No original snapshot found for {original_key}"}
            
        # Take a snapshot of the current state
        current_snapshot = StateSnapshot(component)
        self.snapshots[reset_key] = current_snapshot
        
        # Compare with original
        is_same, differences = self.snapshots[original_key].compare_with(current_snapshot)
        
        # Handle verification result
        if not is_same:
            strict_mode = GlobalConfig.get('state.strict_verification', False)
            if strict_mode:
                raise StateVerificationError(
                    f"Component {component.name} not properly reset", 
                    differences
                )
                
        return is_same, differences
        
    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots = {}
```

### Enhanced Component Base Class

Enhance the Component base class to support state verification:

```python
class Component:
    """Base class for all system components with state verification."""
    
    def __init__(self, name, parameters=None):
        """Initialize component with name and parameters."""
        self.name = name
        self.parameters = parameters or {}
        self.initialized = False
        self.running = False
        
        # Define state attributes that should be tracked
        self.__state_attributes__ = [
            'parameters', 'initialized', 'running'
        ]
        
    def initialize(self, context):
        """Set up component with dependencies from context."""
        self.event_bus = context.get('event_bus')
        self.logger = context.get('logger')
        self.config = context.get('config')
        
        # Get state verifier if available
        self.state_verifier = context.get('state_verifier')
        
        # Take initial state snapshot if verification enabled
        if self.state_verifier:
            self.state_verifier.take_snapshot(self, f"{self.name}_initial")
        
        # Initialize event subscriptions
        if self.event_bus:
            self.initialize_event_subscriptions()
            
        self.initialized = True
        
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        pass
        
    def start(self):
        """Begin component operation."""
        if not self.initialized:
            raise ComponentError("Component must be initialized before starting")
            
        self.running = True
        
    def stop(self):
        """End component operation."""
        self.running = False
        
    def reset(self):
        """
        Clear component state for a new run.
        
        If state verification is enabled, the verifier will check
        that all state has been properly cleared after this method.
        
        Returns:
            bool: True if reset was successful
        """
        # Take pre-reset snapshot if verification enabled
        if hasattr(self, 'state_verifier') and self.state_verifier:
            self.state_verifier.take_snapshot(self, f"{self.name}_before_reset")
            
        # Perform reset logic (to be implemented by subclasses)
        
        # Verify reset if enabled
        if hasattr(self, 'state_verifier') and self.state_verifier:
            # Take post-reset snapshot and compare with initial state
            is_reset, differences = self.state_verifier.verify_reset(
                self,
                f"{self.name}_initial",
                f"{self.name}_after_reset"
            )
            
            if not is_reset:
                self.logger.warning(f"Component {self.name} not properly reset: {differences}")
                
            return is_reset
            
        return True
        
    def teardown(self):
        """Release resources."""
        # Unsubscribe from events
        if self.event_bus:
            self.event_bus.unsubscribe_all(self)
            
        self.initialized = False
        self.running = False
```

### StateVerificationError

Add a new exception for state verification failures:

```python
class StateVerificationError(Exception):
    """Exception for state verification failures."""
    
    def __init__(self, message, differences=None):
        """Initialize with message and differences."""
        super().__init__(message)
        self.differences = differences or {}
```

### Usage in Optimization

Here's how to use state verification during optimization:

```python
def optimize_strategy(parameter_space, components):
    """Run optimization with state verification."""
    # Create state verifier
    state_verifier = StateVerifier()
    
    # Add to component context
    context = Context()
    context.register('state_verifier', state_verifier)
    
    # Configure verification
    GlobalConfig.set('state.strict_verification', False)  # Log warnings instead of raising exceptions
    
    # Take initial snapshots
    for component in components:
        state_verifier.take_snapshot(component, f"{component.name}_original")
    
    # Run optimization iterations
    for params in parameter_space.get_combinations():
        # Reset all components
        for component in components:
            component.reset()
            
        # Run backtest with these parameters
        result = run_backtest(params, components)
        
        # Process result
        process_result(result)
    
    # Report verification statistics
    verification_stats = {
        "total_components": len(components),
        "reset_failures": sum(1 for c in components if not c.reset())
    }
    
    return verification_stats
```

### Framework Integration

To integrate state verification into the larger system:

1. **Component Registry**: Maintain a registry of all components for centralized reset verification
2. **Lifecycle Hooks**: Add hooks to automatically verify state after reset operations
3. **Reporting**: Generate reports of state differences for debugging
4. **Configuration**: Make verification behavior configurable

By implementing this state verification framework, you ensure that component state is properly reset between optimization runs, preventing data leakage and ensuring reliable results.

By following these guidelines, you will create a robust Core module that provides a solid foundation for the entire ADMF-Trader system.