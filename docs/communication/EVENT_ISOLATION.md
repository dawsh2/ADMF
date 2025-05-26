# Event Isolation

## Overview

Event isolation is a critical feature of the ADMF-Trader system that prevents event leakage between different execution contexts, particularly during optimization runs. This mechanism ensures that events from one backtest or optimization run don't interfere with another, enabling reliable, deterministic results.

## The Need for Event Isolation

During optimization, the system runs multiple backtests with different parameters. Without proper isolation:

1. **Event Leakage**: Events from one run could affect another
2. **Cross-Contamination**: Data from different runs could get mixed
3. **Non-Deterministic Results**: Results would be dependent on run order
4. **Inconsistent Outcomes**: Optimization would produce unreliable results

## Event Context System

The event isolation system is built around the concept of event contexts:

```python
class EventContext:
    """Context for event isolation between runs."""
    
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

## Current Context Tracking

The system tracks the current context using thread-local storage:

```python
# Thread-local storage for current context
_current_context = threading.local()

def get_current_context():
    """Get the current event context."""
    return getattr(_current_context, 'value', None)
    
def set_current_context(context):
    """Set the current event context."""
    _current_context.value = context
```

## Context-Aware Event Bus

The EventBus is context-aware, only delivering events to handlers in the appropriate context:

```python
class EventBus:
    """Context-aware event distribution system."""
    
    def __init__(self):
        """Initialize event bus."""
        self.subscribers = {}          # Global subscribers
        self.context_subscribers = {}  # Context-specific subscribers
        self._lock = threading.RLock()
        self.event_tracer = EventTracer()
        
    def publish(self, event):
        """Publish an event to subscribers with context validation."""
        event_type = event.get_type()
        
        # Get current context
        current_context = get_current_context()
        
        # Validate event context
        if not EventContextValidator.validate_event_context(event, current_context):
            # If validation fails and we're not in strict mode, continue but log it
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

## Event Context Validation

The system includes validation to ensure events respect context boundaries:

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

## Event Tracing

The EventTracer helps debug context issues by tracking event flow:

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

## Exception Types

Custom exceptions for event context validation:

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

## Using Event Contexts

Event contexts are used to isolate event flows during execution:

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

## Context-Aware Components

Components can be context-aware to help with isolation:

```python
class ContextAwareComponent(Component):
    """A component aware of event contexts."""
    
    def initialize(self, context):
        """Initialize with context awareness."""
        super().initialize(context)
        self.contexts = {}
        
    def initialize_event_subscriptions(self):
        """Set up context-aware event subscriptions."""
        self.subscription_manager = SubscriptionManager(self.event_bus)
        
        # Get current context
        current_context = get_current_context()
        
        # Subscribe with current context
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar, current_context)
        
    def on_bar(self, event):
        """Handle bar event."""
        # Get the context of this event
        context = event.get_context()
        
        # Store data in context-specific storage
        if context:
            context_name = context.name
            if context_name not in self.contexts:
                self.contexts[context_name] = {
                    'data': [],
                    'counters': {}
                }
                
            # Update context-specific data
            self.contexts[context_name]['data'].append(event.get_data())
            
            # Update context-specific counters
            counters = self.contexts[context_name]['counters']
            counters['bar_count'] = counters.get('bar_count', 0) + 1
```

## Debugging Context Issues

Debugging tools for context issues:

```python
class ContextDebugger:
    """Debugger for context issues."""
    
    def __init__(self, event_tracer):
        """Initialize with event tracer."""
        self.event_tracer = event_tracer
        
    def analyze_context_boundaries(self):
        """
        Analyze context boundaries for leaks.
        
        Returns:
            dict: Analysis results
        """
        # Find all contexts
        contexts = set()
        for event, context_name in self.event_tracer.events:
            if event.get_context():
                contexts.add(event.get_context().name)
                
        # Check each context for leaks
        results = {}
        for context_name in contexts:
            leaked_events = [
                (event, processed_in) 
                for event, processed_in in self.event_tracer.events
                if event.get_context() and event.get_context().name == context_name and processed_in != context_name
            ]
            
            results[context_name] = {
                'total_events': sum(1 for e, _ in self.event_tracer.events if e.get_context() and e.get_context().name == context_name),
                'leaked_events': len(leaked_events),
                'leaks': [
                    {
                        'event_type': e.get_type(),
                        'processed_in': ctx
                    }
                    for e, ctx in leaked_events
                ]
            }
            
        return results
```

## Configuration Options

The event isolation system can be configured:

```python
# Configure event context validation
GlobalConfig.set('event.strict_context_validation', True)  # Raise exceptions for context mismatches
GlobalConfig.set('event.verify_context_boundaries', True)  # Verify boundaries when contexts are exited
GlobalConfig.set('event.trace_enabled', True)              # Enable event tracing
```

## Troubleshooting

Common issues and solutions:

1. **Events not being delivered**: Ensure the context of the subscriber matches the context of the publisher
2. **Context boundary violations**: Check for handlers that store events and re-publish them in different contexts
3. **Missing context**: Ensure events are created within a context (using the with EventContext pattern)

## Best Practices

1. **Always Use Contexts**: Wrap backtest and optimization runs in event contexts
2. **Reset Between Runs**: Always reset component state between runs
3. **Avoid Context Mixing**: Don't publish events from one context in another
4. **Test Context Isolation**: Verify isolation during testing
5. **Enable Validation**: Enable strict validation during development and testing
6. **Trace Events**: Use event tracing to debug isolation issues

## Advanced Pattern: Context Hierarchy

For complex scenarios, contexts can be hierarchical:

```python
class HierarchicalEventContext(EventContext):
    """Hierarchical event context for nested isolation."""
    
    def __init__(self, name, parent=None):
        """
        Initialize hierarchical context.
        
        Args:
            name: Context name
            parent: Optional parent context
        """
        super().__init__(name)
        self.parent = parent
        self.full_name = f"{parent.full_name}.{name}" if parent else name
        
    def is_ancestor_of(self, context):
        """
        Check if this context is an ancestor of another context.
        
        Args:
            context: Context to check
            
        Returns:
            bool: Whether this context is an ancestor
        """
        if not context:
            return False
            
        if not isinstance(context, HierarchicalEventContext):
            return False
            
        current = context.parent
        while current:
            if current == self:
                return True
            current = current.parent
            
        return False
```

The hierarchical validation would then allow events to flow from parent contexts to child contexts but not vice versa:

```python
def validate_hierarchical_context(event, current_context):
    """
    Validate event context in a hierarchical context system.
    
    Args:
        event: Event to validate
        current_context: Current context
        
    Returns:
        bool: Whether the event context is valid
    """
    event_context = event.get_context()
    
    # If no context or same context, valid
    if not event_context or event_context == current_context:
        return True
        
    # If both are hierarchical
    if isinstance(event_context, HierarchicalEventContext) and isinstance(current_context, HierarchicalEventContext):
        # Events can flow downward (parent to child)
        if event_context.is_ancestor_of(current_context):
            return True
            
    return False
```

By implementing robust event isolation, the ADMF-Trader system ensures reliable, deterministic behavior during optimization and parallel execution, enabling consistent, reproducible results.