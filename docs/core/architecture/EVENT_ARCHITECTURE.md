# Event Architecture

## Overview

The ADMF-Trader system is built on an event-driven architecture that enables loose coupling between components, clear separation of concerns, and flexible system behavior. The event system serves as the communication backbone of the entire application.

## Key Architectural Principles

1. **Loose Coupling**: Components interact through events, not direct method calls
2. **Publisher-Subscriber Pattern**: Components publish events and subscribe to events of interest
3. **Event Typing**: Events have explicit types for routing and handling
4. **Context Isolation**: Events can be confined to specific contexts for isolation
5. **Thread Safety**: Event handling is thread-safe for concurrent execution
6. **Scalability**: The event system scales with system complexity

## Core Event Types

The event system defines standard event types that establish the primary flows of information through the system:

```python
class EventType(Enum):
    """Standard event types for the system."""
    BAR = "BAR"               # Market data bar event
    SIGNAL = "SIGNAL"         # Strategy signal event
    ORDER = "ORDER"           # Order request event
    FILL = "FILL"             # Order fill event
    PORTFOLIO = "PORTFOLIO"   # Portfolio update event
    ERROR = "ERROR"           # Error event
    SYSTEM = "SYSTEM"         # System control event
    BACKTEST_START = "BACKTEST_START"  # Backtest start event
    BACKTEST_END = "BACKTEST_END"      # Backtest end event
```

## Event Structure

Each event has a consistent structure:

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

## Event Bus

The EventBus is the central hub for event distribution:

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

## Subscription Management

Components use a SubscriptionManager to manage their event subscriptions:

```python
class SubscriptionManager:
    """Manages event subscriptions for a component."""
    
    def __init__(self, event_bus):
        """Initialize with event bus."""
        self.event_bus = event_bus
        self.subscriptions = []
        
    def subscribe(self, event_type, handler, context=None):
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, handler, context)
        self.subscriptions.append((event_type, handler, context))
        
    def unsubscribe_all(self):
        """Unsubscribe from all subscribed events."""
        for event_type, handler, context in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler, context)
        self.subscriptions = []
```

## Event Flow

The standard event flow in the system:

```
┌─────────┐     ┌──────────┐     ┌───────────────┐     ┌────────┐     ┌────────────┐
│ DataHandler │─────▶│ Strategy │─────▶│ RiskManager │─────▶│ Broker │─────▶│ Portfolio │
└─────────┘     └──────────┘     └───────────────┘     └────────┘     └────────────┘
    │                │                  │                  │                │
    ▼                ▼                  ▼                  ▼                ▼
┌─────────┐     ┌──────────┐     ┌───────────────┐     ┌────────┐     ┌────────────┐
│  BAR    │─────▶│  SIGNAL  │─────▶│    ORDER     │─────▶│  FILL  │─────▶│ PORTFOLIO  │
└─────────┘     └──────────┘     └───────────────┘     └────────┘     └────────────┘
```

1. The DataHandler emits BAR events with market data
2. The Strategy consumes BAR events and emits SIGNAL events
3. The RiskManager consumes SIGNAL events and emits ORDER events
4. The Broker consumes ORDER events and emits FILL events
5. The Portfolio consumes FILL events and emits PORTFOLIO events

## Event Handlers

Components implement event handlers for events they're interested in:

```python
class Strategy(Component):
    """Strategy component example."""
    
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event):
        """Handle bar event."""
        bar_data = event.get_data()
        # Process bar data
        signals = self.calculate_signals(bar_data)
        
        # Emit signal events
        for signal in signals:
            self.emit_signal(signal)
```

## Event Context Isolation

Events can be isolated to specific contexts to prevent cross-contamination:

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

Usage:

```python
with EventContext("backtest_1"):
    # Events published here are confined to the backtest_1 context
    run_backtest(params)

with EventContext("backtest_2"):
    # Events published here are confined to the backtest_2 context
    run_backtest(params)
```

## Event Tracing and Debugging

The event system includes tracing for debugging:

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

## Event Context Validation

The system includes validation to ensure event context integrity:

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
```

## Best Practices

1. **Define Clear Event Types**: Use explicit event types for clear routing
2. **Use Consistent Event Structure**: Follow the standard event structure
3. **Handle Exceptions**: Always handle exceptions in event handlers to prevent cascading failures
4. **Manage Subscriptions**: Use the SubscriptionManager to track and clean up subscriptions
5. **Context Isolation**: Use event contexts to isolate event flows during parallel operations
6. **Event Validation**: Enable event context validation during testing and debugging
7. **Minimize Event Size**: Keep event data small and focused
8. **Avoid Cycles**: Prevent circular event dependencies

By following these patterns consistently, we create a robust, scalable event architecture that enables loose coupling, clear separation of concerns, and flexible system behavior.