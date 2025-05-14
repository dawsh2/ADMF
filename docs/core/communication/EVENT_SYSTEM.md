# Event System

## Overview

The Event System is the communication backbone of the ADMF-Trader application, enabling loosely coupled interactions between components. It implements the publisher-subscriber (pub/sub) pattern, allowing components to communicate without direct dependencies.

## Core Components

The Event System consists of three main components:

1. **Events**: Data structures that carry information between components
2. **EventBus**: Central hub for event distribution
3. **SubscriptionManager**: Helper for managing component subscriptions

## Event Structure

Events have a standard structure:

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
        
    def get_timestamp(self):
        """Get event timestamp."""
        return self.timestamp
        
    def get_context(self):
        """Get event context."""
        return self.context
```

## Event Types

The system defines standard event types:

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
    LIFECYCLE = "LIFECYCLE"   # Component lifecycle event
    BACKTEST_START = "BACKTEST_START"  # Backtest start event
    BACKTEST_END = "BACKTEST_END"      # Backtest end event
```

## EventBus Implementation

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
        """
        Publish an event to subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: Whether the event was delivered to any handlers
        """
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
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events occur
            context: Optional context to associate with the subscription
        """
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
                
    def unsubscribe(self, event_type, handler, context=None):
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to unsubscribe
            context: Optional context associated with the subscription
            
        Returns:
            bool: Whether the handler was successfully unsubscribed
        """
        with self._lock:
            if context:
                # Context-specific unsubscription
                if (
                    context.name in self.context_subscribers and
                    event_type in self.context_subscribers[context.name]
                ):
                    if handler in self.context_subscribers[context.name][event_type]:
                        self.context_subscribers[context.name][event_type].remove(handler)
                        return True
            else:
                # Global unsubscription
                if event_type in self.subscribers:
                    if handler in self.subscribers[event_type]:
                        self.subscribers[event_type].remove(handler)
                        return True
                        
            return False
            
    def unsubscribe_all(self, handler):
        """
        Unsubscribe a handler from all event types.
        
        Args:
            handler: Handler to unsubscribe
        """
        with self._lock:
            # Unsubscribe from global subscriptions
            for event_type in list(self.subscribers.keys()):
                if handler in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(handler)
                    
            # Unsubscribe from context-specific subscriptions
            for context_name in list(self.context_subscribers.keys()):
                for event_type in list(self.context_subscribers[context_name].keys()):
                    if handler in self.context_subscribers[context_name][event_type]:
                        self.context_subscribers[context_name][event_type].remove(handler)
                        
    def reset(self):
        """Reset the event bus, clearing all subscriptions."""
        with self._lock:
            self.subscribers = {}
            self.context_subscribers = {}
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
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events occur
            context: Optional context to associate with the subscription
            
        Returns:
            self: For method chaining
        """
        self.event_bus.subscribe(event_type, handler, context)
        self.subscriptions.append((event_type, handler, context))
        return self
        
    def unsubscribe(self, event_type, handler, context=None):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to unsubscribe
            context: Optional context associated with the subscription
            
        Returns:
            bool: Whether the handler was successfully unsubscribed
        """
        result = self.event_bus.unsubscribe(event_type, handler, context)
        if result:
            self.subscriptions.remove((event_type, handler, context))
        return result
        
    def unsubscribe_all(self):
        """Unsubscribe from all subscribed events."""
        for event_type, handler, context in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler, context)
        self.subscriptions = []
```

## Component Integration

Components integrate with the event system through the Component base class:

```python
class Component:
    """Base class for all system components."""
    
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
        
    def teardown(self):
        """Release resources."""
        # Unsubscribe from events
        if self.event_bus:
            self.event_bus.unsubscribe_all(self)
            
        self.initialized = False
        self.running = False
```

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
        """
        Handle bar event.
        
        Args:
            event: Bar event
        """
        bar_data = event.get_data()
        signals = self.calculate_signals(bar_data)
        
        # Emit signal events
        for signal in signals:
            self.emit_signal(signal)
            
    def emit_signal(self, signal):
        """
        Emit a signal event.
        
        Args:
            signal: Signal data
            
        Returns:
            bool: Whether the signal was successfully emitted
        """
        signal_event = Event(
            event_type=EventType.SIGNAL,
            data=signal,
            timestamp=datetime.now(),
            context=None  # Will inherit current context
        )
        return self.event_bus.publish(signal_event)
```

## Event Helpers

The system includes helper functions for common event operations:

```python
def create_bar_event(symbol, timestamp, open_price, high, low, close, volume):
    """
    Create a bar event.
    
    Args:
        symbol: Instrument symbol
        timestamp: Bar timestamp
        open_price: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume
        
    Returns:
        Event: Bar event
    """
    data = {
        'symbol': symbol,
        'timestamp': timestamp,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }
    return Event(EventType.BAR, data)

def create_signal_event(strategy, symbol, direction, price, quantity, timestamp=None):
    """
    Create a signal event.
    
    Args:
        strategy: Strategy name
        symbol: Instrument symbol
        direction: Signal direction ('BUY' or 'SELL')
        price: Signal price
        quantity: Signal quantity
        timestamp: Optional timestamp
        
    Returns:
        Event: Signal event
    """
    data = {
        'strategy': strategy,
        'symbol': symbol,
        'direction': direction,
        'price': price,
        'quantity': quantity,
        'timestamp': timestamp or datetime.now()
    }
    return Event(EventType.SIGNAL, data)
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

## Event Debugging and Monitoring

The system includes features for debugging and monitoring events:

```python
class EventMonitor:
    """Monitors events flowing through the system."""
    
    def __init__(self, event_bus):
        """Initialize with event bus."""
        self.event_bus = event_bus
        self.events = []
        self.max_events = 1000
        self.enabled = False
        self.subscription_manager = SubscriptionManager(event_bus)
        
    def start(self):
        """Start monitoring events."""
        # Subscribe to all event types
        for event_type in EventType:
            self.subscription_manager.subscribe(event_type, self._on_event)
        self.enabled = True
        
    def stop(self):
        """Stop monitoring events."""
        self.subscription_manager.unsubscribe_all()
        self.enabled = False
        
    def clear(self):
        """Clear captured events."""
        self.events = []
        
    def _on_event(self, event):
        """Handle an event."""
        if not self.enabled:
            return
            
        # Capture event
        self.events.append({
            'type': event.get_type(),
            'timestamp': event.get_timestamp(),
            'data': event.get_data(),
            'context': event.get_context().name if event.get_context() else None
        })
        
        # Limit event history
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
            
    def get_events(self, event_type=None, context=None, limit=None):
        """
        Get captured events, optionally filtered.
        
        Args:
            event_type: Optional event type to filter by
            context: Optional context to filter by
            limit: Optional limit on number of events
            
        Returns:
            list: Matching events
        """
        # Filter events
        result = self.events
        
        if event_type:
            result = [e for e in result if e['type'] == event_type]
            
        if context:
            result = [e for e in result if e['context'] == context]
            
        # Apply limit
        if limit and limit < len(result):
            result = result[-limit:]
            
        return result
        
    def get_statistics(self):
        """
        Get event statistics.
        
        Returns:
            dict: Event statistics
        """
        # Count events by type
        counts = {}
        for event in self.events:
            event_type = event['type']
            counts[event_type] = counts.get(event_type, 0) + 1
            
        # Calculate rates
        rates = {}
        if self.events:
            first_time = self.events[0]['timestamp']
            last_time = self.events[-1]['timestamp']
            duration = (last_time - first_time).total_seconds()
            
            if duration > 0:
                for event_type, count in counts.items():
                    rates[event_type] = count / duration
                    
        return {
            'total': len(self.events),
            'counts': counts,
            'rates': rates
        }
```

## Best Practices

1. **Use Event Constants**: Always use the EventType constants for event types
2. **Clear Event Data**: Include all necessary information in event data
3. **Handle Exceptions**: Always handle exceptions in event handlers
4. **Unsubscribe Properly**: Use SubscriptionManager to ensure proper unsubscription
5. **Event Context**: Use event contexts to isolate event flows
6. **Minimal Processing**: Keep event handler processing minimal and fast
7. **Thread Safety**: Be aware of thread safety issues in event handlers
8. **Event Validation**: Validate event data before processing

## Performance Considerations

1. **Handler Efficiency**: Keep event handlers efficient and minimal
2. **Event Size**: Keep event payload size small
3. **Handler Count**: Limit the number of handlers per event type
4. **Publish Frequency**: Avoid excessive event publication
5. **Batching**: Consider batching events for high-frequency data

By following these patterns, the event system provides a robust communication backbone for the ADMF-Trader application, enabling loosely coupled interactions between components.