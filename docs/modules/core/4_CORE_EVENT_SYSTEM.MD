
# ADMF Core: Event System

## 1. Overview

The Event System is the primary communication backbone of the ADMF-Trader application. It implements a publish-subscribe (pub/sub) pattern, enabling loosely coupled interactions between components. This design allows components to communicate changes in state or trigger actions in other components without requiring direct knowledge of each other, fostering modularity and extensibility.

This document details the architecture of the event system, from basic event structures and bus mechanics to advanced concepts for event isolation and scalability.

## 2. Key Architectural Principles

* **Loose Coupling**: Components interact via events, minimizing direct dependencies.
* **Publisher-Subscriber Pattern**: Components can publish events they generate, and other components can subscribe to event types they are interested in.
* **Event Typing**: Events have explicit types, enabling targeted subscriptions and clear routing of information.
* **Context Isolation (Idealized)**: Events can be confined to specific execution contexts to prevent leakage, particularly during parallel operations like optimization. (See Section 6).
* **Thread Safety**: The event handling mechanisms are designed to be thread-safe for concurrent execution.
* **Scalability (Idealized)**: The event system architecture includes designs for scaling to high-frequency trading scenarios. (See Section 7).

## 3. Core Event System Components

The fundamental parts of the event system include the structure of events themselves, the types of events, the central `EventBus`, and a `SubscriptionManager`.

### 3.1. Event Structure (`Event` class)

Events are data structures that carry information between components. Each event has a consistent structure.

**Idealized Structure (Python):**
```python
from datetime import datetime
from typing import Dict, Any, Optional, Union # Added Union
from enum import Enum # For EventType ideally

# class EventType(Enum): # Defined in the next section
#     BAR = "BAR"
#     # ... other event types

class Event:
    """Event message passed between components."""
    
    def __init__(self, 
                 event_type: Union[Enum, str], # Ideally EventType enum
                 data: Optional[Dict[str, Any]] = None, 
                 timestamp: Optional[datetime] = None, 
                 context: Optional[Any] = None, # For EventContext
                 metadata: Optional[Dict[str, Any]] = None): # Added metadata
        """Initialize event with type, data, and metadata."""
        self.event_type = event_type
        self.payload = data or {} # Renamed data to payload for clarity as in src/core/event.py
        self.timestamp = timestamp or datetime.now()
        self.context = context # For event isolation
        self.metadata = metadata or {} # For additional info like source, trace_id

    def get_type(self) -> Union[Enum, str]:
        """Get event type."""
        return self.event_type
        
    def get_data(self) -> Dict[str, Any]: # Kept for compatibility, but payload is preferred
        """Get event data (payload)."""
        return self.payload

    @property
    def payload(self) -> Dict[str, Any]:
        """Get event payload."""
        return self._payload # Assuming self.payload is set as self._payload internally

    @payload.setter
    def payload(self, value: Dict[str, Any]):
        self._payload = value

    def get_timestamp(self) -> datetime:
        """Get event timestamp."""
        return self.timestamp
        
    def get_context(self) -> Optional[Any]:
        """Get event context."""
        return self.context

    def get_metadata(self) -> Dict[str, Any]:
        """Get event metadata."""
        return self.metadata
```

Your src/core/event.py implements Event with type, timestamp, and payload attributes, which aligns well with this.

### 3.2. Event Types (EventType)

The system defines standard event types to categorize events and facilitate targeted subscriptions.

Standard Event Types (Conceptual Enum):

```python
from enum import Enum # Ideal

class EventType(Enum):
    BAR = "BAR"               # Market data bar event
    SIGNAL = "SIGNAL"         # Strategy signal event
    ORDER = "ORDER"           # Order request event
    FILL = "FILL"             # Order fill event
    PORTFOLIO = "PORTFOLIO"   # Portfolio update event
    ERROR = "ERROR"           # Error event
    SYSTEM = "SYSTEM"         # System control event
    LIFECYCLE = "LIFECYCLE"   # Component lifecycle event (from COMPONENT_LIFECYCLE.MD)
    CLASSIFICATION = "CLASSIFICATION" # From CLASSIFIER.MD
    REGIME_CHANGE = "REGIME_CHANGE" # From REGIME_OPTIMIZATION_IMPLEMENTATION.md
    META_LABEL = "META_LABEL" # From ANALYTICS_COMPONENT_FRAMEWORK.md
    RETRY = "RETRY" # From ERROR_HANDLING.md
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED" # From ERROR_HANDLING.md
    ALERT = "ALERT" # From LOGGING_MONITORING.MD
    BACKTEST_START = "BACKTEST_START"
    BACKTEST_END = "BACKTEST_END"
```

Your src/core/event.py defines EventType as a class with string constants. While functional, using a Python Enum is often preferred for better type safety and clarity.

### 3.3. EventBus

The EventBus is the central dispatcher for all events. Components publish events to the bus, and the bus delivers them to all subscribed handlers.

Basic Implementation (src/core/event_bus.py):

publish(event): Takes an Event object and iterates through subscribers matching the event type, calling their handler functions.
subscribe(event_type, handler): Registers a handler function for a specific event type.
unsubscribe(event_type, handler): Removes a handler.
unsubscribe_all(handler_instance): Removes a specific handler instance from all event types it was subscribed to.
Thread Safety: Uses threading.RLock to protect concurrent access to the subscriber lists.
Error Handling: Includes a try-except block around handler invocation to log errors and prevent one failing handler from stopping others.
The current EventBus in src/core/event_bus.py does not yet implement the context-aware subscription/publishing detailed in docs/core/communication/EVENT_ISOLATION.md or the advanced scalability features from docs/core/communication/EVENT_SCALABILITY.MD.
3.4. SubscriptionManager

To simplify managing subscriptions within components and ensure proper cleanup (especially during teardown), the SubscriptionManager is used.

Implementation (src/core/subscription_manager.py):

__init__(event_bus): Takes an EventBus instance.
subscribe(event_type, handler, context=None): Subscribes the handler via the EventBus and keeps track of the subscription.
unsubscribe(event_type, handler, context=None): Unsubscribes and removes from tracking.
unsubscribe_all(): Iterates through all tracked subscriptions and unsubscribes them from the EventBus. This is typically called in a component's teardown() method.
This is correctly implemented and used in src/core/component_base.py.

## 4. Standard Event Flow

The primary data processing pipeline in ADMF-Trader follows a standard sequence of events:

1. **BAR Event**: The DataHandler loads/receives market data and emits BAR events for each new data point (e.g., a one-minute bar).
2. **SIGNAL Event**: Strategy components subscribe to BAR events. Upon receiving a bar, they update indicators and apply their logic to generate trading signals, which are then published as SIGNAL events.
3. **ORDER Event**: The RiskManager subscribes to SIGNAL events. It processes these signals, applies position sizing rules and risk limits, and if appropriate, generates an ORDER event.
4. **FILL Event**: The ExecutionHandler (e.g., SimulatedBroker or a live broker interface) subscribes to ORDER events. It processes these orders, simulates or executes them, and upon execution (full or partial), publishes FILL events.
5. **PORTFOLIO Event**: The PortfolioManager subscribes to FILL events to update positions, cash, and P&L. It may also subscribe to BAR events for mark-to-market calculations. It can publish PORTFOLIO events to signal updates to its state.

This event-driven flow decouples the modules significantly.

## 5. Event Handlers

Components that need to react to events define handler methods. These methods are then registered with the EventBus for specific event types via the SubscriptionManager, typically during the component's initialize_event_subscriptions() phase.

```python
# Example in a Strategy component
# (from docs/core/communication/EVENT_SYSTEM.md)
class Strategy(ComponentBase):
    # ...
    def initialize_event_subscriptions(self):
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def on_bar(self, event: Event): # Type hint with your Event class
        bar_data = event.payload # Or event.get_data()
        # ... process bar data and generate signals ...
```

## 6. Event Isolation (EventContext) [Idealized Design]

To ensure reliable and deterministic results, especially during parallel operations like optimization or multiple backtests, event isolation is critical. This prevents events from one execution context (e.g., one optimization trial) from interfering with another.

### 6.1. EventContext Class

* **Purpose**: Defines a scope for events.
* **Mechanism**: Used as a context manager (`with EventContext("backtest_run_1"):`).
* **`__enter__`**: Sets itself as the "current context" (e.g., using thread-local storage).
* **`__exit__`**: Restores the previous context.

### 6.2. Current Context Tracking

A mechanism (typically using threading.local()) to store and retrieve the active EventContext for the current thread of execution.

6.3. Context-Aware EventBus (Idealized)

The EventBus needs to be enhanced to:

Publish with Context: When an event is published, it should either inherit the current EventContext or have one explicitly assigned.
Subscribe with Context: Subscriptions can be global or associated with a specific EventContext.
Contextual Delivery: The EventBus delivers an event only to handlers that are either global or belong to the same context as the event (or a compatible one, e.g., in a hierarchical system).
This functionality is not present in src/core/event_bus.py which uses a simpler, global subscription model.
6.4. EventContextValidator

Purpose: A utility to verify that events respect context boundaries, potentially raising errors or warnings if an event's context doesn't match the current processing context.
Can also verify that no events from a specific context "leak" and get processed outside of it.
6.5. Relationship with Scoped Containers

If "Scoped Containers" (as discussed in 3_CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md) are used, and each scoped container has its own instance of an EventBus, then the need for EventContextIsolation between these scopes is naturally handled by the container boundaries. EventContextIsolation would then be more relevant for finer-grained isolation needs within a single scope/container if multiple asynchronous or parallel tasks are managed by a single EventBus instance.

6.6. (Idealized) Hierarchical Event Contexts

For complex scenarios, contexts could be nested, allowing events to flow, for example, from parent contexts to child contexts but not vice-versa.

7. Event System Scalability (High-Frequency Trading) [Idealized Design]
For high-performance scenarios, the basic EventBus might become a bottleneck. docs/core/communication/EVENT_SCALABILITY.MD outlines several advanced designs:

PartitionedEventBus: Divides events into partitions (e.g., by domain: market_data, order, fill) with each partition processed by dedicated threads/tasks. This distributes load.
EventBusPartition Implementation: Each partition would have its own queue and processing logic, potentially its own thread pool.
LockFreeEventQueue: For critical paths, using lock-free queues (e.g., based on compare-and-swap primitives) can minimize contention and improve throughput compared to lock-based queues.
EventBatching: Grouping multiple small events into batches for processing can reduce overhead per event, improving overall throughput, especially for high-volume, less latency-critical events.
EventPrioritization (PriorityEventQueue): Implementing multiple queues based on event priority (e.g., HIGH for orders/fills, MEDIUM for signals, LOW for logging/monitoring events) ensures that critical events are processed with lower latency.
Benchmark-Driven Scaling / AdaptiveEventBus: An event bus that monitors its own performance (throughput, latency, queue backlogs) and dynamically adjusts resources, such as the number of worker threads per partition.
These scalability features are advanced and not yet implemented in src/core/event_bus.py.

8. Event Tracing, Debugging, and Monitoring
Understanding event flow is crucial for debugging.

EventTracer (Idealized Design):
Captures a log of events as they are published, possibly including the context in which they were published and handled.
Can be enabled/disabled and configured for trace length.
EventMonitor (Idealized Design):
Subscribes to various (or all) event types.
Collects statistics about event occurrences (counts, rates per type).
Stores a recent history of events for inspection.
Debugging Context Issues: Using the EventTracer logs to analyze if events are crossing context boundaries incorrectly.
These dedicated tracing and monitoring components are not yet present in src/core/.

9. Event Helpers
Utility functions can simplify the creation of standard event types, ensuring consistent data structures.

Python
# Example helper (from docs/core/communication/EVENT_SYSTEM.md)
# def create_bar_event(symbol, timestamp, open_price, high, low, close, volume):
#     data = { 'symbol': symbol, ... }
#     return Event(EventType.BAR, data)
These would typically reside in a utility module or alongside the Event class definition.

10. Best Practices
Clear Event Types: Use explicit and well-defined event types.
Consistent Event Structure: Adhere to the standard Event fields.
Idempotent Handlers (where applicable): Design handlers to be safe if an event is accidentally processed more than once (though the bus aims for once-only delivery).
Efficient Handlers: Keep event handlers lightweight and fast to avoid blocking the EventBus. Offload heavy processing to background tasks if necessary.
Proper Subscription Management: Use SubscriptionManager to manage the lifecycle of subscriptions, ensuring unsubscribe_all() is called during component teardown().
Context Discipline (Idealized): When event contexts are implemented, ensure events are published and handled within the correct context.
Minimize Event Payload: Include only necessary data in events.
Avoid Event Cycles: Be cautious of designs where components might create infinite loops of events.
11. Custom Event Exceptions
For issues related to event processing, especially context violations:

EventContextError
EventContextBoundaryError
These would be defined in src/core/exceptions.py.

This consolidated document provides a comprehensive view of the ADMF-Trader Event System, from its current basic implementation to the advanced idealized features for isolation and scalability.


---
This draft for `4_CORE_EVENT_SYSTEM.md` should cover the requested aspects. Please review it, and we can then proceed to document #5.
