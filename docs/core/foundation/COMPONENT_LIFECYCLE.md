# Component Lifecycle Management

## Overview

The Component Lifecycle system is a core feature of the ADMF-Trader architecture, providing a standardized way to manage the creation, initialization, operation, and cleanup of all system components. This consistent lifecycle approach ensures proper resource management, state isolation, and deterministic behavior.

## Component States

Components have the following defined states:

1. **CREATED**: Initial state after instantiation
2. **INITIALIZED**: After dependencies are injected and component is ready
3. **RUNNING**: During active operation
4. **STOPPED**: After operation completes
5. **DISPOSED**: After resources are released

## Lifecycle Methods

The Component base class defines standard lifecycle methods:

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

## Lifecycle Method Responsibilities

### 1. `__init__(name, parameters=None)`

The constructor is responsible for:
- Setting component name and parameters
- Initializing internal state variables
- NOT accessing external resources or services
- NOT requiring dependencies

```python
def __init__(self, name, parameters=None):
    """Initialize component with name and parameters."""
    super().__init__(name, parameters or {})
    
    # Initialize internal state
    self.positions = {}
    self.signals = []
    self.current_bar = None
    
    # Extract parameter values with defaults
    self.max_positions = parameters.get('max_positions', 10)
    self.position_size = parameters.get('position_size', 100)
```

### 2. `initialize(context)`

The initialize method is responsible for:
- Extracting dependencies from the context
- Setting up resources and connections
- Initializing event subscriptions
- Validating configuration

```python
def initialize(self, context):
    """Initialize component with dependencies."""
    super().initialize(context)
    
    # Extract dependencies
    self.data_handler = context.get('data_handler')
    if not self.data_handler:
        raise ComponentError(f"Component {self.name} requires data_handler")
        
    self.portfolio = context.get('portfolio')
    
    # Initialize resources
    self.db_connection = self._create_db_connection()
    
    # Validate configuration
    if self.max_positions <= 0:
        raise ValueError("max_positions must be positive")
```

### 3. `initialize_event_subscriptions()`

This method is responsible for:
- Setting up event subscriptions
- Creating subscription manager
- Defining event handlers

```python
def initialize_event_subscriptions(self):
    """Set up event subscriptions."""
    self.subscription_manager = SubscriptionManager(self.event_bus)
    self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
    self.subscription_manager.subscribe(EventType.SIGNAL, self.on_signal)
```

### 4. `start()`

The start method is responsible for:
- Beginning active operations
- Starting background threads or tasks
- Initiating data processing

```python
def start(self):
    """Begin component operation."""
    super().start()
    
    # Start background tasks
    self.processing_thread = threading.Thread(
        target=self._process_queue,
        name=f"{self.name}_processor"
    )
    self.processing_thread.start()
    
    # Signal readiness
    self.logger.info(f"Component {self.name} started")
```

### 5. `stop()`

The stop method is responsible for:
- Halting active operations
- Stopping background threads or tasks
- Preserving state for potential restart

```python
def stop(self):
    """End component operation."""
    # Signal threads to stop
    self._stop_requested = True
    
    # Wait for threads to terminate
    if hasattr(self, 'processing_thread') and self.processing_thread:
        self.processing_thread.join(timeout=5.0)
    
    # Call parent method
    super().stop()
    
    self.logger.info(f"Component {self.name} stopped")
```

### 6. `reset()`

The reset method is responsible for:
- Clearing internal state
- Preserving configuration
- Preparing for a new run
- Ensuring clean state separation between runs

```python
def reset(self):
    """Clear component state for a new run."""
    # Clear state collections
    self.positions = {}
    self.signals = []
    self.current_bar = None
    
    # Reset statistics
    self.trade_count = 0
    self.win_count = 0
    self.loss_count = 0
    
    # Reset state tracking
    self._last_update_time = None
    
    # Note: Configuration (self.parameters) is preserved
    
    self.logger.debug(f"Component {self.name} reset")
```

### 7. `teardown()`

The teardown method is responsible for:
- Releasing external resources
- Closing connections
- Unsubscribing from events
- Final cleanup before destruction

```python
def teardown(self):
    """Release resources and perform final cleanup."""
    # Close database connection
    if hasattr(self, 'db_connection') and self.db_connection:
        self.db_connection.close()
        
    # Release file handles
    if hasattr(self, 'log_file') and self.log_file:
        self.log_file.close()
        
    # Unsubscribe from events (handled by parent)
    super().teardown()
    
    self.logger.info(f"Component {self.name} teardown complete")
```

## Lifecycle Transitions

Components transition between states in a well-defined sequence:

1. **CREATED → INITIALIZED**: Through `initialize(context)` method
2. **INITIALIZED → RUNNING**: Through `start()` method
3. **RUNNING → STOPPED**: Through `stop()` method
4. **STOPPED → RUNNING**: Through `start()` method (restart)
5. **STOPPED → INITIALIZED**: Through `reset()` method
6. **INITIALIZED → DISPOSED**: Through `teardown()` method

## State Verification

The system includes mechanisms to verify proper state transitions and cleanup:

```python
class StateVerifier:
    """Verifies that component state is properly managed."""
    
    def __init__(self):
        """Initialize state verifier."""
        self.snapshots = {}
        self.enabled = True
        
    def take_snapshot(self, component, key=None):
        """Take a snapshot of component state."""
        if not self.enabled:
            return None
            
        key = key or f"{component.name}_{id(component)}"
        snapshot = StateSnapshot(component)
        self.snapshots[key] = snapshot
        return snapshot
        
    def verify_reset(self, component, original_key=None, reset_key=None):
        """Verify that a component has been properly reset."""
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
```

## Lifecycle Events

The system emits events for important lifecycle transitions:

```python
class LifecycleEvent(Event):
    """Event for component lifecycle transitions."""
    
    def __init__(self, component_name, transition, previous_state, current_state, timestamp=None):
        """Initialize lifecycle event."""
        data = {
            'component_name': component_name,
            'transition': transition,
            'previous_state': previous_state,
            'current_state': current_state
        }
        super().__init__(EventType.LIFECYCLE, data, timestamp)
```

## Composite Component Lifecycle

Composite components (components that contain other components) follow special rules:

```python
class CompositeComponent(Component):
    """A component that contains other components."""
    
    def initialize(self, context):
        """Initialize this component and all children."""
        super().initialize(context)
        
        # Initialize all child components
        for component in self.components:
            if not component.initialized:
                component.initialize(context)
                
    def start(self):
        """Start this component and all children."""
        super().start()
        
        # Start all child components
        for component in self.components:
            if not component.running:
                component.start()
                
    def stop(self):
        """Stop this component and all children."""
        # Stop all child components
        for component in self.components:
            if component.running:
                component.stop()
                
        super().stop()
        
    def reset(self):
        """Reset this component and all children."""
        # Reset all child components
        for component in self.components:
            component.reset()
            
        super().reset()
        
    def teardown(self):
        """Teardown this component and all children."""
        # Teardown all child components
        for component in self.components:
            component.teardown()
            
        super().teardown()
```

## Best Practices

1. **Always Call Super**: Child classes should always call the parent class method
2. **Clean Reset**: The `reset()` method should return the component to its post-initialization state
3. **Resource Management**: Acquire resources in `initialize()` and release them in `teardown()`
4. **Validation**: Validate dependencies and configuration in `initialize()`
5. **State Transitions**: Respect the defined state transitions
6. **Idempotence**: Methods should be idempotent (safe to call multiple times)
7. **Thread Safety**: Methods should be thread-safe if components can be accessed from multiple threads
8. **Error Handling**: Handle errors gracefully during lifecycle transitions

## Example Implementation

Here's a complete example of proper lifecycle implementation:

```python
class ExampleStrategy(Component):
    """Example strategy with proper lifecycle implementation."""
    
    def __init__(self, name, parameters=None):
        """Initialize strategy with name and parameters."""
        super().__init__(name, parameters or {})
        
        # Initialize state
        self.positions = {}
        self.signals = []
        self.indicators = {}
        
        # Extract parameters with defaults
        self.max_positions = self.parameters.get('max_positions', 10)
        self.position_size = self.parameters.get('position_size', 100)
        
    def initialize(self, context):
        """Initialize with dependencies."""
        super().initialize(context)
        
        # Extract dependencies
        self.data_handler = context.get('data_handler')
        if not self.data_handler:
            raise ComponentError(f"Strategy {self.name} requires data_handler")
            
        self.portfolio = context.get('portfolio')
        
        # Initialize indicators
        self.indicators['ma_fast'] = MovingAverage(10)
        self.indicators['ma_slow'] = MovingAverage(30)
        
        # Initialize resources
        self.performance_log = open(f"{self.name}_performance.log", "w")
        
        self.logger.info(f"Strategy {self.name} initialized")
        
    def initialize_event_subscriptions(self):
        """Set up event subscriptions."""
        self.subscription_manager = SubscriptionManager(self.event_bus)
        self.subscription_manager.subscribe(EventType.BAR, self.on_bar)
        
    def start(self):
        """Begin strategy operation."""
        super().start()
        self.logger.info(f"Strategy {self.name} started")
        
    def stop(self):
        """Stop strategy operation."""
        super().stop()
        self.logger.info(f"Strategy {self.name} stopped")
        
    def reset(self):
        """Reset strategy state."""
        # Clear state
        self.positions = {}
        self.signals = []
        
        # Reset indicators
        for indicator in self.indicators.values():
            indicator.reset()
            
        self.logger.debug(f"Strategy {self.name} reset")
        
    def teardown(self):
        """Release resources."""
        # Close files
        if hasattr(self, 'performance_log') and self.performance_log:
            self.performance_log.close()
            
        # Call parent method
        super().teardown()
        
        self.logger.info(f"Strategy {self.name} teardown complete")
```

By following these lifecycle patterns consistently, the ADMF-Trader system ensures proper resource management, state isolation, and deterministic behavior across all components.