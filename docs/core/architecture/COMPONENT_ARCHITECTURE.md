# Component Architecture

## Overview

The ADMF-Trader system is built on a comprehensive component-based architecture. This architecture organizes the system into well-defined components with standard interfaces, clear responsibilities, and a consistent lifecycle. This approach enables high modularity, testability, and extensibility.

## Key Architectural Principles

1. **Component-Based Design**: All system elements follow a consistent design pattern with explicit interfaces
2. **Consistent Lifecycle**: Components have a standard lifecycle (initialize, start, stop, reset, teardown)
3. **Dependency Injection**: Components receive dependencies via a context object, promoting loose coupling
4. **Event-Driven Communication**: Components communicate through events, enabling scalability and extensibility
5. **Hierarchical Structure**: Components can contain and manage other components
6. **State Isolation**: Components properly isolate and manage state to ensure clean execution runs

## Component Structure

The component architecture is centered around the `Component` base class, which all system components inherit from:

```
Component (Base)
├── DataHandlerBase
│   ├── HistoricalDataHandler
│   ├── LiveDataHandler
│   └── ...
├── StrategyBase
│   ├── MAStrategy
│   ├── RsiStrategy
│   └── ...
├── RiskManagerBase
│   ├── BasicRiskManager
│   ├── PositionSizer
│   └── ...
└── ...
```

## Component Base Class

The `Component` base class serves as the foundation for all system components:

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

## Component Lifecycle

Components follow a consistent lifecycle:

1. **Construction**: The component is instantiated with a name and parameters
2. **Initialization**: Dependencies are injected and the component is set up
3. **Start**: The component begins active operation
4. **Stop**: The component halts active operation
5. **Reset**: The component's state is cleared, but its configuration remains
6. **Teardown**: The component releases resources and prepares for destruction

## Component Registration and Discovery

Components are registered with the system through a central registry, which enables:

1. **Component Discovery**: Finding components by name, type, or capability
2. **Dependency Resolution**: Automatically resolving component dependencies
3. **Lifecycle Management**: Managing component lifecycle transitions
4. **Configuration**: Applying configuration to components

## Component Composition

Components can be composed into hierarchies:

```python
class CompositeComponent(Component):
    """A component that contains other components."""
    
    def __init__(self, name, parameters=None):
        super().__init__(name, parameters)
        self.components = []
        
    def add_component(self, component):
        """Add a child component."""
        self.components.append(component)
        
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

## Component Context

Components receive a context object during initialization that provides access to dependencies:

```python
class Context:
    """Container for component context and dependencies."""
    
    def __init__(self):
        """Initialize context."""
        self._entries = {}
        
    def register(self, name, instance):
        """Register a dependency."""
        self._entries[name] = instance
        
    def get(self, name, default=None):
        """Get a dependency by name."""
        return self._entries.get(name, default)
        
    def has(self, name):
        """Check if a dependency exists."""
        return name in self._entries
```

## Component Parameters

Components receive parameters during construction and can access them in a standardized way:

```python
# In component initialization
self.max_items = self.parameters.get('max_items', 1000)  # With default
self.enabled = self.parameters.get('enabled')  # Without default
```

## Component Error Handling

Components follow a consistent error handling pattern:

```python
def process_data(self, data):
    """Process data with proper error handling."""
    try:
        result = self._do_process(data)
        return result
    except Exception as e:
        self.logger.error(f"Error processing data: {e}")
        if self.event_bus:
            error_event = ErrorEvent(
                component=self.name,
                error=str(e),
                traceback=traceback.format_exc()
            )
            self.event_bus.publish(error_event)
        return None
```

## Component State Verification

The component architecture includes state verification to ensure proper state management:

```python
def reset(self):
    """Reset component state with verification."""
    # Take pre-reset snapshot
    self.state_verifier.take_snapshot(self, "before_reset")
    
    # Reset state
    self.data = []
    self.calculations = {}
    
    # Verify reset was successful
    reset_status = self.state_verifier.verify_reset(self)
    if not reset_status:
        self.logger.warning(f"Component {self.name} not properly reset")
```

## Component Introspection

Components support introspection to examine their state and capabilities:

```python
def get_status(self):
    """Get component status information."""
    return {
        'name': self.name,
        'type': self.__class__.__name__,
        'initialized': self.initialized,
        'running': self.running,
        'parameters': self.parameters,
        'dependencies': self._get_dependency_info(),
        'state': self._get_state_info()
    }
```

By following these patterns consistently throughout the system, we create a robust component architecture that enables modular, maintainable, and testable code while supporting complex functionality and interactions.