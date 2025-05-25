# ADMF Core: System Architecture & Component Model

## 1. Overview

The Core module of the ADMF-Trader system provides the foundational infrastructure and architectural patterns upon which all other system modules are built. It establishes a robust, component-based design that promotes modularity, testability, extensibility, and maintainability.

This document outlines the fundamental architectural principles, the definition of a "Component," its standardized lifecycle, and how components interact within the system. The goal is to ensure a consistent and understandable structure for all parts of ADMF-Trader.

## 2. Key Architectural Principles

The ADMF-Trader system, and particularly its Core module, is built upon several key architectural principles:

* **Component-Based Design**: All system functionality is encapsulated in modular components with standardized interfaces and lifecycles. Each component has a well-defined responsibility.
* **Consistent Lifecycle**: All components adhere to a standard lifecycle (e.g., CREATED, INITIALIZED, RUNNING, STOPPED, DISPOSED), ensuring predictable state management and resource handling across the system.
* **Dependency Injection (DI)**: Components receive their dependencies through a context object or a DI container during initialization, promoting loose coupling and testability. (Detailed in `CORE_BOOTSTRAP_CONFIGURATION_DEPENDENCIES.md` and `CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md`).
* **Event-Driven Communication**: Components primarily interact through a publish-subscribe event system, enabling scalability, extensibility, and further decoupling. (Detailed in `CORE_EVENT_SYSTEM.md`).
* **Hierarchical Structure / Composition**: Components can be composed of other components, allowing for the construction of complex functionalities from simpler, reusable parts.
* **State Isolation**: Careful management and isolation of component state are critical, especially to ensure clean execution runs for backtesting and optimization. (Strategies for this, including Scoped Containers, are detailed in `CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md`).
* **Interface-Based Design**: Components should, where practical, depend on abstractions (interfaces or Abstract Base Classes) rather than concrete implementations, allowing for greater flexibility and interchangeability.

## 3. The Core Component Model (`ComponentBase`)

The cornerstone of the architecture is the `ComponentBase` class (or an interface defining its contract). All manageable parts of the ADMF-Trader system should inherit from or implement this base.

**Definition and Purpose:**
`ComponentBase` provides a standard contract for all components, ensuring they integrate smoothly into the system's lifecycle and management framework. It defines common properties and methods related to initialization, runtime state, configuration, and cleanup.

**Idealized Code Structure (Python):**
(This structure is based on `docs/core/architecture/INTERFACE_DESIGN.MD` and `src/core/component_base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Forward declaration of SystemContext for type hinting if preferred
# class SystemContext: pass
# Forward declaration for SubscriptionManager
# class SubscriptionManager: pass
# Forward declaration for Logger
# class Logger: pass
# Forward declaration for EventBus
# class EventBus: pass
# Forward declaration for Config
# class Config: pass


class ComponentBase(ABC):
    """
    Base interface for all system components.
    Defines the standard lifecycle and core functionalities.
    """

    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """
        Minimal constructor for a component.
        Sets the instance name and optional configuration key.
        Actual dependencies and full configuration are handled in initialize().

        Args:
            instance_name: Unique name for this component instance.
            config_key: Key in the main configuration file where this
                        component's specific settings can be found.
        """
        self.instance_name: str = instance_name
        self.config_key: Optional[str] = config_key
        self.initialized: bool = False
        self.running: bool = False
        
        # Dependencies to be injected by initialize()
        self.context: Optional[Any] = None # Should be SystemContext
        self.config_loader: Optional[Any] = None # Specific type if available
        self.config: Optional[Any] = None # Specific type if available (e.g., main Config object)
        self.component_config: Dict[str, Any] = {}
        self.event_bus: Optional[Any] = None # Specific type if available
        self.container: Optional[Any] = None # Specific type if available
        self.logger: Optional[Any] = None # Specific type if available (e.g., logging.Logger)
        self.subscription_manager: Optional[Any] = None # Specific type if available

    @abstractmethod
    def initialize(self, context: Any) -> None: # context should ideally be SystemContext
        """
        Initialize component with dependencies from context.
        Responsibilities:
        - Extract dependencies (config, logger, event_bus, other components via container).
        - Load component-specific configuration.
        - Set up resources and connections.
        - Initialize event subscriptions via initialize_event_subscriptions().
        - Perform component-specific setup via _initialize().
        - Validate configuration via _validate_configuration().
        - Set self.initialized = True.

        Args:
            context: System context containing shared services and configuration.
        """
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """
        Component-specific initialization logic.
        To be implemented by subclasses for their specific setup tasks.
        """
        pass

    def initialize_event_subscriptions(self) -> None:
        """
        Set up event subscriptions using self.subscription_manager.
        Called automatically during initialize() if an event bus is available.
        Subclasses should override this if they need to subscribe to events.
        """
        pass

    def _validate_configuration(self) -> None:
        """
        Validate component-specific configuration.
        Subclasses can override this. Raises ValueError on invalid config.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Begin component operation.
        Responsibilities:
        - Start active operations (e.g., processing data, listening for requests).
        - Start background threads or tasks if any.
        - Set self.running = True.
        Prerequisite: Component must be initialized.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        End component operation.
        Responsibilities:
        - Halt active operations gracefully.
        - Stop background threads or tasks.
        - Preserve necessary state for potential restart.
        - Set self.running = False.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Clear component's internal state to prepare for a new run or reuse.
        Responsibilities:
        - Reset internal data structures and state variables.
        - Configuration (parameters) should generally be preserved.
        - Ensure the component is in a state similar to post-initialization but clean.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Release all resources held by the component and prepare for destruction.
        Responsibilities:
        - Unsubscribe from all events.
        - Close any open connections (files, network, databases).
        - Release external resources.
        - Set self.initialized = False and self.running = False.
        """
        pass

    @property
    def name(self) -> str: # From INTERFACE_DESIGN.MD
        """Get component instance name."""
        return self.instance_name
		
Explanation:

instance_name: A unique identifier for the component instance.
config_key: An optional key used by the Bootstrap system to fetch this component's specific configuration from the main application configuration.
4. Component Lifecycle Management
Components in ADMF-Trader follow a well-defined lifecycle, ensuring predictable behavior, proper resource management, and effective state isolation, particularly crucial for backtesting and optimization.

4.1. Component States

A component transitions through the following states:

CREATED: The initial state immediately after the component's __init__ method has been called. At this point, the component has performed minimal setup and should not have acquired external resources or resolved dependencies.
INITIALIZED: The state after the initialize(context) method has successfully completed. Dependencies are injected, component-specific configuration is loaded and validated, resources may be acquired, and event subscriptions are set up. The component is ready to start but is not yet actively operating.
RUNNING: The state after the start() method has been called. The component is actively performing its operations (e.g., processing data, generating signals, handling orders).
STOPPED: The state after the stop() method has been called. Active operations are halted. The component's internal state should be preserved to allow for a potential restart or reset.
DISPOSED: The state after the teardown() method has completed. All resources are released, event subscriptions are cleared, and the component is no longer operational and should be ready for garbage collection.
4.2. Standard Lifecycle Methods

The ComponentBase defines abstract or overridable methods corresponding to these lifecycle phases. Concrete components must implement or extend these.

__init__(self, instance_name: str, config_key: Optional[str] = None)

Responsibility: Minimal setup. Store instance_name and config_key. Initialize internal state variables to their defaults. Crucially, do not attempt to resolve dependencies or access external services/configuration here. This ensures components can be instantiated easily, for example, during unit testing or by a discovery mechanism, before the full system context is available.
From src/core/component_base.py: The constructor correctly adheres to this by only setting instance name, config key, and default states for initialized, running, and dependency placeholders.
initialize(self, context: SystemContext)

Responsibility: Prepare the component for operation. This is where dependencies are injected from the context (which should be a SystemContext object providing access to config_loader, the main config, event_bus, container, and a logger).
It typically involves:
Storing references to dependencies (e.g., self.event_bus = context.event_bus).
Loading its specific configuration using self.config_key and context.config_loader (or context.config). Storing this in self.component_config.
Calling self.initialize_event_subscriptions() if an event_bus is available.
Calling a subclass-specific _initialize() method for custom setup.
Calling self._validate_configuration() for custom config validation.
Finally, setting self.initialized = True.
From src/core/component_base.py: The initialize method correctly extracts standard dependencies from context, loads component-specific config, and calls initialize_event_subscriptions, _initialize, and _validate_configuration.
_initialize(self) -> None (To be implemented by subclasses)

Responsibility: Contains the component-specific initialization logic after standard dependencies are available. This might include setting up internal data structures based on configuration, initializing models, or preparing specific resources.
initialize_event_subscriptions(self) -> None

Responsibility: Set up all event subscriptions required by the component using self.subscription_manager (which should be initialized in ComponentBase.initialize if an event_bus is present). This method is called by initialize().
From docs/core/foundation/COMPONENT_LIFECYCLE.MD: Example shows using a SubscriptionManager. Your src/core/component_base.py correctly initializes self.subscription_manager.
_validate_configuration(self) -> None (To be implemented by subclasses)

Responsibility: Perform any component-specific validation of its configuration parameters. Should raise an appropriate exception (e.g., ConfigurationError from src/core/exceptions.py) if validation fails.
start(self) -> None

Responsibility: Begin active operation of the component. This could involve starting background threads, initiating data processing loops, or signaling readiness to other components. It should only be called after initialize() has successfully completed. Sets self.running = True.
From src/core/foundation/COMPONENT_LIFECYCLE.MD: Mentions starting background tasks and signaling readiness.
stop(self) -> None

Responsibility: Gracefully halt active operations. This includes stopping any background threads or tasks. The component's internal state should generally be preserved, as it might be restarted or reset. Sets self.running = False.
From docs/core/foundation/COMPONENT_LIFECYCLE.MD: Highlights signaling threads to stop and preserving state.
reset(self) -> None

Responsibility: Clear the component's internal state, returning it to a condition similar to just after initialize(), but preserving its configuration and dependencies. This is critical for ensuring state isolation between runs, such as in optimization loops or sequential backtests.
From docs/core/foundation/COMPONENT_LIFECYCLE.MD: Emphasizes clearing internal state while preserving configuration.
teardown(self) -> None

Responsibility: Release all external resources (e.g., file handles, network connections, database connections), unsubscribe from all events (typically handled by SubscriptionManager.unsubscribe_all()), and perform any final cleanup before the component instance is destroyed. After teardown(), the component should not be used further. Sets self.initialized = False and self.running = False.
From src/core/component_base.py: Correctly calls unsubscribe_all() on subscription_manager.
4.3. Lifecycle Transitions

Components transition between states in a defined sequence:

__init__ -> CREATED
CREATED --initialize()--> INITIALIZED
INITIALIZED --start()--> RUNNING
RUNNING --stop()--> STOPPED
STOPPED --start()--> RUNNING (Restart)
STOPPED --reset()--> INITIALIZED (State cleared, ready for new run)
Any operational state (INITIALIZED, RUNNING, STOPPED) --teardown()--> DISPOSED
4.4. State Verification in Lifecycle

To ensure proper state management, particularly the effectiveness of reset(), a StateVerifier component can be used (as designed in docs/core/foundation/COMPONENT_LIFECYCLE.MD and docs/core/infrastructure/TESTING_STRATEGY.MD). This involves taking snapshots of a component's state before and after reset to confirm it has returned to a clean initial state. (Detailed in CORE_INFRASTRUCTURE_SERVICES.md).

4.5. Lifecycle Events

The system can emit LifecycleEvent (e.g., COMPONENT_INITIALIZED, COMPONENT_STARTED) to allow other parts of the system or monitoring tools to react to component state changes. (Detailed in CORE_EVENT_SYSTEM.md).

5. Component Composition
Components can be composed to build more complex functionalities. A CompositeComponent would manage the lifecycle of its child components, ensuring that lifecycle methods (initialize, start, stop, reset, teardown) are propagated to its children in the correct order.

Example CompositeComponent Lifecycle Propagation:

initialize(): Initializes itself, then all children.
start(): Starts itself, then all children.
stop(): Stops all children first, then itself (often in reverse order of start).
reset(): Resets itself, then all children.
teardown(): Tears down all children first (often in reverse order of initialization), then itself.
(A generic CompositeComponent base class is a potential addition to src/core as per the gap analysis).

6. Component Context and Dependency Injection (Introduction)
During the initialize(context) phase, components receive a SystemContext object (or a similar context structure). This object acts as a service locator or provides access to a Dependency Injection (DI) Container.
The SystemContext typically provides:

Access to the global application Config.
The system EventBus.
A Logger instance.
The DI Container for resolving other component dependencies.
This mechanism decouples components from direct knowledge of how their dependencies are created or located. (Detailed in CORE_BOOTSTRAP_CONFIGURATION_DEPENDENCIES.md and CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md).

7. Component Parameters
Components are instantiated with an instance_name and an optional config_key. The config_key is used during the initialize phase to look up component-specific parameters from the main application configuration loaded by the Bootstrap system. Components should provide a way to access these parameters, often with defaults if specific values are not found in the configuration.
Your ComponentBase in src/core/component_base.py stores these in self.component_config.

8. Component Error Handling (Introduction)
Components should follow consistent error handling patterns, typically involving try-except blocks for critical operations and potentially publishing ErrorEvent instances via the EventBus. A more comprehensive error handling strategy, including custom exception hierarchies and ErrorBoundary patterns, is detailed in CORE_INFRASTRUCTURE_SERVICES.md.

9. Component Introspection
Components may support introspection to allow other parts of the system (especially debugging or monitoring tools) to examine their current state, parameters, and capabilities. A standardized get_status() method, as suggested in docs/core/architecture/COMPONENT_ARCHITECTURE.MD, can facilitate this. Your ComponentBase in src/core/component_base.py includes a get_status() method.

10. Interface Design Philosophy
The ADMF-Trader system emphasizes an interface-based design to achieve loose coupling and modularity.
Key tenets include:

Explicit Contracts: Interfaces clearly define the methods and properties a component provides or requires. ComponentBase itself serves as such a contract.
Dependency Inversion: High-level modules should not depend on low-level modules; both should depend on abstractions (interfaces).
Isolation: Interfaces allow modules to be developed and tested independently.
Extensibility: New implementations of an interface can be added without altering existing client code.
Configuration-Driven: The selection of concrete implementations for interfaces can be driven by external configuration.
Type Safety: Interfaces, especially when using Python's type hinting with Abstract Base Classes (ABCs), help enforce type safety.
Execution Mode Agnostic: Interfaces should be designed to allow for both synchronous and asynchronous implementations where applicable, supporting different system run modes. (Asynchronous aspects detailed in CORE_CONCURRENCY_AND_ASYNCHRONOUS.md).
By adhering to these architectural concepts and the component model, ADMF-Trader aims for a robust, maintainable, and scalable system.


---

This consolidated document, `1_CORE_CONCEPTS_AND_ARCHITECTURE.md`, covers the foundational architectural ideas.

**Next Steps:**

1.  **Review this document:** Please check if it accurately captures the essential, non-obvious information from the source documents (`COMPONENT_ARCHITECTURE.MD`, `COMPONENT_LIFECYCLE.MD`, and parts of `INTERFACE_DESIGN.MD` and `IMPLEMENTATION.MD`) and if it aligns with your current vision.
2.  **Identify Missing Nuances:** Let me know if any critical details or design rationales from the original documents were lost or misrepresented.
3.  **Choose the Next Document:** Once you're satisfied with this one, we can move on to the next consolidated document from the plan (e.g., `2_CORE_BOOTSTRAP_CONFIGURATION_DEPENDENCIES.md`).

This iterative process will help build up your target documentation.		
