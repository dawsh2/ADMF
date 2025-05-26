# ADMF Core: Advanced Dependency Management

## 1. Overview

Effective dependency management is crucial for building a modular, maintainable, and testable system like ADMF-Trader. This document details the advanced mechanisms for managing dependencies, including a robust Dependency Injection (DI) `Container`, a `DependencyGraph` for analysis and validation, strategies for state isolation using "Scoped Containers," and rules for ensuring proper dependency direction between modules.

The goal is to create a system where components are loosely coupled, dependencies are explicit and validated, and state leakage between operational scopes is minimized.

## 2. Core Concepts Recap

* **Dependency Injection (DI)**: Components do not create their dependencies; instead, dependencies are "injected" into them, typically during initialization. This promotes loose coupling.
* **Inversion of Control (IoC)**: The framework (specifically the `Bootstrap` system and `Container`) controls the creation and wiring of components, rather than components managing this themselves.

## 3. The `DependencyGraph` Component

To understand and manage the relationships between components, the system utilizes a `DependencyGraph`.

### 3.1. Purpose

* Represents all registered components as nodes and their dependencies as directed edges.
* Enables detection of circular dependencies.
* Facilitates determination of correct initialization and teardown order for components.
* Provides a basis for visualizing system architecture and component relationships.
* Supports validation of dependency integrity.

### 3.2. Key Features & Idealized Structure

The `DependencyGraph` (as implemented in `src/core/dependency_graph.py` and designed in `docs/core/foundation/DEPENDENCY_MANAGEMENT.md`) offers:

* **`add_component(component_name, metadata)`**: Adds a component node.
* **`add_dependency(component, dependency, metadata)`**: Adds a directed edge.
* **`detect_cycles()`**: Uses algorithms (e.g., from `networkx`) to find all circular dependencies.
* **`get_initialization_order()`**: Provides a topologically sorted list of components for safe initialization.
* **`get_dependencies(component)` / `get_dependents(component)`**: Queries direct relationships.
* **`get_component_metadata(component)`**: Retrieves stored metadata about a component (e.g., its module, class, whether it's required).
* **`validate()`**: Checks the integrity of the graph itself.
* **(Idealized) Visualization Hooks**: The design should support exporting the graph data in formats suitable for visualization tools (e.g., Graphviz, D3.js) to help developers understand system structure.

```python
# Conceptual usage within Bootstrap or a validator
# Assuming 'dg' is an instance of DependencyGraph from src/core/dependency_graph.py
dg.add_component("PortfolioService", metadata={"module": "risk", "required": True})
dg.add_component("RiskService", metadata={"module": "risk", "required": True})
dg.add_dependency("RiskService", "PortfolioService")

cycles = dg.detect_cycles()
if cycles:
    logger.error(f"Circular dependencies found: {cycles}")

init_order = dg.get_initialization_order()
```

## 4. Enhanced DI Container

The DI Container (building upon src/core/container.py) is central to managing component instances. The idealized Container integrates deeply with the DependencyGraph for robust dependency management.

### 4.1. Idealized Features

* **Registration with Dependency Information**: When components are registered (register_type, register_instance, register_factory), their declared dependencies should also be registered with the DependencyGraph.
* **Real-time Cycle Detection during Resolution**: The container.get(name, resolution_path) method should use the resolution_path to detect circular dependencies as it tries to resolve a component and its transitive dependencies. This provides immediate feedback if a problematic cycle is encountered during instantiation.
  * Current src/core/container.py does not implement the resolution_path for on-the-fly cycle detection during resolve; cycle detection relies on analyzing the DependencyGraph separately, often done by Bootstrap.
* **Automatic Constructor Injection**: Resolve and inject dependencies based on constructor parameter names and type hints. If a constructor parameter foo: FooService exists, the container attempts to resolve a component registered as "FooService" or "foo".
* **Support for Scoped Containers**: The Container design should be suitable for instantiation within different scopes (see Section 5).
* **Analysis and Reporting**: Methods like analyze_dependencies() and generate_dependency_report() providing insights into the container's state and component relationships.

### 4.2. Container.reset() and State

The reset() method in src/core/container.py clears cached singleton instances. This is vital for scenarios where the same container is reused across multiple isolated operations (e.g., within CleanBacktestEngine for different parameter sets in an optimization), ensuring that subsequent resolutions get fresh singleton instances where appropriate.

## 5. Scoped Containers for Enhanced State Isolation

For operations requiring stringent state isolation (e.g., individual optimization trials, parallel backtest executions), "Scoped Containers" provide a powerful mechanism.

### 5.1. Purpose and Mechanism

**Purpose**: To ensure that each distinct operational scope (like a single backtest run in an optimization loop) has its own completely isolated set of component instances. This definitively prevents state from one scope inadvertently affecting another.

**Mechanism**:
* An orchestrator (e.g., an Optimizer, ParallelBacktestManager, or the CleanBacktestEngine) takes responsibility for creating a new, independent Container instance for each operational scope or trial.
* All components necessary for that single trial (e.g., a specific strategy instance, its own portfolio, data handler view, execution simulator) are registered within this new, trial-specific container.
* Dependencies for these components are resolved only from this scoped container.
* Shared services (like a global configuration object, or a connection to a read-only historical database if truly shareable without side effects) could potentially be registered as existing instances passed into the scoped container, but mutable, stateful components critical to the trial's independent execution (Portfolio, Strategy state, etc.) must be fresh instances within the scope.
* Even the EventBus can be a fresh instance within each scoped container, naturally isolating event flow between scopes without needing complex EventContextIsolation rules between scopes. EventContextIsolation would then only be relevant for finer-grained isolation within a single scope/trial, if needed.
* Upon completion of the trial/scope, the entire scoped Container can be discarded, along with all the component instances it created and managed (that are not externally referenced). This provides a very clean and complete teardown of state.

### 5.2. Contrast with Component.reset()

* **Component.reset()**: Aims to revert a single, existing component instance to a clean state, typically its post-initialization state, while keeping its configuration. This is useful for sequential operations within the same container/scope.
* **Scoped Containers**: Offer a more robust isolation by providing entirely new component instances within a new container for each scope. This is generally preferred for parallelism or when absolute certainty against state pollution is required (as in optimization trials).

### 5.3. Integration with Bootstrap and Orchestrators

The main Bootstrap system initializes the primary application container.
For features like optimization, the Optimizer component (itself managed by the primary Bootstrap) would act as an orchestrator. For each optimization trial, the Optimizer (or a helper like CleanBacktestEngine) would:
Create a new Container() instance.
Potentially run a mini-bootstrap process for this scoped container: register necessary component classes/factories (often the same definitions as the main system but ensuring new instances).
Resolve and initialize components within this scoped container.
Run the backtest using these scoped components.
Teardown and discard the scoped container and its components.
This pattern is evident in CleanBacktestEngine, which creates fresh components for each backtest run, achieving the effect of a scoped container.
6. Dependency Validation
Ensuring dependencies are correctly defined and respected is critical for system stability. The idealized system includes several layers of validation.

6.1. DependencyDirectionValidator

Purpose: To enforce a defined module hierarchy (e.g., Core -> Data -> Strategy -> Risk -> Execution -> Analytics). This prevents lower-level modules from inappropriately depending on higher-level modules, maintaining a clean architectural structure.
Mechanism:
Assigns a "level" to each module.
Defines rules for which levels can depend on which other levels (e.g., Strategy can depend on Data and Core, but Data cannot depend on Strategy).
Validates all declared dependencies in the DependencyGraph against these rules.
Status: This component is part of the idealized design and not yet present in src/core/.
6.2. Early DependencyValidator System

Purpose: To perform comprehensive dependency checks at system load-time or pre-execution, catching issues before runtime.
Mechanism:
Integrates with the Container and DependencyGraph.
Checks Performed:
Missing Dependencies: Ensures all declared dependencies for registered components can be resolved in the container.
Circular Dependencies: Performs a global check for cycles in the DependencyGraph.
Direction Violations: Uses DependencyDirectionValidator.
Interface Compliance (Idealized): Check if concrete components correctly implement the interfaces they claim or are expected for their dependencies.
Reporting: Generates a report of all validation errors.
Status: The Bootstrap.validate_dependencies() in src/core/bootstrap.py performs some of these checks (cycles, missing dependencies using the graph). A separate, more comprehensive DependencyValidator class as detailed in the docs is a more advanced feature.
7. Dependency Injection Patterns & Best Practices
Constructor Injection: Preferred for mandatory dependencies. Components declare what they need in their __init__ (for type hinting and clarity) but receive them in initialize(context).

```python
class MyComponent(ComponentBase):
    # __init__ might hint at dependencies
    # def __init__(self, instance_name, config_key, data_service: DataService):
    #    super().__init__(instance_name, config_key)
    #    self._data_service = data_service # This is conceptual for type hint

    def initialize(self, context):
        super().initialize(context)
        self._data_service = context.container.resolve("DataService") # Actual injection
```

Setter/Method Injection: Can be used for optional dependencies, allowing them to be set after construction/initialization if available.
Interface-Based Dependencies: Depend on abstractions (ABCs, protocols) rather than concrete classes where possible, promoting flexibility. (See 1_CORE_CONCEPTS_AND_ARCHITECTURE.md).
Explicit Dependency Registration: Components (or their registration metadata) should clearly declare their dependencies for the DependencyGraph. src/core/bootstrap.py's STANDARD_COMPONENTS and ComponentDiscovery mechanisms handle this.
Documenting Dependencies: Docstrings or metadata should list a component's dependencies.
Resolving Circular Dependencies: If unavoidable (which should be rare with good design):
Refactor: Often indicates a design flaw that needs addressing by splitting components or redefining responsibilities.
Dependency Inversion Principle: Introduce an interface that both components depend on, inverting the direct dependency.
Mediator Pattern: Introduce a mediator component to manage interaction.
Event-Based Communication: Use events to decouple components instead of direct calls.
Lazy Loading/Resolution: Resolve the dependency only when it's first used (can hide underlying design issues if overused).
8. Custom Exceptions
The system should define custom exceptions for dependency-related issues to allow for specific error handling:

DependencyError (base)
CircularDependencyError
MissingDependencyError
InvalidDependencyDirectionError
(Your src/core/exceptions.py includes DependencyNotFoundError, which aligns with this.)

9. Visualization
The DependencyGraph is designed to be visualizable. While the core graph logic resides in src/core/dependency_graph.py, the actual rendering (e.g., to a PNG using Graphviz or to a JSON for D3.js) might be handled by separate utility scripts or a dedicated visualization component, using the graph data exported by DependencyGraph.

By implementing these advanced dependency management features, ADMF-Trader can achieve a highly modular, robust, and maintainable architecture where component relationships are clear, validated, and adhere to sound design principles. The "Scoped Container" concept is a key part of this for ensuring runtime state isolation.


---
This draft for `3_CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md` attempts to synthesize the extensive design from `docs/core/foundation/DEPENDENCY_MANAGEMENT.md` and integrate the "Scoped Container" idea more explicitly.

Please review this document. Let me know if it captures the key information correctly and where you'd like to refine it or add more detail, especially regarding how "Scoped Containers" should be architecturally positioned and used by systems like the `Optimizer`. Then, we can proceed to document #4.
