# ADMF Core: Bootstrap, Configuration & Basic Dependencies

## 1. Overview

This document details the ADMF-Trader system's startup and configuration management processes. It covers the `Bootstrap` system, which orchestrates application initialization, the comprehensive `Configuration Management` framework, and the basic principles of how components are registered and receive their initial dependencies.

The primary goal is to ensure a consistent, configurable, and robust system startup that correctly assembles all necessary components according to the defined architecture and runtime environment.

## 2. The Bootstrap System

The `Bootstrap` system is the central orchestrator for initializing the entire ADMF-Trader application. It ensures that all components are loaded, configured, and started in a controlled and predictable manner.

### 2.1. Key Responsibilities

* **Single Initialization Path**: Provides a unified way to start the application across different run modes (e.g., Production, Backtest, Optimization).
* **Configuration Loading**: Loads and validates application-wide and component-specific configurations. (See Section 3: Configuration Management).
* **Dependency Injection (DI) Container Setup**: Creates and populates the main DI `Container`. (See Section 4: Basic Dependency Management & Container).
* **Component Discovery and Registration**: Finds available system components (either through explicit definitions or discovery mechanisms) and registers them with the DI `Container`.
* **Component Lifecycle Management**: Manages the lifecycle of all registered components, ensuring they are initialized, started, stopped, and torn down in the correct order and according to the defined `ComponentBase` contract. (Lifecycle details in `1_CORE_CONCEPTS_AND_ARCHITECTURE.md`).
* **System Context Creation**: Establishes a `SystemContext` that provides shared services like the `Config`, `EventBus`, `Logger`, and `Container` to all components.
* **Run Mode Awareness**: Adapts its behavior and component setup based on the current `RunMode` (e.g., `RunMode.OPTIMIZATION`, `RunMode.BACKTEST`).
* **Error Handling**: Manages errors that occur during the startup sequence.
* **Extensibility**: Provides hooks for customizing the bootstrap process.

### 2.2. Bootstrap Process Steps

The typical bootstrap process, as implemented in `src/core/bootstrap.py` and conceptualized in `docs/core/foundation/BOOTSTRAP_SYSTEM.md`, involves:

1.  **Minimal Setup (e.g., `ApplicationLauncher`)**:
    * Parse command-line arguments.
    * Load the initial `Config` object (e.g., from a YAML file specified via CLI).
    * Determine the `RunMode`.
    * Instantiate the `Bootstrap` class.

2.  **`Bootstrap.initialize(config, run_mode, metadata)`**:
    * Sets up initial logging (which might be further refined by the loaded config).
    * Creates the main DI `Container`.
    * Creates the primary `EventBus`.
    * Populates the `SystemContext` with `config`, `container`, `event_bus`, `logger`, `run_mode`, and any passed `metadata` (like CLI args).
    * Registers these core services into the `Container`.

3.  **`Bootstrap.setup_managed_components(component_overrides)`**:
    * **Component Definition Loading**: Merges standard component definitions with those discovered (e.g., from `component_meta.yaml` files or other mechanisms). The `src/core/bootstrap.py` uses `ComponentDiscovery` for this.
    * **Component Creation**: Iterates through the components designated for the current `RunMode`. For each:
        * Runs `pre_component_create` hooks.
        * Instantiates the component using its minimal constructor (as per `ComponentBase`).
        * Stores the component instance and registers it with the `Container`.
        * Adds the component and its declared dependencies to the `DependencyGraph`.
        * Runs `post_component_create` hooks.
    * **Component Initialization (`Bootstrap.initialize_components()`)**:
        * Validates the overall `DependencyGraph` for issues like missing or circular dependencies.
        * Determines the correct initialization order from the `DependencyGraph`.
        * Iterates through components in order, calling their `initialize(system_context)` method, thereby injecting dependencies.

4.  **`Bootstrap.start_components()`**:
    * Runs `pre_start` hooks.
    * Iterates through components in initialization order, calling their `start()` method.
    * Runs `post_start` hooks.

5.  **`Bootstrap.execute_entrypoint(method_name)`**:
    * Identifies the configured "entrypoint component" for the current `RunMode` (e.g., `AppRunner`).
    * Calls the specified method (e.g., `execute()`) on this entrypoint component to begin the main application logic.

6.  **`Bootstrap.teardown()`** (often via context manager `__exit__`):
    * Runs `pre_teardown` hooks.
    * Calls `stop_components()` (in reverse initialization order).
    * Calls `teardown()` on each component (in reverse initialization order).
    * Resets the `Container` and clears event subscriptions.
    * Runs `post_teardown` hooks.

### 2.3. Entrypoint Component (`AppRunner`)

The `Bootstrap` system launches a designated "entrypoint component" (like `AppRunner`) whose `execute()` method contains the main application flow for a given `RunMode`. This keeps `main.py` and `ApplicationLauncher` minimal and delegates application control to a managed component.

### 2.4. Lifecycle Hooks

The `Bootstrap` class includes a hook system (`register_hook`, `_run_hooks`) allowing external code to plug into various stages of the startup and shutdown process, providing extensibility.

## 3. Configuration Management

Effective configuration management is vital for adapting ADMF-Trader to different environments (development, testing, production) and for managing strategy parameters. The idealized system is described in `docs/core/foundation/CONFIGURATION_MANAGEMENT.md`.

### 3.1. Idealized `ConfigurationProvider` Features

The target `ConfigurationProvider` (an evolution of the current `Config` class in `src/core/config.py`) should support:

* **Hierarchical Loading**: A layered approach to loading configurations:
    1.  `defaults.yaml` (base values).
    2.  `{environment}.yaml` (e.g., `development.yaml`, `production.yaml`) overriding defaults.
    3.  `instance.yaml` (instance-specific overrides, not version-controlled).
    4.  Environment Variables (e.g., `ADMF_CONFIG_SECTION_KEY=value`) as final overrides.
* **Schema Validation**: Validate the loaded configuration against a defined schema (e.g., `schema.yaml` using `jsonschema`) to ensure correctness and completeness before system use.
* **Secure Credential Management (`CredentialManager`)**:
    * Encrypt and decrypt sensitive values (e.g., API keys, database passwords) stored in configuration.
    * Use a master key (potentially from an environment variable or a secure vault service) for encryption.
* **Dynamic Configuration Updates (`ConfigurationWatcher`)**:
    * Monitor configuration files for changes at runtime.
    * Reload configuration and notify relevant components if changes are detected (for systems designed to support hot-reloading).
* **Type Conversion and Accessors**: Provide methods to get configuration values with type conversion (e.g., `get_int`, `get_bool`) and access nested sections using dot notation (e.g., `config.get("data_handler.parameters.symbol")`). The current `Config` class in `src/core/config.py` provides basic `get` and `set` with dot notation.
* **Custom Validators**: Allow registration of custom validation functions for specific configuration paths.

### 3.2. Current Implementation (`src/core/config.py`)

* The `SimpleConfigLoader` handles loading from a single YAML file.
* The `Config` class wraps the loader, provides `get`/`set` methods, basic environment variable overriding (for keys starting with `ADMF_`), and can save the current state.
* It lacks the advanced hierarchical loading, schema validation, credential management, and dynamic update features of the idealized `ConfigurationProvider`.

## 4. Basic Dependency Management & Container

The DI `Container` is responsible for instantiating and providing components and their dependencies. While advanced features are covered in `CORE_DEPENDENCY_MANAGEMENT_ADVANCED.md`, the `Bootstrap` system relies on its basic functionalities.

### 4.1. Container Responsibilities in Bootstrap

* **Component Registration**: `Bootstrap` registers all discovered and defined components (or their classes/factories) with the `Container`.
* **Instance Storage**: The `Container` stores singleton instances.
* **Dependency Resolution**: When `ComponentBase.initialize(context)` is called, components can use `context.container.resolve("dependency_name")` to get their dependencies. The current `Container.resolve()` in `src/core/container.py` handles basic resolution.

### 4.2. `src/core/container.py` Implementation

* Provides `register_instance`, `register_type`, and `resolve` methods.
* Supports singleton instances by caching resolved types.
* The `reset()` method clears cached singleton instances (excluding explicitly registered instances), which is crucial for state isolation between runs when the same container instance is reused (e.g., by `CleanBacktestEngine`).

## 5. Scoped Containers for Enhanced State Isolation

To achieve stronger state isolation, particularly for independent execution units like optimization trials or parallel backtests, the concept of "Scoped Containers" is introduced.

### 5.1. Purpose and Mechanism

* **Purpose**: To provide each distinct operational scope (e.g., a single backtest run within an optimization loop) with its own isolated set of component instances. This prevents any possibility of state leakage through shared component instances between these scopes.
* **Mechanism**:
    1.  An orchestrator (like an `Optimizer` or a parallel backtest manager, often utilizing `Bootstrap` functionalities iteratively) creates a *new, fresh `Container` instance* for each scope.
    2.  All components required for that scope are registered within this new, scoped `Container`.
    3.  Dependencies are resolved from this scoped `Container`.
    4.  Crucially, services like the `EventBus` can also be registered as new instances within each scoped container, inherently isolating event flow between scopes.
    5.  Once the operations within that scope are complete, the entire scoped `Container` and all its (non-global) components can be discarded, ensuring complete cleanup.

### 5.2. Relation to `Component.reset()`

* `Component.reset()`: Designed to clear the internal state of a *single component instance*, making it reusable for a subsequent phase of operation *within the same general scope and container*.
* **Scoped Containers**: Provide a more comprehensive isolation by ensuring that different scopes use entirely *different component instances*. This is generally a more robust approach for guaranteeing no state pollution between independent trials.

### 5.3. Integration with Bootstrap

* The `Bootstrap` system itself can be used (or its patterns adopted by orchestrators like `CleanBacktestEngine`) to set up these scoped containers. The orchestrator would call `bootstrap.initialize(config, run_mode, container=new_scoped_container)` for each scope.
* The `Bootstrap.teardown()` method, when applied to a scoped container setup, would ensure all components within that scope are properly torn down.

The use of Scoped Containers, as demonstrated by `CleanBacktestEngine`, is a key strategy for achieving the reproducible and consistent results needed for reliable optimization and testing.

## 6. Best Practices

* **Configuration-Driven Behavior**: System behavior, including component selection and `RunMode`, should be primarily driven by configuration files to allow flexibility without code changes.
* **Clear Dependency Chains**: Define dependencies explicitly so the `DependencyGraph` can accurately determine initialization and teardown order.
* **Minimal `main.py`**: The main entry point (`main.py`) should delegate almost all logic to `ApplicationLauncher` and `Bootstrap`.
* **Idempotent Lifecycle Methods**: Ensure `initialize`, `start`, `stop`, `reset`, and `teardown` can be safely called if the component is already in the target state or can handle repeated calls gracefully.

## 7. Example Bootstrap Configuration

```python
# Example bootstrap configuration
bootstrap_config = {
    "run_mode": "BACKTEST",
    "components": {
        "data_handler": {
            "class": "CSVDataHandler",
            "config": {"file_path": "data/prices.csv"}
        },
        "strategy": {
            "class": "MovingAverageStrategy",
            "config": {"fast_period": 10, "slow_period": 30}
        }
    }
}
```
