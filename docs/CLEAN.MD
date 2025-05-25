# Refactoring Roadmap: Core Infrastructure and Component Updates

This document outlines the steps and considerations for implementing `ComponentDiscovery`, updating components to the new `ComponentBase` contract (including use of `SubscriptionManager`), refining the `Bootstrap` process, and refactoring `main.py`.

## I. Core Infrastructure Finalization & Bootstrap Implementation

**A. Finalize Core Classes (`component_base.py`, `dependency_graph.py`, `subscription_manager.py`):**
* [x] **`component_base.py` Reviewed & Aligned:**
    * [x] Minimal `__init__(instance_name, config_key)`.
    * [x] `initialize(context)` for DI, calling abstract `_initialize()`.
    * [x] Standard lifecycle methods (`start`, `stop`, `reset`, `teardown`) defined.
    * [x] Hooks for `initialize_event_subscriptions()` and `_validate_configuration()`.
* [ ] **Update `ComponentBase` to use `SubscriptionManager`:**
    * In `ComponentBase.initialize_event_subscriptions()`:
        * If `self.event_bus` is available, instantiate `self.subscription_manager = SubscriptionManager(self.event_bus)`. Subclasses will use this manager.
    * In `ComponentBase.teardown()`:
        * Ensure it calls `if hasattr(self, 'subscription_manager') and self.subscription_manager: self.subscription_manager.unsubscribe_all()`.
* [x] **`dependency_graph.py` Reviewed:**
    * [x] Custom graph implementation is functional for current needs.
    * [x] Provides `add_component`, `add_dependency`, `detect_cycles`, `validate`.
* [x] **`subscription_manager.py` Reviewed:**
    * [x] `subscribe`, `unsubscribe`, and `unsubscribe_all` methods are suitable.
    * **Task:** Verify that `SubscriptionManager.subscribe` and `SubscriptionManager.unsubscribe` pass the `context` argument to the underlying `self.event_bus.subscribe/unsubscribe` methods if the `EventBus` supports context-specific subscriptions at that level.

**B. Implement Dynamic `ComponentDiscovery` & Metadata Strategy:**
* [x] **Design `ComponentDiscovery` Class:** Structure for discovering components via `component_meta.yaml` (as seen in the updated `bootstrap.py`) is in place.
* [ ] **Create `component_meta.yaml` for Each Core Component:**
    * For Data Handlers, Strategies, Optimizers, Risk, Portfolio, Execution components.
    * **Content:** `name` (unique definition name), `type` (category), `module_path` (e.g., "src.data.csv_data_handler.CSVDataHandler"), `config_key`, `singleton` (bool), `dependencies` (list of instance names), `required` (bool).
* [ ] **Finalize `ComponentDiscovery.discover_components()` in `bootstrap.py`:**
    * Ensure robust YAML parsing and error handling.

**C. Enhance `Bootstrap` Class (`bootstrap.py`):**
* [ ] **Integrate `ComponentDiscovery` Results:**
    * `Bootstrap` should call `ComponentDiscovery.discover_components()`.
    * Store discovered `ComponentDefinition`s in `self.component_definitions` (merging with/overriding any defaults).
* [ ] **Build Dependency Graph from Discovered Components:**
    * After loading definitions, iterate `self.component_definitions`.
    * Populate `self.dependency_graph` by calling `add_component` and `add_dependency` for each.
* [ ] **Implement Topological Sort for Initialization Order (Critical Next Step):**
    * **Task:** Add a method to `DependencyGraph` (e.g., `get_topological_sort() -> List[str]`) or implement logic within `Bootstrap` (e.g., in `_resolve_initialization_order()`).
        * This method must perform a topological sort on `self.dependency_graph`.
        * It should handle cycle detection (which `DependencyGraph.validate()` already does, so this can be a prerequisite).
        * The result (a list of component *instance names*) must be stored (e.g., in `self.resolved_initialization_order`).
* [ ] **Update Component Creation & Lifecycle Orchestration:**
    * Create a central method in `Bootstrap` (e.g., `_load_and_prepare_components`) that:
        1.  Uses `self.resolved_initialization_order` to iterate through component names.
        2.  Retrieves `ComponentDefinition` for each.
        3.  Calls `self.create_component(name, definition)` (minimal `__init__`).
    * `self.initialize_components()` must use `self.resolved_initialization_order`.
    * `start_components`, `stop_components`, `reset_components`, `teardown` must also use this resolved order (or its reverse for shutdown).
* [ ] **Implement Config-Driven Entrypoint Component Logic (Critical Next Step):**
    * Add `get_entrypoint_component(self) -> ComponentBase:` method to `Bootstrap`.
    * **Functionality:**
        1.  Access `self.context.config` and `self.context.run_mode`.
        2.  Look up the configured entrypoint component name (e.g., from `config['system']['run_modes'][run_mode.value]['entrypoint_component_name']`).
            * **Config Example:**
                ```yaml
                system:
                  run_modes:
                    backtest:
                      entrypoint_component_name: "BacktestCoordinator"
                    optimization:
                      entrypoint_component_name: "MyOptimizer" 
                ```
        3.  Retrieve the component instance using `self.context.container.get(entrypoint_name)`.
        4.  Return the instance.

## II. Component Refactoring/Rewriting

* [ ] **For each core component (Data Handlers, Strategies, Optimizers, Risk, Portfolio, Execution):**
    * **Decision:** Confirm rewrite (your preference) or refactor.
    * Ensure class inherits from `ComponentBase`.
    * Implement minimal `__init__(self, instance_name, config_key)`.
    * Implement `_initialize(self)`: Move dependency acquisition (from `self.container`, `self.config`, etc.) and resource setup here. Log parameter sources.
    * Override `initialize_event_subscriptions(self)`: Use `self.subscription_manager.subscribe(...)`.
    * Override `_validate_configuration(self)` if needed.
    * Implement `start(self)`. Call `super()`.
    * Implement `stop(self)`. Call `super()`.
    * Implement `reset(self)`: CRITICAL for full state reset to post-`_initialize()` condition. Call `super()`.
    * Implement `teardown(self)`: Call `self.subscription_manager.unsubscribe_all()`, release resources, then `super().teardown()`.
* [ ] **Update/Create `component_meta.yaml` for each component with accurate dependencies.**

## III. Refactor `main.py` (Entry Point)

* [ ] **Minimal Logic:** `main.py` should be a lean orchestrator.
* [ ] **Argument Parsing:** For config path, run mode, log levels.
* [ ] **Bootstrap Instantiation:** `bootstrap = Bootstrap()`.
* [ ] **System Context Initialization:** `bootstrap.initialize(config_path, run_mode, ...)`.
* [ ] **Comprehensive Component Setup via Bootstrap:**
    * Call a new central method in `Bootstrap`, e.g., `bootstrap.setup_managed_components(search_paths=["./src"])`. This method internally handles:
        1.  Discovering component definitions.
        2.  Building the dependency graph.
        3.  Resolving initialization order via topological sort.
        4.  Creating component instances (minimal `__init__`).
        5.  Initializing components (calling `component.initialize(context)`).
* [ ] **Start Components:** `bootstrap.start_components()`.
* [ ] **Fetch and Execute Entrypoint:**
    * `entrypoint_component = bootstrap.get_entrypoint_component()`
    * `results = entrypoint_component.execute()` (assuming a standard method).
* [ ] **Graceful Shutdown in `try...finally`:**
    * `bootstrap.stop_components()`
    * `bootstrap.teardown()`
* [ ] **Top-Level Exception Handling.**

---
## IV. Expanded Points & Clarifications (Reiteration)

**A. Event Ordering Consistency**

* **Bootstrap's Role:** Ensures components *start* in a consistent state and order. The `reset()` mechanism managed by an optimizer (which is itself a component) is key for consistency *between* iterative runs like optimization trials.
* **`EventContext` Role:** (From `EVENT_ISOLATION.MD`) Prevents event *leakage* between logically separate runs (e.g., different optimization trials), ensuring one run's events don't affect another. This is vital for independent and reproducible trial results.
* **Intra-Run Ordering:** Determined by component logic and `EventBus` behavior (synchronous/asynchronous processing). A single-threaded, synchronous event bus within an `EventContext` will yield deterministic event order for a given set of inputs and starting state.

**B. Top-Level Application Component Determined by Config**

* **Goal:** `main.py` delegates the decision of *what application to run* to the configuration, based on the current `RunMode`.
* **Mechanism:**
    1.  `config.yaml` specifies an `entrypoint_component_name` for each `RunMode`.
    2.  `Bootstrap.get_entrypoint_component()` reads this name from the config.
    3.  It fetches the already created and initialized component instance from the DI `Container`.
    4.  `main.py` calls a standard method (e.g., `execute()`) on this component.
* **Benefit:** Maximizes flexibility; `main.py` remains unchanged even if the primary component for a run mode (e.g., the specific optimizer implementation) changes.

This updated checklist and the expanded points should provide a clear and actionable path. Prioritizing the Bootstrap enhancements (topological sort, config-driven entrypoint) will be key before widespread component refactoring.
