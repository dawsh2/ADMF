# ADMF-Trader Framework: Phased Implementation Plan for MVP (As of 2025-05-13)

This document outlines a phased approach to developing a Minimum Viable Product (MVP) for the ADMF-Trader framework, focusing on achieving core backtesting, train/test splitting, and basic optimization capabilities.

## Guiding Principles (from `IMPLEMENTATION.MD`):

1.  **Start with minimal working code.**
2.  **Maintain event-driven architecture.**
3.  **Refactor incrementally.**
4.  **Test each step.**
5.  **Use passthrough modes where appropriate for initial development.**

## Phase 0: Setup & Basic Event Loop

* **Goal:** Create a single Python script that runs an extremely basic, hardcoded event loop.
* **Key Activities:**
    * Implement minimal, in-script versions of: Event, EventBus, DataGenerator (e.g., yields `BarEvent`s), a hardcoded Strategy (e.g., MA Crossover emitting `SignalEvent`s), and a simple Portfolio (tracks position/P&L from signals).
    * Basic loop: Data -> Strategy -> Signal -> Portfolio Update.
* **Output:** A script that runs, processes data, and prints a basic result (e.g., final P&L).
* **Focus:** Prove the fundamental event flow and prevent lookahead bias.

## Phase 1: Core Module Extraction & Basic Backtester

* **Goal:** Extract core components into a proper directory structure (`src/core`, `src/data`, etc.) and refactor the Phase 0 script to use these.
* **Core Components (`src/core`):**
    * `EventBus` (standard thread-safe).
    * `Event` structure and basic `EventTypes` (`BAR`, `SIGNAL`, `ORDER`, `FILL`).
    * `Component` base class (simplified lifecycle).
    * `Config` (simple YAML loader).
    * `Container` (basic Dependency Injection).
    * Standard Python `logging`.
* **Application Components:**
    * `DataHandler` (`CSVLoader` for `SPY_1min.csv`, implements `DataHandlerBase`).
    * `MAStrategy` (implements `StrategyBase`).
    * `SimulatedExecutionHandler` (processes `ORDER` events, simulates fills, emits `FILL` events).
    * `BasicPortfolio` (processes `FILL` events, tracks P&L and positions).
    * `BacktestCoordinator` (manages the main backtest loop).
* **Output:** A structured project capable of running the MA Crossover backtest and outputting an equity curve.
* **Testing:** Begin unit tests for core and application components. Setup CI.

## Phase 2: Achieving Primary Goals (Train/Test Split & Grid Search)

* **Goal:** Implement train/test data splitting and basic grid search parameter optimization, addressing primary items from `GOALS.MD`.
* **Data Module Enhancements:**
    * `TrainTestSplitter` (initially using `DEEP_COPY` mode).
    * `DataHandler` to support train/test splits.
* **Strategy Module Enhancements:**
    * `StrategyBase` to support `get_parameters`, `set_parameters`, `parameter_space`.
* **Risk Module (Basic):**
    * `RiskManager` to receive signals.
    * Implement a basic `PositionSizer` (e.g., `FixedSizer`).
    * `RiskManager` applies sizing and emits `ORDER` events.
* **Optimization Framework (Simplified, in `src/strategy/optimization/`):**
    * `OptimizationTarget` for strategy parameters.
    * `GridSearchMethod`.
    * Simple `OptimizationMetric` (e.g., Sharpe Ratio, Total Return).
    * Simplified `OptimizationManager` to run grid search.
* **Analytics (Simplified, in `src/core/analytics/`):**
    * Functions for basic performance metrics (Sharpe, MDD, Total Return).
* **Output:** Ability to run grid search on training data, identify best parameters, and run a validation backtest on test data. Report key performance metrics for both.
* **Testing:** Integration tests for optimization and train/test functionalities.

## Phase 3: Incremental Enhancements & Feature Expansion

* **Goal:** With the MVP and core goals achieved, incrementally add more advanced features and refine existing components, drawing from the detailed design documents.
* **Potential Enhancements (Prioritize based on needs):**
    * More sophisticated `PositionSizer`s and `RiskLimit`s.
    * Advanced `Indicator` library and `Strategy` composition.
    * Improved `SimulatedBroker` (slippage, commissions).
    * Full `MetricsFramework` and `ReportingSystem`.
    * Additional `OptimizationMethod`s (e.g., Walk-Forward).
    * Refined `ConfigurationManager`, `Logging`, `ErrorHandling`.
    * Introduce asynchronous operations where beneficial (e.g., for live data/execution).
    * Implement `ContextAwareThreadSafety` if parallelizing optimizations or for live trading.

## Key Success Factors:

* Strict adherence to the iterative development philosophy.
* Continuous and comprehensive testing.
* Regularly revisit and update `GOALS.MD` to guide priorities.
* Use detailed design documents as a reference for implementing enhancements.




# ADMF-Trader Framework: Phased Implementation Plan for MVP (As of 2025-05-13)

This document outlines a phased approach to developing a Minimum Viable Product (MVP) for the ADMF-Trader framework, focusing on achieving core backtesting, train/test splitting, and basic optimization capabilities.

## Guiding Principles (from `IMPLEMENTATION.MD`):

1.  **Start with minimal working code.**
2.  **Maintain event-driven architecture.**
3.  **Refactor incrementally.**
4.  **Test each step.**
5.  **Use passthrough modes where appropriate for initial development.**

## Phase 0: Setup & Basic Event Loop

* **Goal:** Create a single Python script that runs an extremely basic, hardcoded event loop.
* **Key Activities:**
    * Implement minimal, in-script versions of: Event, EventBus, DataGenerator (e.g., yields `BarEvent`s), a hardcoded Strategy (e.g., MA Crossover emitting `SignalEvent`s), and a simple Portfolio (tracks position/P&L from signals).
    * Basic loop: Data -> Strategy -> Signal -> Portfolio Update.
* **Output:** A script that runs, processes data, and prints a basic result (e.g., final P&L).
* **Focus:** Prove the fundamental event flow and prevent lookahead bias.

## Phase 1: Core Module Extraction & Basic Backtester

* **Goal:** Extract core components into a proper directory structure (`src/core`, `src/data`, etc.) and refactor the Phase 0 script to use these.
* **Core Components (`src/core`):**
    * `EventBus` (standard thread-safe).
    * `Event` structure and basic `EventTypes` (`BAR`, `SIGNAL`, `ORDER`, `FILL`).
    * `Component` base class (simplified lifecycle).
    * `Config` (simple YAML loader).
    * `Container` (basic Dependency Injection).
    * Standard Python `logging`.
* **Application Components:**
    * `DataHandler` (`CSVLoader` for `SPY_1min.csv`, implements `DataHandlerBase`).
    * `MAStrategy` (implements `StrategyBase`).
    * `SimulatedExecutionHandler` (processes `ORDER` events, simulates fills, emits `FILL` events).
    * `BasicPortfolio` (processes `FILL` events, tracks P&L and positions).
    * `BacktestCoordinator` (manages the main backtest loop).
* **Output:** A structured project capable of running the MA Crossover backtest and outputting an equity curve.
* **Testing:** Begin unit tests for core and application components. Setup CI.

## Phase 2: Achieving Primary Goals (Train/Test Split & Grid Search)

* **Goal:** Implement train/test data splitting and basic grid search parameter optimization, addressing primary items from `GOALS.MD`.
* **Data Module Enhancements:**
    * `TrainTestSplitter` (initially using `DEEP_COPY` mode).
    * `DataHandler` to support train/test splits.
* **Strategy Module Enhancements:**
    * `StrategyBase` to support `get_parameters`, `set_parameters`, `parameter_space`.
* **Risk Module (Basic):**
    * `RiskManager` to receive signals.
    * Implement a basic `PositionSizer` (e.g., `FixedSizer`).
    * `RiskManager` applies sizing and emits `ORDER` events.
* **Optimization Framework (Simplified, in `src/strategy/optimization/`):**
    * `OptimizationTarget` for strategy parameters.
    * `GridSearchMethod`.
    * Simple `OptimizationMetric` (e.g., Sharpe Ratio, Total Return).
    * Simplified `OptimizationManager` to run grid search.
* **Analytics (Simplified, in `src/core/analytics/`):**
    * Functions for basic performance metrics (Sharpe, MDD, Total Return).
* **Output:** Ability to run grid search on training data, identify best parameters, and run a validation backtest on test data. Report key performance metrics for both.
* **Testing:** Integration tests for optimization and train/test functionalities.

## Phase 3: Incremental Enhancements & Feature Expansion

* **Goal:** With the MVP and core goals achieved, incrementally add more advanced features and refine existing components, drawing from the detailed design documents.
* **Potential Enhancements (Prioritize based on needs):**
    * More sophisticated `PositionSizer`s and `RiskLimit`s.
    * Advanced `Indicator` library and `Strategy` composition.
    * Improved `SimulatedBroker` (slippage, commissions).
    * Full `MetricsFramework` and `ReportingSystem`.
    * Additional `OptimizationMethod`s (e.g., Walk-Forward).
    * Refined `ConfigurationManager`, `Logging`, `ErrorHandling`.
    * Introduce asynchronous operations where beneficial (e.g., for live data/execution).
    * Implement `ContextAwareThreadSafety` if parallelizing optimizations or for live trading.

## Key Success Factors:

* Strict adherence to the iterative development philosophy.
* Continuous and comprehensive testing.
* Regularly revisit and update `GOALS.MD` to guide priorities.
* Use detailed design documents as a reference for implementing enhancements.

# ADMF-Trader Framework: Role of Existing Detailed Design Documents (As of 2025-05-13)

This document clarifies how the existing comprehensive design documentation for ADMF-Trader should be utilized throughout the development process, especially in conjunction with the phased, MVP-first implementation plan.

## Value of the Detailed Documents:

The extensive design documents (covering Core, Data, Strategy, Risk, Execution, Optimization modules, and various architectural aspects like `COMPONENT_ARCHITECTURE.MD`, `EVENT_ARCHITECTURE.MD`, `THREAD_SAFETY.MD`, etc.) are an invaluable asset. They represent a thorough, well-thought-out vision for a sophisticated and robust trading framework.

## How to Use Them:

1.  **As a Long-Term Architectural Vision (The "North Star"):**
    * These documents define the target state of the framework. Even when implementing simplified MVP versions, keep this end-state in mind to ensure foundational choices don't preclude future enhancements.
    * They provide a common understanding for the team about where the framework is heading.

2.  **As a Library of Pre-Designed Features & Solutions:**
    * When a new feature is prioritized after the MVP, consult the relevant design document. It likely contains detailed plans, interface definitions, best practices, and considerations for that feature.
    * This saves re-design effort and ensures new features integrate well with the overall architecture.
    * Examples:
        * Need more advanced risk management? Refer to `RISK_IMPLEMENTATION.MD` for ideas on `RiskLimitComposition` or specific `RiskLimitBase` types.
        * Performance becoming an issue? Review `RESOURCE_OPTIMIZATION.MD` or `STRATEGIC_CACHING.MD`.
        * Ready to implement parallel optimization? `PARALLEL_EXECUTION.MD` and `OPTIMIZATION_IMPLEMENTATION.MD` offer guidance.

3.  **To Guide Refactoring and Incremental Improvements:**
    * As the MVP evolves, the detailed documents can guide how to refactor and improve components. For instance, moving from a simple DI container to the more feature-rich one described in `DEPENDENCY_MANAGEMENT.MD`.

4.  **To Understand Inter-Component Dependencies and Contracts:**
    * Documents like `COMPONENT_ARCHITECTURE.MD` and `INTERFACE_DESIGN.MD` help maintain clarity on how different parts of the system should interact, even as individual components are built out incrementally.

5.  **NOT as a Strict Specification for Initial Implementation:**
    * Avoid the trap of trying to implement every detail from every document in the initial MVP phases. This would lead to over-engineering and slow progress.
    * The MVP plan deliberately simplifies or defers many of the advanced aspects detailed in these documents.

## Practical Application:

* **During MVP Development (Phases 0-2):**
    * Focus on the simplified requirements outlined in `ADMF_TRADER_IMPLEMENTATION_PLAN_MVP.MD`.
    * Refer to the detailed docs primarily for interface definitions (`*Base` classes) and fundamental architectural patterns (like event-driven flow) that should be established correctly from the start, albeit in a simplified form.
* **Post-MVP (Phase 3 and beyond):**
    * When planning to add a new feature (e.g., a specific type of `RiskLimit`, a new `OptimizationMethod`, or the `MetricsFramework`):
        1.  Consult the relevant detailed design document(s).
        2.  Adapt the design if necessary based on learnings from the MVP.
        3.  Implement the feature, integrating it into the existing, stable codebase.

By using the detailed design documents in this manner, they become a powerful enabler for building a sophisticated system iteratively, rather than an overwhelming upfront burden.
