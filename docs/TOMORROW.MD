# ADMF-Trader: Development & Refactoring Roadmap - Phase [Current Phase Number/Name]

## 1. Introduction

This document outlines the key development and refactoring initiatives planned for the ADMF-Trader system. The primary goals are to introduce advanced optimization techniques, enhance the system's modularity and maintainability, formalize the handling of analytical components, standardize system initialization, and improve overall code health and efficiency. These efforts are aligned with the architectural vision detailed in various system design documents.

## 2. Overarching Goals for This Phase

* **Enhanced Optimization:** Implement Genetic Algorithms for more sophisticated parameter tuning.
* **Improved Modularity & Decoupling:** Refine component responsibilities and interfaces, reducing tight coupling.
* **Increased Clarity & Maintainability:** Restructure code for better readability and easier future modifications.
* **Formalized Analytics:** Introduce a dedicated framework for components that analyze market conditions and system performance without direct trading actions.
* **Standardized Initialization:** Implement a bootstrap process for consistent system setup.
* **System Efficiency:** Optimize processes like event publishing.
* **Alignment with Design:** Bring the codebase closer to the architectural principles outlined in `STRATEGY_IMPLEMENTATION.MD`, `OPTIMIZATION_FRAMEWORK.MD`, `ANALYTICS_COMPONENT_FRAMEWORK.MD`, and `BOOTSTRAP_SYSTEM.MD`.

## 3. Key Development & Refactoring Initiatives

### Initiative 1: Genetic Algorithm (GA) for Ensemble Rule Weight Optimization

* **Goal:** Implement a Genetic Algorithm to optimize the rule weights (e.g., `ma_rule.weight`, `rsi_rule.weight`) within the `EnsembleSignalStrategy`.
* **Key Implementation Steps:**
    1.  **`GeneticOptimizer` Class:**
        * Create `src/strategy/optimization/genetic_optimizer.py`.
        * Class `GeneticOptimizer` to inherit from `BasicOptimizer`.
    2.  **GA Configuration:**
        * Define GA parameters in `config.yaml` (population size, generations, crossover/mutation rates, gene parameter names like `["ma_rule.weight", "rsi_rule.weight"]`, gene ranges, weight normalization flag).
    3.  **Chromosome Design & Utilities:**
        * Represent chromosomes as lists of floats (weights).
        * Implement `_create_individual()`, `_initialize_population()`, `_normalize_weights()` (if applicable), and `_chromosome_to_params()` (to convert chromosome to strategy parameters, incorporating other fixed strategy params).
    4.  **Core GA Operations:**
        * **Fitness Evaluation (`_evaluate_fitness`):** Use `_perform_single_backtest_run` with parameters derived from the chromosome.
        * **Selection (`_selection`):** Implement tournament selection or similar.
        * **Crossover (`_crossover`):** Implement single-point or other crossover methods.
        * **Mutation (`_mutate`):** Implement random reset or creep mutation for genes.
    5.  **Main GA Loop (`run_optimization`):**
        * Orchestrate generations: evaluation, selection, elitism (optional), crossover, mutation.
        * Track best individual/fitness over generations.
        * Implement termination criteria.
    6.  **Integration:**
        * Modify `main.py` to select and run `GeneticOptimizer`.
        * Standardize optimizer result reporting and saving.

### Initiative 2: Analytics Component Framework Implementation

* **Goal:** Establish a formal framework for components that analyze market conditions, signals, or performance without direct trading actions, as per `ANALYTICS_COMPONENT_FRAMEWORK.MD`.
* **Key Implementation Steps:**
    1.  **`AnalyticsComponent` Base Class:**
        * Create `src/analytics/base_analytics_component.py`.
        * Inherit from `BaseComponent`.
        * Define abstract `analyze(event)` method and common functionalities (event subscription, `get_analysis_results()`, potentially publishing `ANALYSIS_RESULT` events).
    2.  **New Event Type:** Define `EventType.ANALYSIS_RESULT` in `src/core/event.py`.
    3.  **Refactor `Classifier`/`RegimeDetector`:**
        * Have `Classifier` inherit from `AnalyticsComponent`.
        * Move these components to an `src/analytics/` subdirectory (e.g., `src/analytics/market_condition/`).
    4.  **Implement New Concrete Analytics Component:**
        * Example: `SignalQualityAssessor` in `src/analytics/signal_quality_assessor.py`.
        * Subscribes to `SIGNAL` (and possibly `FILL`, `CLASSIFICATION`) events.
        * Publishes `ANALYSIS_RESULT` events with quality/confidence scores.
    5.  **Integration:**
        * Register analytics components in `main.py` / `bootstrap.py`.
        * Modify consumers (e.g., `RiskManager`) to use analysis results (e.g., from `SignalQualityAssessor` or `RegimeDetector`) for decision-making, aligning with `SIGNAL_PROCESSING.MD`.

### Initiative 3: Refactoring `EnhancedOptimizer`

* **Goal:** Improve modularity, readability, and maintainability of `EnhancedOptimizer`, aligning with the common `OptimizerInterface`.
* **Key Refactoring Areas:**
    1.  **Modularize `run_grid_search()` (or common `run_optimization()`):** Break down into logical phases (init, param generation, training loop, test phase, adaptive test, reporting).
    2.  **`_run_regime_adaptive_test()`:** Simplify its internal structure, ensure robust component lifecycle management (no re-running `setup` unnecessarily). Consider if parts are reusable.
    3.  **Clarity:** Ensure methods like `_process_regime_performance()` are focused and easy to understand.
    4.  **State Management:** Review and clarify initialization and use of member variables storing results.
    5.  **Adherence to `OptimizerInterface`:** Ensure consistency with `BasicOptimizer` and the new `GeneticOptimizer` regarding common methods and result structures.

### Initiative 4: Implementing `bootstrap.py`

* **Goal:** Centralize and standardize system initialization for consistency and cleaner `main.py`, based on `docs/core/foundation/BOOTSTRAP_SYSTEM.MD`.
* **Key Implementation Steps:**
    1.  **Create `src/core/bootstrap.py`**.
    2.  **Bootstrap Functions:**
        * `setup_logging(config_loader, cli_args)`
        * `initialize_core_services(config_loader) -> Tuple[Container, EventBus]`
        * `register_core_components(container, config_loader, event_bus, cli_args)`: For components common to all modes.
        * `setup_application_mode(container, config_loader, event_bus, cli_args)`: For mode-specific components (e.g., which optimizer, which main strategy).
    3.  **Refactor `main.py`:**
        * Delegate initialization tasks to `bootstrap.py` functions.
        * `main.py` focuses on CLI argument parsing and high-level orchestration of selected mode (e.g., calling optimizer's `run_optimization()`).
    4.  **Test Environment Usage:** Enable test scripts to use `bootstrap.py` for consistent setup.

### Initiative 5: Optimizing Classifier Event Publishing

* **Goal:** Publish `CLASSIFICATION` events from `Classifier` only when the actual classification state changes, improving efficiency.
* **Key Implementation Steps:**
    1.  **Modify `Classifier.on_bar()`:**
        * Call `new_classification = self.classify(data)`.
        * Publish event only if `new_classification != self._current_classification` or if it's the first classification.
        * Update `self._current_classification` *before* publishing.
    2.  **Remove Redundant Publishing from `RegimeDetector`:**
        * Delete direct calls to `_publish_classification_event` within `RegimeDetector._apply_stabilization`. `RegimeDetector.classify()` will return the stabilized regime, and `Classifier.on_bar` will handle the event publishing.

### Initiative 6: General Code Health and Strategy Component Refinement

* **Goal:** Further align code with design documents (e.g., `STRATEGY_IMPLEMENTATION.MD`), improve overall modularity and code quality.
* **Key Actions:**
    1.  **`Strategy` Base Class:** Refine/implement the abstract `Strategy` base class as envisioned, potentially with an `add_component` mechanism for indicators/rules.
    2.  **Refactor `MAStrategy` & `EnsembleSignalStrategy`:** Ensure they align with the `Strategy` base class and utilize component-based design internally where appropriate.
    3.  **Standardize Interfaces:** Ensure `Indicator` and `Rule` components adhere to consistent interfaces (e.g., `update()`, `value`, `ready`, parameter methods).
    4.  **Component Lifecycles:** Thoroughly review and ensure `setup()`, `start()`, `stop()`, and `reset()` methods are consistently implemented and correctly manage state for all core components. `start()` should allow transitions from `STOPPED` or `INITIALIZED`.
    5.  **Docstrings & Type Hinting:** Improve across the codebase.
    6.  **Dead Code Removal:** Identify and remove unused code.
    7.  **Testing:** Expand unit and integration test coverage for refactored and new components.

## 4. Process & General Recommendations

* **Version Control:** Utilize a dedicated branch for these refactoring and development efforts. Commit changes frequently with clear messages.
* **Iterative Development:** Tackle initiatives and their sub-steps one at a time. Test each change thoroughly.
* **Prioritization:** Address foundational changes (like `bootstrap.py`, base class refinements) early if they unblock or simplify other tasks. The order presented above is a suggestion.
* **Continuous Reference to Design Documents:** Regularly consult `PLAN.MD`, `IMPLEMENTATION.MD`, `ARCH.MD`, and other specific `.MD` files (like those for Analytics, Optimization, Strategy) to ensure development aligns with the intended architecture.
* **Code Reviews:** If possible, have code reviewed to catch issues early and ensure clarity.

### Initiative 7: Multi-Strategy Testing from Single Training Run

* **Goal:** Enable execution of multiple strategy variants using derived optimal parameters and classifier data from a single backtest/optimization run.
* **Key Implementation Steps:**
    1. **Training Phase Output Framework:**
        * Persist optimization results (optimal parameters for each strategy/rule combination from grid search).
        * Save regime classifier training data and trained model.
        * Store performance rankings and metrics for all tested combinations.
    2. **Testing Phase Capabilities:**
        * **Top N Strategy**: Run ensemble of top N performing strategies from optimization.
        * **Adaptive Regime Switching**: Use regime classifier to dynamically switch between strategies.
        * **Risk-Adjusted Ensemble**: Weight strategies based on risk-adjusted performance metrics.
        * **Regime-Specific Strategies**: Deploy different strategies for different market regimes.
        * **Performance Bracket Testing**: Test strategies grouped by performance tiers (top 10%, top 25%, etc.).
    3. **Implementation Requirements:**
        * Results persistence layer for optimization outputs and classifier data.
        * Framework to instantiate multiple strategy variants from saved parameters.
        * Test harness to run various ensemble/selection methodologies.
        * Comparative analysis across different strategy selection approaches.
    4. **Test Orchestration Framework:**
        * **`StrategyTestRunner` Class**: Orchestrates multiple strategy tests from single optimization run.
        * **Test Configuration**: Define multiple test scenarios in config (Top N with different N values, risk-adjusted weightings, regime adaptive switching, performance brackets).
        * **Test Execution Pipeline**:
            - Load optimization results once (`optimization_results.json`)
            - For each test scenario: build strategy config from optimization results, create fresh system with test data, run backtest on out-of-sample data
        * **Strategy Config Builders**:
            - `EnsembleConfig`: For Top N strategies with various weighting schemes (equal, performance-based, risk-adjusted)
            - `AdaptiveConfig`: For regime-based strategy switching using pre-trained classifier
            - `BracketConfig`: For performance percentile-based strategy selection
        * **Comparative Analysis**: Aggregate results across all test scenarios for comprehensive performance comparison.
    5. **Benefits:**
        * Comprehensive strategy evaluation without re-running expensive optimization processes.
        * Enable rapid testing of multiple ensemble approaches from single training run.
        * Facilitate robust strategy selection and validation methodologies.

### Initiative 8: Live Trading Integration with Configuration-Driven Mode Selection

* **Goal:** Enable seamless transition between backtesting and live trading using the same codebase with configuration-driven mode selection.
* **Key Implementation Steps:**
    1. **Execution Handler Abstraction:**
        * Create `LiveExecutionHandler` class alongside existing `SimulatedExecutionHandler`.
        * Both implement same interface but handle orders differently (real broker vs simulation).
        * Live handler integrates with broker APIs (Alpaca, Interactive Brokers, etc.).
    2. **Data Handler Abstraction:**
        * Create `LiveDataHandler` class alongside existing `CSVDataHandler`.
        * Live handler connects to real-time data feeds.
        * Maintain same data interface for strategy components.
    3. **Configuration-Driven Architecture:**
        * Add execution mode flag to config: `execution.mode: "live"` or `"backtest"`.
        * Add data source configuration: `data.source: "live"` or `"csv"`.
        * Include live trading parameters: broker credentials, data feed settings.
        * Strategy deployment configuration: `strategy.load_from_optimization: true`, `deployment_method: "top_n"`.
    4. **Bootstrap Integration:**
        * Modify `bootstrap.py` to instantiate appropriate handlers based on config.
        * Enable loading of pre-optimized parameters from optimization results.
        * Support multiple deployment methods (Top N, adaptive regime, risk-adjusted ensemble).
    5. **Strategy State Management:**
        * Ensure strategy logic remains identical across both modes.
        * Same strategy components work with both simulated and live data/execution.
        * Maintain consistent internal state management regardless of execution context.
    6. **Live Trading Deployment Options:**
        * **Pre-Selection Deployment**: Deploy single chosen ensemble method from optimization results.
        * **Real-Time Strategy Selection**: Use live regime classifier to switch between strategies.
        * **Ensemble Execution**: Run multiple strategies simultaneously with optimized position weights.
    7. **Safety & Testing:**
        * Easy mode switching via configuration for validation.
        * Separate config files for development/staging/production environments.
        * Parameter persistence: optimization results automatically flow to live trading.

## 5. Conclusion

Successfully completing these initiatives will significantly enhance the ADMF-Trader system's capabilities, robustness, maintainability, and extensibility. It will pave the way for more advanced strategy development, optimization techniques, and in-depth performance analysis.

## 6. Additional Items

- Explore how indicators are reset during regime change, in particular when changed, ensure they are retrieving previous bar data rather than waiting for period_len bars until valid
- Implement indicator price history preservation: indicators should maintain price history equal to the maximum period length in the parameter space (e.g., if RSI periods range from 9-21, always keep 21+ prices) to ensure valid readings immediately after parameter changes