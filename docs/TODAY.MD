# Design Document: Refactored Optimization & Testing Framework

## 1. Introduction and Goals

This document outlines a refactored architecture for a comprehensive trading strategy optimization and testing framework. The design aims to evolve the existing system, exemplified by components like the `EnhancedOptimizer`, into a more modular, configurable, and powerful platform.

The primary goals of this refactoring are:

* **Enhanced Modularity**: Clearly define and separate the concerns of optimization, backtest execution, and multi-scenario testing.
* **Efficient Broad Research**: Enable the optimization of multiple related strategy variations (e.g., a core strategy with different risk profiles or signal configurations, or even a few distinct base strategies) within a single, consolidated optimization job, maximizing insights from computationally intensive processes.
* **Streamlined Strategy Updates**: Provide a straightforward mechanism for targeted parameter updates for specific strategies, suitable for scenarios like nightly recalibrations of live strategies.
* **Comprehensive Multi-Strategy Testing (Initiative 7)**: Establish a robust, distinct phase for rigorous out-of-sample testing and comparison of various strategy configurations derived from optimization outputs. This includes evaluating ensembles, adaptive strategies, risk variations, and other comparative analyses.
* **Configuration-Driven Operation**: Ensure all major operations and component behaviors are defined and controlled through a clear and flexible system of configuration files.
* **Clean and Maintainable Codebase**: Promote reusability and minimize unnecessary specialized classes by making core components more configurable and by clearly defining responsibilities.

## 2. Core Architectural Principles

* **Separation of Concerns**: Distinct components will manage different stages of the workflow:
    * System Bootstrap & Global Configuration.
    * Optimization Job Execution (parameter discovery and model training).
    * Single Backtest Execution (the core engine for running one strategy instance against data).
    * Batch Backtest Execution (managing the execution of multiple pre-defined single backtests).
    * Multi-Scenario Test Execution (post-optimization validation and comparison).
* **Configuration-Driven**: All operations are controlled via YAML configuration files.
* **Modularity and Reusability**: Components like the `BacktestRunner` and `BacktestManager` are designed for general applicability.
* **Configurable Components**: Favor making core components highly configurable to handle variations in behavior, rather than creating many narrowly specialized classes.

## 3. Key Components and Their Roles

This architecture revolves around several key components:

### 3.1. `SystemOrchestrator` (`main.py`)

* **Responsibilities**:
    * Parses Command-Line Interface (CLI) arguments to determine the operational mode (e.g., `--optimize_job <job_config_path>`, `--update_strategy <StrategyComponentName>`, `--run_multi_tests <multi_test_config_path>`).
    * Loads the global `system_config.yaml`.
    * Initializes the Dependency Injection (DI) container, registers all system components (strategies, runners, managers, builders, data handlers, etc.), and sets up logging.
    * Invokes the appropriate top-level runner component (`OptimizationRunner` or `MultiTestRunner`) based on the determined operational mode and its specific configuration file.

### 3.2. `OptimizationRunner` (Evolved from `EnhancedOptimizer`)

* **Purpose**: To discover optimal configurations by executing "Optimization Jobs." This component is an evolution of the logic within your existing `EnhancedOptimizer`.
* **Input**: An "Optimization Job Configuration" YAML file.
* **Process**:
    * The job config can define one or multiple `strategy_optimization_tasks`. Each task specifies:
        * A `task_name` (unique identifier for its results).
        * A `strategy_service_to_optimize` (the DI name of the strategy class to be used).
        * Its specific `parameter_space` for that task. This space can include complex, nested structures to define parameters for risk modules, signal processing behaviors, or other internal configurable aspects of the strategy.
        * Optional overrides for data sources or regime detector configurations for this task.
    * For each defined task, the `OptimizationRunner` generates all parameter trial combinations.
    * It translates these trial combinations into a list of "backtest specifications."
    * It then utilizes the `BacktestManager` to execute this batch of trial backtests.
    * After receiving results from the `BacktestManager`, it performs its optimization logic (finding best parameters overall and per-regime, evaluating metrics) for each task.
    * It may also orchestrate the training and saving of auxiliary models (e.g., regime classifiers) as defined in the job config.
* **Output**: Produces one comprehensive JSON output file per executed "Optimization Job." This file contains all discovered optimal parameters, performance metrics for trials, paths to any trained models, etc., with results clearly delineated by `task_name` for each optimized variation.

### 3.3. `BacktestRunner`

* **Purpose**: To execute *one single, complete backtest instance*. This is the core workhorse for any simulation.
* **Input**: A full configuration for a single backtest run, including the strategy component (or its DI name), its specific parameters (which can be a complex dictionary), data source details, portfolio setup, and any adaptive mode settings.
* **Process**:
    * Manages the lifecycle (setup, reset, execution, teardown) of the necessary component instances (e.g., `DataHandler`, `Strategy`, `Portfolio`, `ExecutionHandler`) for that single run. These components are typically resolved via DI and reset to a clean state.
    * Executes the event loop or data stream.
    * If configured for an adaptive run (`is_adaptive_run=True` and `adaptive_config` provided):
        * It monitors regime change events (from a `RegimeDetector`).
        * It calls a designated method on the running strategy instance (e.g., `strategy.adapt_to_regime(new_regime, params_for_regime)`) to allow the strategy to dynamically change its parameters, active internal logic, or behavior.
* **Output**: Detailed results for that single backtest run (performance metrics, trade log, final portfolio state, etc.).

### 3.4. `BacktestManager`

* **Purpose**: A shared module responsible for efficiently executing a *batch* of pre-defined, independent backtest specifications.
* **Input**: A list of "backtest specifications." Each specification is a complete configuration object ready to be passed to `BacktestRunner.run()`.
* **Process**:
    * Iterates through the input list of backtest specifications.
    * For each specification, it invokes `BacktestRunner.run()`.
    * Collects the individual result from each completed `BacktestRunner.run()` call.
    * (This component is the natural place to implement future enhancements like parallel execution of backtests using multiprocessing or threading).
* **Output**: A list of results, corresponding to each input backtest specification.
* **Usage**: Utilized by both the `OptimizationRunner` (for running optimization trials) and the `MultiTestRunner` (for running test scenarios).

### 3.5. `MultiTestRunner` (Python Class - The Engine for "Initiative 7")

* **Purpose**: To validate, compare, and analyze various strategy configurations using the outputs generated by the `OptimizationRunner`.
* **Input**:
    * A "Multi-Test Scenario Configuration" YAML file.
    * Paths to one or more comprehensive JSON output files from previous `OptimizationRunner` jobs.
* **Process**:
    * Loads the specified optimization output JSON(s).
    * For each test scenario defined in its configuration file:
        * It resolves and uses the appropriate `ConfigBuilder` component.
        * The `ConfigBuilder` interprets the scenario definition and uses data from the loaded optimization output(s) to prepare one or more "backtest specifications."
        * It passes these specifications to the `BacktestManager` for execution.
    * Receives the batch of results from the `BacktestManager`.
* **Output**: Aggregated comparative reports, analyses, and performance statistics for all executed test scenarios.

### 3.6. `ConfigBuilder`s

* **Purpose**: Helper components (e.g., `EnsembleConfigBuilder`, `AdaptiveConfigBuilder`, `StaticSingleConfigBuilder`, `ParameterVariationConfigBuilder`) primarily used by the `MultiTestRunner`.
* **Input**: A specific test scenario definition (from the multi-test config) and the loaded optimization result data.
* **Process**: Translate these inputs into concrete "backtest specifications" suitable for the `BacktestManager` (and thus, the `BacktestRunner`). They bridge the gap between a high-level test idea and an executable backtest configuration.

### 3.7. `IStrategy` Implementations (e.g., `AdaptiveEnsembleStrategy`)

* **Adaptability**: Strategy classes (like your `AdaptiveEnsembleStrategy` or other core strategies) can be designed to manage internal sub-logics, modules, or different parameter sets. This allows a single strategy component to embody multiple "personalities" or behaviors.
* **Interface for Adaptation**: They should expose a method (e.g., `adapt_to_regime(self, regime: str, regime_specific_parameters: Dict)`) that the `BacktestRunner` can call during adaptive runs. This method enables the strategy to reconfigure its internal state, switch active internal logic, or apply new parameters based on the detected regime and the parameters provided for that regime.
* **Complex Parameterization**: Strategies should be designed to accept and interpret rich, potentially nested, parameter structures. This allows their internal risk modules, signal processing logic, or other complex behaviors to be configured and tuned via the optimization process.

## 4. Configuration Strategy

The system's operation is driven by a layered set of YAML configuration files:

* **`system_config.yaml` (Global/Bootstrap)**:
    * Central Dependency Injection (DI) registration for all component *types* (e.g., "MAStrategy" maps to `src.strategy.ma_strategy.MAStrategy`).
    * Default paths for persistence (e.g., templates for naming optimization output JSONs, trained model files).
    * Definitions of available named `data_sources` (e.g., "SPY_1min_Train_Data").
    * May define "update profiles" for specific, registered strategy components to simplify CLI-driven nightly updates.

* **Optimization Job Configuration (`config/optimization_jobs/<job_name>.yaml`)**:
    * Defines a single, potentially broad, optimization run, executed by the `OptimizationRunner`.
    * Specifies a `job_name` for identification.
    * May define global data and regime detector configurations applicable to all tasks within the job.
    * Contains a list of `strategy_optimization_tasks`. Each task includes:
        * `task_name`: A unique identifier for this specific variation's results within the job's output file.
        * `strategy_service_to_optimize`: The DI name of the strategy class to be instantiated and optimized.
        * `parameter_space`: The search space for this task. This can be a complex, nested structure to define parameters for core strategy logic, as well as for internal risk modules, signal processing configurations, etc.
        * Optional task-specific overrides for data sources or regime detector settings.
    * Defines an `output_config` specifying how the single, comprehensive JSON output for this entire job is named and saved (often using templates from `system_config.yaml`).

* **Multi-Test Scenario Configuration (`config/multi_test_scenarios.yaml`)**:
    * Input for the `MultiTestRunner`.
    * Specifies which `OptimizationRunner` output JSON file(s) to load (these contain the results of previous optimization jobs).
    * Defines a list of `test_scenarios`. Each scenario dictates:
        * A descriptive `name`.
        * An `enabled` flag (true/false).
        * The `config_builder_service` to use (DI name of the appropriate builder).
        * `builder_config`: Specific instructions for that builder, including how to select/derive parameters from the loaded optimization JSONs (e.g., which `task_name`'s results to use, how to form an ensemble, parameters for an adaptive setup, which parameters to vary for sensitivity analysis).

* **CLI-Driven Invocation (via `main.py`)**:
    * `python main.py --optimize_job <job_config_path.yaml>`: Initiates the `OptimizationRunner` with the specified job configuration.
    * `python main.py --update_strategy <RegisteredStrategyComponentName>`: `main.py` looks up an "update profile" for the component in `system_config.yaml`. This profile typically points to a specialized (often simpler) "Optimization Job Config" file or defines a minimal inline optimization task, which is then executed by the `OptimizationRunner`.
    * `python main.py --run_multi_tests [--multi_test_config <path.yaml>]`: Initiates the `MultiTestRunner`, which uses a default or specified multi-test scenario configuration file.

## 5. Handling Complex Risk/Meta Configurations

* Advanced risk management rules (e.g., regime-based trade vetos, dynamic stop-loss calculations based on MAE) and meta-strategy functionalities (e.g., signal smoothing, confidence scoring) are treated as configurable aspects of an `IStrategy` implementation.
* The parameters controlling these complex behaviors are defined within the `parameter_space` of a `strategy_optimization_task` in an "Optimization Job Config."
* The strategy class itself is responsible for parsing these (potentially nested or structured) parameters and configuring its internal logic, modules, or behavioral modes accordingly.
* This approach allows the `OptimizationRunner` to tune these sophisticated features as integral parts of the overall strategy optimization.

## 6. Key Benefits of This Design

* **True Modularity**: Clear separation of concerns between optimization, batch execution, single backtest execution, multi-testing, and strategy logic.
* **Configuration-Driven Flexibility**: The system's behavior is primarily controlled by declarative YAML files, making it adaptable without code changes for many scenarios.
* **Efficient Broad Research**: The "Optimization Job Config" allows for optimizing multiple related strategy variations in a single, unified run, producing a rich dataset for subsequent analysis.
* **Streamlined Targeted Updates**: Clear and simple mechanisms for updating parameters of specific strategies.
* **Robust "Initiative 7" Capabilities**: The `MultiTestRunner`, `ConfigBuilder`s, and `BacktestManager` provide a solid foundation for comprehensive post-optimization testing and validation.
* **Leverages and Extends Existing Architecture**: Builds upon concepts like adaptable strategies (e.g., `AdaptiveEnsembleStrategy`) by providing clear interfaces for interaction with the `BacktestRunner`.
* **Clean Codebase**: Minimizes redundancy by introducing the `BacktestManager` for shared batch execution logic and by making core components highly configurable.

This refined architecture provides a robust and scalable framework for developing, optimizing, and validating complex trading strategies.