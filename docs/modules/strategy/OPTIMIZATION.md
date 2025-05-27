# ADMF-Trader Optimization Framework (OPTIMIZATION.MD)

## I. Implementation Philosophy & Core Concepts

The ADMF-Trader optimization framework is designed for flexibility, power, and granular control over the optimization of algorithmic trading strategies and their components. The core philosophy centers on:

1.  **Flexible Component Integration**: Optimizers should work seamlessly with any component in the trading system (strategies, rules, indicators, regime detectors, etc.).
2.  **Minimal Boilerplate**: Reduce repetitive code when defining and executing optimization workflows.
3.  **Configuration-Driven**: Enable complex optimization processes to be defined declaratively (e.g., via YAML), promoting reproducibility and ease of modification.
4.  **Granular Control**: Provide fine-grained control over which parameters are optimized, the search spaces, the methods used, and the evaluation criteria.
5.  **Sequential & Staged Optimization**: Support multi-stage optimization where results from one stage can inform subsequent stages.
6.  **Joint Optimization**: Allow multiple interdependent components or parameters to be optimized simultaneously.
7.  **Regime-Specific Optimization**: Enable parameters to be optimized separately for different identified market regimes.

**Core Optimization Components:**
* **Parameter Spaces**: Define the search space (ranges, discrete values, distributions) for each parameter being optimized.
* **Objective Functions**: Specify the metrics to be optimized (e.g., Sharpe ratio, net profit, win rate, drawdown). These define "what to achieve."
* **Optimizers (Methods)**: Encapsulate the algorithms used to search the parameter space (e.g., Grid Search, Genetic Algorithms, Bayesian Optimization). These define "how to search."
* **Constraints**: Define rules that valid parameter combinations must adhere to (e.g., short window < long window).
* **Workflows**: Orchestrate sequences of optimization steps, potentially involving multiple components, methods, and objectives.

## II. Framework Architecture

The framework is designed with a modular class hierarchy:

```
OptimizationBase
├── ParameterSpace
│   ├── DiscreteParameterSpace
│   ├── ContinuousParameterSpace
│   └── MixedParameterSpace
├── Objective
│   ├── ReturnObjective
│   ├── RiskObjective
│   ├── CompositeObjective  // For multi-objective optimization
│   └── CustomObjective
├── Optimizer           // Base class for optimization algorithms
│   ├── GridOptimizer
│   ├── RandomOptimizer
│   ├── BayesianOptimizer
│   ├── GeneticOptimizer
│   └── CustomOptimizer
├── Constraint
│   ├── RangeConstraint
│   ├── RelationalConstraint
│   └── CustomConstraint
├── ComponentOptimizer      // Specializes in optimizing individual system components
│   ├── IndicatorOptimizerMixin
│   ├── RuleOptimizerMixin
│   ├── StrategyOptimizerMixin // Can handle rule weights within a strategy
├── JointOptimizer          // For simultaneous optimization of multiple components
├── OptimizerManager        // Central orchestrator for all optimization processes
└── Workflow                // Defines and executes sequences of optimization tasks
    ├── SingleStageWorkflow
    ├── SequentialWorkflow  // Multi-stage optimization
    ├── ParallelWorkflow
    ├── RegimeBasedWorkflow // Coordinates regime-specific optimization
    ├── WalkForwardWorkflow
    └── CustomWorkflow
```

### A. Key Architectural Components

1.  **`OptimizerManager`**:
    * **Responsibilities**: Central orchestration of the optimization process. Manages registration and creation of optimizers and workflows. Can run complex optimization processes defined in configuration files. Handles saving and loading of optimization results.
    * **Key Methods**: 
        * `register_optimizer(name, optimizer)`: Register an optimizer instance
        * `register_workflow(name, workflow)`: Register a workflow
        * `create_workflow(workflow_type, config)`: Factory method for workflows
        * `run_workflow(workflow_name, **kwargs)`: Execute a registered workflow
        * `run_from_config(config)`: Run optimization from configuration
        * `save_optimization_results(strategy_name, version, variant, base_dir, metadata)`: Save results with strategy-aware naming
        * `load_optimization_results(strategy_name, version, variant, date, filename, base_dir)`: Load saved parameters
        * `list_available_parameters(strategy_name, base_dir)`: Query available parameter sets

2.  **`ComponentOptimizer`**:
    * **Purpose**: A specialized optimizer designed to work with specific trading system components (indicators, rules, strategies). It extracts the parameter space from the component and uses a configured `Optimizer` algorithm and `Objective` function.
    * **Key Methods**:
        * `optimize(**kwargs)`: Run optimization for the component
        * `from_config(component, config, backtest_runner)`: Create from configuration
    * **Mixins**: Can use mixins (`IndicatorOptimizerMixin`, `RuleOptimizerMixin`, `StrategyOptimizerMixin`) to tailor optimization logic, such as how to evaluate parameters (e.g., re-calculating indicators vs. full backtest) or how to optimize internal aspects like rule weights within an ensemble strategy.

3.  **`JointOptimizer`**:
    * **Purpose**: Facilitates the simultaneous optimization of parameters across multiple components, building a combined parameter space and applying a single optimization process. Useful when parameters of different components are interdependent.
    * **Implementation Details**:
        * Accepts a list of `(component, prefix)` tuples to namespace parameters
        * Builds a combined parameter space with prefixed parameter names
        * Distributes optimized parameters back to individual components

4.  **`ParameterSpace`**:
    * **Purpose**: Defines the searchable parameters for an optimization task, including their names, types (discrete, continuous), ranges or possible values, and step sizes.
    * **Methods**: 
        * `from_component(component)`: Derive space from a component's `get_parameter_space()` method
        * `combine(other_space)`: Merge with another parameter space for joint optimization
        * `sample(method='grid', n_samples=None)`: Generate parameter combinations
        * `validate()`: Ensure parameter space configuration is valid

5.  **`Objective` (Function/Metric)**:
    * **Purpose**: Encapsulates a performance metric used to evaluate a set of parameters. The optimizer seeks to maximize or minimize this value.
    * **Common Implementations**:
        * `SharpeObjective`: Risk-adjusted returns (return/volatility)
        * `SortinoObjective`: Downside risk-adjusted returns
        * `DrawdownObjective`: Maximum peak-to-trough decline
        * `WinRateObjective`: Percentage of profitable trades
        * `ProfitFactorObjective`: Gross profit / gross loss
        * `CompositeObjective`: Weighted combination of multiple objectives
    * **Interface**: `calculate(results) -> float`, `normalize(value) -> float`

6.  **`Optimizer` (Method/Algorithm)**:
    * **Purpose**: Base class for various search algorithms.
    * **Implementations**:
        * `GridOptimizer`: Exhaustive search over parameter combinations
        * `GeneticOptimizer`: Evolutionary approach with mutation/crossover
        * `BayesianOptimizer`: Probabilistic model-guided search  
        * `ParticleSwarmOptimizer`: Swarm intelligence-based optimization
    * **Interface**: `optimize(evaluate_func, n_trials=None, **kwargs)`, `get_best_parameters()`, `get_best_score()`

7.  **`Constraint`**:
    * **Purpose**: Defines rules that parameter combinations must satisfy to be considered valid.
    * **Common Implementations**:
        * `RangeConstraint`: Limits parameter to specific range
        * `RelationalConstraint`: Enforces relationships between parameters (e.g., `param1 < param2`)
    * **Interface**: `is_satisfied(params) -> bool`, `validate_and_adjust(params) -> params`

8.  **`Workflow`**:
    * **Purpose**: Defines and executes sequences of optimization tasks. Allows for complex, multi-stage optimization processes.
    * **Implementations**:
        * `SingleStageWorkflow`: Simple one-step optimization
        * `SequentialWorkflow`: Multi-stage where results feed forward
        * `RegimeBasedWorkflow`: Optimizes separately for each market regime
        * `WalkForwardWorkflow`: Rolling window optimization with out-of-sample validation
    * **Interface**: `run() -> results`

## III. Regime-Specific Optimization

Optimizing strategy parameters for different market regimes is a core capability.

### A. Concept and Workflow

1.  **Regime Detection**: A `RegimeDetector` (implementing the `Classifier` interface) identifies the market regime.
2.  **Performance Tracking by Regime**: The portfolio manager tracks performance metrics segmented by the active regime during a backtest.
3.  **Targeted Optimization (`EnhancedOptimizer` / `RegimeBasedWorkflow`)**:
    * The `EnhancedOptimizer` (current implementation) or a `RegimeBasedWorkflow` (new framework) uses this regime-specific performance data to find the optimal parameters for each regime.
    * A minimum number of trades/samples per regime is often required (typically 5-10).
4.  **Parameter Storage**: Optimal parameters for each regime are saved in a structured format (e.g., `regime_optimized_parameters.json`).
5.  **Adaptive Strategy Execution (`RegimeAdaptiveStrategy`)**: This strategy type loads regime-specific parameters and switches them dynamically based on `CLASSIFICATION` events from the `RegimeDetector`.

### B. Architectural Considerations

* The `RegimeDetector` functions as an independent `Classifier`.
* Strategies generate "pure" signals; regime-based filtering can be a separate step, possibly in the Risk module. This allows for independent optimization of parameters and filtering rules.

## IV. Weight Optimization for Ensemble Strategies

For strategies combining multiple rules (e.g., `EnsembleStrategy`), the contribution of each rule (its weight) can be optimized.

* **Genetic Algorithms (`GeneticOptimizer`)**: Currently used for weight optimization, typically after initial parameter optimization of individual rules. The `StrategyOptimizerMixin` provides an `optimize_rule_weights` method for this purpose.
* **Signal-Based Weight Optimization (Proposed)**:
    * **Concept**: Instead of full backtests per weight set, this method first captures signal events (with component strengths and regime context) during parameter optimization. Weight optimization then replays these stored signals, applying different weights to simulate performance efficiently.
    * **Components Required**: `SignalCollector` during parameter optimization, `RegimeSignalReplayer` for weight optimization phase
    * **Advantages**: Maintains temporal integrity, massive efficiency gains, allows parallel regime optimization.

## V. Advanced Optimization Workflows & Patterns

The framework supports several advanced patterns:

* **Multi-Stage Optimization (Sequential Workflow)**: Results from one optimization stage (e.g., optimizing a Regime Detector) are used to configure subsequent stages (e.g., optimizing rule parameters within detected regimes).
* **Walk-Forward Optimization (`WalkForwardWorkflow`)**: Optimizes parameters on a rolling window of training data and validates on subsequent out-of-sample data to test parameter stability and robustness over time.
* **Hierarchical Optimization**: A complex sequence where different levels or aspects of a strategy (e.g., market regime identification, rule set selection, rule parameter tuning) are optimized in a structured hierarchy.
* **Adaptive Workflow**: Changes optimization approach based on intermediate results (e.g., switching from grid search to genetic algorithm if search space is too large).

## VI. Optimization Methodology & Challenges

### A. Current vs. Proposed Approaches

* **Current (Retroactive Regime Analysis)**: Strategies are backtested over the entire dataset, and performance is subsequently segmented by regime to find regime-optimal parameters.
* **Proposed (Regime-Specific Training & Signal Replay)**: For weight optimization, using regime-specific data subsets (like signal history) is proposed for efficiency and accuracy. This offers guaranteed isolation and efficiency gains.

### B. Boundary Trade Handling & Optimization

Trades spanning regime transitions ("boundary trades") can suffer performance degradation due to parameter/weight changes. This is a critical consideration for optimization.

* **Impact**: Optimization might inadvertently select parameters that exacerbate boundary issues if not accounted for.
* **Optimization-Related Solutions**:
    1.  **Boundary-Aware Fitness Functions**: Incorporate the performance of boundary trades into the optimization objective function.
    2.  **Cross-Regime Robustness Optimization**: Optimize parameters not just for their primary regime but also for acceptable performance in other (potentially misclassified or transitional) regimes. This promotes more stable parameters.
    3.  **Trade-Locked Parameters**: Use parameters from trade entry for the entire trade lifecycle.
    4.  **Gradual Parameter Transitions**: Blend parameters over multiple bars during regime changes.

## VII. Strategy Lifecycle Management & Parameter Versioning

Effective optimization requires robust management of the resulting parameters.

### A. Parameter Storage Structure

Optimized parameters are stored in a strategy-specific directory structure:

```
optimized_params/
├── ma_crossover/
│   ├── v1.0_2025-04-25.yaml
│   ├── v1.1_2025-04-26.yaml
│   └── v1.0_aggressive_2025-04-25.yaml
├── regime_ensemble/
│   ├── v1.0_2025-04-25.yaml
│   ├── v1.0_trend_only_2025-04-26.yaml
│   └── v2.0_with_volatility_2025-04-27.yaml
└── mean_reversion/
    ├── v1.0_2025-04-25.yaml
    └── v1.1_with_filter_2025-04-26.yaml
```

### B. Parameter File Format

Each parameter file follows a standardized YAML format:

```yaml
strategy:
  name: "regime_ensemble"
  version: "1.0"
  variant: "standard"
  optimization_date: "2025-04-25"
  in_sample_period: ["2020-01-01", "2023-01-01"]
  metrics:
    sharpe_ratio: 1.82
    win_rate: 0.56
    max_drawdown: -0.12
  parameters:
    regime_detector:
      volatility_window: 20
      volatility_threshold: 0.015
    regimes:
      trending_up:
        rules:
          trend_rule:
            fast_ma_window: 10
            slow_ma_window: 30
        weights:
          trend_rule: 1.0
          mean_reversion_rule: 0.2
```

### C. Parameter Management Methods

The `OptimizerManager` provides comprehensive parameter management:

* **Saving**: `save_optimization_results()` creates versioned parameter files with metadata
* **Loading**: `load_optimization_results()` supports filtering by version, variant, date
* **Querying**: `list_available_parameters()` shows all available parameter sets
* **Integration**: `BacktestCoordinator` can automatically load optimized parameters

## VIII. Configuration System

The optimization framework is highly configurable via YAML:

```yaml
optimization:
  output_dir: ./results/optimization
  save_results: true
  parallel: true
  n_jobs: 4
  
  components:
    regime_detector:
      parameter_space:
        volatility_window: [15, 20, 25]
        volatility_threshold: [0.01, 0.015, 0.02]
    
    trend_rule:
      parameter_space:
        fast_ma_window: [5, 10, 15]
        slow_ma_window: [20, 30, 40]
  
  objectives:
    sharpe:
      class: SharpeObjective
      weight: 1.0
      direction: maximize
    
  optimizers:
    grid:
      class: GridOptimizer
    genetic:
      class: GeneticOptimizer
      population_size: 50
      n_generations: 20
  
  workflow:
    type: sequential
    stages:
      - component: regime_detector
        optimizer: grid
        objective: win_rate
      
      - component: trend_rule
        optimizer: genetic
        objective: sharpe
        regime: trending_up
```

## IX. Advanced Optimization Patterns (Future Considerations)

### A. Queue-Based Component Optimization

An advanced pattern for even more efficient optimization involves "unpacking" strategies into their constituent components and processing them through a parameter queue:

```python
# Conceptual implementation
class QueueBasedOptimizer:
    """Process component optimizations through a parameter queue."""
    
    def optimize_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        # 1. Unpack strategy into optimizable components
        components = strategy.unpack_optimizable_components()
        
        # 2. Build parameter queue
        param_queue = []
        for component in components:
            param_space = component.get_parameter_space()
            for params in param_space.get_combinations():
                param_queue.append({
                    'component': component,
                    'parameters': params,
                    'component_id': component.instance_name
                })
        
        # 3. Process queue (potentially in parallel)
        results = self.backtest_engine.process_optimization_queue(param_queue)
        
        # 4. Aggregate and return results
        return self.aggregate_results(results)
```

**Advantages:**
- Enables true parallel processing of parameter combinations
- Minimal overhead per test (no strategy wrapper creation)
- Could batch similar computations for efficiency
- Natural fit for distributed optimization

**Challenges:**
- Components need standardized signal generation interface
- Dependency resolution between components (e.g., rules depending on indicators)
- Loss of strategy-level context and filters
- Requires significant changes to backtest infrastructure

**Implementation Considerations:**
- Could coexist with current wrapper-based isolation
- Best suited for simple, independent components
- May require component "contracts" for signal generation
- Consider hybrid approach: simple components use queue, complex ones use wrappers

This pattern represents a potential future optimization once the current isolation approach proves insufficient for scale.

## X. Component Interface for Optimization

Components should implement the `OptimizableComponent` interface:

```python
class OptimizableComponent:
    """Mixin that makes a component optimizable."""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for optimization."""
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters on the component."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameter values."""
        return True, ""
```

## X. Best Practices for Optimization

* **Parameter Space Definition**: Define realistic and constrained parameter spaces to make optimization feasible and avoid nonsensical parameter combinations.
* **Objective Function Selection**: Choose metrics that align with the strategy's goals and risk tolerance. Composite objectives can balance multiple criteria.
* **Preventing Overfitting**:
    * Use rigorous train/test/validation splits
    * Employ Walk-Forward Optimization
    * Be mindful of the number of parameters being optimized relative to the data size
    * Require a sufficient number of trades/samples (minimum 5-10 per regime)
* **Computational Efficiency**:
    * Use parallel processing when possible
    * Implement early stopping for poor-performing parameter sets
    * Cache indicator calculations when testing multiple combinations
    * For ensemble strategies, optimize base parameters before weights
* **Regime-Specific Considerations**:
    * Ensure adequate samples per regime before trusting optimized parameters
    * Consider cross-regime robustness in objective functions
    * Monitor and analyze boundary trade performance

This optimization framework provides the foundation for systematic strategy improvement while maintaining robustness and preventing overfitting.