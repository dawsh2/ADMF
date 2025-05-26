# Optimization Framework

## Overview

This document defines the enhanced optimization framework for the ADMF-Trader system. The framework enables flexible, modular, and configurable optimization of trading system components, supporting multiple optimization methods, targets, metrics, and sequences that can be composed to create sophisticated optimization workflows.

## Core Architecture

The optimization framework follows a modular design with five key interface types:

1. **Optimization Targets** - Components that can be optimized
2. **Optimization Methods** - Algorithms for finding optimal parameters
3. **Optimization Metrics** - Measures for evaluating performance
4. **Optimization Sequences** - Orchestration of optimization workflows
5. **Optimization Constraints** - Restrictions on parameter values

These interfaces are coordinated by the `OptimizationManager`, which provides a central point of configuration and execution.

## Module Structure

```
strategy/optimization/
├── __init__.py
├── targets/
│   ├── __init__.py
│   ├── target_base.py              # Base optimization target interface
│   ├── rule_parameters.py          # Rule parameter optimization target
│   ├── rule_weights.py             # Rule weight optimization target
│   ├── regime_detector.py          # Regime detector optimization target
│   └── position_sizing.py          # Position sizing optimization target
├── methods/
│   ├── __init__.py
│   ├── method_base.py              # Base optimization method interface
│   ├── grid_search.py              # Grid search optimization
│   ├── genetic.py                  # Genetic algorithm optimization
│   ├── bayesian.py                 # Bayesian optimization
│   └── particle_swarm.py           # Particle swarm optimization
├── metrics/
│   ├── __init__.py
│   ├── metric_base.py              # Base metric interface
│   ├── return_metrics.py           # Return-based metrics (total, annualized)
│   ├── risk_metrics.py             # Risk-based metrics (drawdown, volatility)
│   ├── risk_adjusted_metrics.py    # Risk-adjusted metrics (Sharpe, Sortino)
│   └── custom_metrics.py           # Custom metric definitions
├── sequences/
│   ├── __init__.py
│   ├── sequence_base.py            # Base sequence interface
│   ├── sequential.py               # Sequential optimization
│   ├── parallel.py                 # Parallel optimization
│   ├── hierarchical.py             # Hierarchical optimization
│   ├── regime_specific.py          # Regime-specific optimization
│   └── walk_forward.py             # Walk-forward optimization
├── constraints/
│   ├── __init__.py
│   ├── constraint_base.py          # Base constraint interface
│   ├── parameter_constraints.py    # Parameter value constraints
│   ├── relationship_constraints.py # Parameter relationship constraints
│   └── performance_constraints.py  # Performance-based constraints
├── results/
│   ├── __init__.py
│   ├── optimization_result.py      # Optimization result container
│   ├── result_analysis.py          # Result analysis utilities
│   └── visualization.py            # Result visualization utilities
├── manager.py                      # Optimization manager
└── utils.py                        # Optimization utilities
```

## Core Interfaces

### 1. Optimization Target

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

class OptimizationTarget(ABC):
    """Base interface for any component that can be optimized."""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get the parameter space for optimization.
        
        Returns:
            Dict mapping parameter names to lists of possible values
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dict mapping parameter names to current values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters to specified values.
        
        Args:
            params: Dict mapping parameter names to values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a set of parameters.
        
        Args:
            params: Dict mapping parameter names to values
            
        Returns:
            (is_valid, error_message)
        """
        pass
    
    def reset(self) -> None:
        """Reset target to initial state."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this optimization target.
        
        Returns:
            Dict of metadata including target type, description, etc.
        """
        return {
            "type": self.__class__.__name__,
            "description": self.__doc__ or "No description available"
        }
```

### 2. Optimization Method

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, Tuple

class OptimizationMethod(ABC):
    """Base interface for optimization methods."""
    
    @abstractmethod
    def optimize(self, 
                parameter_space: Dict[str, List[Any]],
                objective_function: Callable[[Dict[str, Any]], float],
                constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform optimization.
        
        Args:
            parameter_space: Dict mapping parameter names to possible values
            objective_function: Function that evaluates parameter combinations
            constraints: Optional list of constraint functions
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dict containing optimization results
        """
        pass
    
    @abstractmethod
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the best result found during optimization.
        
        Returns:
            Dict with best parameters and score, or None if not optimized
        """
        pass
```

### 3. Optimization Metric

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class OptimizationMetric(ABC):
    """Base interface for optimization metrics."""
    
    @abstractmethod
    def calculate(self, 
                 equity_curve: Any, 
                 trades: List[Dict[str, Any]], 
                 **kwargs) -> float:
        """
        Calculate metric value.
        
        Args:
            equity_curve: Equity curve data
            trades: List of trade records
            **kwargs: Additional parameters
            
        Returns:
            Metric value (higher is better)
        """
        pass
    
    @property
    def higher_is_better(self) -> bool:
        """
        Whether higher values of this metric are better.
        
        Returns:
            True if higher is better, False otherwise
        """
        return True
```

### 4. Optimization Sequence

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class OptimizationSequence(ABC):
    """Base interface for optimization sequences."""
    
    @abstractmethod
    def execute(self, 
               manager: 'OptimizationManager',
               targets: List[str],
               methods: Dict[str, str], 
               metrics: Dict[str, str],
               **kwargs) -> Dict[str, Any]:
        """
        Execute the optimization sequence.
        
        Args:
            manager: Optimization manager instance
            targets: List of target names to optimize
            methods: Dict mapping target names to method names
            metrics: Dict mapping target names to metric names
            **kwargs: Additional sequence-specific parameters
            
        Returns:
            Dict containing optimization results
        """
        pass
```

## Key Component Implementations

### 1. Optimization Manager

The `OptimizationManager` centralizes optimization configuration and execution:

```python
class OptimizationManager:
    """
    Central manager for coordination of optimization activities.
    """
    
    def __init__(self, container=None, config=None):
        """
        Initialize the optimization manager.
        
        Args:
            container: Optional DI container
            config: Optional configuration
        """
        self.container = container
        self.config = config
        
        # Component registries
        self.targets = {}      # name -> OptimizationTarget
        self.methods = {}      # name -> OptimizationMethod
        self.metrics = {}      # name -> OptimizationMetric
        self.sequences = {}    # name -> OptimizationSequence
        self.constraints = {}  # name -> constraint function
        
        # Results storage
        self.results = {}      # key -> optimization result
        
        # Initialize from container/config if provided
        if container and config:
            self._initialize_from_container()
    
    def run_optimization(self, 
                        sequence_name: str,
                        targets: List[str],
                        methods: Dict[str, str] = None,
                        metrics: Dict[str, str] = None,
                        constraints: Dict[str, List[str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Run an optimization sequence.
        
        Args:
            sequence_name: Name of sequence to run
            targets: List of target names to optimize
            methods: Dict mapping target names to method names (default: use first registered method)
            metrics: Dict mapping target names to metric names (default: use first registered metric)
            constraints: Dict mapping target names to lists of constraint names
            **kwargs: Additional sequence-specific parameters
            
        Returns:
            Dict containing optimization results
        """
        # Implementation details omitted for brevity
        pass
```

### 2. Target Implementation Example: Rule Parameters

```python
class RuleParametersTarget(OptimizationTarget):
    """Optimization target for trading rule parameters."""
    
    def __init__(self, rules=None, parameter_space=None):
        """
        Initialize the rule parameters target.
        
        Args:
            rules: List of rules or rule container
            parameter_space: Optional explicit parameter space
        """
        self.rules = rules or []
        self._parameter_space = parameter_space or self._build_parameter_space()
        
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """Get the parameter space for optimization."""
        return self._parameter_space
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {}
        for rule in self.rules:
            rule_params = rule.get_parameters()
            for name, value in rule_params.items():
                # Prefix parameters with rule name to avoid conflicts
                params[f"{rule.name}.{name}"] = value
        return params
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters to specified values."""
        # Group parameters by rule
        rule_params = {}
        for full_name, value in params.items():
            if '.' in full_name:
                rule_name, param_name = full_name.split('.', 1)
                if rule_name not in rule_params:
                    rule_params[rule_name] = {}
                rule_params[rule_name][param_name] = value
        
        # Apply parameters to rules
        for rule in self.rules:
            if rule.name in rule_params:
                rule.set_parameters(rule_params[rule.name])
```

### 3. Method Implementation Example: Grid Search

```python
class GridSearchMethod(OptimizationMethod):
    """
    Grid search optimization method.
    
    Evaluates all combinations of parameter values.
    """
    
    def __init__(self, verbose=False):
        """
        Initialize grid search method.
        
        Args:
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.best_result = None
        
    def optimize(self, 
                parameter_space: Dict[str, List[Any]],
                objective_function: Callable[[Dict[str, Any]], float],
                constraints: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            parameter_space: Dict mapping parameter names to possible values
            objective_function: Function that evaluates parameter combinations
            constraints: Optional list of constraint functions
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dict containing optimization results
        """
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(parameter_space)
        
        # Track best result
        best_params = None
        best_score = float('-inf')
        all_results = []
        
        # Evaluate each combination
        total_combinations = len(param_combinations)
        for i, params in enumerate(param_combinations):
            if self.verbose:
                print(f"Evaluating combination {i+1}/{total_combinations}: {params}")
                
            # Skip if constraints are violated
            if constraints and not self._check_constraints(params, constraints):
                if self.verbose:
                    print("  Skipped due to constraint violation")
                continue
                
            # Evaluate objective function
            try:
                score = objective_function(params)
                all_results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                # Update best result if better
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    if self.verbose:
                        print(f"  New best score: {best_score}")
            except Exception as e:
                if self.verbose:
                    print(f"  Error: {str(e)}")
        
        # Store and return results
        self.best_result = {
            'best_params': best_params,
            'best_score': best_score
        }
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'method': 'grid_search',
            'parameter_space': parameter_space
        }
```

### 4. Sequence Implementation Example: Regime-Specific Optimization

```python
class RegimeSpecificOptimization(OptimizationSequence):
    """
    Optimize targets separately for each detected market regime.
    """
    
    def execute(self, 
               manager: 'OptimizationManager',
               targets: List[str],
               methods: Dict[str, str], 
               metrics: Dict[str, str],
               **kwargs) -> Dict[str, Any]:
        """
        Execute regime-specific optimization.
        
        Args:
            manager: Optimization manager instance
            targets: List of target names to optimize
            methods: Dict mapping target names to method names
            metrics: Dict mapping target names to metric names
            **kwargs: Additional parameters including:
                - data_handler: Data handler for backtesting
                - regime_detector: Regime detector to use
                - regime_detector_target: Alternative - name of target that is a regime detector
                - constraints: Dict mapping target names to lists of constraint functions
                - min_regime_data: Minimum data points required for regime optimization
                
        Returns:
            Dict containing optimization results
        """
        # Implementation details omitted for brevity
        pass
```

## Advanced Optimization Patterns

### 1. Multi-Stage Optimization

```python
# Stage 1: Optimize Regime Detector Parameters
regime_result = optimization_manager.run_optimization(
    sequence_name="sequential",
    targets=["regime_detector"],
    methods={"regime_detector": "grid_search"},
    metrics={"regime_detector": "classification_accuracy"},
    data_handler=data_handler
)

# Stage 2: Optimize Rule Parameters by Regime
rule_params_result = optimization_manager.run_optimization(
    sequence_name="regime_specific",
    targets=["ma_rule_params", "rsi_rule_params"],
    methods={"ma_rule_params": "grid_search", "rsi_rule_params": "grid_search"},
    metrics={"ma_rule_params": "sharpe_ratio", "rsi_rule_params": "sharpe_ratio"},
    data_handler=data_handler,
    regime_detector_target="regime_detector"
)

# Stage 3: Optimize Rule Weights by Regime
rule_weights_result = optimization_manager.run_optimization(
    sequence_name="regime_specific",
    targets=["rule_weights"],
    methods={"rule_weights": "genetic"},
    metrics={"rule_weights": "sharpe_ratio"},
    data_handler=data_handler,
    regime_detector_target="regime_detector"
)
```

### 2. Walk-Forward Optimization

```python
result = optimization_manager.run_optimization(
    sequence_name="walk_forward",
    targets=["strategy_params"],
    methods={"strategy_params": "grid_search"},
    metrics={"strategy_params": "sharpe_ratio"},
    data_handler=data_handler,
    window_size=126,  # 6 months of trading days
    step_size=21,     # 1 month step
    min_train_size=252,  # Minimum 1 year training data
    validation_size=63   # 3 months validation window
)
```

### 3. Hierarchical Optimization

```python
result = optimization_manager.run_optimization(
    sequence_name="hierarchical",
    targets=["strategy_group"],
    methods={"strategy_group": "hierarchical"},
    metrics={"strategy_group": "sharpe_ratio"},
    data_handler=data_handler,
    hierarchy={
        "level_1": {
            "target": "market_regime",
            "method": "grid_search"
        },
        "level_2": {
            "target": "rule_set",
            "method": "genetic"
        },
        "level_3": {
            "target": "rule_parameters",
            "method": "bayesian"
        }
    }
)
```

## Integration with Strategy Lifecycle

The Optimization Framework integrates with the Strategy Lifecycle Management system to provide:

1. **Automated Re-optimization** - Schedule regular re-optimization based on performance metrics
2. **Parameter Version Management** - Track all optimized parameter sets with their performance metrics
3. **Optimization Audit Trail** - Record full details of optimization runs for auditing
4. **Performance Comparison** - Compare performance across different optimization approaches
5. **Deployment Integration** - Streamlined path from optimization to parameter deployment

```python
# Example integration with Strategy Lifecycle Management
optimization_workflow = StrategyLifecycle.create_optimization_workflow(
    strategy_id="trend_following_strategy",
    optimization_manager=optimization_manager,
    data_handler=data_handler
)

# Run optimization
result = optimization_workflow.run_optimization(
    targets=["rule_parameters"],
    methods={"rule_parameters": "grid_search"},
    metrics={"rule_parameters": "sharpe_ratio"}
)

# Create parameter version from optimization result
parameter_version = optimization_workflow.create_parameter_version(
    strategy_id="trend_following_strategy",
    optimization_result=result,
    description="Monthly re-optimization"
)

# Validate on out-of-sample data
validation_result = optimization_workflow.validate_parameters(
    parameter_version=parameter_version,
    data_handler=validation_data_handler
)

# Deploy if validation successful
if validation_result['passed']:
    optimization_workflow.deploy_parameters(
        parameter_version=parameter_version,
        approver="system"
    )
```

## Configuration Examples

### YAML Configuration

```yaml
# optimization_config.yaml
optimization:
  targets:
    ma_rule_params:
      class: RuleParametersTarget
      parameters:
        rules:
          - ma_crossover_rule
          - ma_filter_rule
  
  methods:
    grid_search:
      class: GridSearchMethod
      parameters:
        verbose: true
    
    genetic:
      class: GeneticMethod
      parameters:
        population_size: 50
        generations: 20
        mutation_rate: 0.1
  
  metrics:
    sharpe_ratio:
      class: SharpeRatioMetric
      parameters:
        risk_free_rate: 0.0
        annualization_factor: 252
    
    sortino_ratio:
      class: SortinoRatioMetric
      parameters:
        risk_free_rate: 0.0
        annualization_factor: 252
  
  sequences:
    sequential:
      class: SequentialOptimization
      parameters: {}
    
    regime_specific:
      class: RegimeSpecificOptimization
      parameters:
        min_regime_data: 100
```

## Benefits

The enhanced optimization framework provides several key benefits:

1. **Modularity** - Components can be combined and configured in various ways
2. **Flexibility** - Support for multiple optimization methods, metrics, and sequences
3. **Extensibility** - New components can be added without changing the core framework
4. **Reproducibility** - Comprehensive tracking of optimization runs and results
5. **Advanced Workflows** - Support for sophisticated optimization patterns
6. **Integration** - Seamless operation with the Strategy Lifecycle Management system

## Implementation Plan

The optimization framework will be implemented in phases:

1. **Phase 1: Core Interfaces** - Define and implement all interface classes
2. **Phase 2: Basic Components** - Implement essential optimization components
3. **Phase 3: Regime Components** - Implement regime detection and optimization
4. **Phase 4: Advanced Components** - Implement advanced optimization methods and sequences
5. **Phase 5: Integration** - Integrate with Strategy Lifecycle Management

## Conclusion

The Enhanced Optimization Framework provides a comprehensive, flexible system for optimizing trading strategy components. By supporting multiple optimization targets, methods, metrics, and sequences, it enables sophisticated optimization workflows while maintaining a clean, modular architecture. The integration with Strategy Lifecycle Management ensures that optimization is a systematic, ongoing process rather than a one-time activity, helping to maintain strategy performance over time.