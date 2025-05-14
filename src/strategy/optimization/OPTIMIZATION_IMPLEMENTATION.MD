# Strategy Optimization Implementation Guide

## Overview

The Optimization framework provides a comprehensive, flexible system for optimizing trading strategy components in the ADMF-Trader system. It supports multiple optimization targets, methods, metrics, and sequences that can be composed to create sophisticated optimization workflows, enabling systematic improvement of strategy performance while preventing overfitting.

## Directory Structure

```
src/strategy/optimization/
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
├── lifecycle/
│   ├── __init__.py
│   ├── parameter_store.py          # Parameter version storage
│   ├── versioned_parameters.py     # Versioned parameter set
│   ├── metadata.py                 # Parameter metadata definitions
│   └── deployment_manager.py       # Parameter deployment management
├── manager.py                      # Optimization manager
└── utils.py                        # Optimization utilities
```

## Core Interfaces

### 1. Optimization Target

The `OptimizationTarget` interface defines components that can be optimized:

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

The `OptimizationMethod` interface defines algorithms for finding optimal parameters:

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

The `OptimizationMetric` interface defines measures for evaluating performance:

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

The `OptimizationSequence` interface defines orchestration of optimization workflows:

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

### 5. Optimization Constraint

The `OptimizationConstraint` interface defines restrictions on parameter values:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class OptimizationConstraint(ABC):
    """Base interface for optimization constraints."""
    
    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter values against constraint.
        
        Args:
            params: Dict mapping parameter names to values
            
        Returns:
            (is_valid, error_message)
        """
        pass
    
    def get_description(self) -> str:
        """
        Get constraint description.
        
        Returns:
            Description of this constraint
        """
        return self.__doc__ or "No description available"
```

## Implementation Examples

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
        # Get sequence
        if sequence_name not in self.sequences:
            raise ValueError(f"Unknown sequence: {sequence_name}")
            
        sequence = self.sequences[sequence_name]
        
        # Validate targets
        for target_name in targets:
            if target_name not in self.targets:
                raise ValueError(f"Unknown target: {target_name}")
                
        # Set default methods if not provided
        if methods is None:
            methods = {}
            default_method = next(iter(self.methods))
            for target_name in targets:
                methods[target_name] = default_method
                
        # Set default metrics if not provided
        if metrics is None:
            metrics = {}
            default_metric = next(iter(self.metrics))
            for target_name in targets:
                metrics[target_name] = default_metric
                
        # Execute sequence
        result = sequence.execute(
            manager=self,
            targets=targets,
            methods=methods,
            metrics=metrics,
            constraints=constraints,
            **kwargs
        )
        
        # Store result with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_key = f"{sequence_name}_{timestamp}"
        self.results[result_key] = result
        
        return result
```

### 2. Rule Parameters Target

The `RuleParametersTarget` optimizes parameters for trading rules:

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
        
    def _build_parameter_space(self) -> Dict[str, List[Any]]:
        """Build parameter space from rules."""
        space = {}
        
        for rule in self.rules:
            # Get rule parameters
            rule_params = rule.get_parameter_space()
            
            # Add to space with rule name prefix
            for param_name, param_values in rule_params.items():
                space[f"{rule.name}.{param_name}"] = param_values
                
        return space
        
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
                
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameter values."""
        # Group parameters by rule
        rule_params = {}
        for full_name, value in params.items():
            if '.' in full_name:
                rule_name, param_name = full_name.split('.', 1)
                if rule_name not in rule_params:
                    rule_params[rule_name] = {}
                rule_params[rule_name][param_name] = value
                
        # Validate each rule's parameters
        for rule in self.rules:
            if rule.name in rule_params:
                is_valid, error = rule.validate_parameters(rule_params[rule.name])
                if not is_valid:
                    return False, f"Invalid parameters for rule '{rule.name}': {error}"
                    
        return True, None
```

### 3. Grid Search Method

The `GridSearchMethod` evaluates all combinations of parameter values:

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
            'parameter_space': parameter_space,
            'evaluated_combinations': len(all_results),
            'total_combinations': total_combinations
        }
        
    def _generate_combinations(self, parameter_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        # Extract parameter names and values
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        # Generate combinations using recursion
        combinations = []
        current_combo = {}
        
        def generate_recursive(index):
            if index == len(param_names):
                combinations.append(current_combo.copy())
                return
                
            param_name = param_names[index]
            values = param_values[index]
            
            for value in values:
                current_combo[param_name] = value
                generate_recursive(index + 1)
                
        generate_recursive(0)
        return combinations
        
    def _check_constraints(self, params: Dict[str, Any], 
                          constraints: List[Callable[[Dict[str, Any]], bool]]) -> bool:
        """Check if parameters satisfy all constraints."""
        for constraint in constraints:
            if not constraint(params):
                return False
        return True
        
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get the best result found during optimization."""
        return self.best_result
```

### 4. Sharpe Ratio Metric

The `SharpeRatioMetric` calculates the Sharpe ratio from equity curve and trades:

```python
import numpy as np

class SharpeRatioMetric(OptimizationMetric):
    """
    Sharpe ratio metric.
    
    Calculates Sharpe ratio from equity curve.
    """
    
    def __init__(self, risk_free_rate=0.0, annualization_factor=252):
        """
        Initialize with parameters.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            annualization_factor: Annualization factor (default: 252 for daily data)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        
    def calculate(self, 
                 equity_curve: List[Dict[str, Any]], 
                 trades: List[Dict[str, Any]], 
                 **kwargs) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            equity_curve: Equity curve data
            trades: List of trade records
            **kwargs: Additional parameters
            
        Returns:
            float: Sharpe ratio
        """
        # Extract equity values
        equity_values = [point['equity'] for point in equity_curve]
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] / equity_values[i-1]) - 1
            returns.append(ret)
            
        # Calculate Sharpe ratio
        if not returns or np.std(returns) == 0:
            return 0.0
            
        excess_returns = np.mean(returns) - self.risk_free_rate
        sharpe = excess_returns / np.std(returns) * np.sqrt(self.annualization_factor)
        
        return sharpe
        
    @property
    def higher_is_better(self) -> bool:
        """Whether higher values of this metric are better."""
        return True
```

### 5. Walk-Forward Optimization Sequence

The `WalkForwardOptimization` sequence uses a rolling window approach:

```python
from datetime import datetime, timedelta

class WalkForwardOptimization(OptimizationSequence):
    """
    Walk-forward optimization sequence.
    
    Optimizes parameters using a rolling window approach.
    """
    
    def execute(self, 
               manager: 'OptimizationManager',
               targets: List[str],
               methods: Dict[str, str], 
               metrics: Dict[str, str],
               **kwargs) -> Dict[str, Any]:
        """
        Execute walk-forward optimization.
        
        Args:
            manager: Optimization manager instance
            targets: List of target names to optimize
            methods: Dict mapping target names to method names
            metrics: Dict mapping target names to metric names
            **kwargs: Additional parameters including:
                - data_handler: Data handler for backtesting
                - window_size: Size of each walk-forward window (in trading days)
                - step_size: Window step size (in trading days)
                - min_train_size: Minimum training data size
                - validation_size: Validation window size
                
        Returns:
            Dict containing optimization results
        """
        # Extract required parameters
        data_handler = kwargs.get('data_handler')
        if not data_handler:
            raise ValueError("data_handler is required for walk-forward optimization")
            
        window_size = kwargs.get('window_size', 252)  # Default: 1 year
        step_size = kwargs.get('step_size', 63)      # Default: 3 months
        min_train_size = kwargs.get('min_train_size', 252)  # Minimum training data
        validation_size = kwargs.get('validation_size', 63)  # Validation window
        
        # Get data range
        data_range = data_handler.get_data_range()
        start_date = data_range['start_date']
        end_date = data_range['end_date']
        
        # Initialize result containers
        window_results = []
        combined_equity_curve = []
        
        # Create window iterator
        current_date = start_date + timedelta(days=min_train_size)
        end_training_date = end_date - timedelta(days=validation_size)
        
        # Process each window
        while current_date <= end_training_date:
            # Define window boundaries
            train_start = current_date - timedelta(days=window_size)
            train_end = current_date
            validation_start = current_date
            validation_end = current_date + timedelta(days=validation_size)
            
            # Configure data handler for this window
            data_handler.set_training_range(train_start, train_end)
            data_handler.set_validation_range(validation_start, validation_end)
            
            # Optimize for this window
            window_result = self._optimize_window(
                manager=manager,
                targets=targets,
                methods=methods,
                metrics=metrics,
                data_handler=data_handler,
                window_info={
                    'train_start': train_start,
                    'train_end': train_end,
                    'validation_start': validation_start,
                    'validation_end': validation_end
                },
                **kwargs
            )
            
            # Store window result
            window_results.append(window_result)
            
            # Add validation equity curve to combined result
            combined_equity_curve.extend(window_result['validation_equity_curve'])
            
            # Move to next window
            current_date += timedelta(days=step_size)
            
        # Calculate combined performance metrics
        combined_metrics = self._calculate_combined_metrics(combined_equity_curve)
        
        # Return results
        return {
            'window_results': window_results,
            'combined_equity_curve': combined_equity_curve,
            'combined_metrics': combined_metrics,
            'sequence': 'walk_forward',
            'parameters': {
                'window_size': window_size,
                'step_size': step_size,
                'min_train_size': min_train_size,
                'validation_size': validation_size
            }
        }
```

### 6. Parameter Relationship Constraint

The `ParameterRelationshipConstraint` enforces relationships between parameters:

```python
class ParameterRelationshipConstraint(OptimizationConstraint):
    """
    Constraint for parameter relationships.
    
    Enforces relationships between multiple parameters.
    """
    
    def __init__(self, relationship_function, error_message=None):
        """
        Initialize with relationship function.
        
        Args:
            relationship_function: Function that validates parameter relationships
            error_message: Optional error message
        """
        self.relationship_function = relationship_function
        self.error_message = error_message
        
    def validate(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter relationships.
        
        Args:
            params: Dict mapping parameter names to values
            
        Returns:
            (is_valid, error_message)
        """
        is_valid = self.relationship_function(params)
        
        if not is_valid:
            return False, self.error_message or "Parameter relationship constraint violated"
            
        return True, None
```

## Strategy Lifecycle Management and Parameter Versioning

The framework integrates with the Strategy Lifecycle Management system to provide comprehensive parameter versioning and lifecycle management:

### 1. Versioned Parameter Set

```python
from datetime import datetime

class VersionedParameterSet:
    """Immutable, versioned set of strategy parameters."""
    
    def __init__(self, strategy_id, parameters, version=None, metadata=None):
        """
        Initialize a versioned parameter set.
        
        Args:
            strategy_id: Identifier of the strategy
            parameters: Strategy parameters
            version: Version identifier (auto-generated if None)
            metadata: Parameter metadata
        """
        self.strategy_id = strategy_id
        self.parameters = dict(parameters)  # Make a copy to ensure immutability
        self.version = version or self._generate_version()
        self.metadata = metadata or {}
        self.creation_timestamp = datetime.now()
        
    def _generate_version(self):
        """Generate a unique version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.strategy_id}_{timestamp}"
        
    def get_parameter(self, name, default=None):
        """Get a parameter value."""
        return self.parameters.get(name, default)
        
    def get_all_parameters(self):
        """Get a copy of all parameters."""
        return dict(self.parameters)
        
    def get_metadata(self):
        """Get parameter metadata."""
        return dict(self.metadata)
        
    def with_updated_metadata(self, metadata_updates):
        """Create a new instance with updated metadata."""
        new_metadata = {**self.metadata, **metadata_updates}
        return VersionedParameterSet(
            self.strategy_id,
            self.parameters,
            self.version,
            new_metadata
        )
```

### 2. Parameter Metadata

```python
class ParameterMetadata:
    """Standard metadata for parameter sets."""
    
    # Strategy information
    STRATEGY_VERSION = "strategy_version"
    STRATEGY_CODE_HASH = "strategy_code_hash"
    
    # Optimization information
    OPTIMIZATION_DATE = "optimization_date"
    OPTIMIZATION_OBJECTIVE = "optimization_objective"
    OPTIMIZATION_METHOD = "optimization_method"
    
    # Data information
    TRAIN_DATA_START = "train_data_start"
    TRAIN_DATA_END = "train_data_end"
    TEST_DATA_START = "test_data_start"
    TEST_DATA_END = "test_data_end"
    DATA_RESOLUTION = "data_resolution"
    SYMBOLS = "symbols"
    
    # Performance metrics
    TRAIN_SHARPE_RATIO = "train_sharpe_ratio"
    TRAIN_RETURNS = "train_returns"
    TRAIN_DRAWDOWN = "train_drawdown"
    TEST_SHARPE_RATIO = "test_sharpe_ratio"
    TEST_RETURNS = "test_returns"
    TEST_DRAWDOWN = "test_drawdown"
    
    # Deployment information
    DEPLOYMENT_STATUS = "deployment_status"
    DEPLOYMENT_DATE = "deployment_date"
    APPROVED_BY = "approved_by"
    
    # Lifecycle information
    ACTIVE = "active"
    RETIRED_DATE = "retired_date"
    RETIREMENT_REASON = "retirement_reason"
```

### 3. Parameter Repository

```python
class ParameterRepository:
    """Repository for versioned parameter sets."""
    
    def __init__(self, storage_adapter):
        """
        Initialize repository.
        
        Args:
            storage_adapter: Storage backend adapter
        """
        self.storage = storage_adapter
        
    def save(self, parameter_set):
        """
        Save a parameter set.
        
        Args:
            parameter_set: VersionedParameterSet to save
            
        Returns:
            str: Version identifier
        """
        return self.storage.save(parameter_set)
        
    def get_by_version(self, strategy_id, version):
        """
        Get parameter set by version.
        
        Args:
            strategy_id: Strategy identifier
            version: Version identifier
            
        Returns:
            VersionedParameterSet: The parameter set
        """
        return self.storage.get_by_version(strategy_id, version)
        
    def get_latest(self, strategy_id, filter_metadata=None):
        """
        Get latest parameter set, optionally filtered by metadata.
        
        Args:
            strategy_id: Strategy identifier
            filter_metadata: Optional metadata filter
            
        Returns:
            VersionedParameterSet: The latest parameter set
        """
        return self.storage.get_latest(strategy_id, filter_metadata)
        
    def get_version_history(self, strategy_id):
        """
        Get version history for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            list: Version history information
        """
        return self.storage.get_version_history(strategy_id)
        
    def update_metadata(self, strategy_id, version, metadata_updates):
        """
        Update metadata for a parameter set.
        
        Args:
            strategy_id: Strategy identifier
            version: Version identifier
            metadata_updates: Metadata updates to apply
            
        Returns:
            bool: Success status
        """
        return self.storage.update_metadata(strategy_id, version, metadata_updates)
```

### 4. Optimization Workflow

```python
class OptimizationWorkflow:
    """Workflow for strategy optimization and validation."""
    
    def __init__(self, parameter_repository, optimizer, validator):
        """
        Initialize workflow.
        
        Args:
            parameter_repository: ParameterRepository instance
            optimizer: Strategy optimizer
            validator: Strategy validator
        """
        self.repository = parameter_repository
        self.optimizer = optimizer
        self.validator = validator
        
    def run_optimization(self, strategy_id, config, train_data, test_data):
        """
        Run optimization workflow.
        
        Args:
            strategy_id: Strategy identifier
            config: Optimization configuration
            train_data: Training data
            test_data: Test data
            
        Returns:
            dict: Optimization results
        """
        # Step 1: Run optimization on training data
        optimization_results = self.optimizer.optimize(
            strategy_id=strategy_id,
            config=config,
            data=train_data
        )
        
        # Step 2: Extract best parameters
        best_parameters = optimization_results['best_parameters']
        train_metrics = optimization_results['metrics']
        
        # Step 3: Validate on test data
        validation_results = self.validator.validate(
            strategy_id=strategy_id,
            parameters=best_parameters,
            data=test_data
        )
        
        test_metrics = validation_results['metrics']
        
        # Step 4: Create and store parameter set with metadata
        metadata = {
            ParameterMetadata.STRATEGY_VERSION: config.get('strategy_version', '1.0.0'),
            ParameterMetadata.OPTIMIZATION_DATE: datetime.now().isoformat(),
            ParameterMetadata.OPTIMIZATION_OBJECTIVE: config.get('objective', 'sharpe_ratio'),
            ParameterMetadata.OPTIMIZATION_METHOD: config.get('method', 'grid_search'),
            ParameterMetadata.TRAIN_DATA_START: train_data.get_start_date().isoformat(),
            ParameterMetadata.TRAIN_DATA_END: train_data.get_end_date().isoformat(),
            ParameterMetadata.TEST_DATA_START: test_data.get_start_date().isoformat(),
            ParameterMetadata.TEST_DATA_END: test_data.get_end_date().isoformat(),
            ParameterMetadata.SYMBOLS: ','.join(train_data.get_symbols()),
            ParameterMetadata.TRAIN_SHARPE_RATIO: train_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TRAIN_RETURNS: train_metrics.get('returns', 0),
            ParameterMetadata.TRAIN_DRAWDOWN: train_metrics.get('max_drawdown', 0),
            ParameterMetadata.TEST_SHARPE_RATIO: test_metrics.get('sharpe_ratio', 0),
            ParameterMetadata.TEST_RETURNS: test_metrics.get('returns', 0),
            ParameterMetadata.TEST_DRAWDOWN: test_metrics.get('max_drawdown', 0),
            ParameterMetadata.DEPLOYMENT_STATUS: 'pending_approval'
        }
        
        parameter_set = VersionedParameterSet(
            strategy_id=strategy_id,
            parameters=best_parameters,
            metadata=metadata
        )
        
        # Step 5: Save parameter set
        version = self.repository.save(parameter_set)
        
        return {
            'version': version,
            'parameters': best_parameters,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'validation_passed': validation_results['passed']
        }
```

### 5. Deployment Manager

```python
class DeploymentManager:
    """Manages strategy deployment and rollback."""
    
    def __init__(self, parameter_repository, system_manager, change_logger):
        """
        Initialize manager.
        
        Args:
            parameter_repository: ParameterRepository instance
            system_manager: Trading system manager
            change_logger: Change log service
        """
        self.repository = parameter_repository
        self.system_manager = system_manager
        self.change_logger = change_logger
        
    def deploy_parameters(self, strategy_id, version, approver=None):
        """
        Deploy a parameter version.
        
        Args:
            strategy_id: Strategy identifier
            version: Parameter version to deploy
            approver: Person approving the deployment
            
        Returns:
            bool: Success status
        """
        # Get parameter set
        param_set = self.repository.get_by_version(strategy_id, version)
        if param_set is None:
            raise ValueError(f"Invalid parameter version: {version}")
            
        # Record current version for potential rollback
        current_version = self._get_current_version(strategy_id)
        
        # Update deployment metadata
        metadata_updates = {
            ParameterMetadata.DEPLOYMENT_STATUS: 'deployed',
            ParameterMetadata.DEPLOYMENT_DATE: datetime.now().isoformat()
        }
        if approver:
            metadata_updates[ParameterMetadata.APPROVED_BY] = approver
            
        self.repository.update_metadata(strategy_id, version, metadata_updates)
        
        # Update system configuration
        success = self.system_manager.update_strategy_parameters(
            strategy_id=strategy_id,
            parameters=param_set.get_all_parameters()
        )
        
        # Log the change
        self.change_logger.log_change(
            component="strategy",
            action="parameter_deployment",
            details={
                "strategy_id": strategy_id,
                "version": version,
                "previous_version": current_version,
                "approved_by": approver,
                "success": success
            }
        )
        
        return success
        
    def rollback(self, strategy_id, target_version=None):
        """
        Rollback to a previous version.
        
        Args:
            strategy_id: Strategy identifier
            target_version: Target version (or previous deployed if None)
            
        Returns:
            bool: Success status
        """
        if target_version is None:
            # Find previous deployed version
            history = self.repository.get_version_history(strategy_id)
            deployed_versions = [
                v for v in history 
                if v.get('metadata', {}).get(ParameterMetadata.DEPLOYMENT_STATUS) == 'deployed'
            ]
            
            # Sort by deployment date (descending)
            deployed_versions.sort(
                key=lambda v: v.get('metadata', {}).get(ParameterMetadata.DEPLOYMENT_DATE, ''),
                reverse=True
            )
            
            # Get previous version (second most recent)
            if len(deployed_versions) < 2:
                raise ValueError("No previous version available for rollback")
                
            target_version = deployed_versions[1]['version']
        
        # Deploy the target version
        return self.deploy_parameters(
            strategy_id=strategy_id,
            version=target_version,
            approver="SYSTEM_ROLLBACK"
        )
```

## Advanced Optimization Patterns

### 1. Multi-Stage Optimization

```python
# Example multi-stage optimization workflow
def run_multi_stage_optimization(optimization_manager, data_handler):
    """
    Run multi-stage optimization workflow.
    
    1. Optimize regime detector parameters
    2. Optimize rule parameters by regime
    3. Optimize rule weights by regime
    """
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
    
    return {
        'regime_result': regime_result,
        'rule_params_result': rule_params_result,
        'rule_weights_result': rule_weights_result
    }
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

## Best Practices

### 1. Parameter Space Definition

Define reasonable parameter bounds to prevent excessive search space:

```python
# For a moving average, very large windows rarely make sense
parameter_space.add_parameter('fast_window', 2, 50, 1, int)
parameter_space.add_parameter('slow_window', 10, 200, 5, int)

# Use constraints to enforce relationships
constraint = ParameterRelationshipConstraint(
    lambda params: params['fast_window'] < params['slow_window'],
    "Fast window must be smaller than slow window"
)
```

### 2. Objective Function Definition

Define appropriate performance metrics for your strategy type:

```python
def sharpe_ratio_objective(backtest_result):
    """Calculate Sharpe ratio from equity curve."""
    returns = calculate_returns(backtest_result['equity_curve'])
    sharpe = calculate_sharpe_ratio(returns)
    return sharpe

def sortino_ratio_objective(backtest_result):
    """Calculate Sortino ratio from equity curve."""
    returns = calculate_returns(backtest_result['equity_curve'])
    downside_deviation = calculate_downside_deviation(returns)
    sortino = calculate_sortino_ratio(returns, downside_deviation)
    return sortino
```

### 3. Preventing Overfitting

Use train/test splitting to evaluate out-of-sample performance:

```python
# Train parameters
data_handler.set_active_split('train')
train_results = optimizer.optimize(strategy, parameter_space, objective_function)

# Validate on test set
data_handler.set_active_split('test')
test_results = optimizer.evaluate(strategy, train_results['best_parameters'])

# Check validation threshold
validation_threshold = 0.7  # 70% of training performance
relative_performance = test_results['sharpe_ratio'] / train_results['best_score']
if relative_performance < validation_threshold:
    print(f"Warning: Potential overfitting - test performance is {relative_performance*100:.1f}% of training performance")
```

### 4. Parameter Versioning

Track parameter versions with full context and metadata:

```python
# Create parameter version from optimization result
parameter_set = VersionedParameterSet(
    strategy_id="trend_following_strategy",
    parameters=optimization_result['best_parameters'],
    metadata={
        "optimization_date": datetime.now().isoformat(),
        "train_period": "2020-01-01 to 2021-01-01",
        "test_period": "2021-01-01 to 2022-01-01",
        "train_sharpe": 1.85,
        "test_sharpe": 1.42,
        "optimization_method": "grid_search"
    }
)

# Save parameter version
version = parameter_repository.save(parameter_set)
```

### 5. Deployment Management

Implement a controlled deployment process:

```python
# Deploy parameters with approval
deployment_manager.deploy_parameters(
    strategy_id="trend_following_strategy",
    version="trend_following_strategy_20230101120000",
    approver="John Doe"
)

# Monitor performance after deployment
performance_monitor.monitor_strategy(
    strategy_id="trend_following_strategy",
    version="trend_following_strategy_20230101120000",
    window_days=30
)

# Rollback if needed
if performance_degradation_detected:
    deployment_manager.rollback(
        strategy_id="trend_following_strategy"
    )
```

## Implementation Timeline

### Phase 1: Core Interfaces (Week 1)
   - Implement base interfaces for targets, methods, metrics, and sequences
   - Create optimization manager
   - Develop parameter space definition
   - Implement constraints framework

### Phase 2: Basic Components (Week 2)
   - Implement grid search method
   - Create rule parameters target
   - Develop Sharpe ratio and other basic metrics
   - Implement sequential optimization sequence
   - Create constraint implementations

### Phase 3: Advanced Components (Week 3)
   - Implement genetic algorithm optimization
   - Create regime-specific optimization sequence
   - Develop advanced metrics (Sortino ratio, Calmar ratio)
   - Implement walk-forward optimization sequence
   - Create hierarchical optimization sequence

### Phase 4: Parameter Versioning (Week 4)
   - Implement versioned parameter sets
   - Create parameter metadata framework
   - Develop parameter repository
   - Implement optimization workflow

### Phase 5: Deployment and Monitoring (Week 5)
   - Implement deployment manager
   - Create performance monitoring framework
   - Develop rollback capabilities
   - Implement change logging system

### Phase 6: Integration and Testing (Week 6)
   - Integrate with strategy lifecycle
   - Create result analysis utilities
   - Develop visualization components
   - Implement comprehensive testing framework