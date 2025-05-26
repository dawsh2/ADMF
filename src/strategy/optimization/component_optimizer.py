"""
Component-level optimizer for ADMF system.

This module implements the ComponentOptimizer that can optimize individual
components or groups of components independently, as described in
OPTIMIZATION_FRAMEWORK.md.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Type, Union
from datetime import datetime
import json
from pathlib import Path

from src.core.component_base import ComponentBase
from src.strategy.base.parameter import ParameterSpace, Parameter
from .mixins.base import OptimizationMixin
from .mixins.grid_search import GridSearchMixin
from .mixins.genetic import GeneticOptimizationMixin
from .core.parameter_manager import ParameterManager, VersionedParameterSet


class OptimizableComponent(ComponentBase, GridSearchMixin):
    """Example of a component that supports optimization."""
    pass


class ComponentOptimizer:
    """
    Optimizer for individual components or component groups.
    
    This class orchestrates the optimization of components by:
    - Managing component lifecycle during optimization
    - Coordinating with backtest engine for evaluation
    - Tracking optimization progress and results
    - Supporting different optimization methods
    """
    
    def __init__(self, 
                 backtest_engine: Optional[Any] = None,
                 results_dir: str = "component_optimization_results",
                 parameter_manager: Optional[ParameterManager] = None):
        """
        Initialize the ComponentOptimizer.
        
        Args:
            backtest_engine: Engine for evaluating component performance
            results_dir: Directory for storing optimization results
            parameter_manager: Optional parameter manager for versioning
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backtest_engine = backtest_engine
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.parameter_manager = parameter_manager or ParameterManager(
            storage_dir=str(self.results_dir / "parameter_versions")
        )
        
        # Optimization state
        self._current_optimization: Optional[Dict[str, Any]] = None
        self._optimization_history: List[Dict[str, Any]] = []
        
    def optimize_component(self,
                         component: Union[ComponentBase, Type[ComponentBase]],
                         method: str = "grid_search",
                         objective_metric: str = "sharpe_ratio",
                         minimize: bool = False,
                         constraints: Optional[Dict[str, Any]] = None,
                         **optimization_params) -> Dict[str, Any]:
        """
        Optimize a single component.
        
        Args:
            component: Component instance or class to optimize
            method: Optimization method ('grid_search', 'genetic', etc.)
            objective_metric: Metric to optimize
            minimize: Whether to minimize the objective
            constraints: Optional constraints on parameters
            **optimization_params: Method-specific parameters
            
        Returns:
            Optimization results including best parameters
        """
        optimization_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Starting {method} optimization for component {component}")
        
        # Initialize optimization
        self._current_optimization = {
            'id': optimization_id,
            'component': component.__class__.__name__ if hasattr(component, '__class__') else str(component),
            'method': method,
            'objective_metric': objective_metric,
            'minimize': minimize,
            'start_time': datetime.now(),
            'iterations': []
        }
        
        try:
            # Create optimizable component if needed
            if isinstance(component, type):
                # It's a class, need to instantiate
                opt_component = self._create_optimizable_component(component, method)
            else:
                # It's an instance, wrap it
                opt_component = self._wrap_component_for_optimization(component, method)
                
            # Run optimization
            if method == "grid_search":
                results = self._run_grid_search(
                    opt_component, 
                    objective_metric, 
                    minimize,
                    constraints,
                    **optimization_params
                )
            elif method == "genetic":
                results = self._run_genetic_optimization(
                    opt_component,
                    objective_metric,
                    minimize,
                    constraints,
                    **optimization_params
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
            # Store results
            self._current_optimization['end_time'] = datetime.now()
            self._current_optimization['results'] = results
            self._optimization_history.append(self._current_optimization)
            
            # Save results
            self._save_optimization_results(optimization_id, self._current_optimization)
            
            # Create versioned parameter set
            if results['best_parameters']:
                self._create_parameter_version(
                    component=opt_component,
                    parameters=results['best_parameters'],
                    performance=results['best_performance'],
                    optimization_id=optimization_id,
                    method=method
                )
                
            return results
            
        finally:
            self._current_optimization = None
            
    def optimize_component_group(self,
                               components: List[ComponentBase],
                               method: str = "grid_search",
                               objective_metric: str = "sharpe_ratio",
                               minimize: bool = False,
                               joint: bool = False,
                               **optimization_params) -> Dict[str, Any]:
        """
        Optimize a group of components.
        
        Args:
            components: List of components to optimize
            method: Optimization method
            objective_metric: Metric to optimize
            minimize: Whether to minimize the objective
            joint: If True, optimize jointly; if False, optimize independently
            **optimization_params: Method-specific parameters
            
        Returns:
            Optimization results for all components
        """
        if joint:
            return self._optimize_components_jointly(
                components, method, objective_metric, minimize, **optimization_params
            )
        else:
            return self._optimize_components_independently(
                components, method, objective_metric, minimize, **optimization_params
            )
            
    def _create_optimizable_component(self, 
                                    component_class: Type[ComponentBase],
                                    method: str) -> ComponentBase:
        """Create an optimizable version of a component class."""
        # Dynamically create a class that inherits from both the component and optimization mixin
        if method == "grid_search":
            mixin_class = GridSearchMixin
        elif method == "genetic":
            mixin_class = GeneticOptimizationMixin
        else:
            mixin_class = OptimizationMixin
            
        # Create new class with multiple inheritance
        optimizable_class = type(
            f"Optimizable{component_class.__name__}",
            (component_class, mixin_class),
            {}
        )
        
        # Instantiate
        return optimizable_class(
            instance_name=f"opt_{component_class.__name__.lower()}"
        )
        
    def _wrap_component_for_optimization(self,
                                       component: ComponentBase,
                                       method: str) -> ComponentBase:
        """Wrap an existing component instance for optimization."""
        # Add optimization mixin methods to the instance
        if method == "grid_search":
            mixin = GridSearchMixin()
        elif method == "genetic":
            mixin = GeneticOptimizationMixin()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Copy mixin methods to component instance
        for attr_name in dir(mixin):
            if not attr_name.startswith('_') or attr_name.startswith('_optimization'):
                attr = getattr(mixin, attr_name)
                if callable(attr):
                    setattr(component, attr_name, attr.__get__(component, component.__class__))
                else:
                    setattr(component, attr_name, attr)
                    
        # Initialize optimization state
        component._parameter_history = []
        component._performance_history = []
        component._current_optimization_id = None
        component._optimization_metadata = {}
        
        return component
        
    def _run_grid_search(self,
                        component: ComponentBase,
                        objective_metric: str,
                        minimize: bool,
                        constraints: Optional[Dict[str, Any]],
                        **params) -> Dict[str, Any]:
        """Run grid search optimization."""
        # Initialize grid search
        num_combinations = component.initialize_grid_search()
        
        self.logger.info(f"Grid search initialized with {num_combinations} combinations")
        
        # Start optimization run
        component.start_optimization_run(
            self._current_optimization['id'],
            {'method': 'grid_search', 'combinations': num_combinations}
        )
        
        best_params = None
        best_performance = None
        best_metric = float('inf') if minimize else float('-inf')
        
        iteration = 0
        while True:
            # Get next parameters
            params = component.suggest_next_parameters()
            if params is None:
                break
                
            # Apply constraints if any
            if constraints and not self._check_constraints(params, constraints):
                continue
                
            # Evaluate parameters
            performance = self._evaluate_parameters(component, params)
            
            # Record results
            param_id = component.record_parameter_set(params, performance)
            
            # Check if best
            metric_value = performance.get(objective_metric)
            if metric_value is not None:
                is_better = (minimize and metric_value < best_metric) or \
                           (not minimize and metric_value > best_metric)
                           
                if is_better:
                    best_metric = metric_value
                    best_params = params.copy()
                    best_performance = performance.copy()
                    
            # Record iteration
            self._current_optimization['iterations'].append({
                'iteration': iteration,
                'parameters': params,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            })
            
            iteration += 1
            
            # Log progress
            if iteration % 10 == 0:
                progress = component.get_grid_progress()
                self.logger.info(f"Grid search progress: {progress['completed']}/{progress['total_combinations']} "
                               f"({progress['progress_percent']:.1f}%)")
                               
        # End optimization run
        component.end_optimization_run()
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'total_iterations': iteration,
            'method': 'grid_search',
            'search_info': component.get_search_space_info()
        }
        
    def _run_genetic_optimization(self,
                                component: ComponentBase,
                                objective_metric: str,
                                minimize: bool,
                                constraints: Optional[Dict[str, Any]],
                                **params) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        # Initialize genetic algorithm
        population_size = params.get('population_size', 50)
        max_generations = params.get('max_generations', 20)
        
        component.initialize_genetic_search(
            population_size=population_size,
            mutation_rate=params.get('mutation_rate', 0.1),
            crossover_rate=params.get('crossover_rate', 0.8),
            elitism_count=params.get('elitism_count', 2)
        )
        
        self.logger.info(f"Genetic algorithm initialized with population size {population_size}")
        
        # Start optimization run
        component.start_optimization_run(
            self._current_optimization['id'],
            {'method': 'genetic', 'population_size': population_size, 'max_generations': max_generations}
        )
        
        best_params = None
        best_performance = None
        best_metric = float('inf') if minimize else float('-inf')
        
        iteration = 0
        generation = 0
        
        while generation < max_generations:
            generation_complete = True
            
            # Evaluate population
            while True:
                params = component.suggest_next_parameters()
                if params is None:
                    break
                    
                generation_complete = False
                
                # Apply constraints
                if constraints and not self._check_constraints(params, constraints):
                    # Record as invalid
                    fitness = float('-inf') if not minimize else float('inf')
                    component.record_fitness(params, fitness)
                    continue
                    
                # Evaluate parameters
                performance = self._evaluate_parameters(component, params)
                
                # Calculate fitness
                metric_value = performance.get(objective_metric, 
                                              float('-inf') if not minimize else float('inf'))
                fitness = -metric_value if minimize else metric_value
                
                # Record results
                component.record_parameter_set(params, performance)
                component.record_fitness(params, fitness)
                
                # Check if best
                is_better = (minimize and metric_value < best_metric) or \
                           (not minimize and metric_value > best_metric)
                           
                if is_better:
                    best_metric = metric_value
                    best_params = params.copy()
                    best_performance = performance.copy()
                    
                # Record iteration
                self._current_optimization['iterations'].append({
                    'iteration': iteration,
                    'generation': generation,
                    'parameters': params,
                    'performance': performance,
                    'fitness': fitness,
                    'timestamp': datetime.now().isoformat()
                })
                
                iteration += 1
                
            if generation_complete:
                generation += 1
                progress = component.get_genetic_progress()
                self.logger.info(f"Generation {progress['generation']} complete. "
                               f"Best fitness: {progress['best_fitness']}")
                               
        # End optimization run
        component.end_optimization_run()
        
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'total_iterations': iteration,
            'generations': generation,
            'method': 'genetic',
            'final_population': component._population if hasattr(component, '_population') else []
        }
        
    def _evaluate_parameters(self,
                           component: ComponentBase,
                           parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate component with given parameters.
        
        Args:
            component: Component to evaluate
            parameters: Parameters to apply
            
        Returns:
            Performance metrics
        """
        # Apply parameters to component
        component.apply_parameters(parameters)
        
        # If we have a backtest engine, use it
        if self.backtest_engine:
            # Run backtest with component
            results = self.backtest_engine.run_component_backtest(component)
            return results.get('metrics', {})
        else:
            # Dummy evaluation for testing
            import random
            return {
                'sharpe_ratio': random.uniform(-1, 3),
                'total_return': random.uniform(-0.5, 1.5),
                'max_drawdown': random.uniform(0.05, 0.5),
                'win_rate': random.uniform(0.3, 0.7)
            }
            
    def _check_constraints(self, 
                         parameters: Dict[str, Any],
                         constraints: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraints."""
        # Example constraints:
        # - min/max bounds
        # - relationships between parameters
        # - excluded combinations
        
        for constraint_type, constraint_value in constraints.items():
            if constraint_type == 'bounds':
                for param, bounds in constraint_value.items():
                    if param in parameters:
                        value = parameters[param]
                        if 'min' in bounds and value < bounds['min']:
                            return False
                        if 'max' in bounds and value > bounds['max']:
                            return False
                            
            elif constraint_type == 'relationships':
                # e.g., "fast_ma < slow_ma"
                for relationship in constraint_value:
                    if not eval(relationship, {}, parameters):
                        return False
                        
        return True
        
    def _optimize_components_independently(self,
                                         components: List[ComponentBase],
                                         method: str,
                                         objective_metric: str,
                                         minimize: bool,
                                         **params) -> Dict[str, Any]:
        """Optimize each component independently."""
        results = {}
        
        for component in components:
            self.logger.info(f"Optimizing component {component.instance_name}")
            
            component_results = self.optimize_component(
                component=component,
                method=method,
                objective_metric=objective_metric,
                minimize=minimize,
                **params
            )
            
            results[component.instance_name] = component_results
            
        return {
            'method': 'independent',
            'component_results': results,
            'summary': self._summarize_group_results(results)
        }
        
    def _optimize_components_jointly(self,
                                   components: List[ComponentBase],
                                   method: str,
                                   objective_metric: str,
                                   minimize: bool,
                                   **params) -> Dict[str, Any]:
        """Optimize components jointly as a system."""
        # Create joint parameter space
        joint_space = ParameterSpace("joint_space")
        
        for component in components:
            if hasattr(component, 'get_parameter_space'):
                comp_space = component.get_parameter_space()
                if comp_space:
                    # Add as subspace
                    joint_space.add_subspace(component.instance_name, comp_space)
                    
        # Run optimization on joint space
        # This would require a more complex evaluation that considers all components together
        # For now, this is a placeholder
        
        raise NotImplementedError("Joint optimization not yet implemented")
        
    def _summarize_group_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize results from component group optimization."""
        summary = {
            'total_components': len(results),
            'successful': sum(1 for r in results.values() if r.get('best_parameters')),
            'total_iterations': sum(r.get('total_iterations', 0) for r in results.values())
        }
        
        # Aggregate performance metrics
        all_metrics = {}
        for comp_name, comp_results in results.items():
            if comp_results.get('best_performance'):
                for metric, value in comp_results['best_performance'].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
                    
        # Calculate average metrics
        summary['average_metrics'] = {}
        for metric, values in all_metrics.items():
            if values:
                summary['average_metrics'][metric] = sum(values) / len(values)
                
        return summary
        
    def _create_parameter_version(self,
                                component: ComponentBase,
                                parameters: Dict[str, Any],
                                performance: Dict[str, float],
                                optimization_id: str,
                                method: str) -> None:
        """Create a versioned parameter set."""
        self.parameter_manager.create_version(
            parameters=parameters,
            strategy_name=component.instance_name,
            optimization_method=method,
            training_period={'start': 'N/A', 'end': 'N/A'},  # Would come from backtest
            performance_metrics=performance,
            dataset_info={'type': 'component_optimization'},
            notes=f"Optimization run {optimization_id}"
        )
        
    def _save_optimization_results(self, 
                                 optimization_id: str,
                                 results: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        filename = f"component_opt_{optimization_id}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            # Convert datetime objects to strings
            serializable_results = json.loads(
                json.dumps(results, default=str)
            )
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Saved optimization results to {filepath}")
        
    def get_optimization_history(self,
                               component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization history, optionally filtered by component."""
        if component_name:
            return [
                opt for opt in self._optimization_history
                if opt['component'] == component_name
            ]
        return self._optimization_history.copy()