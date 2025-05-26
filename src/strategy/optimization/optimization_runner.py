# src/strategy/optimization/optimization_runner.py
"""
Optimization Runner - Central entry point for the modular optimization framework.

This module provides a clean interface for running optimizations using the 
component-based optimization framework defined in OPTIMIZATION_FRAMEWORK.md.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

from src.core.component_base import ComponentBase
from src.core.container import Container


class OptimizationRunner(ComponentBase):
    """
    Main runner for the optimization framework.
    
    This component coordinates optimization activities by:
    1. Loading optimization configuration
    2. Instantiating optimization components (targets, methods, metrics, sequences)
    3. Running optimization workflows
    4. Managing optimization results
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        """Minimal constructor following ComponentBase pattern."""
        super().__init__(instance_name, config_key)
        
        # Component registries
        self.targets: Dict[str, 'OptimizationTarget'] = {}
        self.methods: Dict[str, 'OptimizationMethod'] = {}
        self.metrics: Dict[str, 'OptimizationMetric'] = {}
        self.sequences: Dict[str, 'OptimizationSequence'] = {}
        self.constraints: Dict[str, 'OptimizationConstraint'] = {}
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.optimization_config: Dict[str, Any] = {}
        self.output_dir: Path = Path("optimization_results")
    
    def _initialize(self):
        """Component-specific initialization logic."""
        # Load optimization configuration
        self.optimization_config = self.get_specific_config("optimization", {})
        self.output_dir = Path(self.get_specific_config("output_dir", "optimization_results"))
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"OptimizationRunner '{self.instance_name}' initialized")
    
    def setup(self):
        """Set up optimization components from configuration."""
        super().setup()
        
        # Load and instantiate optimization components
        self._load_targets()
        self._load_methods()
        self._load_metrics()
        self._load_sequences()
        self._load_constraints()
        
        self.logger.info(f"OptimizationRunner setup complete. Loaded: "
                        f"{len(self.targets)} targets, {len(self.methods)} methods, "
                        f"{len(self.metrics)} metrics, {len(self.sequences)} sequences, "
                        f"{len(self.constraints)} constraints")
    
    def _load_targets(self):
        """Load optimization targets from configuration."""
        targets_config = self.optimization_config.get("targets", {})
        
        for target_name, target_config in targets_config.items():
            try:
                # Get target class
                target_class_name = target_config.get("class")
                if not target_class_name:
                    self.logger.warning(f"No class specified for target '{target_name}'")
                    continue
                
                # Import and instantiate target
                target_class = self._import_class(f"targets.{target_class_name}")
                target_params = target_config.get("parameters", {})
                
                # Resolve any component references
                target_params = self._resolve_component_references(target_params)
                
                # Create target instance
                target = target_class(**target_params)
                self.targets[target_name] = target
                
                self.logger.debug(f"Loaded target '{target_name}' ({target_class_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load target '{target_name}': {e}", exc_info=True)
    
    def _load_methods(self):
        """Load optimization methods from configuration."""
        methods_config = self.optimization_config.get("methods", {})
        
        for method_name, method_config in methods_config.items():
            try:
                # Get method class
                method_class_name = method_config.get("class")
                if not method_class_name:
                    self.logger.warning(f"No class specified for method '{method_name}'")
                    continue
                
                # Import and instantiate method
                method_class = self._import_class(f"methods.{method_class_name}")
                method_params = method_config.get("parameters", {})
                
                # Create method instance
                method = method_class(**method_params)
                self.methods[method_name] = method
                
                self.logger.debug(f"Loaded method '{method_name}' ({method_class_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load method '{method_name}': {e}", exc_info=True)
    
    def _load_metrics(self):
        """Load optimization metrics from configuration."""
        metrics_config = self.optimization_config.get("metrics", {})
        
        for metric_name, metric_config in metrics_config.items():
            try:
                # Get metric class
                metric_class_name = metric_config.get("class")
                if not metric_class_name:
                    self.logger.warning(f"No class specified for metric '{metric_name}'")
                    continue
                
                # Import and instantiate metric
                metric_class = self._import_class(f"metrics.{metric_class_name}")
                metric_params = metric_config.get("parameters", {})
                
                # Create metric instance
                metric = metric_class(**metric_params)
                self.metrics[metric_name] = metric
                
                self.logger.debug(f"Loaded metric '{metric_name}' ({metric_class_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load metric '{metric_name}': {e}", exc_info=True)
    
    def _load_sequences(self):
        """Load optimization sequences from configuration."""
        sequences_config = self.optimization_config.get("sequences", {})
        
        for sequence_name, sequence_config in sequences_config.items():
            try:
                # Get sequence class
                sequence_class_name = sequence_config.get("class")
                if not sequence_class_name:
                    self.logger.warning(f"No class specified for sequence '{sequence_name}'")
                    continue
                
                # Import and instantiate sequence
                sequence_class = self._import_class(f"sequences.{sequence_class_name}")
                sequence_params = sequence_config.get("parameters", {})
                
                # Create sequence instance
                sequence = sequence_class(**sequence_params)
                self.sequences[sequence_name] = sequence
                
                self.logger.debug(f"Loaded sequence '{sequence_name}' ({sequence_class_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load sequence '{sequence_name}': {e}", exc_info=True)
    
    def _load_constraints(self):
        """Load optimization constraints from configuration."""
        constraints_config = self.optimization_config.get("constraints", {})
        
        for constraint_name, constraint_config in constraints_config.items():
            try:
                # Get constraint class
                constraint_class_name = constraint_config.get("class")
                if not constraint_class_name:
                    self.logger.warning(f"No class specified for constraint '{constraint_name}'")
                    continue
                
                # Import and instantiate constraint
                constraint_class = self._import_class(f"constraints.{constraint_class_name}")
                constraint_params = constraint_config.get("parameters", {})
                
                # Create constraint instance
                constraint = constraint_class(**constraint_params)
                self.constraints[constraint_name] = constraint
                
                self.logger.debug(f"Loaded constraint '{constraint_name}' ({constraint_class_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load constraint '{constraint_name}': {e}", exc_info=True)
    
    def _import_class(self, class_path: str):
        """
        Dynamically import a class from the optimization package.
        
        Args:
            class_path: Relative path to class (e.g., "methods.GridSearchMethod")
            
        Returns:
            The imported class
        """
        module_path, class_name = class_path.rsplit(".", 1)
        full_module_path = f"src.strategy.optimization.{module_path}"
        
        # Import module
        import importlib
        module = importlib.import_module(full_module_path)
        
        # Get class from module
        return getattr(module, class_name)
    
    def _resolve_component_references(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve component references in parameters.
        
        Component references are specified as strings starting with '@'.
        For example: "@ma_crossover_rule" would be resolved to the actual component.
        
        Args:
            params: Parameters potentially containing component references
            
        Returns:
            Parameters with resolved references
        """
        resolved_params = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("@"):
                # This is a component reference
                component_name = value[1:]  # Remove @ prefix
                
                # Try to resolve from container
                try:
                    resolved_component = self.container.resolve(component_name)
                    resolved_params[key] = resolved_component
                    self.logger.debug(f"Resolved component reference '{value}' -> {resolved_component}")
                except Exception as e:
                    self.logger.warning(f"Failed to resolve component reference '{value}': {e}")
                    resolved_params[key] = None
            
            elif isinstance(value, list):
                # Handle lists of potential references
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and item.startswith("@"):
                        try:
                            resolved_component = self.container.resolve(item[1:])
                            resolved_list.append(resolved_component)
                        except Exception as e:
                            self.logger.warning(f"Failed to resolve component reference '{item}': {e}")
                    else:
                        resolved_list.append(item)
                resolved_params[key] = resolved_list
            
            else:
                # Not a reference, keep as is
                resolved_params[key] = value
        
        return resolved_params
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute optimization (entrypoint method for Bootstrap).
        
        This method is called by Bootstrap when in optimization mode.
        It runs the optimization based on the configuration.
        
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting optimization execution")
        
        # Get optimization configuration
        opt_config = self.optimization_config.get('default', {})
        
        # Run the optimization
        results = self.run_optimization(
            target_name=opt_config.get('target', 'regime_strategy'),
            method_name=opt_config.get('method', 'grid_search'),
            metric_name=opt_config.get('metric', 'sharpe_ratio'),
            sequence_name=opt_config.get('sequence', 'sequential')
        )
        
        self.logger.info(f"Optimization completed with {len(results)} results")
        
        # Return the best result
        if results:
            best_result = max(results.values(), key=lambda x: x.get('metric_value', float('-inf')))
            return {
                'status': 'success',
                'best_result': best_result,
                'all_results': results
            }
        else:
            return {
                'status': 'no_results',
                'message': 'Optimization produced no results'
            }
    
    def run_optimization(self,
                        sequence_name: str,
                        targets: List[str],
                        methods: Optional[Dict[str, str]] = None,
                        metrics: Optional[Dict[str, str]] = None,
                        constraints: Optional[Dict[str, List[str]]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Run an optimization workflow.
        
        Args:
            sequence_name: Name of the optimization sequence to run
            targets: List of target names to optimize
            methods: Dict mapping target names to method names (optional)
            metrics: Dict mapping target names to metric names (optional)
            constraints: Dict mapping target names to constraint names (optional)
            **kwargs: Additional parameters passed to the sequence
            
        Returns:
            Dict containing optimization results
        """
        self.logger.info(f"Starting optimization: sequence='{sequence_name}', targets={targets}")
        
        # Validate sequence
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
            if self.methods:
                default_method = next(iter(self.methods))
                for target_name in targets:
                    methods[target_name] = default_method
                self.logger.info(f"Using default method '{default_method}' for all targets")
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = {}
            if self.metrics:
                default_metric = next(iter(self.metrics))
                for target_name in targets:
                    metrics[target_name] = default_metric
                self.logger.info(f"Using default metric '{default_metric}' for all targets")
        
        # Validate methods and metrics
        for target_name in targets:
            if target_name in methods and methods[target_name] not in self.methods:
                raise ValueError(f"Unknown method '{methods[target_name]}' for target '{target_name}'")
            if target_name in metrics and metrics[target_name] not in self.metrics:
                raise ValueError(f"Unknown metric '{metrics[target_name]}' for target '{target_name}'")
        
        # Get constraint functions
        constraint_functions = None
        if constraints:
            constraint_functions = {}
            for target_name, constraint_names in constraints.items():
                constraint_functions[target_name] = []
                for constraint_name in constraint_names:
                    if constraint_name in self.constraints:
                        constraint_functions[target_name].append(self.constraints[constraint_name])
                    else:
                        self.logger.warning(f"Unknown constraint '{constraint_name}' for target '{target_name}'")
        
        # Execute sequence
        try:
            result = sequence.execute(
                runner=self,
                targets=targets,
                methods=methods,
                metrics=metrics,
                constraints=constraint_functions,
                **kwargs
            )
            
            # Store result with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_key = f"{sequence_name}_{timestamp}"
            self.results[result_key] = result
            
            # Save result to file
            self._save_result(result_key, result)
            
            self.logger.info(f"Optimization complete. Result key: {result_key}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
    
    def create_objective_function(self,
                                target_name: str,
                                metric_name: str,
                                data_handler: Any,
                                **kwargs) -> Callable[[Dict[str, Any]], float]:
        """
        Create an objective function for optimization.
        
        This function creates a callable that:
        1. Sets parameters on the target
        2. Runs a backtest
        3. Calculates the metric
        4. Returns the metric value
        
        Args:
            target_name: Name of the optimization target
            metric_name: Name of the metric to optimize
            data_handler: Data handler for backtesting
            **kwargs: Additional parameters for backtesting
            
        Returns:
            Objective function callable
        """
        # Get target and metric
        target = self.targets[target_name]
        metric = self.metrics[metric_name]
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Evaluate parameters and return metric value."""
            try:
                # Set parameters on target
                target.set_parameters(params)
                
                # Run backtest
                backtest_result = self._run_backtest(
                    target=target,
                    data_handler=data_handler,
                    **kwargs
                )
                
                # Calculate metric
                equity_curve = backtest_result.get('equity_curve', [])
                trades = backtest_result.get('trades', [])
                metric_value = metric.calculate(equity_curve, trades)
                
                # Invert if lower is better
                if not metric.higher_is_better:
                    metric_value = -metric_value
                
                return metric_value
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {e}", exc_info=True)
                return float('-inf')  # Return worst possible value on error
        
        return objective_function
    
    def _run_backtest(self, target: Any, data_handler: Any, **kwargs) -> Dict[str, Any]:
        """
        Run a backtest with the current target parameters.
        
        Args:
            target: Optimization target (e.g., strategy)
            data_handler: Data handler
            **kwargs: Additional backtest parameters
            
        Returns:
            Backtest results including equity curve and trades
        """
        # This is a simplified backtest runner
        # In practice, this would integrate with the full backtesting system
        
        # Get required components from container
        portfolio = self.container.resolve(kwargs.get('portfolio_key', 'portfolio_manager'))
        risk_manager = self.container.resolve(kwargs.get('risk_manager_key', 'risk_manager'))
        execution_handler = self.container.resolve(kwargs.get('execution_handler_key', 'execution_handler'))
        
        # Reset components
        portfolio.reset()
        data_handler.reset()
        
        # Setup components
        portfolio.setup()
        risk_manager.setup()
        execution_handler.setup()
        target.setup()
        
        # Start components
        portfolio.start()
        risk_manager.start()
        execution_handler.start()
        target.start()
        data_handler.start()
        
        # Data handler will stream all data (it manages the event loop)
        # Components will process events as they arrive
        
        # Stop components
        data_handler.stop()
        target.stop()
        execution_handler.stop()
        risk_manager.stop()
        portfolio.stop()
        
        # Get results
        equity_curve = portfolio.get_equity_curve()
        trades = portfolio.get_trade_log()
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'final_value': portfolio.get_portfolio_value(),
            'total_trades': len(trades)
        }
    
    def _save_result(self, result_key: str, result: Dict[str, Any]):
        """
        Save optimization result to file.
        
        Args:
            result_key: Unique key for this result
            result: Result data to save
        """
        result_file = self.output_dir / f"{result_key}.json"
        
        try:
            # Convert non-serializable objects to strings
            serializable_result = self._make_serializable(result)
            
            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
                
            self.logger.info(f"Saved optimization result to {result_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}", exc_info=True)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def get_result(self, result_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific optimization result.
        
        Args:
            result_key: Key of the result to retrieve
            
        Returns:
            Result data or None if not found
        """
        return self.results.get(result_key)
    
    def list_results(self) -> List[str]:
        """
        List all available result keys.
        
        Returns:
            List of result keys
        """
        return list(self.results.keys())
    
    def start(self):
        """Start the optimization runner."""
        super().start()
        self.logger.info(f"OptimizationRunner '{self.instance_name}' started")
    
    def stop(self):
        """Stop the optimization runner."""
        self.logger.info(f"Stopping OptimizationRunner '{self.instance_name}'...")
        super().stop()
    
    def teardown(self):
        """Clean up resources."""
        # Clear registries
        self.targets.clear()
        self.methods.clear()
        self.metrics.clear()
        self.sequences.clear()
        self.constraints.clear()
        
        # Clear results
        self.results.clear()
        
        # Call parent teardown
        super().teardown()


def main():
    """
    Example usage of the OptimizationRunner.
    """
    # This would typically be called from main.py or a specific optimization script
    
    # Create container and load configuration
    from src.core.container import Container
    container = Container()
    
    # Create and setup optimization runner
    runner = OptimizationRunner('OptimizationRunner', 'optimization')
    
    # Initialize with context
    context = {
        'container': container,
        'config': container.get_config()
    }
    runner.initialize(context)
    runner.setup()
    runner.start()
    
    # Example: Run a simple grid search optimization
    try:
        result = runner.run_optimization(
            sequence_name="sequential",
            targets=["ma_rule_params"],
            methods={"ma_rule_params": "grid_search"},
            metrics={"ma_rule_params": "sharpe_ratio"},
            data_handler=container.resolve('data_handler')
        )
        
        print(f"Optimization complete. Best parameters: {result.get('best_params')}")
        
    finally:
        runner.stop()
        runner.teardown()


if __name__ == "__main__":
    main()