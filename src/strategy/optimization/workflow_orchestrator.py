"""
Optimization Workflow Orchestrator

Implements config-driven optimization workflows where the --optimize flag
triggers a sequence of optimization processes defined in the configuration.

Supported workflows:
1. Component optimization (rulewise optimization for each component)
2. Ensemble weight optimization (optimize weights after component optimization)
3. Sequential workflows (e.g., rulewise → genetic → ensemble weights)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import hashlib
from pathlib import Path

from src.core.component_base import ComponentBase
from src.core.exceptions import ConfigurationError
from .component_optimizer import ComponentOptimizer

logger = logging.getLogger(__name__)


class OptimizationWorkflowOrchestrator(ComponentBase):
    """
    Orchestrates complex optimization workflows based on configuration.
    
    Example config:
    ```yaml
    optimization:
      workflow:
        - name: "component_optimization"
          type: "rulewise"
          targets:
            - "rsi_indicator_*"
            - "ma_indicator_*"
          method: "grid_search"
          
        - name: "ensemble_weight_optimization"
          type: "ensemble_weights"
          method: "genetic"
          depends_on: ["component_optimization"]
    ```
    """
    
    def __init__(self, instance_name: str, config_key: Optional[str] = None):
        super().__init__(instance_name, config_key)
        self.workflow_steps = []
        self.results = {}
        self.output_dir = Path("optimization_results")
        
    def _initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load workflow configuration from optimization section
        config = self._context.config if hasattr(self._context, 'config') else None
        if config:
            # Try different ways to access config
            if hasattr(config, 'get_all_config'):
                full_config = config.get_all_config()
                opt_config = full_config.get("optimization", {})
                self.workflow_steps = opt_config.get("workflow", [])
            elif hasattr(config, 'get'):
                # Legacy method - get optimization section directly
                opt_config = config.get("optimization", {})
                if isinstance(opt_config, dict):
                    self.workflow_steps = opt_config.get("workflow", [])
                else:
                    self.workflow_steps = []
            else:
                # Fallback to component config
                self.workflow_steps = self.component_config.get("workflow", []) if self.component_config else []
        else:
            # Try getting from component config
            self.workflow_steps = self.component_config.get("workflow", []) if self.component_config else []
        
        if not self.workflow_steps:
            self.logger.warning(f"{self.instance_name}: No optimization workflow defined in config")
            
        # Validate workflow
        self._validate_workflow()
        
        self.logger.info(f"{self.instance_name}: Initialized with {len(self.workflow_steps)} workflow steps")
    
    def _load_workflow_config(self) -> None:
        """Load or reload workflow configuration from context."""
        self.logger.debug(f"Loading workflow config. Context type: {type(self._context)}")
        
        # Load workflow configuration from optimization section
        config = self._context.config if hasattr(self._context, 'config') else None
        self.logger.debug(f"Config found: {config is not None}")
        
        if config:
            # Try different ways to access config
            if hasattr(config, 'get_all_config'):
                self.logger.debug("Using get_all_config method")
                full_config = config.get_all_config()
                opt_config = full_config.get("optimization", {})
                self.workflow_steps = opt_config.get("workflow", [])
            elif hasattr(config, 'get'):
                self.logger.debug("Using get method")
                # Legacy method - get optimization section directly
                opt_config = config.get("optimization", {})
                if isinstance(opt_config, dict):
                    self.workflow_steps = opt_config.get("workflow", [])
                    self.logger.debug(f"Workflow steps from optimization: {len(self.workflow_steps)}")
                else:
                    self.workflow_steps = []
            else:
                self.logger.debug("Config has no get methods, using component config")
                # Fallback to component config
                self.workflow_steps = self.component_config.get("workflow", []) if self.component_config else []
        else:
            self.logger.debug("No config in context, using component config")
            # Try getting from component config
            self.workflow_steps = self.component_config.get("workflow", []) if self.component_config else []
        
        self.logger.info(f"Loaded {len(self.workflow_steps)} workflow steps")
        
        # Validate the reloaded workflow
        if self.workflow_steps:
            self._validate_workflow()
        
    def _validate_workflow(self) -> None:
        """Validate workflow configuration."""
        step_names = set()
        
        for step in self.workflow_steps:
            # Check required fields
            if "name" not in step:
                raise ConfigurationError("Workflow step missing 'name' field")
            if "type" not in step:
                raise ConfigurationError(f"Workflow step '{step['name']}' missing 'type' field")
                
            # Check for duplicate names
            if step["name"] in step_names:
                raise ConfigurationError(f"Duplicate workflow step name: {step['name']}")
            step_names.add(step["name"])
            
            # Validate dependencies
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep not in step_names:
                        raise ConfigurationError(
                            f"Step '{step['name']}' depends on unknown step '{dep}'"
                        )
    
    def run_optimization_workflow(self, data_handler, portfolio_manager, 
                                strategy, risk_manager, execution_handler,
                                train_dates: Tuple[str, str],
                                test_dates: Tuple[str, str]) -> Dict[str, Any]:
        """
        Run the complete optimization workflow.
        
        Returns:
            Dictionary of optimization results
        """
        logger.info(f"{self.instance_name}: Starting optimization workflow")
        
        # Reload workflow configuration if not loaded
        if not self.workflow_steps:
            self._load_workflow_config()
            logger.info(f"{self.instance_name}: Reloaded workflow configuration, found {len(self.workflow_steps)} steps")
        
        workflow_results = {}
        completed_steps = set()
        
        for step in self.workflow_steps:
            # Check dependencies
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep not in completed_steps:
                        logger.error(f"Cannot run step '{step['name']}' - dependency '{dep}' not completed")
                        continue
            
            logger.info(f"{self.instance_name}: Running workflow step: {step['name']}")
            
            try:
                if step["type"] == "rulewise":
                    result = self._run_rulewise_optimization(
                        step, data_handler, portfolio_manager, strategy, 
                        risk_manager, execution_handler, train_dates, test_dates
                    )
                elif step["type"] == "ensemble_weights":
                    result = self._run_ensemble_weight_optimization(
                        step, data_handler, portfolio_manager, strategy,
                        risk_manager, execution_handler, train_dates, test_dates,
                        workflow_results
                    )
                elif step["type"] in ["regime_optimization", "regime"]:
                    result = self._run_regime_optimization(
                        step, data_handler, portfolio_manager, strategy,
                        risk_manager, execution_handler, train_dates, test_dates
                    )
                else:
                    logger.warning(f"Unknown optimization type: {step['type']}")
                    continue
                    
                workflow_results[step["name"]] = result
                completed_steps.add(step["name"])
                
                # Save intermediate results
                self._save_step_results(step["name"], result)
                
            except Exception as e:
                logger.error(f"Error in workflow step '{step['name']}': {str(e)}")
                workflow_results[step["name"]] = {"error": str(e)}
        
        # Display training set performance summary before test
        self._display_training_summary(workflow_results)
        
        # FINAL STEP: Run complete ensemble on TEST dataset with regime-adaptive parameters
        self.logger.info("===== FINAL TEST EVALUATION WITH COMPLETE REGIME-ADAPTIVE ENSEMBLE =====")
        
        # Ensure all best parameters have been applied to the strategy
        self._apply_best_parameters_to_strategy(strategy, workflow_results)
        
        # Switch to test dataset
        if hasattr(data_handler, 'set_active_dataset'):
            data_handler.set_active_dataset('test')
            self.logger.info("Set data handler to TEST dataset for final ensemble evaluation")
        
        # Run final test evaluation
        try:
            # Reset portfolio for clean test run
            portfolio_manager.reset()
            
            # Get backtest runner - try multiple approaches
            backtest_runner = None
            
            # Try getting from container
            if hasattr(self._context, 'container'):
                try:
                    backtest_runner = self._context.container.resolve('backtest_runner')
                except:
                    pass
            
            # If not found, try getting from context directly
            if not backtest_runner and isinstance(self._context, dict):
                backtest_runner = self._context.get('backtest_runner')
                
            # If still not found, create one with proper configuration
            if not backtest_runner:
                from src.execution.backtest_runner import BacktestRunner
                backtest_runner = BacktestRunner(instance_name="final_test_runner")
                
                # Create a proper context for the backtest runner
                if isinstance(self._context, dict):
                    test_context = self._context.copy()
                else:
                    # Create context dict from object
                    test_context = {
                        'config_loader': getattr(self._context, 'config_loader', None),
                        'config': getattr(self._context, 'config', None),
                        'event_bus': getattr(self._context, 'event_bus', None),
                        'container': getattr(self._context, 'container', None),
                        'logger': getattr(self._context, 'logger', self.logger),
                        'metadata': getattr(self._context, 'metadata', {})
                    }
                
                # Ensure the backtest runner uses test dataset
                if 'metadata' not in test_context:
                    test_context['metadata'] = {}
                if 'cli_args' not in test_context['metadata']:
                    test_context['metadata']['cli_args'] = {}
                test_context['metadata']['cli_args']['dataset'] = 'test'
                
                # Initialize with test context
                backtest_runner.initialize(test_context)
                
                # Start the backtest runner (must be started before execute)
                if not backtest_runner.running:
                    backtest_runner.start()
                    
            if backtest_runner:
                self.logger.info(f"BacktestRunner state - initialized: {backtest_runner.initialized}, running: {backtest_runner.running}")
                
                if hasattr(backtest_runner, 'execute'):
                    self.logger.info("Running final test backtest with regime-adaptive ensemble")
                    
                    # Double-check the state
                    if not backtest_runner.initialized:
                        self.logger.error("BacktestRunner not initialized!")
                    if not backtest_runner.running:
                        self.logger.error("BacktestRunner not running!")
                        
                    test_results = backtest_runner.execute()
                    
                    # Add test results to workflow results
                    workflow_results['final_test_evaluation'] = {
                        'dataset': 'test',
                        'results': test_results,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"Final test Sharpe ratio: {test_results.get('performance_metrics', {}).get('portfolio_sharpe_ratio', 'N/A')}")
                    self.logger.info(f"Final test return: {test_results.get('total_return', 'N/A')}")
                else:
                    self.logger.error("BacktestRunner does not have execute method")
            else:
                self.logger.error("No backtest runner available for final test evaluation")
        except Exception as e:
            self.logger.error(f"Error in final test evaluation: {e}")
            workflow_results['final_test_evaluation'] = {'error': str(e)}
        
        # Restore train dataset
        if hasattr(data_handler, 'set_active_dataset'):
            data_handler.set_active_dataset('train')
        
        # Save complete workflow results
        self._save_workflow_results(workflow_results)
        
        return workflow_results
    
    def _run_rulewise_optimization(self, step_config: Dict[str, Any],
                                 data_handler, portfolio_manager, strategy,
                                 risk_manager, execution_handler,
                                 train_dates: Tuple[str, str],
                                 test_dates: Tuple[str, str]) -> Dict[str, Any]:
        """Run component-based (rulewise) optimization."""
        targets = step_config.get("targets", [])
        method = step_config.get("method", "grid_search")
        
        self.logger.info(f"Running component-based optimization for targets {targets}")
        
        # Create component optimizer
        component_optimizer = ComponentOptimizer(
            instance_name=f"{self.instance_name}_component_optimizer"
        )
        component_optimizer.initialize(self._context)
        
        # Set up isolated evaluator if requested
        isolate = step_config.get("isolate", False)
        if isolate:
            # Get backtest runner from container
            backtest_runner = None
            try:
                if hasattr(self._context, 'container') and hasattr(self._context.container, 'resolve'):
                    backtest_runner = self._context.container.resolve('backtest_runner')
                    self.logger.info("Using existing backtest_runner from container")
            except Exception as e:
                self.logger.warning(f"Could not resolve backtest_runner from container: {e}")
            
            if not backtest_runner:
                # Create a backtest runner if not available
                from src.execution.backtest_runner import BacktestRunner
                backtest_runner = BacktestRunner(instance_name="isolated_backtest_runner")
                backtest_runner.initialize(self._context)
            
            # Configure isolated evaluator with required components
            component_optimizer.set_isolated_evaluator(
                backtest_runner=backtest_runner,
                data_handler=data_handler,
                portfolio=portfolio_manager,
                risk_manager=risk_manager,
                execution_handler=execution_handler
            )
        
        # Get components to optimize
        components_to_optimize = []
        
        # If targets is empty or contains "*", optimize all components of the strategy
        if not targets or "*" in targets:
            # Get all optimizable components from strategy
            if hasattr(strategy, '_indicators'):
                components_to_optimize.extend(strategy._indicators.values())
            if hasattr(strategy, '_rules'):
                components_to_optimize.extend(strategy._rules.values())
            if hasattr(strategy, '_features'):
                components_to_optimize.extend(strategy._features.values())
        else:
            # Get specific components matching patterns
            for pattern in targets:
                if "*" in pattern:
                    # Pattern matching - look in strategy components
                    prefix = pattern.replace("*", "")
                    if hasattr(strategy, '_indicators'):
                        for name, comp in strategy._indicators.items():
                            if name.startswith(prefix):
                                components_to_optimize.append(comp)
                    if hasattr(strategy, '_rules'):
                        for name, comp in strategy._rules.items():
                            if name.startswith(prefix):
                                components_to_optimize.append(comp)
                else:
                    # Exact match - try to get from strategy
                    component = None
                    if hasattr(strategy, '_indicators') and pattern in strategy._indicators:
                        component = strategy._indicators[pattern]
                    elif hasattr(strategy, '_rules') and pattern in strategy._rules:
                        component = strategy._rules[pattern]
                    elif hasattr(strategy, '_features') and pattern in strategy._features:
                        component = strategy._features[pattern]
                    
                    if component:
                        components_to_optimize.append(component)
        
        if not components_to_optimize:
            # If no components found in strategy, try the strategy itself
            components_to_optimize = [strategy]
        
        self.logger.info(f"Found {len(components_to_optimize)} components to optimize")
        
        # For now, optimize each component independently
        # In the future, we could optimize jointly
        results = {}
        
        for component in components_to_optimize:
            self.logger.info(f"Optimizing component: {component.instance_name}")
            
            # STEP 1: Optimize on TRAIN dataset
            self.logger.info(f"===== OPTIMIZING {component.instance_name} ON TRAIN DATASET =====")
            
            # Ensure we're on train dataset
            if hasattr(data_handler, 'set_active_dataset'):
                data_handler.set_active_dataset('train')
                self.logger.info(f"Set data handler to TRAIN dataset for {component.instance_name} optimization")
            
            # Create evaluator based on whether we're isolating or not
            if isolate and hasattr(component_optimizer, '_isolated_evaluator'):
                # Use the isolated evaluator
                evaluator = component_optimizer._isolated_evaluator.create_evaluator_function(
                    metric=step_config.get("metric", "sharpe_ratio")
                )
            else:
                # Create a standard evaluator that runs a backtest
                def evaluator(comp):
                    # For now, return a mock score
                    # In production, this would run a full backtest
                    self.logger.info(f"Evaluating component {comp.instance_name}")
                    return 0.5  # Placeholder score
            
            # Optimize this component on training data
            comp_results = component_optimizer.optimize_component(
                component=component,
                evaluator=evaluator,
                method=method,
                isolate=isolate,
                metric=step_config.get("metric", "sharpe_ratio")
            )
            
            # NO TEST EVALUATION HERE - only training optimization
            # Test evaluation happens once at the very end with the complete ensemble
            
            # Check if we have regime-specific results
            if 'regime_best_parameters' in comp_results:
                self.logger.info(f"Regime-specific best parameters found for {component.instance_name}")
                for regime, regime_data in comp_results.get('regime_best_parameters', {}).items():
                    self.logger.info(f"  {regime}: params={regime_data.get('parameters', {})}, "
                                   f"sharpe={regime_data.get('sharpe_ratio', 0):.4f}")
            
            results[component.instance_name] = comp_results
        
        return {
            "optimization_type": "component",
            "targets": targets,
            "method": method,
            "components_optimized": len(components_to_optimize),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_ensemble_weight_optimization(self, step_config: Dict[str, Any],
                                        data_handler, portfolio_manager, strategy,
                                        risk_manager, execution_handler,
                                        train_dates: Tuple[str, str],
                                        test_dates: Tuple[str, str],
                                        previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble weight optimization."""
        self.logger.info("Running ensemble weight optimization")
        
        container = self._context.container if hasattr(self._context, 'container') else None
        if not container:
            return {"error": "Container not available"}
        
        # Check if strategy has weight parameters
        if not hasattr(strategy, 'get_optimizable_parameters'):
            self.logger.warning("Strategy does not support parameter optimization")
            return {
                "status": "not_supported", 
                "message": "Strategy does not implement optimization interface"
            }
        
        # Get all parameters to find weight parameters
        all_params = strategy.get_optimizable_parameters()
        weight_params = [k for k in all_params.keys() if 'weight' in k.lower()]
        
        if not weight_params:
            self.logger.warning("No weight parameters found in strategy")
            return {
                "status": "no_weights",
                "message": "Strategy has no weight parameters to optimize"
            }
        
        self.logger.info(f"Found {len(weight_params)} weight parameters: {weight_params}")
        
        # Create component optimizer
        component_optimizer = ComponentOptimizer(
            instance_name=f"{self.instance_name}_weight_optimizer"
        )
        component_optimizer.initialize(self._context)
        
        # Create evaluator for weights
        def weight_evaluator(weights: Dict[str, float]) -> float:
            # Apply weights to strategy
            strategy.apply_parameters(weights)
            
            # Run evaluation (simplified - should run backtest)
            optimizer = container.get("optimizer")
            if optimizer and hasattr(optimizer, 'execute'):
                opt_results = optimizer.execute()
                if opt_results and 'best_performance' in opt_results:
                    return opt_results['best_performance'].get('sharpe_ratio', 0.0)
            return 0.0
        
        # STEP 1: Optimize weights on TRAIN dataset
        self.logger.info("===== OPTIMIZING ENSEMBLE WEIGHTS ON TRAIN DATASET =====")
        
        # Disable regime switching during optimization
        if hasattr(strategy, '_enable_regime_switching'):
            original_regime_switching = strategy._enable_regime_switching
            strategy._enable_regime_switching = False
            self.logger.info("Disabled regime switching for weight optimization")
        
        # Ensure we're on train dataset
        if hasattr(data_handler, 'set_active_dataset'):
            data_handler.set_active_dataset('train')
            self.logger.info("Set data handler to TRAIN dataset for weight optimization")
        
        # Run weight optimization on training data
        results = component_optimizer.optimize_weights(
            strategy=strategy,
            evaluator=weight_evaluator,
            weight_params=weight_params,
            method=step_config.get("method", "grid_search")
        )
        
        # NO TEST EVALUATION HERE - only training optimization
        # Test evaluation happens once at the very end with the complete ensemble
        
        # Restore regime switching setting
        if hasattr(strategy, '_enable_regime_switching'):
            strategy._enable_regime_switching = original_regime_switching
            self.logger.info(f"Restored regime switching to {original_regime_switching}")
        
        # Tag results
        results['optimization_type'] = 'ensemble_weights'
        results['weight_parameters'] = weight_params
        
        return results
    
    def _run_regime_optimization(self, step_config: Dict[str, Any],
                               data_handler, portfolio_manager, strategy,
                               risk_manager, execution_handler,
                               train_dates: Tuple[str, str],
                               test_dates: Tuple[str, str]) -> Dict[str, Any]:
        """Run regime-specific optimization."""
        logger.info("Running regime optimization")
        
        container = self._context.container if hasattr(self._context, 'container') else None
        if not container:
            return {"error": "Container not available"}
            
        optimizer = container.get("optimizer")
        if not optimizer:
            self.logger.warning("No optimizer component found")
        
        # For now, return placeholder
        self.logger.info("Regime optimization not yet fully implemented via workflow")
        return {
            "status": "pending_implementation",
            "message": "Regime optimization will be implemented"
        }
    
    def _evaluate_component_params(self, component, params, data_handler,
                                 portfolio_manager, strategy, risk_manager,
                                 execution_handler, train_dates) -> float:
        """Evaluate component with specific parameters."""
        # This is a simplified evaluation - in practice you'd run a backtest
        # For now, return a mock score
        return 0.5
    
    def _save_step_results(self, step_name: str, results: Dict[str, Any]) -> None:
        """Save results from a workflow step."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{step_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved {step_name} results to {filename}")
    
    def _apply_best_parameters_to_strategy(self, strategy, workflow_results: Dict[str, Any]) -> None:
        """Apply best parameters from optimization results to strategy."""
        self.logger.info("Applying best parameters to strategy for final test evaluation")
        
        # Collect all best parameters from component optimizations
        all_params = {}
        
        # Get parameters from rulewise optimizations
        for step_name, results in workflow_results.items():
            if results.get('optimization_type') == 'component':
                # Get parameters from each optimized component
                component_results = results.get('results', {})
                for comp_name, comp_data in component_results.items():
                    best_params = comp_data.get('best_parameters', {})
                    if best_params:
                        # Add with proper namespacing
                        for param, value in best_params.items():
                            all_params[param] = value
                        self.logger.debug(f"Added parameters from {comp_name}: {best_params}")
        
        # Get weights from ensemble optimization
        for step_name, results in workflow_results.items():
            if results.get('optimization_type') == 'ensemble_weights':
                best_weights = results.get('best_parameters', {})
                if best_weights:
                    all_params.update(best_weights)
                    self.logger.debug(f"Added ensemble weights: {best_weights}")
        
        # Apply all parameters to strategy
        if all_params:
            strategy.apply_parameters(all_params)
            self.logger.info(f"Applied {len(all_params)} parameters to strategy")
        else:
            self.logger.warning("No best parameters found to apply to strategy")
    
    def _display_training_summary(self, workflow_results: Dict[str, Any]) -> None:
        """Display comprehensive training set performance summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("===== TRAINING SET OPTIMIZATION SUMMARY =====")
        self.logger.info("="*80)
        
        # Collect all regime-specific best parameters
        regime_params = self._collect_regime_parameters(workflow_results)
        
        if regime_params:
            self.logger.info("\n===== REGIME-SPECIFIC BEST PARAMETERS =====")
            for regime, params in regime_params.items():
                self.logger.info(f"\nRegime: {regime.upper()}")
                self.logger.info("-" * 40)
                
                # Group parameters by component
                component_params = {}
                for param_name, value in params.items():
                    if '.' in param_name:
                        component, param = param_name.rsplit('.', 1)
                        if component not in component_params:
                            component_params[component] = {}
                        component_params[component][param] = value
                    else:
                        if 'general' not in component_params:
                            component_params['general'] = {}
                        component_params['general'][param_name] = value
                
                # Display parameters by component
                for component, comp_params in component_params.items():
                    self.logger.info(f"  {component}:")
                    for param, value in comp_params.items():
                        self.logger.info(f"    {param}: {value}")
            
            # Display regime performance statistics
            self.logger.info("\n===== REGIME PERFORMANCE STATISTICS =====")
            for step_name, results in workflow_results.items():
                if results.get('optimization_type') == 'component':
                    component_results = results.get('results', {})
                    for comp_name, comp_data in component_results.items():
                        regime_stats = comp_data.get('regime_statistics', {})
                        if regime_stats:
                            self.logger.info(f"\n{comp_name} regime statistics:")
                            for regime, stats in regime_stats.items():
                                self.logger.info(f"  {regime}:")
                                self.logger.info(f"    Avg Sharpe: {stats.get('avg_sharpe', 0):.4f}")
                                self.logger.info(f"    Avg Win Rate: {stats.get('avg_win_rate', 0):.2%}")
                                self.logger.info(f"    Total Trades: {stats.get('total_trades', 0)}")
        
        self.logger.info("\n" + "="*80 + "\n")
    
    def _collect_regime_parameters(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect all regime-specific parameters from optimization results."""
        regime_params = {}
        
        # Collect from component optimizations
        for step_name, results in workflow_results.items():
            if results.get('optimization_type') == 'component':
                component_results = results.get('results', {})
                for comp_name, comp_data in component_results.items():
                    regime_best = comp_data.get('regime_best_parameters', {})
                    if regime_best:
                        for regime, regime_data in regime_best.items():
                            if regime not in regime_params:
                                regime_params[regime] = {}
                            # Add component parameters with proper namespacing
                            comp_params = regime_data.get('parameters', {})
                            for param, value in comp_params.items():
                                # Namespace by component if needed
                                if comp_name in param:
                                    regime_params[regime][param] = value
                                else:
                                    regime_params[regime][f"{comp_name}.{param}"] = value
        
        return regime_params
    
    def _save_regime_parameters(self, regime_params: Dict[str, Any]) -> None:
        """Save regime-specific parameters to file for strategy loading."""
        if not regime_params:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"regime_optimized_parameters_{timestamp}.json"
        
        # Format for easy loading by strategy
        formatted_params = {
            "timestamp": timestamp,
            "regimes": regime_params,
            "metadata": {
                "source": "workflow_optimization",
                "workflow": self.instance_name
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(formatted_params, f, indent=2)
            
        self.logger.info(f"Saved regime-specific parameters to {filename}")
        
        # Also save to a fixed filename for easy loading
        latest_filename = self.output_dir / "regime_optimized_parameters.json"
        with open(latest_filename, 'w') as f:
            json.dump(formatted_params, f, indent=2)
            
        self.logger.info(f"Updated latest regime parameters at {latest_filename}")
    
    def _save_workflow_results(self, results: Dict[str, Any]) -> None:
        """Save complete workflow results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create workflow hash for tracking
        workflow_str = json.dumps(self.workflow_steps, sort_keys=True)
        workflow_hash = hashlib.md5(workflow_str.encode()).hexdigest()[:8]
        
        filename = self.output_dir / f"workflow_{timestamp}_{workflow_hash}.json"
        
        workflow_data = {
            "timestamp": timestamp,
            "workflow_hash": workflow_hash,
            "workflow_steps": self.workflow_steps,
            "results": results
        }
        
        with open(filename, 'w') as f:
            json.dump(workflow_data, f, indent=2, default=str)
            
        logger.info(f"Saved complete workflow results to {filename}")
        
        # Also save regime parameters separately
        regime_params = self._collect_regime_parameters(results)
        if regime_params:
            self._save_regime_parameters(regime_params)