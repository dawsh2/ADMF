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
from src.core.event import EventType
from src.core.exceptions import ConfigurationError
from .component_optimizer import ComponentOptimizer

# Create custom log level for optimization progress
PROGRESS_LEVEL = 35  # Just above WARNING (30) but below ERROR (40)
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")

def progress(self, message, *args, **kwargs):
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)

# Add the progress method to Logger
logging.Logger.progress = progress

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
            self.logger.info(f"{self.instance_name}: No optimization workflow defined in config")
            
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
        
        # Ensure data handler has an active dataset
        if hasattr(data_handler, 'set_active_dataset'):
            # Start with train dataset for optimization
            data_handler.set_active_dataset('train')
            logger.info("Set data handler to TRAIN dataset for optimization workflow")
        
        # Reload workflow configuration if not loaded
        if not self.workflow_steps:
            self._load_workflow_config()
            logger.info(f"{self.instance_name}: Reloaded workflow configuration, found {len(self.workflow_steps)} steps")
        
        # Calculate total backtests upfront
        total_backtests, backtest_breakdown = self._calculate_total_backtests(strategy)
        logger.warning(f"\n{'='*80}")
        logger.warning(f"OPTIMIZATION WORKFLOW SUMMARY")
        logger.warning(f"{'='*80}")
        logger.warning(f"Total workflow steps: {len(self.workflow_steps)}")
        logger.warning(f"Total backtests to run: {total_backtests}")
        logger.warning(f"\nBreakdown by step:")
        for step_name, count in backtest_breakdown.items():
            logger.warning(f"  - {step_name}: {count} backtests")
        logger.warning(f"{'='*80}\n")
        
        workflow_results = {}
        completed_steps = set()
        self._current_backtest = 0
        self._total_backtests = total_backtests
        
        for step in self.workflow_steps:
            # Check dependencies
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep not in completed_steps:
                        logger.error(f"Cannot run step '{step['name']}' - dependency '{dep}' not completed")
                        continue
            
            # Calculate backtests for this step
            step_backtests = self._calculate_step_backtests(step, strategy)
            logger.warning(f"\n{'='*80}")
            logger.warning(f"STARTING: {step['name'].upper()}")
            logger.warning(f"Progress: {self._current_backtest + 1}-{self._current_backtest + step_backtests} of {self._total_backtests} total backtests")
            logger.warning(f"{'='*80}")
            
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
                
                # Update progress counter
                self._current_backtest += step_backtests
                
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
        
        # Enable regime switching for test evaluation
        if hasattr(strategy, '_enable_regime_switching'):
            strategy._enable_regime_switching = True
            self.logger.info("ENABLED regime switching for test evaluation")
            
        # Apply regime-specific parameters directly to strategy
        regime_params = self._collect_regime_parameters(workflow_results)
        if regime_params and hasattr(strategy, '_regime_specific_params'):
            strategy._regime_specific_params = regime_params
            self.logger.info(f"Applied regime-specific parameters for {len(regime_params)} regimes directly to strategy")
            for regime, params in regime_params.items():
                weight_params = {k: v for k, v in params.items() if 'weight' in k}
                if weight_params:
                    self.logger.info(f"  {regime}: {weight_params}")
            
        # Force strategy to subscribe to classification events for test phase
        if hasattr(strategy, 'event_bus') and strategy.event_bus:
            try:
                # First try to unsubscribe in case already subscribed
                strategy.event_bus.unsubscribe(EventType.CLASSIFICATION, strategy._on_classification)
            except:
                pass
            
            # Now subscribe
            strategy.event_bus.subscribe(EventType.CLASSIFICATION, strategy._on_classification)
            self.logger.info("Strategy manually subscribed to CLASSIFICATION events for test evaluation")
        
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
                    
                    # Display comprehensive test results
                    self._display_test_results(test_results)
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
        
        # Display final optimization summary
        self._display_final_summary(workflow_results)
        
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
        
        # Set global progress tracking
        if hasattr(self, '_current_backtest') and hasattr(self, '_total_backtests'):
            component_optimizer._global_progress = {
                'current': self._current_backtest,
                'total': self._total_backtests
            }
        
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
                    # Check rules first since we usually want to optimize rules not indicators
                    component = None
                    if hasattr(strategy, '_rules') and pattern in strategy._rules:
                        component = strategy._rules[pattern]
                    elif hasattr(strategy, '_indicators') and pattern in strategy._indicators:
                        component = strategy._indicators[pattern]
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
            self.logger.info(f"\nOptimizing component: {component.instance_name}")
            
            # Log parameter space info for BB components
            if 'bb' in component.instance_name.lower() or 'bollinger' in component.instance_name.lower():
                if hasattr(component, 'get_parameter_space'):
                    param_space = component.get_parameter_space()
                    self.logger.warning(f"[BB Debug] Component {component.instance_name} parameter space:")
                    if hasattr(param_space, 'to_dict'):
                        space_dict = param_space.to_dict()
                        self.logger.warning(f"  Parameters: {space_dict.get('parameters', {})}")
                        self.logger.warning(f"  Subspaces: {list(space_dict.get('subspaces', {}).keys())}")
            
            # STEP 1: Optimize on TRAIN dataset
            self.logger.debug(f"===== OPTIMIZING {component.instance_name} ON TRAIN DATASET =====")
            
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
            
            # Component optimizer will have updated its own progress counter internally
            
            # NO TEST EVALUATION HERE - only training optimization
            # Test evaluation happens once at the very end with the complete ensemble
            
            # Show brief results summary
            if 'best_parameters' in comp_results:
                best_params = comp_results['best_parameters']
                best_score = comp_results.get('best_score', 0)
                combinations = comp_results.get('combinations_tested', 0)
                param_str = ", ".join([f"{k}={v}" for k, v in sorted(best_params.items())])
                self.logger.warning(f"\n✓ {component.instance_name} optimization complete: Best score={best_score:.4f}")
                self.logger.warning(f"  Best params: {param_str}\n")
            
            # Check if we have regime-specific results
            if 'regime_best_parameters' in comp_results:
                self.logger.debug(f"Regime-specific best parameters found for {component.instance_name}")
                for regime, regime_data in comp_results.get('regime_best_parameters', {}).items():
                    self.logger.debug(f"  {regime}: params={regime_data.get('parameters', {})}, "
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
        """Run ensemble weight optimization using grid search."""
        # Check if we should do per-regime optimization or overall
        per_regime = step_config.get('per_regime', True)
        
        if per_regime:
            self.logger.info("Running regime-specific ensemble weight optimization")
            return self._run_regime_specific_weight_optimization(
                step_config, data_handler, portfolio_manager, strategy,
                risk_manager, execution_handler, train_dates, test_dates, previous_results
            )
        else:
            self.logger.info("Running overall ensemble weight optimization (single set of weights)")
            return self._run_overall_weight_optimization(
                step_config, data_handler, portfolio_manager, strategy,
                risk_manager, execution_handler, train_dates, test_dates, previous_results
            )
    
    def _run_overall_weight_optimization(self, step_config: Dict[str, Any],
                                       data_handler, portfolio_manager, strategy,
                                       risk_manager, execution_handler,
                                       train_dates: Tuple[str, str],
                                       test_dates: Tuple[str, str],
                                       previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run weight optimization to find a single set of weights for all regimes."""
        self.logger.progress("Starting overall weight optimization for ensemble strategy")
        
        # CRITICAL: Apply regime-specific parameters from previous optimization steps
        # This ensures the strategy will switch parameters dynamically during weight optimization
        regime_params = self._collect_regime_parameters(previous_results)
        if regime_params and hasattr(strategy, '_regime_specific_params'):
            strategy._regime_specific_params = regime_params
            self.logger.info(f"Applied regime-specific parameters for {len(regime_params)} regimes to strategy")
            self.logger.info("Strategy will dynamically switch parameters as regimes change during weight optimization")
        
        # Enable regime switching for this optimization phase
        original_regime_switching = None
        if hasattr(strategy, '_enable_regime_switching'):
            original_regime_switching = strategy._enable_regime_switching
            strategy._enable_regime_switching = True
            self.logger.info("ENABLED regime switching for overall weight optimization")
        
        # Get weight combinations from config
        weight_config = step_config.get('weight_combinations', step_config.get('config', {}).get('weight_combinations', []))
        
        if weight_config:
            # Parse weight combinations from config (handles list/dict formats)
            rule_names = list(strategy._rules.keys())
            weight_combinations = self._parse_weight_combinations(weight_config, rule_names)
            self.logger.progress(f"Found {len(weight_combinations)} weight combinations to test")
        else:
            # Generate default combinations if none provided
            weight_combinations = self._generate_weight_combinations(strategy)
            self.logger.progress(f"Generated {len(weight_combinations)} weight combinations to test")
        
        # Ensure we're on train dataset
        if hasattr(data_handler, 'set_active_dataset'):
            data_handler.set_active_dataset('train')
            
        best_score = float('-inf')
        best_weights = None
        all_results = []
        
        # Test each weight combination on the full training set
        for i, weights in enumerate(weight_combinations):
            # Format weights nicely
            weight_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(weights.items())])
            
            # Apply weights to strategy
            for rule_name, weight in weights.items():
                if rule_name in strategy._component_weights:
                    strategy._component_weights[rule_name] = weight
                    
            # Normalize weights after update
            if hasattr(strategy, '_normalize_weights'):
                strategy._normalize_weights()
            
            # Reset portfolio for clean evaluation
            portfolio_manager.reset()
            
            # Always create a fresh backtest runner for each weight combination
            # This ensures no state carryover between tests
            from src.execution.backtest_runner import BacktestRunner
            backtest_runner = BacktestRunner(instance_name=f"weight_opt_{i}")
            
            # Create proper context
            if isinstance(self._context, dict):
                opt_context = self._context.copy()
            else:
                opt_context = {
                    'config_loader': getattr(self._context, 'config_loader', None),
                    'config': getattr(self._context, 'config', None),
                    'event_bus': getattr(self._context, 'event_bus', None),
                    'container': getattr(self._context, 'container', None),
                    'logger': getattr(self._context, 'logger', self.logger),
                    'metadata': getattr(self._context, 'metadata', {})
                }
            
            # Ensure using train dataset
            if 'metadata' not in opt_context:
                opt_context['metadata'] = {}
            if 'cli_args' not in opt_context['metadata']:
                opt_context['metadata']['cli_args'] = {}
            opt_context['metadata']['cli_args']['dataset'] = 'train'
            
            backtest_runner.initialize(opt_context)
            if not backtest_runner.running:
                backtest_runner.start()
            
            try:
                test_results = backtest_runner.execute()
                metrics = test_results.get('performance_metrics', {})
                score = metrics.get('portfolio_sharpe_ratio', float('-inf'))
                
                result_entry = {
                    'weights': weights.copy(),
                    'score': score,
                    'trade_count': metrics.get('num_trades', 0),
                    'metrics': metrics
                }
                all_results.append(result_entry)
                
                # Track best and log progress
                is_new_best = False
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    is_new_best = True
                
                # Show the test and result on one line with global progress
                global_progress = ""
                if hasattr(self, '_current_backtest') and hasattr(self, '_total_backtests'):
                    current = self._current_backtest + i + 1
                    global_progress = f"[{current}/{self._total_backtests}] "
                    
                result_str = f"{global_progress}Testing {i+1}/{len(weight_combinations)}: {weight_str} → Score: {score:.4f}"
                if is_new_best:
                    result_str += " ⭐ NEW BEST!"
                self.logger.progress(result_str)
                    
            except Exception as e:
                global_progress = ""
                if hasattr(self, '_current_backtest') and hasattr(self, '_total_backtests'):
                    current = self._current_backtest + i + 1
                    global_progress = f"[{current}/{self._total_backtests}] "
                    
                error_str = f"{global_progress}Testing {i+1}/{len(weight_combinations)}: {weight_str} → Error: {str(e)}"
                self.logger.progress(error_str)
            finally:
                # Clean up backtest runner
                if backtest_runner and backtest_runner.running:
                    backtest_runner.stop()
                backtest_runner.teardown()
        
        # Restore original regime switching state
        if original_regime_switching is not None and hasattr(strategy, '_enable_regime_switching'):
            strategy._enable_regime_switching = original_regime_switching
            self.logger.info(f"Restored regime switching to: {original_regime_switching}")
        
        # Log completion
        if best_weights:
            weight_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(best_weights.items())])
            self.logger.progress(f"✓ Overall weight optimization complete: Best score={best_score:.4f}, Weights: {weight_str}")
        
        # Return results
        return {
            'optimization_type': 'ensemble_weights',
            'method': 'overall_grid_search',
            'best_weights': best_weights,
            'best_score': best_score,
            'all_results': all_results,
            'combinations_tested': len(weight_combinations),
            'timestamp': datetime.now().isoformat()
        }
        
    def _run_regime_specific_weight_optimization(self, step_config: Dict[str, Any],
                                               data_handler, portfolio_manager, strategy,
                                               risk_manager, execution_handler,
                                               train_dates: Tuple[str, str],
                                               test_dates: Tuple[str, str],
                                               previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Original per-regime weight optimization method."""
        
        # Get rule names from strategy
        if not hasattr(strategy, '_rules') or not strategy._rules:
            self.logger.warning("No rules found in strategy to optimize weights")
            return {
                "status": "no_rules",
                "message": "Strategy has no rules for weight optimization"
            }
        
        rule_names = list(strategy._rules.keys())
        self.logger.info(f"Found {len(rule_names)} rules for weight optimization: {rule_names}")
        
        # Get weight combinations from config or generate them
        weight_config = step_config.get('weight_combinations', step_config.get('config', {}).get('weight_combinations', []))
        
        if weight_config:
            # Parse weight combinations from config (handles list/dict formats)
            weight_combinations = self._parse_weight_combinations(weight_config, rule_names)
            self.logger.info(f"Parsed {len(weight_combinations)} weight combinations from config")
        else:
            # Use the same weight generation logic as overall optimization
            weight_combinations = self._generate_weight_combinations(strategy)
            self.logger.info(f"Generated {len(weight_combinations)} weight combinations for {len(rule_names)} rules")
        
        # Collect regime-specific optimized parameters from previous steps
        regime_params = self._collect_regime_parameters(previous_results)
        if not regime_params:
            self.logger.warning("No regime-specific parameters found from previous optimization steps")
            return {
                "status": "no_regime_params",
                "message": "Weight optimization requires regime-specific parameters from previous steps"
            }
        
        # Get list of regimes to optimize
        regimes_to_optimize = list(regime_params.keys())
        self.logger.progress(f"Starting regime-specific weight optimization for {len(regimes_to_optimize)} regimes: {regimes_to_optimize}")
        
        # Ensure we're on train dataset
        if hasattr(data_handler, 'set_active_dataset'):
            data_handler.set_active_dataset('train')
            self.logger.info("Set data handler to TRAIN dataset for weight optimization")
        
        # Store results for each regime
        regime_weight_results = {}
        
        # For each regime, find optimal weights
        for regime in regimes_to_optimize:
            self.logger.progress(f"\nOptimizing weights for regime: {regime.upper()}")
            
            # Apply the regime-specific parameters
            regime_specific_params = regime_params[regime]
            self.logger.debug(f"Applying regime-specific parameters: {regime_specific_params}")
            
            # Display the optimal parameters for each rule in this regime
            self.logger.info(f"Optimal parameters for {regime.upper()} regime:")
            for rule_name in ['ma_crossover', 'rsi', 'bb', 'macd']:
                rule_params = {}
                for param_key, param_value in regime_specific_params.items():
                    if param_key.startswith(f"{rule_name}.") or param_key.startswith(f"{rule_name}_"):
                        clean_key = param_key.replace(f"{rule_name}.", "").replace(f"{rule_name}_", "")
                        rule_params[clean_key] = param_value
                if rule_params:
                    param_str = ", ".join([f"{k}={v}" for k, v in sorted(rule_params.items())])
                    self.logger.info(f"  {rule_name}: {param_str}")
            
            # Apply parameters to strategy (but not weights yet)
            param_only = {k: v for k, v in regime_specific_params.items() if 'weight' not in k}
            if hasattr(strategy, '_apply_regime_parameters'):
                strategy._apply_regime_parameters(param_only)
            
            best_score = float('-inf')
            best_weights = None
            all_results = []
            
            # Test each weight combination for this regime
            for i, weights in enumerate(weight_combinations):
                # Format weights nicely
                weight_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(weights.items())])
                
                # Apply weights to strategy rules
                for rule_name, weight in weights.items():
                    if rule_name in strategy._component_weights:
                        strategy._component_weights[rule_name] = weight
                        
                # Normalize weights after update
                if hasattr(strategy, '_normalize_weights'):
                    strategy._normalize_weights()
                
                # Reset portfolio for clean evaluation
                portfolio_manager.reset()
                
                # Always create a fresh backtest runner for each weight combination
                # This ensures no state carryover between tests
                from src.execution.backtest_runner import BacktestRunner
                backtest_runner = BacktestRunner(instance_name=f"weight_opt_runner_{regime}_{i}")
                
                # Create a proper context for the backtest runner
                if isinstance(self._context, dict):
                    opt_context = self._context.copy()
                else:
                    # Create context dict from object
                    opt_context = {
                        'config_loader': getattr(self._context, 'config_loader', None),
                        'config': getattr(self._context, 'config', None),
                        'event_bus': getattr(self._context, 'event_bus', None),
                        'container': getattr(self._context, 'container', None),
                        'logger': getattr(self._context, 'logger', self.logger),
                        'metadata': getattr(self._context, 'metadata', {})
                    }
                
                # Ensure the backtest runner uses train dataset
                if 'metadata' not in opt_context:
                    opt_context['metadata'] = {}
                if 'cli_args' not in opt_context['metadata']:
                    opt_context['metadata']['cli_args'] = {}
                opt_context['metadata']['cli_args']['dataset'] = 'train'
                
                # Initialize with context
                backtest_runner.initialize(opt_context)
                
                # Start the backtest runner
                if not backtest_runner.running:
                    backtest_runner.start()
                
                # Get data handler and filter to regime-specific data
                if hasattr(data_handler, 'get_regime_filtered_data'):
                    # If data handler supports regime filtering
                    original_data = data_handler.active_df
                    regime_data = data_handler.get_regime_filtered_data(regime)
                    data_handler.active_df = regime_data
                    self.logger.info(f"Filtered data to {len(regime_data)} bars for regime {regime}")
                
                # Run backtest with current weights
                try:
                    test_results = backtest_runner.execute()
                    metrics = test_results.get('performance_metrics', {})
                    
                    # For regime-specific optimization, we look at the regime-specific performance
                    regime_perf = metrics.get('regime_performance', {}).get(regime, {})
                    score = regime_perf.get('sharpe_ratio', metrics.get('portfolio_sharpe_ratio', float('-inf')))
                    trade_count = regime_perf.get('count', metrics.get('num_trades', 0))
                    
                    # Only consider if we have enough trades
                    min_trades = 5
                    if trade_count < min_trades:
                        self.logger.warning(f"Only {trade_count} trades for {regime} with weights {weights}, skipping (min: {min_trades})")
                        score = float('-inf')
                    
                    result_entry = {
                        'weights': weights.copy(),
                        'score': score,
                        'trade_count': trade_count,
                        'metrics': regime_perf if regime_perf else metrics
                    }
                    all_results.append(result_entry)
                    
                    # Track best for this regime
                    is_new_best = False
                    if score > best_score and trade_count >= min_trades:
                        best_score = score
                        best_weights = weights.copy()
                        is_new_best = True
                    
                    # Show the test and result on one line with global progress
                    global_progress = ""
                    if hasattr(self, '_current_backtest') and hasattr(self, '_total_backtests'):
                        # Calculate current position across all regimes
                        regime_idx = regimes_to_optimize.index(regime)
                        current = self._current_backtest + (regime_idx * len(weight_combinations)) + i + 1
                        global_progress = f"[{current}/{self._total_backtests}] "
                        
                    result_str = f"{global_progress}Testing {i+1}/{len(weight_combinations)} for {regime}: {weight_str} → Score: {score:.4f}, Trades: {trade_count}"
                    if trade_count < min_trades:
                        result_str += " (insufficient trades)"
                    elif is_new_best:
                        result_str += " ⭐ NEW BEST!"
                    self.logger.progress(result_str)
                        
                except Exception as e:
                    global_progress = ""
                    if hasattr(self, '_current_backtest') and hasattr(self, '_total_backtests'):
                        regime_idx = regimes_to_optimize.index(regime)
                        current = self._current_backtest + (regime_idx * len(weight_combinations)) + i + 1
                        global_progress = f"[{current}/{self._total_backtests}] "
                        
                    error_str = f"{global_progress}Testing {i+1}/{len(weight_combinations)} for {regime}: {weight_str} → Error: {str(e)}"
                    self.logger.progress(error_str)
                    continue
                finally:
                    # Clean up backtest runner if we created one
                    if backtest_runner and hasattr(backtest_runner, 'instance_name') and 'weight_opt_runner' in backtest_runner.instance_name:
                        try:
                            if backtest_runner.running:
                                backtest_runner.stop()
                            backtest_runner.teardown()
                        except:
                            pass
                    
                    # Restore original data if we filtered it
                    if hasattr(data_handler, 'get_regime_filtered_data'):
                        data_handler.active_df = original_data
            
            # Store results for this regime
            regime_weight_results[regime] = {
                'best_weights': best_weights,
                'best_score': best_score,
                'all_results': all_results,
                'combinations_tested': len(weight_combinations)
            }
            
            if best_weights:
                weight_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(best_weights.items())])
                self.logger.progress(f"✓ {regime.upper()} optimization complete: Best score={best_score:.4f}, Weights: {weight_str}")
            else:
                self.logger.warning(f"No valid weights found for {regime} (insufficient trades)")
        
        # Prepare final results
        results = {
            'optimization_type': 'ensemble_weights',
            'method': 'regime_specific_grid_search',
            'regimes_optimized': len(regimes_to_optimize),
            'regime_weight_results': regime_weight_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert to format expected by other methods
        results['regime_best_weights'] = {}
        for regime, data in regime_weight_results.items():
            if data['best_weights']:
                results['regime_best_weights'][regime] = {
                    'weights': data['best_weights'],
                    'sharpe_ratio': data['best_score'],
                    'metrics': {'sharpe_ratio': data['best_score']}
                }
        
        # Display weight optimization summary
        self.logger.info("\n===== REGIME-SPECIFIC WEIGHT OPTIMIZATION SUMMARY =====")
        for regime, data in regime_weight_results.items():
            self.logger.info(f"\n{regime.upper()}:")
            if data['best_weights']:
                self.logger.info(f"  Best weights: {data['best_weights']}")
                self.logger.info(f"  Best Sharpe: {data['best_score']:.4f}")
            else:
                self.logger.info(f"  No valid weights found (insufficient trades)")
        
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
                                    
        # Collect from weight optimization
        for step_name, results in workflow_results.items():
            if results.get('optimization_type') == 'ensemble_weights':
                regime_best_weights = results.get('regime_best_weights', {})
                if regime_best_weights:
                    for regime, weight_data in regime_best_weights.items():
                        if regime not in regime_params:
                            regime_params[regime] = {}
                        # Add weights with proper namespacing
                        weights = weight_data.get('weights', {})
                        for rule_name, weight in weights.items():
                            regime_params[regime][f"{rule_name}.weight"] = weight
                            
                # Also use overall best weights as fallback for regimes without specific weights
                best_weights = results.get('best_weights', {})
                if best_weights:
                    # Add to regimes that don't have weights yet
                    for regime in ['default', 'trending_up', 'trending_down', 'volatile']:
                        if regime in regime_params and not any('weight' in k for k in regime_params[regime].keys()):
                            for rule_name, weight in best_weights.items():
                                regime_params[regime][f"{rule_name}.weight"] = weight
        
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
    
    def _display_test_results(self, test_results: Dict[str, Any]) -> None:
        """Display comprehensive test set evaluation results."""
        self.logger.info("\n" + "="*80)
        self.logger.info("===== FINAL TEST SET RESULTS =====")
        self.logger.info("="*80)
        
        # Basic metrics
        metrics = test_results.get('performance_metrics', {})
        self.logger.info("\n📊 OVERALL PERFORMANCE:")
        self.logger.info(f"  Initial Portfolio Value: ${metrics.get('initial_value', 100000):,.2f}")
        self.logger.info(f"  Final Portfolio Value: ${metrics.get('final_value', 0):,.2f}")
        self.logger.info(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        self.logger.info(f"  Sharpe Ratio: {metrics.get('portfolio_sharpe_ratio', 0):.2f}")
        self.logger.info(f"  Number of Trades: {metrics.get('num_trades', 0)}")
        self.logger.info(f"  Realized PnL: ${metrics.get('realized_pnl', 0):,.2f}")
        
        # Regime-specific performance
        regime_perf = metrics.get('regime_performance', {})
        if regime_perf:
            self.logger.info("\n📈 REGIME-SPECIFIC PERFORMANCE:")
            
            # Sort regimes for consistent display
            regime_order = ['default', 'trending_up', 'trending_down', 'volatile']
            sorted_regimes = sorted(regime_perf.keys(), 
                                  key=lambda x: regime_order.index(x) if x in regime_order else 999)
            
            for regime in sorted_regimes:
                if regime == '_boundary_trades_summary':
                    continue
                    
                perf = regime_perf[regime]
                self.logger.info(f"\n  {regime.upper()}:")
                self.logger.info(f"    PnL: ${perf.get('pnl', 0) or 0:,.2f}")
                self.logger.info(f"    Trades: {perf.get('count', 0) or 0}")
                # Handle potential None values in win_rate
                win_rate = perf.get('win_rate', 0)
                if win_rate is None:
                    win_rate = 0
                self.logger.info(f"    Win Rate: {win_rate:.1%}")
                self.logger.info(f"    Sharpe Ratio: {perf.get('sharpe_ratio', 0) or 0:.2f}")
                self.logger.info(f"    Avg PnL/Trade: ${perf.get('avg_pnl', 0) or 0:,.2f}")
                
                # Check for boundary trades
                boundary_count = perf.get('boundary_trade_count', 0)
                if boundary_count > 0:
                    self.logger.info(f"    ⚠️  Boundary Trades: {boundary_count}")
                    self.logger.info(f"    Boundary PnL: ${perf.get('boundary_trades_pnl', 0):,.2f}")
        
        # Summary comparison with training
        self.logger.info("\n" + "-"*80)
        self.logger.info("💡 TEST EVALUATION COMPLETE")
        self.logger.info("   - Parameters dynamically switched based on regime")
        self.logger.info("   - No data leakage: test set was held out during optimization")
        self.logger.info("   - Results reflect true out-of-sample performance")
        
        self.logger.info("\n" + "="*80 + "\n")
    
    def _display_final_summary(self, workflow_results: Dict[str, Any]) -> None:
        """Display final optimization completion summary."""
        # Count optimizations
        total_backtests = 0
        
        for step_name, results in workflow_results.items():
            if results.get('optimization_type') == 'component':
                component_results = results.get('results', {})
                for comp_name, comp_data in component_results.items():
                    total_backtests += comp_data.get('combinations_tested', 0)
        
        # Test results summary
        test_eval = workflow_results.get('final_test_evaluation', {})
        if test_eval and 'results' in test_eval:
            test_metrics = test_eval['results'].get('performance_metrics', {})
            return_pct = test_metrics.get('total_return_pct', 0)
            sharpe = test_metrics.get('portfolio_sharpe_ratio', 0)
            trades = test_metrics.get('num_trades', 0)
            
            # One line summary
            self.logger.warning(f"\n✅ Optimization complete: {total_backtests} backtests | "
                              f"Test: {return_pct:.2f}% return, {sharpe:.2f} Sharpe, {trades} trades")
    
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
    
    
    def _parse_weight_combinations(self, weight_config: List, rule_names: List[str]) -> List[Dict[str, float]]:
        """Parse weight combinations from config, handling both list and dict formats."""
        combinations = []
        
        for config_item in weight_config:
            if isinstance(config_item, dict):
                # Already in dict format
                combinations.append(config_item)
            elif isinstance(config_item, list):
                # Convert list to dict using rule names
                if len(config_item) == len(rule_names):
                    weight_dict = dict(zip(rule_names, config_item))
                    combinations.append(weight_dict)
                else:
                    self.logger.warning(f"Weight list length {len(config_item)} doesn't match "
                                      f"number of rules {len(rule_names)}, skipping")
            else:
                self.logger.warning(f"Unknown weight config format: {type(config_item)}")
        
        return combinations
    
    def _generate_weight_combinations(self, strategy) -> List[Dict[str, float]]:
        """Generate weight combinations for rules."""
        rule_names = list(strategy._rules.keys())
        n_rules = len(rule_names)
        
        if n_rules == 0:
            return []
        elif n_rules == 1:
            return [{rule_names[0]: 1.0}]
        elif n_rules == 2:
            # For 2 rules, use 0.1 increments
            combinations = []
            for w1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                w2 = 1.0 - w1
                combinations.append({rule_names[0]: w1, rule_names[1]: w2})
            return combinations
        elif n_rules == 3:
            # For 3 rules, use 0.2 increments to keep it manageable
            combinations = []
            for w1 in [0.2, 0.4, 0.6]:
                for w2 in [0.2, 0.4, 0.6]:
                    w3 = 1.0 - w1 - w2
                    if w3 > 0 and abs(w3) > 0.05:  # Ensure valid weight
                        combinations.append({
                            rule_names[0]: w1,
                            rule_names[1]: w2,
                            rule_names[2]: w3
                        })
            return combinations
        else:
            # For 4+ rules, use a smaller set of predefined combinations
            combinations = []
            
            # Equal weights
            equal_weight = 1.0 / n_rules
            combinations.append({rule: equal_weight for rule in rule_names})
            
            # Favor first rule
            weights = [0.4] + [0.6 / (n_rules - 1)] * (n_rules - 1)
            combinations.append(dict(zip(rule_names, weights)))
            
            # Favor last rule
            weights = [0.6 / (n_rules - 1)] * (n_rules - 1) + [0.4]
            combinations.append(dict(zip(rule_names, weights)))
            
            # Favor middle rules
            if n_rules == 4:
                combinations.extend([
                    {rule_names[0]: 0.3, rule_names[1]: 0.3, rule_names[2]: 0.2, rule_names[3]: 0.2},
                    {rule_names[0]: 0.2, rule_names[1]: 0.3, rule_names[2]: 0.3, rule_names[3]: 0.2},
                    {rule_names[0]: 0.1, rule_names[1]: 0.3, rule_names[2]: 0.3, rule_names[3]: 0.3},
                    {rule_names[0]: 0.4, rule_names[1]: 0.2, rule_names[2]: 0.2, rule_names[3]: 0.2},
                ])
                
            self.logger.info(f"Generated {len(combinations)} weight combinations for {n_rules} rules")
            return combinations
            
    def _calculate_total_backtests(self, strategy) -> Tuple[int, Dict[str, int]]:
        """Calculate total number of backtests that will be run."""
        total = 0
        breakdown = {}
        
        for step in self.workflow_steps:
            step_count = self._calculate_step_backtests(step, strategy)
            breakdown[step['name']] = step_count
            total += step_count
            
        return total, breakdown
        
    def _calculate_step_backtests(self, step: Dict[str, Any], strategy) -> int:
        """Calculate number of backtests for a single step."""
        if step["type"] == "rulewise":
            # Get the rule and its parameter space
            rule_name = step["targets"][0] if step["targets"] else None
            if not rule_name:
                return 0
                
            # Get rule from strategy
            rule = getattr(strategy, f"_{rule_name}_rule", None)
            if not rule or not hasattr(rule, 'parameter_space'):
                return 0
                
            # Calculate combinations
            param_space = rule.parameter_space
            combinations = 1
            for param, values in param_space.items():
                combinations *= len(values)
            return combinations
            
        elif step["type"] == "ensemble_weights":
            # Count weight combinations
            if "weight_combinations" in step:
                return len(step["weight_combinations"])
            else:
                # Use default generation logic
                n_rules = len(getattr(strategy, '_rules', {}))
                if n_rules <= 2:
                    return 9  # 0.1 increments
                elif n_rules == 3:
                    return 6  # 0.2 increments
                else:
                    return 7  # Predefined set
                    
        elif step["type"] == "regime":
            # Regime optimization runs all parameter combinations for each regime
            # This is complex - for now return an estimate
            if hasattr(self._context, 'config_loader') and self._context.config_loader:
                regimes = self._context.config_loader.get('optimization.regimes', ['default'])
            else:
                regimes = ['default']
            # Assume ~20 parameter combinations per regime as estimate
            return len(regimes) * 20
            
        return 0