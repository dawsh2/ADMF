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
            opt_config = config.get("optimization", {})
            self.workflow_steps = opt_config.get("workflow", [])
        else:
            # Try getting from component config
            self.workflow_steps = self.component_config.get("workflow", []) if self.component_config else []
        
        if not self.workflow_steps:
            self.logger.warning(f"{self.instance_name}: No optimization workflow defined in config")
            
        # Validate workflow
        self._validate_workflow()
        
        self.logger.info(f"{self.instance_name}: Initialized with {len(self.workflow_steps)} workflow steps")
        
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
                elif step["type"] == "regime_optimization":
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
        
        # Component optimization will be implemented
        self.logger.info(f"Component-based optimization for targets {targets} not yet fully implemented")
        
        # For now, use the existing optimization infrastructure
        container = self._context.container if hasattr(self._context, 'container') else None
        if not container:
            return {"error": "Container not available"}
            
        # Get the standard optimizer and run it
        optimizer = container.get("optimizer")
        if optimizer and hasattr(optimizer, 'execute'):
            self.logger.info("Delegating to standard optimizer for now")
            return optimizer.execute()
        
        return {
            "status": "pending_implementation",
            "targets": targets,
            "method": method,
            "message": "Component optimization will be implemented"
        }
    
    def _run_ensemble_weight_optimization(self, step_config: Dict[str, Any],
                                        data_handler, portfolio_manager, strategy,
                                        risk_manager, execution_handler,
                                        train_dates: Tuple[str, str],
                                        test_dates: Tuple[str, str],
                                        previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble weight optimization."""
        logger.info("Running ensemble weight optimization")
        
        # Use existing optimizer for ensemble weights
        container = self._context.container if hasattr(self._context, 'container') else None
        if not container:
            return {"error": "Container not available"}
            
        # Get existing optimizer
        optimizer = container.get("optimizer")
        if not optimizer:
            self.logger.warning("No optimizer component found")
        
        # For now, return placeholder until we implement ensemble optimization
        self.logger.info("Ensemble weight optimization not yet fully implemented")
        return {
            "status": "pending_implementation",
            "message": "Ensemble weight optimization will be implemented"
        }
    
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