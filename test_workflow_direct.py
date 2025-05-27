#!/usr/bin/env python3
"""
Direct test of workflow orchestrator to verify isolated optimization works.
"""

import sys
sys.path.append('/Users/daws/ADMF')

import logging
from src.core.bootstrap import Bootstrap
from src.strategy.optimization.workflow_orchestrator import OptimizationWorkflowOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_workflow_orchestrator():
    """Test the workflow orchestrator directly."""
    
    print("=" * 70)
    print("Direct Workflow Orchestrator Test")
    print("=" * 70)
    
    # Initialize bootstrap and config
    config_path = "config/test_ensemble_optimization.yaml"
    
    from src.core.config import SimpleConfigLoader
    from src.core.container import Container
    from src.core.bootstrap import Bootstrap, RunMode
    
    # Load configuration
    config_loader = SimpleConfigLoader(config_path)
    
    # Create bootstrap
    bootstrap = Bootstrap()
    
    # Initialize with config
    container = Container()
    context = bootstrap.initialize(
        config=config_loader,
        run_mode=RunMode.OPTIMIZATION,
        container=container
    )
    
    # Setup managed components
    bootstrap.setup_managed_components()
    
    # Get required components
    # Container is the one we created and passed to bootstrap
    
    # Get or create workflow orchestrator
    orchestrator = container.resolve('workflow_orchestrator')
    if not orchestrator:
        print("Creating workflow orchestrator manually...")
        orchestrator = OptimizationWorkflowOrchestrator(
            instance_name="manual_orchestrator"
        )
        orchestrator.initialize(context)
        
    # Manually load workflow from config if needed
    if not orchestrator.workflow_steps:
        print("\nManually loading workflow from config...")
        opt_config = config_loader.get("optimization", {})
        workflow_steps = opt_config.get("workflow", [])
        if workflow_steps:
            orchestrator.workflow_steps = workflow_steps
            print(f"Loaded {len(workflow_steps)} workflow steps manually")
    
    # Check if workflow loaded
    print(f"\nWorkflow steps loaded: {len(orchestrator.workflow_steps)}")
    for i, step in enumerate(orchestrator.workflow_steps):
        print(f"  Step {i+1}: {step['name']} (type: {step['type']}, isolate: {step.get('isolate', False)})")
    
    if not orchestrator.workflow_steps:
        print("\nNo workflow steps found! Check config.")
        return False
    
    # Get required components for optimization
    data_handler = container.resolve('data_handler')
    portfolio = container.resolve('portfolio_manager')
    strategy = container.resolve('strategy')
    risk_manager = container.resolve('risk_manager')
    execution_handler = container.resolve('execution_handler')
    
    # Run workflow
    print("\nRunning optimization workflow...")
    try:
        results = orchestrator.run_optimization_workflow(
            data_handler=data_handler,
            portfolio_manager=portfolio,
            strategy=strategy,
            risk_manager=risk_manager,
            execution_handler=execution_handler,
            train_dates=("2024-01-01", "2024-06-30"),
            test_dates=("2024-07-01", "2024-12-31")
        )
        
        print("\nWorkflow completed!")
        print(f"Results: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        bootstrap.cleanup()

if __name__ == "__main__":
    success = test_workflow_orchestrator()
    sys.exit(0 if success else 1)