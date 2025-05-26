#!/usr/bin/env python3
"""Test if workflow orchestrator can be imported"""

import sys
sys.path.insert(0, '/Users/daws/ADMF')

try:
    from src.strategy.optimization.workflow_orchestrator import OptimizationWorkflowOrchestrator
    print("✓ Successfully imported OptimizationWorkflowOrchestrator")
    
    # Test instantiation
    orchestrator = OptimizationWorkflowOrchestrator("test_orchestrator")
    print("✓ Successfully created instance")
    
    # Check methods exist
    methods = ['_initialize', 'run_optimization_workflow', '_validate_workflow']
    for method in methods:
        if hasattr(orchestrator, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Method {method} missing")
            
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()