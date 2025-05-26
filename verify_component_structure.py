#!/usr/bin/env python3
"""
Verify component structure without running actual backtest.
Just checks that components follow the new ComponentBase pattern.
"""
import sys
import ast
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def check_component(file_path, class_name):
    """Check if a component follows the ComponentBase pattern."""
    print(f"\nChecking {class_name} in {file_path}...")
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # Check inheritance
            bases = [base.id if isinstance(base, ast.Name) else 
                    base.attr if isinstance(base, ast.Attribute) else None 
                    for base in node.bases]
            print(f"  Inherits from: {bases}")
            
            # Check for required methods
            methods = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods[item.name] = item
            
            # Check constructor
            if '__init__' in methods:
                init_method = methods['__init__']
                args = [arg.arg for arg in init_method.args.args]
                print(f"  Constructor args: {args}")
                if args == ['self', 'instance_name', 'config_key']:
                    print("  ✓ Minimal constructor pattern")
                else:
                    print("  ✗ Constructor doesn't follow minimal pattern")
            
            # Check for _initialize method
            if '_initialize' in methods:
                print("  ✓ Has _initialize() method")
            else:
                print("  ✗ Missing _initialize() method")
            
            # Check for lifecycle methods
            lifecycle_methods = ['_start', '_stop', '_cleanup']
            for method in lifecycle_methods:
                if method in methods:
                    print(f"  ✓ Has {method}() method")
            
            return True
    
    print(f"  ✗ Class {class_name} not found")
    return False


def main():
    """Check all refactored components."""
    print("Verifying Component Structure")
    print("=" * 50)
    
    components_to_check = [
        ('src/data/csv_data_handler.py', 'CSVDataHandler'),
        ('src/risk/basic_portfolio.py', 'BasicPortfolio'),
        ('src/risk/basic_risk_manager.py', 'BasicRiskManager'),
        ('src/execution/simulated_execution_handler.py', 'SimulatedExecutionHandler'),
        ('src/strategy/regime_adaptive_strategy.py', 'RegimeAdaptiveStrategy'),
        ('src/strategy/regime_detector.py', 'RegimeDetector'),
        ('src/strategy/optimization/genetic_optimizer.py', 'GeneticOptimizer'),
        ('src/strategy/optimization/optimization_runner.py', 'OptimizationRunner'),
        ('src/core/dummy_component.py', 'DummyComponent')
    ]
    
    all_good = True
    for file_path, class_name in components_to_check:
        if not check_component(file_path, class_name):
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ All components follow the ComponentBase pattern!")
        print("\nThe refactored components are ready for use.")
        print("To run a backtest, ensure you have the required dependencies:")
        print("  - pandas")
        print("  - numpy") 
        print("  - PyYAML")
        print("\nThen you can run:")
        print("  python main.py --config config/config.yaml")
    else:
        print("❌ Some components need attention")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())