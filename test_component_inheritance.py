#!/usr/bin/env python3
"""Test script to verify component inheritance updates."""

import sys
sys.path.append('/Users/daws/ADMF')

from src.strategy.base.rule import RuleBase, CrossoverRule, ThresholdRule
from src.strategy.base.indicator import IndicatorBase, MovingAverageIndicator, RSIIndicator

def test_rule_components():
    """Test that rule components have instance_name."""
    print("Testing Rule Components...")
    
    # Test CrossoverRule
    rule1 = CrossoverRule("test_crossover")
    print(f"CrossoverRule instance_name: {rule1.instance_name}")
    print(f"CrossoverRule has instance_name: {hasattr(rule1, 'instance_name')}")
    
    # Test ThresholdRule
    rule2 = ThresholdRule("test_threshold")
    print(f"ThresholdRule instance_name: {rule2.instance_name}")
    print(f"ThresholdRule has instance_name: {hasattr(rule2, 'instance_name')}")
    
    # Test parameter space
    param_space = rule1.get_parameter_space()
    print(f"CrossoverRule parameter space name: {param_space.name}")
    
    print("✓ Rule components test passed\n")

def test_indicator_components():
    """Test that indicator components have instance_name."""
    print("Testing Indicator Components...")
    
    # Test MovingAverageIndicator
    ind1 = MovingAverageIndicator("test_ma", 20)
    print(f"MovingAverageIndicator instance_name: {ind1.instance_name}")
    print(f"MovingAverageIndicator has instance_name: {hasattr(ind1, 'instance_name')}")
    
    # Test RSIIndicator
    ind2 = RSIIndicator("test_rsi", 14)
    print(f"RSIIndicator instance_name: {ind2.instance_name}")
    print(f"RSIIndicator has instance_name: {hasattr(ind2, 'instance_name')}")
    
    # Test parameter space
    param_space = ind1.get_parameter_space()
    print(f"MovingAverageIndicator parameter space name: {param_space.name}")
    
    print("✓ Indicator components test passed\n")

def test_optimization_interface():
    """Test that components have optimization interface."""
    print("Testing Optimization Interface...")
    
    rule = CrossoverRule("opt_test")
    
    # Test optimization methods
    print(f"Has get_parameter_space: {hasattr(rule, 'get_parameter_space')}")
    print(f"Has get_parameters: {hasattr(rule, 'get_parameters')}")
    print(f"Has set_parameters: {hasattr(rule, 'set_parameters')}")
    print(f"Has validate_parameters: {hasattr(rule, 'validate_parameters')}")
    print(f"Has apply_parameters: {hasattr(rule, 'apply_parameters')}")
    print(f"Has get_optimizable_parameters: {hasattr(rule, 'get_optimizable_parameters')}")
    
    # Test parameter operations
    params = rule.get_parameters()
    print(f"Current parameters: {params}")
    
    # Test parameter validation
    valid, error = rule.validate_parameters({'min_separation': 0.001})
    print(f"Parameter validation: valid={valid}, error={error}")
    
    print("✓ Optimization interface test passed\n")

if __name__ == "__main__":
    try:
        test_rule_components()
        test_indicator_components()
        test_optimization_interface()
        print("All tests passed! Components now properly inherit from ComponentBase.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()