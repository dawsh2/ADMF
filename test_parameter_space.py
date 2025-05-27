#!/usr/bin/env python3
"""Quick test to verify parameter space generation."""

import sys
sys.path.append('/Users/daws/ADMF')

from src.strategy.base.parameter import Parameter, ParameterSpace

# Test parameter space generation
space = ParameterSpace("test_space")

# Add MA parameters
space.add_parameter(Parameter(
    name="short_window",
    param_type="discrete", 
    values=[5, 10, 15],
    default=10
))

space.add_parameter(Parameter(
    name="long_window",
    param_type="discrete",
    values=[20, 30, 40], 
    default=20
))

# Generate combinations
combinations = space.sample(method='grid')

print(f"Generated {len(combinations)} combinations:")
for i, combo in enumerate(combinations):
    print(f"  {i+1}: {combo}")

print(f"\nExpected: 3 x 3 = 9 combinations")
print(f"Actual: {len(combinations)} combinations")