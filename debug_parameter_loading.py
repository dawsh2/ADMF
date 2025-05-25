#!/usr/bin/env python3
"""
Debug why parameters might be loaded differently.
"""

import json
import sys
sys.path.append('.')

print("DEBUGGING PARAMETER LOADING")
print("="*60)

# Load the regime parameters
with open('regime_optimized_parameters.json', 'r') as f:
    params = json.load(f)

print("\nRegime parameters from JSON:")
for regime, regime_params in params.items():
    print(f"\n{regime}:")
    for k, v in regime_params.items():
        print(f"  {k}: {v}")

print("\n" + "="*60)
print("KEY OBSERVATIONS:")
print("="*60)

# Check for differences between regimes
all_params_same = True
first_params = None
for regime, regime_params in params.items():
    if first_params is None:
        first_params = regime_params
    elif regime_params != first_params:
        all_params_same = False
        break

if all_params_same:
    print("⚠️  ALL REGIMES HAVE IDENTICAL PARAMETERS!")
    print("This means regime-adaptive behavior is effectively disabled.")
    print("The strategy will behave the same regardless of regime.")
else:
    print("✓ Regimes have different parameters (adaptive behavior enabled)")

print("\nPOSSIBLE ISSUE:")
print("If all regimes have the same parameters, then:")
print("- Optimizer might use different defaults during optimization")
print("- Production uses these identical parameters for all regimes")
print("- This could explain the difference in results")

# Check specific differences
print("\n" + "="*60)
print("PARAMETER ANALYSIS:")
print("="*60)

# Count unique parameter sets
unique_sets = {}
for regime, regime_params in params.items():
    param_str = json.dumps(regime_params, sort_keys=True)
    if param_str not in unique_sets:
        unique_sets[param_str] = []
    unique_sets[param_str].append(regime)

print(f"\nNumber of unique parameter sets: {len(unique_sets)}")
for i, (param_set, regimes) in enumerate(unique_sets.items()):
    print(f"\nParameter Set {i+1} used by: {', '.join(regimes)}")
    params_dict = json.loads(param_set)
    for k, v in params_dict.items():
        print(f"  {k}: {v}")