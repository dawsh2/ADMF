#!/usr/bin/env python3
"""Debug script to verify production parameter loading"""

import json
import sys
sys.path.append('/Users/daws/ADMF')

# Load the JSON file
with open('regime_optimized_parameters.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("REGIME OPTIMIZED PARAMETERS ANALYSIS")
print("=" * 80)

# Overall best parameters (from test set)
print("\nOVERALL BEST PARAMETERS (Test Set Winner):")
print(f"  {data['overall_best_parameters']}")
print(f"  Metric: {data['overall_best_metric']['name']} = {data['overall_best_metric']['value']:.2f}")

# Check if we have weights
print("\nREGIME-SPECIFIC PARAMETERS AND WEIGHTS:")
for regime, regime_data in data['regime_best_parameters'].items():
    print(f"\n{regime}:")
    params = regime_data['parameters']['parameters'] if 'parameters' in regime_data['parameters'] else regime_data['parameters']
    print(f"  Parameters: {params}")
    
    weights = regime_data.get('weights', {})
    if weights:
        print(f"  Weights: {weights}")
    else:
        print(f"  Weights: NOT FOUND (will use hard-coded 0.341/0.659)")
    
    if 'metric' in regime_data:
        print(f"  Metric: {regime_data['metric']['name']} = {regime_data['metric']['value']:.4f}")

# Check test results
if 'test_results' in data:
    print("\n\nTOP TEST RESULTS:")
    for i, result in enumerate(data['test_results'][:3]):
        print(f"  #{i+1}: Test={result['test_metric']:.2f}, Train={result['train_metric']:.2f}")
        print(f"       Params: {result['parameters']}")

# Check adaptive test results
if 'regime_adaptive_test_results' in data:
    print("\n\nADAPTIVE TEST RESULTS:")
    adaptive = data['regime_adaptive_test_results']
    if 'adaptive_metric' in adaptive:
        print(f"  Final portfolio value: {adaptive['adaptive_metric']:.2f}")
    if 'regime_performance' in adaptive:
        print(f"  Regimes with trades: {list(adaptive['regime_performance'].keys())}")

print("\n" + "=" * 80)
print("PRODUCTION IMPLICATIONS:")
print("=" * 80)
print("\n1. Production should achieve ~100,826.54 (best test result)")
print("2. But adaptive test only achieved 99,396.03")
print("3. This suggests regime switching may reduce performance")
print("4. Missing weights in JSON means hard-coded 0.341/0.659 are used")
print("\nTO FIX: Run genetic optimization to find optimal weights per regime")