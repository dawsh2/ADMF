#!/usr/bin/env python3
"""Debug script to check component dependencies."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.bootstrap import Bootstrap
from src.core.dependency_graph import DependencyGraph

# Create a bootstrap instance
bootstrap = Bootstrap()

# Print all component definitions
print("Component Definitions:")
print("=" * 50)
for name, comp_def in bootstrap.STANDARD_COMPONENTS.items():
    deps = comp_def.get('dependencies', [])
    print(f"{name}: depends on {deps}")

# Create dependency graph
graph = DependencyGraph()

# Add all components to graph
for name, comp_def in bootstrap.STANDARD_COMPONENTS.items():
    graph.add_component(name, metadata=comp_def)
    
# Add dependencies
for name, comp_def in bootstrap.STANDARD_COMPONENTS.items():
    for dep in comp_def.get('dependencies', []):
        try:
            graph.add_dependency(name, dep)
        except Exception as e:
            print(f"Error adding dependency {name} -> {dep}: {e}")

# Check for cycles
print("\nChecking for cycles...")
cycles = graph.detect_cycles()
if cycles:
    print("Found cycles:")
    for cycle in cycles:
        print(f"  {' -> '.join(cycle)}")
else:
    print("No cycles found")

# Try to get initialization order
print("\nTrying to get initialization order...")
try:
    order = graph.get_initialization_order()
    print("Initialization order:")
    for i, comp in enumerate(order):
        print(f"  {i+1}. {comp}")
except Exception as e:
    print(f"Error: {e}")
    
    # Debug: print the graph state
    print("\nGraph nodes:", graph._nodes)
    print("Graph edges:", graph._edges)
    print("Reverse edges:", graph._reverse_edges)