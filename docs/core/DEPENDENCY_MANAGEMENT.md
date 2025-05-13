# Dependency Management

## Overview

This document outlines the design for dependency management in the ADMF-Trader system, with a focus on circular dependency prevention, dependency validation, and proper dependency direction principles.

## Problem Statement

Complex systems like ADMF-Trader face several dependency management challenges:

1. **Circular Dependencies**: Components depending on each other directly or indirectly create fragile, tightly coupled systems that are difficult to test and maintain

2. **Late Dependency Detection**: Discovering missing dependencies at runtime leads to cryptic errors and makes debugging difficult

3. **Unclear Dependency Relationships**: Without explicit dependency documentation, it becomes hard to understand the system's structure

4. **Dependency Direction Violations**: Lower-level components sometimes inappropriately depend on higher-level components

We need a comprehensive dependency management approach that:
- Detects and prevents circular dependencies
- Validates dependencies early in the development process
- Visualizes dependency relationships for better understanding
- Enforces proper dependency direction

## Design Solution

### 1. Circular Dependency Detection

The core of our circular dependency prevention is a robust detection algorithm:

```python
import networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple

class DependencyGraph:
    """
    Graph representation of component dependencies.
    
    This class builds and analyzes a directed graph of component dependencies,
    enabling circular dependency detection and visualization.
    """
    
    def __init__(self):
        """Initialize dependency graph."""
        self._graph = nx.DiGraph()
        
    def add_component(self, component_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a component to the graph.
        
        Args:
            component_name: Name of component
            metadata: Optional component metadata
        """
        self._graph.add_node(component_name, **(metadata or {}))
        
    def add_dependency(self, component: str, dependency: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a dependency relationship.
        
        Args:
            component: Component that depends on another
            dependency: Component being depended on
            metadata: Optional relationship metadata
        """
        # Ensure nodes exist
        if component not in self._graph:
            self.add_component(component)
            
        if dependency not in self._graph:
            self.add_component(dependency)
            
        # Add edge representing dependency
        self._graph.add_edge(component, dependency, **(metadata or {}))
        
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the dependency graph.
        
        Returns:
            List of cycles found (each cycle is a list of component names)
        """
        try:
            cycles = list(nx.simple_cycles(self._graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []
            
    def get_cycles_for_component(self, component: str) -> List[List[str]]:
        """
        Get cycles involving a specific component.
        
        Args:
            component: Component to check
            
        Returns:
            List of cycles involving the component
        """
        all_cycles = self.detect_cycles()
        return [cycle for cycle in all_cycles if component in cycle]
        
    def visualize(self, output_file: str = 'dependency_graph.png') -> None:
        """
        Create a visual representation of the dependency graph.
        
        Args:
            output_file: Output file path
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Create layout
            pos = nx.spring_layout(self._graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(self._graph, pos, node_color='lightblue', node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(self._graph, pos, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(self._graph, pos)
            
            # Save to file
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
        except ImportError:
            print("Matplotlib is required for visualization")
            
    def get_dependencies(self, component: str) -> List[str]:
        """
        Get direct dependencies of a component.
        
        Args:
            component: Component to check
            
        Returns:
            List of dependencies
        """
        if component not in self._graph:
            return []
            
        return list(self._graph.neighbors(component))
        
    def get_dependents(self, component: str) -> List[str]:
        """
        Get components that depend on the specified component.
        
        Args:
            component: Component to check
            
        Returns:
            List of dependent components
        """
        return [pred for pred in self._graph.predecessors(component)]
        
    def get_all_dependencies(self, component: str) -> Set[str]:
        """
        Get all dependencies (direct and indirect) of a component.
        
        Args:
            component: Component to check
            
        Returns:
            Set of all dependencies
        """
        if component not in self._graph:
            return set()
            
        return set(nx.descendants(self._graph, component))
        
    def get_dependency_path(self, from_component: str, to_component: str) -> Optional[List[str]]:
        """
        Get dependency path between components.
        
        Args:
            from_component: Starting component
            to_component: Target component
            
        Returns:
            List representing path or None if no path exists
        """
        if from_component not in self._graph or to_component not in self._graph:
            return None
            
        try:
            return nx.shortest_path(self._graph, from_component, to_component)
        except nx.NetworkXNoPath:
            return None
            
    def to_dict(self) -> Dict[str, List[str]]:
        """
        Convert graph to dictionary representation.
        
        Returns:
            Dict mapping components to their dependencies
        """
        result = {}
        
        for node in self._graph.nodes():
            result[node] = list(self._graph.neighbors(node))
            
        return result
        
    def from_dict(self, data: Dict[str, List[str]]) -> None:
        """
        Build graph from dictionary representation.
        
        Args:
            data: Dict mapping components to their dependencies
        """
        # Clear existing graph
        self._graph.clear()
        
        # Add components and dependencies
        for component, dependencies in data.items():
            self.add_component(component)
            
            for dependency in dependencies:
                self.add_dependency(component, dependency)
                
    def get_component_metadata(self, component: str) -> Dict[str, Any]:
        """
        Get component metadata.
        
        Args:
            component: Component to get metadata for
            
        Returns:
            Component metadata
        """
        if component not in self._graph:
            return {}
            
        return dict(self._graph.nodes[component])
```

### 2. Enhanced Container with Cycle Detection

The dependency injection container is enhanced with cycle detection:

```python
class Container:
    """
    Dependency injection container with cycle detection.
    
    This class manages component creation and resolution,
    with built-in detection of circular dependencies.
    """
    
    def __init__(self):
        """Initialize container."""
        self._components = {}  # name -> component class/factory
        self._singletons = {}  # name -> component instance
        self._factories = {}  # name -> factory function
        self._metadata = {}  # name -> component metadata
        self._dependency_graph = DependencyGraph()
        
    def register(self, name: str, component_class: Any, singleton: bool = True,
                dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component.
        
        Args:
            name: Component name
            component_class: Component class
            singleton: Whether to use singleton pattern
            dependencies: List of dependency names
            metadata: Component metadata
        """
        self._components[name] = component_class
        
        # Store metadata
        self._metadata[name] = metadata or {}
        self._metadata[name]['singleton'] = singleton
        
        # Add to dependency graph
        self._dependency_graph.add_component(name, self._metadata[name])
        
        # Add dependencies to graph
        if dependencies:
            for dependency in dependencies:
                self._dependency_graph.add_dependency(name, dependency)
                
    def register_instance(self, name: str, instance: Any, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component instance.
        
        Args:
            name: Component name
            instance: Component instance
            metadata: Component metadata
        """
        self._singletons[name] = instance
        
        # Store metadata
        self._metadata[name] = metadata or {}
        self._metadata[name]['singleton'] = True
        
        # Add to dependency graph
        self._dependency_graph.add_component(name, self._metadata[name])
        
    def register_factory(self, name: str, factory: Any,
                        dependencies: Optional[List[str]] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a component factory.
        
        Args:
            name: Component name
            factory: Factory function
            dependencies: List of dependency names
            metadata: Component metadata
        """
        self._factories[name] = factory
        
        # Store metadata
        self._metadata[name] = metadata or {}
        self._metadata[name]['factory'] = True
        
        # Add to dependency graph
        self._dependency_graph.add_component(name, self._metadata[name])
        
        # Add dependencies to graph
        if dependencies:
            for dependency in dependencies:
                self._dependency_graph.add_dependency(name, dependency)
                
    def get(self, name: str, resolution_path: Optional[List[str]] = None) -> Any:
        """
        Get a component by name.
        
        Args:
            name: Component name
            resolution_path: Current resolution path (for cycle detection)
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component not found
            CircularDependencyError: If circular dependency detected
        """
        # Initialize resolution path if not provided
        if resolution_path is None:
            resolution_path = []
            
        # Check for circular dependency
        if name in resolution_path:
            cycle = resolution_path[resolution_path.index(name):] + [name]
            raise CircularDependencyError(f"Circular dependency detected: {' -> '.join(cycle)}")
            
        # Add to resolution path
        current_path = resolution_path + [name]
        
        # Check if singleton instance exists
        if name in self._singletons:
            return self._singletons[name]
            
        # Check if factory exists
        if name in self._factories:
            factory = self._factories[name]
            instance = factory()
            
            # Cache instance for future use
            self._singletons[name] = instance
            
            return instance
            
        # Check if component class exists
        if name in self._components:
            component_class = self._components[name]
            
            # Check for constructor parameter requirements
            instance = self._create_instance(component_class, current_path)
            
            # Cache instance if singleton
            if self._metadata.get(name, {}).get('singleton', True):
                self._singletons[name] = instance
                
            return instance
            
        raise ValueError(f"Component not found: {name}")
        
    def has(self, name: str) -> bool:
        """
        Check if a component exists.
        
        Args:
            name: Component name
            
        Returns:
            bool: Whether component exists
        """
        return name in self._singletons or name in self._factories or name in self._components
        
    def _create_instance(self, component_class: Any, resolution_path: List[str]) -> Any:
        """
        Create a component instance.
        
        Args:
            component_class: Component class
            resolution_path: Current resolution path
            
        Returns:
            Component instance
        """
        # Check if dependencies are needed
        if hasattr(component_class, '__init__'):
            import inspect
            
            # Get constructor signature
            sig = inspect.signature(component_class.__init__)
            
            # Get parameters (excluding self)
            params = list(sig.parameters.values())[1:]
            
            # Collect arguments for constructor
            args = []
            kwargs = {}
            
            for param in params:
                # Skip args with default values
                if param.default is not inspect.Parameter.empty:
                    continue
                    
                # Try to resolve dependency by parameter name
                dependency_name = param.name
                
                if self.has(dependency_name):
                    # Resolve dependency
                    dependency = self.get(dependency_name, resolution_path)
                    
                    if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                        args.append(dependency)
                    elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                        kwargs[param.name] = dependency
                        
            # Create instance
            return component_class(*args, **kwargs)
        else:
            # No constructor, create instance directly
            return component_class()
            
    def reset(self) -> None:
        """Reset container."""
        self._singletons.clear()
        
    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze dependencies.
        
        Returns:
            Dict with dependency analysis
        """
        cycles = self._dependency_graph.detect_cycles()
        
        return {
            'cycles': cycles,
            'component_count': len(self._components) + len(self._factories) + len(self._singletons),
            'dependency_count': sum(1 for _ in self._dependency_graph._graph.edges()),
            'singleton_count': sum(1 for meta in self._metadata.values() if meta.get('singleton', False)),
            'factory_count': sum(1 for meta in self._metadata.values() if meta.get('factory', False)),
            'has_cycles': len(cycles) > 0
        }
        
    def get_dependency_graph(self) -> DependencyGraph:
        """
        Get dependency graph.
        
        Returns:
            DependencyGraph instance
        """
        return self._dependency_graph
        
    def validate_dependencies(self) -> List[str]:
        """
        Validate all component dependencies.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for missing dependencies
        for component in self._components.keys():
            for dependency in self._dependency_graph.get_dependencies(component):
                if not self.has(dependency):
                    errors.append(f"Missing dependency: {component} -> {dependency}")
                    
        # Check for circular dependencies
        cycles = self._dependency_graph.detect_cycles()
        for cycle in cycles:
            errors.append(f"Circular dependency: {' -> '.join(cycle + [cycle[0]])}")
            
        # Check for optional interface implementations
        # ...
        
        return errors
        
    def generate_dependency_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate dependency report.
        
        Args:
            output_file: Optional file to write report to
            
        Returns:
            Dict with report data
        """
        report = {
            'components': {},
            'analysis': self.analyze_dependencies()
        }
        
        # Collect component information
        for name in sorted(set(list(self._components.keys()) + list(self._factories.keys()) + list(self._singletons.keys()))):
            dependencies = self._dependency_graph.get_dependencies(name)
            dependents = self._dependency_graph.get_dependents(name)
            
            report['components'][name] = {
                'dependencies': dependencies,
                'dependents': dependents,
                'is_singleton': self._metadata.get(name, {}).get('singleton', False),
                'is_factory': self._metadata.get(name, {}).get('factory', False),
                'metadata': self._metadata.get(name, {}),
                'cycles': self._dependency_graph.get_cycles_for_component(name)
            }
            
        # Write to file if requested
        if output_file:
            import json
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
```

### 3. Dependency Direction Validation

To enforce proper dependency direction:

```python
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple

class DependencyDirection(Enum):
    """Dependency direction rules."""
    CORE = 0
    DATA = 1
    STRATEGY = 2
    RISK = 3
    EXECUTION = 4
    ANALYTICS = 5

class DependencyDirectionValidator:
    """
    Validator for dependency direction rules.
    
    This class enforces proper dependency direction between modules,
    preventing lower-level components from depending on higher-level ones.
    """
    
    def __init__(self, dependency_graph: DependencyGraph):
        """
        Initialize validator.
        
        Args:
            dependency_graph: Dependency graph to validate
        """
        self._graph = dependency_graph
        
        # Define module levels
        self._module_levels = {
            'core': DependencyDirection.CORE,
            'data': DependencyDirection.DATA,
            'strategy': DependencyDirection.STRATEGY,
            'risk': DependencyDirection.RISK,
            'execution': DependencyDirection.EXECUTION,
            'analytics': DependencyDirection.ANALYTICS
        }
        
        # Define allowed dependency directions
        self._allowed_directions = {
            # Core can only depend on other core components
            DependencyDirection.CORE: [DependencyDirection.CORE],
            
            # Data can depend on core
            DependencyDirection.DATA: [DependencyDirection.CORE, DependencyDirection.DATA],
            
            # Strategy can depend on core and data
            DependencyDirection.STRATEGY: [DependencyDirection.CORE, DependencyDirection.DATA, DependencyDirection.STRATEGY],
            
            # Risk can depend on core, data, and strategy
            DependencyDirection.RISK: [DependencyDirection.CORE, DependencyDirection.DATA, DependencyDirection.STRATEGY, DependencyDirection.RISK],
            
            # Execution can depend on core, data, strategy, and risk
            DependencyDirection.EXECUTION: [DependencyDirection.CORE, DependencyDirection.DATA, DependencyDirection.STRATEGY, DependencyDirection.RISK, DependencyDirection.EXECUTION],
            
            # Analytics can depend on any module
            DependencyDirection.ANALYTICS: [DependencyDirection.CORE, DependencyDirection.DATA, DependencyDirection.STRATEGY, DependencyDirection.RISK, DependencyDirection.EXECUTION, DependencyDirection.ANALYTICS]
        }
        
    def get_component_level(self, component_name: str) -> DependencyDirection:
        """
        Get component level.
        
        Args:
            component_name: Component name
            
        Returns:
            Component level
        """
        # Extract module from component name (e.g., 'core.event_bus' -> 'core')
        if '.' in component_name:
            module = component_name.split('.')[0]
        else:
            # Default to core if no module prefix
            module = 'core'
            
        return self._module_levels.get(module, DependencyDirection.CORE)
        
    def is_valid_dependency(self, component: str, dependency: str) -> bool:
        """
        Check if dependency direction is valid.
        
        Args:
            component: Component name
            dependency: Dependency name
            
        Returns:
            bool: Whether dependency is valid
        """
        component_level = self.get_component_level(component)
        dependency_level = self.get_component_level(dependency)
        
        # Check if dependency level is allowed for this component
        return dependency_level in self._allowed_directions[component_level]
        
    def validate_all(self) -> List[str]:
        """
        Validate all dependencies.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Get all edges from graph
        for component, dependency in self._graph._graph.edges():
            if not self.is_valid_dependency(component, dependency):
                component_level = self.get_component_level(component)
                dependency_level = self.get_component_level(dependency)
                
                errors.append(
                    f"Invalid dependency direction: {component} ({component_level.name}) -> "
                    f"{dependency} ({dependency_level.name})"
                )
                
        return errors
        
    def get_allowed_dependencies(self, component: str) -> List[DependencyDirection]:
        """
        Get allowed dependency levels for a component.
        
        Args:
            component: Component name
            
        Returns:
            List of allowed dependency levels
        """
        component_level = self.get_component_level(component)
        return self._allowed_directions[component_level]
```

### 4. Early Dependency Validation System

To detect missing dependencies early:

```python
class DependencyValidator:
    """
    System for early dependency validation.
    
    This class validates dependencies at load time,
    detecting missing or invalid dependencies before runtime.
    """
    
    def __init__(self, container: Container):
        """
        Initialize validator.
        
        Args:
            container: Container to validate
        """
        self._container = container
        self._direction_validator = DependencyDirectionValidator(container.get_dependency_graph())
        
    def validate_container(self) -> List[str]:
        """
        Validate container dependencies.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for missing dependencies
        errors.extend(self._container.validate_dependencies())
        
        # Check for direction violations
        errors.extend(self._direction_validator.validate_all())
        
        return errors
        
    def validate_component(self, component_name: str) -> List[str]:
        """
        Validate a specific component's dependencies.
        
        Args:
            component_name: Component to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check if component exists
        if not self._container.has(component_name):
            errors.append(f"Component not found: {component_name}")
            return errors
            
        # Get component dependencies
        graph = self._container.get_dependency_graph()
        dependencies = graph.get_dependencies(component_name)
        
        # Check for missing dependencies
        for dependency in dependencies:
            if not self._container.has(dependency):
                errors.append(f"Missing dependency: {component_name} -> {dependency}")
                
        # Check for direction violations
        for dependency in dependencies:
            if not self._direction_validator.is_valid_dependency(component_name, dependency):
                errors.append(
                    f"Invalid dependency direction: {component_name} -> {dependency}"
                )
                
        # Check for circular dependencies
        cycles = graph.get_cycles_for_component(component_name)
        for cycle in cycles:
            errors.append(f"Circular dependency: {' -> '.join(cycle + [cycle[0]])}")
            
        return errors
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate validation report.
        
        Returns:
            Dict with validation report
        """
        all_errors = self.validate_container()
        
        # Categorize errors
        missing_deps = []
        circular_deps = []
        direction_violations = []
        other_errors = []
        
        for error in all_errors:
            if error.startswith("Missing dependency:"):
                missing_deps.append(error)
            elif error.startswith("Circular dependency:"):
                circular_deps.append(error)
            elif error.startswith("Invalid dependency direction:"):
                direction_violations.append(error)
            else:
                other_errors.append(error)
                
        return {
            'all_errors': all_errors,
            'error_count': len(all_errors),
            'missing_dependencies': missing_deps,
            'circular_dependencies': circular_deps,
            'direction_violations': direction_violations,
            'other_errors': other_errors,
            'is_valid': len(all_errors) == 0
        }
        
    def test_load_all_components(self) -> Dict[str, Any]:
        """
        Test loading all components.
        
        Returns:
            Dict with load test results
        """
        results = {
            'success': [],
            'failure': {},
            'total': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        # Get all component names
        graph = self._container.get_dependency_graph()
        all_components = list(graph._graph.nodes())
        
        # Try to load each component
        for component in all_components:
            results['total'] += 1
            
            try:
                # Attempt to get component
                self._container.get(component)
                results['success'].append(component)
                results['success_count'] += 1
            except Exception as e:
                # Record failure
                results['failure'][component] = str(e)
                results['failure_count'] += 1
                
        return results
```

### 5. Dependency Graph Visualization

For dependency visualization:

```python
class DependencyVisualizer:
    """
    Visualization tools for dependency graphs.
    
    This class provides various visualizations of dependency relationships,
    helping developers understand the system's structure.
    """
    
    def __init__(self, dependency_graph: DependencyGraph):
        """
        Initialize visualizer.
        
        Args:
            dependency_graph: Dependency graph to visualize
        """
        self._graph = dependency_graph
        
    def generate_graphviz(self, output_file: str) -> None:
        """
        Generate Graphviz visualization.
        
        Args:
            output_file: Output file path
        """
        try:
            import graphviz as gv
            
            # Create digraph
            dot = gv.Digraph(comment='Dependency Graph')
            
            # Add nodes
            for node in self._graph._graph.nodes():
                # Extract module from name
                module = node.split('.')[0] if '.' in node else 'core'
                
                # Set color based on module
                colors = {
                    'core': 'lightblue',
                    'data': 'lightgreen',
                    'strategy': 'lightyellow',
                    'risk': 'lightcoral',
                    'execution': 'lightgrey',
                    'analytics': 'plum'
                }
                
                color = colors.get(module, 'white')
                
                dot.node(node, node, style='filled', fillcolor=color)
                
            # Add edges
            for u, v in self._graph._graph.edges():
                dot.edge(u, v)
                
            # Render to file
            dot.render(output_file, format='png')
        except ImportError:
            print("Graphviz Python library is required for visualization")
            
    def generate_d3_visualization(self, output_file: str) -> None:
        """
        Generate D3.js visualization.
        
        Args:
            output_file: Output file path
        """
        # Create JSON representation for D3
        data = {
            'nodes': [],
            'links': []
        }
        
        # Add nodes
        for i, node in enumerate(self._graph._graph.nodes()):
            # Extract module from name
            module = node.split('.')[0] if '.' in node else 'core'
            
            # Add node data
            data['nodes'].append({
                'id': i,
                'name': node,
                'module': module,
                'group': list(self._graph._graph.neighbors(node)) or 1
            })
            
            # Map node name to ID
            node_map = {node: i for i, node in enumerate(self._graph._graph.nodes())}
            
        # Add edges
        for u, v in self._graph._graph.edges():
            data['links'].append({
                'source': node_map[u],
                'target': node_map[v],
                'value': 1
            })
            
        # Write to file
        import json
        
        with open(output_file, 'w') as f:
            json.dump(data, f)
            
    def generate_module_dependency_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Generate module dependency matrix.
        
        Returns:
            Dict with module dependencies
        """
        # Get all modules
        modules = set()
        
        for node in self._graph._graph.nodes():
            module = node.split('.')[0] if '.' in node else 'core'
            modules.add(module)
            
        # Create matrix
        matrix = {module: {dep: 0 for dep in modules} for module in modules}
        
        # Count dependencies
        for u, v in self._graph._graph.edges():
            u_module = u.split('.')[0] if '.' in u else 'core'
            v_module = v.split('.')[0] if '.' in v else 'core'
            
            matrix[u_module][v_module] += 1
            
        return matrix
        
    def print_dependency_summary(self) -> None:
        """Print dependency summary."""
        # Count components by module
        modules = {}
        
        for node in self._graph._graph.nodes():
            module = node.split('.')[0] if '.' in node else 'core'
            modules[module] = modules.get(module, 0) + 1
            
        # Count dependencies
        dependencies = {}
        
        for u, v in self._graph._graph.edges():
            u_module = u.split('.')[0] if '.' in u else 'core'
            dependencies[u_module] = dependencies.get(u_module, 0) + 1
            
        # Print summary
        print("Dependency Summary:")
        print("-" * 40)
        print(f"Total components: {len(self._graph._graph.nodes())}")
        print(f"Total dependencies: {len(self._graph._graph.edges())}")
        print()
        
        print("Components by module:")
        for module, count in sorted(modules.items()):
            print(f"  {module}: {count}")
        print()
        
        print("Dependencies by module:")
        for module, count in sorted(dependencies.items()):
            print(f"  {module}: {count}")
        print()
        
        # Print cycles
        cycles = self._graph.detect_cycles()
        if cycles:
            print(f"Circular dependencies: {len(cycles)}")
            for i, cycle in enumerate(cycles):
                print(f"  {i+1}. {' -> '.join(cycle + [cycle[0]])}")
            print()
        else:
            print("No circular dependencies detected")
```

### 6. Dependency Injection Context

For dependency tracking during component creation:

```python
class DependencyContext:
    """
    Context for dependency tracking.
    
    This class tracks the dependencies being resolved during
    component creation, enabling circular dependency detection.
    """
    
    def __init__(self):
        """Initialize dependency context."""
        self.resolution_path = []
        self.resolution_stack = []
        
    def enter_resolution(self, component_name: str) -> None:
        """
        Enter resolution scope.
        
        Args:
            component_name: Component being resolved
        """
        self.resolution_path.append(component_name)
        self.resolution_stack.append(component_name)
        
    def exit_resolution(self) -> None:
        """Exit resolution scope."""
        if self.resolution_stack:
            self.resolution_stack.pop()
            
    def get_current_path(self) -> List[str]:
        """
        Get current resolution path.
        
        Returns:
            List of component names in resolution path
        """
        return self.resolution_path[:]
        
    def check_circular_dependency(self, component_name: str) -> Optional[List[str]]:
        """
        Check for circular dependency.
        
        Args:
            component_name: Component to check
            
        Returns:
            Cycle if circular dependency detected, None otherwise
        """
        if component_name in self.resolution_stack:
            # Find cycle
            start_idx = self.resolution_stack.index(component_name)
            cycle = self.resolution_stack[start_idx:] + [component_name]
            return cycle
            
        return None
        
    def __enter__(self):
        """Enter context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self.resolution_stack:
            self.resolution_stack.pop()
```

## Custom Exceptions

Custom exceptions for dependency management:

```python
class DependencyError(Exception):
    """Base class for dependency errors."""
    pass

class CircularDependencyError(DependencyError):
    """Exception for circular dependency detection."""
    
    def __init__(self, message, cycle=None):
        """Initialize exception."""
        super().__init__(message)
        self.cycle = cycle

class MissingDependencyError(DependencyError):
    """Exception for missing dependency detection."""
    
    def __init__(self, message, component=None, dependency=None):
        """Initialize exception."""
        super().__init__(message)
        self.component = component
        self.dependency = dependency

class InvalidDependencyDirectionError(DependencyError):
    """Exception for invalid dependency direction."""
    
    def __init__(self, message, component=None, dependency=None):
        """Initialize exception."""
        super().__init__(message)
        self.component = component
        self.dependency = dependency
```

## Implementation Strategy

### 1. Core Implementation

1. Implement `DependencyGraph` for cycle detection:
   - Node and edge management
   - Cycle detection algorithm
   - Dependency path analysis

2. Enhance `Container` with cycle detection:
   - Track dependencies during resolution
   - Detect and report cycles
   - Generate dependency reports

3. Implement `DependencyDirectionValidator`:
   - Define module levels
   - Define allowed dependency directions
   - Validate components against rules

### 2. Integration

1. Update system bootstrap to validate dependencies:
   - Early validation of dependencies
   - Reporting of validation errors
   - Display of dependency graphs

2. Add dependency documentation tools:
   - Automatic generation of dependency diagrams
   - Creation of dependency matrices
   - Documentation of module boundaries

### 3. Testing

1. Create dependency validation tests:
   - Unit tests for cycle detection
   - Tests for direction validation
   - Tests for missing dependency detection

2. Add load tests for all components:
   - Verify that all components can be loaded
   - Identify dependency resolution errors
   - Measure resolution performance

## Dependency Direction Principles

### 1. Module Hierarchy

The ADMF-Trader system follows a strict module hierarchy:

1. **Core**: Basic infrastructure (event system, container, etc.)
2. **Data**: Data handling and manipulation
3. **Strategy**: Trading strategy implementation
4. **Risk**: Risk management and portfolio tracking
5. **Execution**: Order execution and brokerage
6. **Analytics**: Performance analysis and reporting

### 2. Direction Rules

Components can only depend on components at the same level or lower:

- Core components can only depend on other core components
- Data components can depend on core components
- Strategy components can depend on core and data components
- Risk components can depend on core, data, and strategy components
- Execution components can depend on core, data, strategy, and risk components
- Analytics components can depend on any component

### 3. Interface-Based Dependencies

To minimize coupling, components should depend on interfaces rather than implementations:

```python
# Bad: Direct dependency on implementation
class Strategy:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        
# Good: Dependency on interface
class Strategy:
    def __init__(self, data_handler: DataHandlerBase):
        self.data_handler = data_handler
```

### 4. Dependency Inversion

For special cases where lower-level components need functions from higher-level ones, use dependency inversion:

```python
# Define interface in lower-level module
class EventHandlerBase:
    def handle_event(self, event):
        pass
        
# Implement in higher-level module
class StrategyEventHandler(EventHandlerBase):
    def handle_event(self, event):
        # Strategy-specific handling
        pass
        
# Use in lower-level module
class EventBus:
    def __init__(self, handler: EventHandlerBase):
        self.handler = handler
        
    def publish(self, event):
        self.handler.handle_event(event)
```

## Best Practices

### 1. Dependency Registration

Register all dependencies explicitly:

```python
# Register components with dependencies
container.register(
    'strategy',
    MovingAverageStrategy,
    dependencies=['data_handler', 'event_bus'],
    metadata={'module': 'strategy'}
)

# Register interfaces and implementations separately
container.register('data_handler_interface', DataHandlerBase)
container.register(
    'csv_data_handler',
    CsvDataHandler,
    dependencies=['data_handler_interface'],
    metadata={'implements': 'data_handler_interface'}
)
```

### 2. Dependency Injection

Use constructor injection for required dependencies:

```python
class RiskManager:
    def __init__(self, portfolio, event_bus):
        # Both dependencies are required
        self.portfolio = portfolio
        self.event_bus = event_bus
        
    # Method injection for optional dependencies
    def set_logger(self, logger):
        # Logger is optional
        self.logger = logger
```

### 3. Dependency Documentation

Document dependencies for each component:

```python
class Strategy:
    """Trading strategy implementation.
    
    Dependencies:
    - data_handler: DataHandlerBase
    - event_bus: EventBus
    - logger: Logger (optional)
    """
```

### 4. Circular Dependency Resolution

When circular dependencies are unavoidable, resolve them using these techniques:

1. **Dependency Inversion**: Extract an interface that both components can depend on
2. **Mediator Pattern**: Create a mediator component that coordinates interaction
3. **Event-Based Communication**: Use events instead of direct method calls
4. **Lazy Loading**: Defer dependency resolution until actually needed

```python
# Example: Resolving circular dependency with mediator
class Mediator:
    def __init__(self):
        self.component_a = None
        self.component_b = None
        
    def set_component_a(self, component_a):
        self.component_a = component_a
        
    def set_component_b(self, component_b):
        self.component_b = component_b
        
    def notify_a(self, message):
        if self.component_a:
            self.component_a.receive(message)
            
    def notify_b(self, message):
        if self.component_b:
            self.component_b.receive(message)

class ComponentA:
    def __init__(self, mediator):
        self.mediator = mediator
        self.mediator.set_component_a(self)
        
    def send_to_b(self, message):
        self.mediator.notify_b(message)
        
    def receive(self, message):
        # Process message from B
        pass

class ComponentB:
    def __init__(self, mediator):
        self.mediator = mediator
        self.mediator.set_component_b(self)
        
    def send_to_a(self, message):
        self.mediator.notify_a(message)
        
    def receive(self, message):
        # Process message from A
        pass
```

## Conclusion

This comprehensive dependency management approach provides robust tools for preventing circular dependencies, validating dependency relationships, and enforcing proper dependency direction. The early validation ensures that dependency issues are caught during development rather than at runtime, improving the system's reliability and maintainability.

The visualization and documentation tools enhance developers' understanding of the system's structure, making it easier to reason about component relationships and dependencies. By enforcing strict dependency direction principles, the system maintains a clean architecture with clear boundaries between modules.

Overall, these improvements will lead to a more robust, maintainable, and understandable ADMF-Trader system.