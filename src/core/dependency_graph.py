#!/usr/bin/env python3
"""
Dependency graph for component dependency management.

Based on DEPENDENCY_MANAGEMENT.md, this module provides robust
dependency tracking and circular dependency detection.
"""

from typing import Dict, List, Set, Any, Optional, Tuple
import logging


class DependencyGraph:
    """
    Graph representation of component dependencies.
    
    This class builds and analyzes a directed graph of component dependencies,
    enabling circular dependency detection and visualization.
    """
    
    def __init__(self):
        """Initialize dependency graph."""
        self._nodes: Dict[str, Dict[str, Any]] = {}  # node -> metadata
        self._edges: Dict[str, Set[str]] = {}  # node -> set of dependencies
        self._reverse_edges: Dict[str, Set[str]] = {}  # node -> set of dependents
        self.logger = logging.getLogger(__name__)
        
    def add_component(self, component_name: str, 
                     dependencies: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a component with its dependencies to the graph.
        
        This method adds a component and optionally all its dependencies
        in one call.
        
        Args:
            component_name: Name of component
            dependencies: Optional list of component names this component depends on
            metadata: Optional component metadata
        """
        # Add the component first
        if component_name not in self._nodes:
            self._nodes[component_name] = metadata or {}
            self._edges[component_name] = set()
            self._reverse_edges[component_name] = set()
            self.logger.debug(f"Added component to dependency graph: {component_name}")
        else:
            # Update metadata if provided
            if metadata:
                self._nodes[component_name].update(metadata)
        
        # Add dependencies
        if dependencies:
            for dep in dependencies:
                self.add_dependency(component_name, dep)
            
    def add_dependency(self, component: str, dependency: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a dependency relationship.
        
        Args:
            component: Component that depends on another
            dependency: Component being depended on
            metadata: Optional relationship metadata
        """
        # Ensure nodes exist
        if component not in self._nodes:
            self.add_component(component)
            
        if dependency not in self._nodes:
            self.add_component(dependency)
            
        # Add edge
        self._edges[component].add(dependency)
        self._reverse_edges[dependency].add(component)
        
        self.logger.debug(f"Added dependency: {component} -> {dependency}")
        
    def remove_component(self, component_name: str) -> None:
        """
        Remove a component and its relationships from the graph.
        
        Args:
            component_name: Component to remove
        """
        if component_name not in self._nodes:
            return
            
        # Remove edges from dependents
        for dependent in self._reverse_edges.get(component_name, set()).copy():
            self._edges[dependent].discard(component_name)
            
        # Remove edges to dependencies
        for dependency in self._edges.get(component_name, set()).copy():
            self._reverse_edges[dependency].discard(component_name)
            
        # Remove node
        del self._nodes[component_name]
        del self._edges[component_name]
        del self._reverse_edges[component_name]
        
        self.logger.debug(f"Removed component from dependency graph: {component_name}")
        
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the dependency graph using DFS.
        
        Note: This implementation may report overlapping cycles or parts of
        cycles multiple times depending on traversal order. This is acceptable
        for identifying the presence and participants of cycles. For unique
        elementary cycles, more complex algorithms would be needed.
        
        Returns:
            List of cycles found (each cycle is a list of component names)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> None:
            """Depth-first search to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    
            path.pop()
            rec_stack.remove(node)
            
        # Check each component
        for node in self._nodes:
            if node not in visited:
                dfs(node)
                
        return cycles
        
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
        
    def get_dependencies(self, component: str) -> List[str]:
        """
        Get direct dependencies of a component.
        
        Args:
            component: Component to check
            
        Returns:
            List of dependencies
        """
        return list(self._edges.get(component, set()))
        
    def get_dependents(self, component: str) -> List[str]:
        """
        Get components that depend on the specified component.
        
        Args:
            component: Component to check
            
        Returns:
            List of dependent components
        """
        return list(self._reverse_edges.get(component, set()))
        
    def get_all_dependencies(self, component: str) -> Set[str]:
        """
        Get all dependencies (direct and indirect) of a component.
        
        Args:
            component: Component to check
            
        Returns:
            Set of all dependencies
        """
        if component not in self._nodes:
            return set()
            
        all_deps = set()
        to_visit = list(self._edges.get(component, set()))
        
        while to_visit:
            dep = to_visit.pop()
            if dep not in all_deps:
                all_deps.add(dep)
                to_visit.extend(self._edges.get(dep, set()))
                
        return all_deps
        
    def get_initialization_order(self) -> List[str]:
        """
        Get topologically sorted initialization order.
        
        Components with no dependencies come first, followed by
        components that depend only on already-initialized components.
        
        Returns:
            List of component names in initialization order
            
        Raises:
            ValueError: If cycles exist in the graph
        """
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(f"Cannot determine initialization order due to cycles: {cycles}")
            
        # Kahn's algorithm for topological sort
        in_degree = {node: 0 for node in self._nodes}
        
        # Calculate in-degrees
        for node in self._nodes:
            for dep in self._edges.get(node, set()):
                in_degree[dep] += 1
                
        # Queue of nodes with no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Remove edges from this node
            for dependent in self._reverse_edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    
        if len(result) != len(self._nodes):
            raise ValueError("Failed to determine initialization order - graph may have cycles")
            
        return result
        
    def has_path(self, from_component: str, to_component: str) -> bool:
        """
        Check if there's a dependency path between components.
        
        Args:
            from_component: Starting component
            to_component: Target component
            
        Returns:
            True if from_component depends on to_component (directly or indirectly)
        """
        if from_component not in self._nodes or to_component not in self._nodes:
            return False
            
        visited = set()
        to_visit = [from_component]
        
        while to_visit:
            current = to_visit.pop()
            if current == to_component:
                return True
                
            if current not in visited:
                visited.add(current)
                to_visit.extend(self._edges.get(current, set()))
                
        return False
        
    def to_dict(self) -> Dict[str, List[str]]:
        """
        Convert graph to dictionary representation.
        
        Returns:
            Dict mapping components to their dependencies
        """
        return {node: list(deps) for node, deps in self._edges.items()}
        
    def from_dict(self, data: Dict[str, List[str]]) -> None:
        """
        Build graph from dictionary representation.
        
        Args:
            data: Dict mapping components to their dependencies
        """
        # Clear existing graph
        self._nodes.clear()
        self._edges.clear()
        self._reverse_edges.clear()
        
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
        return self._nodes.get(component, {}).copy()
        
    def validate(self) -> List[str]:
        """
        Validate the dependency graph.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for cycles
        cycles = self.detect_cycles()
        for cycle in cycles:
            errors.append(f"Circular dependency: {' -> '.join(cycle)}")
            
        # Check for missing nodes in edges
        for node, deps in self._edges.items():
            if node not in self._nodes:
                errors.append(f"Edge references non-existent node: {node}")
            for dep in deps:
                if dep not in self._nodes:
                    errors.append(f"Dependency references non-existent node: {node} -> {dep}")
                    
        return errors
        
    def __str__(self) -> str:
        """String representation of the dependency graph."""
        lines = ["Dependency Graph:"]
        for component in sorted(self._nodes.keys()):
            deps = sorted(self._edges.get(component, set()))
            if deps:
                lines.append(f"  {component} -> {', '.join(deps)}")
            else:
                lines.append(f"  {component} (no dependencies)")
        return "\n".join(lines)