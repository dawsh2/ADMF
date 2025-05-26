---

## 4. Advanced Caching

### 4.1 Caching Protocols

```python
from typing import TypeVar, Generic, Optional, Dict, Any, Callable, Union
from abc import abstractmethod
from enum import Enum
import threading
import time
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')

class CacheStrategy(Enum):
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used  
    TLRU = "tlru"         # Time-aware LRU
    FIFO = "fifo"         # First In, First Out
    UNBOUNDED = "unbounded"  # No size limit

class CacheLevel(Enum):
    MEMORY = "memory"     # In-memory cache
    DISK = "disk"         # Disk-based cache
    DISTRIBUTED = "distributed"  # Redis/distributed cache
    HYBRID = "hybrid"     # Multi-level cache

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable operations"""
    
    @abstractmethod
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for arguments"""
        ...
    
    @abstractmethod
    def is_cacheable(self, *args, **kwargs) -> bool:
        """Check if result should be cached"""
        ...

@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        ...
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ...
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        ...
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        ...
```

### 4.2 Cache Implementation

```python
class MemoryCache:
    """High-performance in-memory cache"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_# ADMF-Trader: Missing Critical Features Documentation

This document covers the five critical features identified as gaps in the Protocol + Composition architecture documentation.

## Table of Contents

1. [State Management](#1-state-management)
2. [System Visualization](#2-system-visualization)
3. [Health Monitoring](#3-health-monitoring)
4. [Advanced Caching](#4-advanced-caching)
5. [Component Discovery](#5-component-discovery)

---

## 1. State Management

### 1.1 State Management Protocols

```python
from typing import Dict, Any, Optional, TypeVar, Generic
from abc import abstractmethod
from enum import Enum
import threading
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime

T = TypeVar('T')

class StateScope(Enum):
    """Defines the scope of state management"""
    GLOBAL = "global"           # Shared across entire system
    SESSION = "session"         # Per trading session
    COMPONENT = "component"     # Per component instance
    THREAD = "thread"          # Per thread
    REQUEST = "request"        # Per operation/request

@runtime_checkable
class Stateful(Protocol):
    """Protocol for components that manage state"""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current component state"""
        ...
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state"""
        ...
    
    @abstractmethod
    def reset_state(self) -> None:
        """Reset component to initial state"""
        ...

@runtime_checkable
class Serializable(Protocol):
    """Protocol for components that can be serialized"""
    
    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize component to bytes"""
        ...
    
    @abstractmethod
    def deserialize(self, data: bytes) -> None:
        """Deserialize component from bytes"""
        ...

@runtime_checkable
class Snapshotable(Protocol):
    """Protocol for components that support snapshots"""
    
    @abstractmethod
    def create_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Create a state snapshot"""
        ...
    
    @abstractmethod
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore from a state snapshot"""
        ...
    
    @abstractmethod
    def list_snapshots(self) -> List[str]:
        """List available snapshots"""
        ...
```

### 1.2 State Manager Implementation

```python
class StateManager:
    """Central state management for the trading system"""
    
    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.state_history: List[Dict[str, Any]] = []
        self.watchers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # State persistence
        self.persistence_enabled = False
        self.persistence_path = "state"
        self.auto_save_interval = 300  # seconds
        self.save_thread: Optional[threading.Thread] = None
    
    def register_component(self, component_id: str, component: Any, 
                          scope: StateScope = StateScope.COMPONENT) -> None:
        """Register component for state management"""
        with self._lock:
            self.states[component_id] = {
                'component': component,
                'scope': scope,
                'state': {},
                'last_updated': datetime.now(),
                'version': 0
            }
    
    def save_state(self, component_id: str, state: Dict[str, Any]) -> None:
        """Save component state"""
        with self._lock:
            if component_id in self.states:
                old_state = self.states[component_id]['state'].copy()
                self.states[component_id]['state'] = state
                self.states[component_id]['last_updated'] = datetime.now()
                self.states[component_id]['version'] += 1
                
                # Record state change in history
                self.state_history.append({
                    'component_id': component_id,
                    'timestamp': datetime.now(),
                    'old_state': old_state,
                    'new_state': state.copy(),
                    'version': self.states[component_id]['version']
                })
                
                # Limit history size
                if len(self.state_history) > 1000:
                    self.state_history.pop(0)
                
                # Notify watchers
                self._notify_watchers(component_id, old_state, state)
    
    def load_state(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Load component state"""
        with self._lock:
            if component_id in self.states:
                return self.states[component_id]['state'].copy()
            return None
    
    def create_snapshot(self, snapshot_id: str, 
                       component_ids: Optional[List[str]] = None) -> None:
        """Create system-wide snapshot"""
        with self._lock:
            snapshot_data = {
                'timestamp': datetime.now(),
                'components': {}
            }
            
            target_components = component_ids or list(self.states.keys())
            
            for comp_id in target_components:
                if comp_id in self.states:
                    snapshot_data['components'][comp_id] = {
                        'state': self.states[comp_id]['state'].copy(),
                        'version': self.states[comp_id]['version']
                    }
            
            self.snapshots[snapshot_id] = snapshot_data
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore system from snapshot"""
        if snapshot_id not in self.snapshots:
            return False
        
        with self._lock:
            snapshot = self.snapshots[snapshot_id]
            
            for comp_id, comp_data in snapshot['components'].items():
                if comp_id in self.states:
                    component = self.states[comp_id]['component']
                    
                    # Restore state
                    if isinstance(component, Stateful):
                        component.set_state(comp_data['state'])
                    
                    # Update our records
                    self.states[comp_id]['state'] = comp_data['state'].copy()
                    self.states[comp_id]['version'] = comp_data['version']
                    self.states[comp_id]['last_updated'] = datetime.now()
            
            return True
    
    def watch_state(self, component_id: str, callback: Callable) -> None:
        """Watch for state changes"""
        with self._lock:
            if component_id not in self.watchers:
                self.watchers[component_id] = []
            self.watchers[component_id].append(callback)
    
    def _notify_watchers(self, component_id: str, old_state: Dict[str, Any], 
                        new_state: Dict[str, Any]) -> None:
        """Notify watchers of state changes"""
        if component_id in self.watchers:
            for callback in self.watchers[component_id]:
                try:
                    callback(component_id, old_state, new_state)
                except Exception as e:
                    print(f"Error in state watcher: {e}")
    
    def enable_persistence(self, path: str = "state", 
                          auto_save_interval: int = 300) -> None:
        """Enable persistent state storage"""
        self.persistence_enabled = True
        self.persistence_path = path
        self.auto_save_interval = auto_save_interval
        
        # Start auto-save thread
        self.save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.save_thread.start()
    
    def _auto_save_loop(self) -> None:
        """Auto-save state periodically"""
        import time
        import os
        
        while self.persistence_enabled:
            try:
                time.sleep(self.auto_save_interval)
                self.persist_to_disk()
            except Exception as e:
                print(f"Error in auto-save: {e}")
    
    def persist_to_disk(self) -> None:
        """Save state to disk"""
        import os
        
        if not self.persistence_enabled:
            return
        
        os.makedirs(self.persistence_path, exist_ok=True)
        
        with self._lock:
            # Save current states
            state_file = os.path.join(self.persistence_path, "current_state.json")
            with open(state_file, 'w') as f:
                serializable_states = {}
                for comp_id, comp_info in self.states.items():
                    serializable_states[comp_id] = {
                        'state': comp_info['state'],
                        'last_updated': comp_info['last_updated'].isoformat(),
                        'version': comp_info['version']
                    }
                json.dump(serializable_states, f, indent=2)
            
            # Save snapshots
            snapshots_file = os.path.join(self.persistence_path, "snapshots.json")
            with open(snapshots_file, 'w') as f:
                serializable_snapshots = {}
                for snap_id, snap_data in self.snapshots.items():
                    serializable_snapshots[snap_id] = {
                        'timestamp': snap_data['timestamp'].isoformat(),
                        'components': snap_data['components']
                    }
                json.dump(serializable_snapshots, f, indent=2)
    
    def load_from_disk(self) -> bool:
        """Load state from disk"""
        import os
        
        if not os.path.exists(self.persistence_path):
            return False
        
        try:
            # Load current states
            state_file = os.path.join(self.persistence_path, "current_state.json")
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    serializable_states = json.load(f)
                
                with self._lock:
                    for comp_id, comp_data in serializable_states.items():
                        if comp_id in self.states:
                            self.states[comp_id]['state'] = comp_data['state']
                            self.states[comp_id]['version'] = comp_data['version']
                            self.states[comp_id]['last_updated'] = datetime.fromisoformat(comp_data['last_updated'])
            
            # Load snapshots
            snapshots_file = os.path.join(self.persistence_path, "snapshots.json")
            if os.path.exists(snapshots_file):
                with open(snapshots_file, 'r') as f:
                    serializable_snapshots = json.load(f)
                
                with self._lock:
                    for snap_id, snap_data in serializable_snapshots.items():
                        self.snapshots[snap_id] = {
                            'timestamp': datetime.fromisoformat(snap_data['timestamp']),
                            'components': snap_data['components']
                        }
            
            return True
            
        except Exception as e:
            print(f"Error loading state from disk: {e}")
            return False
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        with self._lock:
            return {
                'component_count': len(self.states),
                'snapshot_count': len(self.snapshots),
                'history_entries': len(self.state_history),
                'persistence_enabled': self.persistence_enabled,
                'components': {
                    comp_id: {
                        'scope': info['scope'].value,
                        'version': info['version'],
                        'last_updated': info['last_updated'].isoformat(),
                        'state_size': len(str(info['state']))
                    }
                    for comp_id, info in self.states.items()
                }
            }

class StatefulCapability(Capability):
    """Adds state management to components"""
    
    def get_name(self) -> str:
        return "stateful"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_state_manager_ref'):
            component_id = spec.get('name', component.__class__.__name__)
            scope = StateScope(spec.get('state_scope', 'component'))
            
            # Add state management methods
            component._state_manager_ref = None  # Will be set during initialization
            component._component_id = component_id
            component._state_scope = scope
            
            # Add state methods if they don't exist
            if not hasattr(component, 'get_state'):
                component.get_state = lambda: self._get_component_state(component)
            
            if not hasattr(component, 'set_state'):
                component.set_state = lambda state: self._set_component_state(component, state)
            
            if not hasattr(component, 'reset_state'):
                component.reset_state = lambda: self._reset_component_state(component)
            
            # Enhance initialize method to register with state manager
            original_init = getattr(component, 'initialize', lambda ctx: None)
            def enhanced_init(ctx):
                state_manager = ctx.resolve("state_manager")
                component._state_manager_ref = state_manager
                state_manager.register_component(component._component_id, component, scope)
                original_init(ctx)
            component.initialize = enhanced_init
        
        return component
    
    def _get_component_state(self, component: Any) -> Dict[str, Any]:
        """Get component state"""
        if component._state_manager_ref:
            return component._state_manager_ref.load_state(component._component_id) or {}
        return {}
    
    def _set_component_state(self, component: Any, state: Dict[str, Any]) -> None:
        """Set component state"""
        if component._state_manager_ref:
            component._state_manager_ref.save_state(component._component_id, state)
    
    def _reset_component_state(self, component: Any) -> None:
        """Reset component state"""
        if component._state_manager_ref:
            component._state_manager_ref.save_state(component._component_id, {})
```

### 1.3 State Management Usage

```yaml
# Configuration
components:
  portfolio_manager:
    class: "Portfolio"
    capabilities: ["lifecycle", "events", "stateful"]
    state_scope: "session"  # Persist across component restarts
    
  trend_strategy:
    class: "TrendStrategy"
    capabilities: ["lifecycle", "events", "optimization", "stateful"]
    state_scope: "component"  # Component-specific state

system:
  state_management:
    persistence_enabled: true
    persistence_path: "./state"
    auto_save_interval: 300
    max_history_entries: 1000
    enable_snapshots: true
```

```python
# Usage example
class TrendStrategy:
    def __init__(self):
        self.signal_count = 0
        self.last_signal_time = None
    
    def on_bar(self, event):
        # Update internal state
        self.signal_count += 1
        self.last_signal_time = event.payload['timestamp']
        
        # Save state automatically
        if hasattr(self, 'set_state'):
            self.set_state({
                'signal_count': self.signal_count,
                'last_signal_time': self.last_signal_time.isoformat()
            })
    
    def reset(self):
        # Reset internal state
        self.signal_count = 0
        self.last_signal_time = None
        
        # Reset managed state
        if hasattr(self, 'reset_state'):
            self.reset_state()

# Manual state operations
state_manager = context.resolve("state_manager")

# Create system snapshot before risky operation
state_manager.create_snapshot("before_live_trading")

# Restore if something goes wrong
if error_occurred:
    state_manager.restore_snapshot("before_live_trading")

# Watch for critical state changes
def on_portfolio_change(comp_id, old_state, new_state):
    if new_state.get('total_value', 0) < old_state.get('total_value', 0) * 0.95:
        print("WARNING: Portfolio value dropped by more than 5%!")

state_manager.watch_state("portfolio_manager", on_portfolio_change)
```

---

## 2. System Visualization

### 2.1 Visualization Protocols

```python
@runtime_checkable
class Visualizable(Protocol):
    """Protocol for components that can be visualized"""
    
    @abstractmethod
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization"""
        ...
    
    @abstractmethod
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        ...

@runtime_checkable
class GraphNode(Protocol):
    """Protocol for components that can be nodes in a graph"""
    
    @abstractmethod
    def get_node_info(self) -> Dict[str, Any]:
        """Get node information for graph visualization"""
        ...
    
    @abstractmethod
    def get_connections(self) -> List[str]:
        """Get list of connected component IDs"""
        ...
```

### 2.2 System Visualizer Implementation

```python
import networkx as nx
from typing import List, Dict, Any, Optional
import json

class SystemVisualizer:
    """Generates visualizations of the trading system"""
    
    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.dependency_graph = nx.DiGraph()
        self.event_flow_graph = nx.DiGraph()
        self.state_graph = nx.DiGraph()
    
    def register_component(self, component_id: str, component: Any, 
                          metadata: Dict[str, Any] = None) -> None:
        """Register component for visualization"""
        self.components[component_id] = {
            'component': component,
            'metadata': metadata or {},
            'type': component.__class__.__name__,
            'capabilities': self._extract_capabilities(component),
            'connections': self._extract_connections(component)
        }
        
        # Add to graphs
        self._update_graphs(component_id, component)
    
    def _extract_capabilities(self, component: Any) -> List[str]:
        """Extract capabilities from component"""
        capabilities = []
        
        if hasattr(component, '_lifecycle'):
            capabilities.append('lifecycle')
        if hasattr(component, '_events'):
            capabilities.append('events')
        if hasattr(component, '_optimization'):
            capabilities.append('optimization')
        if hasattr(component, '_state_manager_ref'):
            capabilities.append('stateful')
        
        return capabilities
    
    def _extract_connections(self, component: Any) -> List[str]:
        """Extract component connections"""
        connections = []
        
        # Check for explicit dependencies
        if hasattr(component, '_dependencies'):
            connections.extend(component._dependencies)
        
        # Check for event subscriptions
        if hasattr(component, '_events') and component._events:
            if hasattr(component._events, 'subscriptions'):
                for event_type, handler in component._events.subscriptions:
                    connections.append(f"event:{event_type}")
        
        return connections
    
    def _update_graphs(self, component_id: str, component: Any) -> None:
        """Update visualization graphs"""
        # Add to dependency graph
        self.dependency_graph.add_node(component_id, 
                                     type=component.__class__.__name__,
                                     capabilities=self._extract_capabilities(component))
        
        # Add dependencies
        for connection in self._extract_connections(component):
            if not connection.startswith('event:'):
                self.dependency_graph.add_edge(component_id, connection)
        
        # Add to event flow graph
        self.event_flow_graph.add_node(component_id)
        
        # Add event connections
        if hasattr(component, '_events') and component._events:
            if hasattr(component._events, 'subscriptions'):
                for event_type, handler in component._events.subscriptions:
                    event_node = f"event:{event_type}"
                    if not self.event_flow_graph.has_node(event_node):
                        self.event_flow_graph.add_node(event_node, type='event')
                    self.event_flow_graph.add_edge(event_node, component_id)
    
    def generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview visualization data"""
        return {
            'component_count': len(self.components),
            'component_types': self._get_component_type_summary(),
            'capability_distribution': self._get_capability_distribution(),
            'dependency_complexity': self._analyze_dependency_complexity(),
            'event_flow_summary': self._analyze_event_flow()
        }
    
    def _get_component_type_summary(self) -> Dict[str, int]:
        """Get summary of component types"""
        type_counts = {}
        for comp_info in self.components.values():
            comp_type = comp_info['type']
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        return type_counts
    
    def _get_capability_distribution(self) -> Dict[str, int]:
        """Get distribution of capabilities"""
        capability_counts = {}
        for comp_info in self.components.values():
            for capability in comp_info['capabilities']:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        return capability_counts
    
    def _analyze_dependency_complexity(self) -> Dict[str, Any]:
        """Analyze dependency graph complexity"""
        return {
            'total_dependencies': self.dependency_graph.number_of_edges(),
            'avg_dependencies_per_component': self.dependency_graph.number_of_edges() / max(1, self.dependency_graph.number_of_nodes()),
            'circular_dependencies': list(nx.simple_cycles(self.dependency_graph)),
            'critical_components': self._find_critical_components()
        }
    
    def _find_critical_components(self) -> List[str]:
        """Find components that many others depend on"""
        in_degrees = dict(self.dependency_graph.in_degree())
        # Sort by in-degree (how many depend on this component)
        sorted_components = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        # Return top 5 most depended-upon components
        return [comp for comp, degree in sorted_components[:5] if degree > 0]
    
    def _analyze_event_flow(self) -> Dict[str, Any]:
        """Analyze event flow patterns"""
        event_nodes = [node for node in self.event_flow_graph.nodes() if node.startswith('event:')]
        
        event_stats = {}
        for event_node in event_nodes:
            subscribers = list(self.event_flow_graph.successors(event_node))
            event_stats[event_node] = {
                'subscriber_count': len(subscribers),
                'subscribers': subscribers
            }
        
        return {
            'total_event_types': len(event_nodes),
            'total_subscriptions': sum(len(stats['subscribers']) for stats in event_stats.values()),
            'event_details': event_stats
        }
    
    def export_to_graphviz(self, graph_type: str = "dependency") -> str:
        """Export graph to Graphviz format"""
        if graph_type == "dependency":
            graph = self.dependency_graph
            title = "Component Dependencies"
        elif graph_type == "event_flow":
            graph = self.event_flow_graph
            title = "Event Flow"
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        dot_lines = [f'digraph "{title}" {{']
        dot_lines.append('  rankdir=TB;')
        dot_lines.append('  node [shape=box, style=rounded];')
        
        # Add nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            if node.startswith('event:'):
                # Event nodes
                dot_lines.append(f'  "{node}" [shape=ellipse, color=blue, label="{node}"];')
            else:
                # Component nodes
                capabilities = node_data.get('capabilities', [])
                label = f"{node}\\n({', '.join(capabilities)})"
                color = self._get_node_color(node_data.get('type', ''))
                dot_lines.append(f'  "{node}" [color={color}, label="{label}"];')
        
        # Add edges
        for edge in graph.edges():
            dot_lines.append(f'  "{edge[0]}" -> "{edge[1]}";')
        
        dot_lines.append('}')
        return '\n'.join(dot_lines)
    
    def _get_node_color(self, component_type: str) -> str:
        """Get color for component type"""
        color_map = {
            'Strategy': 'green',
            'DataHandler': 'blue',
            'Portfolio': 'orange',
            'RiskManager': 'red',
            'Broker': 'purple',
            'default': 'gray'
        }
        return color_map.get(component_type, color_map['default'])
    
    def export_to_d3(self, graph_type: str = "dependency") -> Dict[str, Any]:
        """Export graph to D3.js format"""
        if graph_type == "dependency":
            graph = self.dependency_graph
        elif graph_type == "event_flow":
            graph = self.event_flow_graph
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        nodes = []
        links = []
        
        # Create node list with indices
        node_indices = {node: i for i, node in enumerate(graph.nodes())}
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            nodes.append({
                'id': node,
                'name': node,
                'type': node_data.get('type', 'unknown'),
                'capabilities': node_data.get('capabilities', []),
                'group': 1 if node.startswith('event:') else 2
            })
        
        for edge in graph.edges():
            links.append({
                'source': node_indices[edge[0]],
                'target': node_indices[edge[1]],
                'value': 1
            })
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def generate_html_report(self) -> str:
        """Generate HTML visualization report"""
        overview = self.generate_system_overview()
        dependency_d3 = self.export_to_d3("dependency")
        event_flow_d3 = self.export_to_d3("event_flow")
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ADMF-Trader System Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .overview {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
        .graph-container {{ border: 1px solid #ddd; margin: 20px 0; }}
        .graph {{ width: 100%; height: 400px; }}
    </style>
</head>
<body>
    <h1>ADMF-Trader System Visualization</h1>
    
    <div class="overview">
        <h2>System Overview</h2>
        <p><strong>Components:</strong> {overview['component_count']}</p>
        <p><strong>Component Types:</strong> {', '.join(f"{k}: {v}" for k, v in overview['component_types'].items())}</p>
        <p><strong>Critical Components:</strong> {', '.join(overview['dependency_complexity']['critical_components'])}</p>
    </div>
    
    <div class="graph-container">
        <h2>Component Dependencies</h2>
        <div id="dependency-graph" class="graph"></div>
    </div>
    
    <div class="graph-container">
        <h2>Event Flow</h2>
        <div id="event-flow-graph" class="graph"></div>
    </div>
    
    <script>
        // Dependency graph data
        const dependencyData = {json.dumps(dependency_d3)};
        
        // Event flow graph data
        const eventFlowData = {json.dumps(event_flow_d3)};
        
        // Create visualizations
        createForceGraph("#dependency-graph", dependencyData);
        createForceGraph("#event-flow-graph", eventFlowData);
        
        function createForceGraph(selector, data) {{
            const width = 800;
            const height = 400;
            
            const svg = d3.select(selector)
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));
            
            const link = svg.append("g")
                .selectAll("line")
                .data(data.links)
                .enter().append("line")
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.6);
            
            const node = svg.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .enter().append("circle")
                .attr("r", 8)
                .attr("fill", d => d.group === 1 ? "#ff7f0e" : "#1f77b4")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            node.append("title")
                .text(d => d.name);
            
            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            }});
            
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
        }}
    </script>
</body>
</html>
        """
        
        return html_template

class VisualizableCapability(Capability):
    """Adds visualization support to components"""
    
    def get_name(self) -> str:
        return "visualizable"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'get_visualization_data'):
            component.get_visualization_data = lambda: self._get_default_visualization_data(component)
        
        if not hasattr(component, 'get_visualization_config'):
            component.get_visualization_config = lambda: self._get_default_visualization_config(component, spec)
        
        return component
    
    def _get_default_visualization_data(self, component: Any) -> Dict[str, Any]:
        """Get default visualization data"""
        data = {
            'type': component.__class__.__name__,
            'status': 'unknown'
        }
        
        # Add state if available
        if hasattr(component, 'get_state'):
            data['state'] = component.get_state()
        
        # Add lifecycle status
        if hasattr(component, '_lifecycle'):
            data['status'] = component._lifecycle.state.value
        
        return data
    
    def _get_default_visualization_config(self, component: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Get default visualization config"""
        return {
            'display_name': spec.get('name', component.__class__.__name__),
            'color': self._get_component_color(component.__class__.__name__),
            'shape': 'box',
            'size': 'medium'
        }
    
    def _get_component_color(self, class_name: str) -> str:
        """Get color for component class"""
        color_map = {
            'Strategy': '#2ecc71',      # Green
            'DataHandler': '#3498db',   # Blue
            'Portfolio': '#f39c12',     # Orange
            'RiskManager': '#e74c3c',   # Red
            'Broker': '#9b59b6',        # Purple
            'default': '#95a5a6'        # Gray
        }
        return color_map.get(class_name, color_map['default'])
```

---

## 3. Health Monitoring

### 3.1 Health Check Protocols

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import time
import threading
import psutil
from datetime import datetime, timedelta

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    check_duration_ms: float = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}

@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for components that can perform health checks"""
    
    @abstractmethod
    def health_check(self) -> HealthCheckResult:
        """Perform health check and return result"""
        ...
    
    @abstractmethod
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics"""
        ...

class HealthCheck(ABC):
    """Base class for health checks"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.last_result: Optional[HealthCheckResult] = None
        self.enabled = True
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check"""
        ...
    
    def run(self) -> HealthCheckResult:
        """Run the health check with timing"""
        if not self.enabled:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{self.name}' is disabled"
            )
        
        start_time = time.time()
        try:
            result = self.check()
            result.check_duration_ms = (time.time() - start_time) * 1000
            self.last_result = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Health check '{self.name}' failed: {str(e)}",
                check_duration_ms=(time.time() - start_time) * 1000
            )
            self.last_result = result
            return result
```

### 3.2 Specific Health Checks

```python
class ComponentHealthCheck(HealthCheck):
    """Health check for individual components"""
    
    def __init__(self, component_id: str, component: Any):
        super().__init__(f"component_{component_id}", f"Health check for {component_id}")
        self.component_id = component_id
        self.component = component
    
    def check(self) -> HealthCheckResult:
        """Check component health"""
        issues = []
        details = {}
        
        # Check lifecycle state
        if hasattr(self.component, '_lifecycle'):
            lifecycle = self.component._lifecycle
            details['lifecycle_state'] = lifecycle.state.value
            details['initialized'] = lifecycle.initialized
            details['running'] = lifecycle.running
            
            if not lifecycle.initialized:
                issues.append("Component not initialized")
            elif lifecycle.state == ComponentState.DISPOSED:
                issues.append("Component has been disposed")
        
        # Check if component is responsive
        if hasattr(self.component, 'health_check'):
            try:
                component_result = self.component.health_check()
                details['component_health'] = component_result.__dict__
                if component_result.status != HealthStatus.HEALTHY:
                    issues.append(f"Component reports {component_result.status.value}: {component_result.message}")
            except Exception as e:
                issues.append(f"Component health check failed: {str(e)}")
        
        # Check memory usage if trackable
        if hasattr(self.component, 'get_memory_usage'):
            try:
                memory_usage = self.component.get_memory_usage()
                details['memory_usage_mb'] = memory_usage
                if memory_usage > 100:  # 100MB threshold
                    issues.append(f"High memory usage: {memory_usage:.1f}MB")
            except Exception as e:
                details['memory_check_error'] = str(e)
        
        # Determine overall status
        if not issues:
            status = HealthStatus.HEALTHY
            message = "Component is healthy"
        elif any("not initialized" in issue or "disposed" in issue for issue in issues):
            status = HealthStatus.CRITICAL
            message = f"Critical issues: {'; '.join(issues)}"
        else:
            status = HealthStatus.WARNING
            message = f"Warning issues: {'; '.join(issues)}"
        
        return HealthCheckResult(
            status=status,
            message=message,
            details=details
        )

class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources"""
    
    def __init__(self):
        super().__init__("system_resources", "System CPU, memory, and disk health")
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.disk_threshold = 90.0  # %
    
    def check(self) -> HealthCheckResult:
        """Check system resource health"""
        issues = []
        details = {}
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            details['cpu_percent'] = cpu_percent
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            details['memory_percent'] = memory.percent
            details['memory_available_gb'] = memory.available / (1024**3)
            if memory.percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            details['disk_percent'] = disk_percent
            details['disk_free_gb'] = disk.free / (1024**3)
            if disk_percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            # Process count
            process_count = len(psutil.pids())
            details['process_count'] = process_count
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Failed to get system resources: {str(e)}"
            )
        
        # Determine status
        if not issues:
            status = HealthStatus.HEALTHY
            message = "System resources are healthy"
        elif len(issues) == 1:
            status = HealthStatus.WARNING
            message = issues[0]
        else:
            status = HealthStatus.CRITICAL
            message = f"Multiple resource issues: {'; '.join(issues)}"
        
        return HealthCheckResult(
            status=status,
            message=message,
            details=details
        )

class EventSystemHealthCheck(HealthCheck):
    """Health check for event system"""
    
    def __init__(self, event_bus: Any):
        super().__init__("event_system", "Event bus and messaging health")
        self.event_bus = event_bus
        self.max_queue_size = 1000
        self.max_processing_time_ms = 100
    
    def check(self) -> HealthCheckResult:
        """Check event system health"""
        issues = []
        details = {}
        
        try:
            # Check if event bus is responsive
            test_event = Event("HEALTH_CHECK", {"test": True})
            start_time = time.time()
            
            # Try to publish test event
            if hasattr(self.event_bus, 'publish'):
                self.event_bus.publish(test_event)
                publish_time = (time.time() - start_time) * 1000
                details['publish_time_ms'] = publish_time
                
                if publish_time > self.max_processing_time_ms:
                    issues.append(f"Slow event publishing: {publish_time:.1f}ms")
            
            # Check queue sizes if available
            if hasattr(self.event_bus, 'get_queue_sizes'):
                queue_sizes = self.event_bus.get_queue_sizes()
                details['queue_sizes'] = queue_sizes
                
                for queue_name, size in queue_sizes.items():
                    if size > self.max_queue_size:
                        issues.append(f"Large event queue '{queue_name}': {size} events")
            
            # Check subscriber counts
            if hasattr(self.event_bus, 'get_subscriber_counts'):
                subscriber_counts = self.event_bus.get_subscriber_counts()
                details['subscriber_counts'] = subscriber_counts
                details['total_subscribers'] = sum(subscriber_counts.values())
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Event system check failed: {str(e)}"
            )
        
        # Determine status
        if not issues:
            status = HealthStatus.HEALTHY
            message = "Event system is healthy"
        else:
            status = HealthStatus.WARNING if len(issues) == 1 else HealthStatus.CRITICAL
            message = '; '.join(issues)
        
        return HealthCheckResult(
            status=status,
            message=message,
            details=details
        )

class TradingSystemHealthCheck(HealthCheck):
    """Trading-specific health check"""
    
    def __init__(self, portfolio: Any, data_handler: Any):
        super().__init__("trading_system", "Trading system operational health")
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.max_data_lag_minutes = 5
        self.max_drawdown_threshold = 0.20  # 20%
    
    def check(self) -> HealthCheckResult:
        """Check trading system health"""
        issues = []
        details = {}
        
        try:
            # Check data freshness
            if hasattr(self.data_handler, 'get_last_update_time'):
                last_update = self.data_handler.get_last_update_time()
                if last_update:
                    data_lag = (datetime.now() - last_update).total_seconds() / 60
                    details['data_lag_minutes'] = data_lag
                    
                    if data_lag > self.max_data_lag_minutes:
                        issues.append(f"Stale market data: {data_lag:.1f} minutes old")
            
            # Check portfolio health
            if self.portfolio:
                portfolio_value = self.portfolio.get_portfolio_value()
                details['portfolio_value'] = portfolio_value
                
                # Check for excessive drawdown
                if hasattr(self.portfolio, 'get_peak_equity'):
                    peak_equity = self.portfolio.get_peak_equity()
                    if peak_equity > 0:
                        drawdown = (peak_equity - portfolio_value) / peak_equity
                        details['current_drawdown'] = drawdown
                        
                        if drawdown > self.max_drawdown_threshold:
                            issues.append(f"Excessive drawdown: {drawdown:.1%}")
                
                # Check for negative portfolio value
                if portfolio_value <= 0:
                    issues.append("Portfolio value is zero or negative")
            
            # Check for recent trading activity
            if hasattr(self.portfolio, 'get_recent_trade_count'):
                recent_trades = self.portfolio.get_recent_trade_count(hours=24)
                details['trades_last_24h'] = recent_trades
                
                # This could be either good or bad depending on strategy
                if recent_trades == 0:
                    details['no_recent_trades'] = True
        
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Trading system check failed: {str(e)}"
            )
        
        # Determine status
        critical_issues = [issue for issue in issues if "zero or negative" in issue or "Excessive drawdown" in issue]
        
        if critical_issues:
            status = HealthStatus.CRITICAL
            message = '; '.join(critical_issues)
        elif issues:
            status = HealthStatus.WARNING
            message = '; '.join(issues)
        else:
            status = HealthStatus.HEALTHY
            message = "Trading system is healthy"
        
        return HealthCheckResult(
            status=status,
            message=message,
            details=details
        )
```

### 3.3 Health Monitor Implementation

```python
class HealthMonitor:
    """Central health monitoring system"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_enabled = False
        self.check_interval = 30  # seconds
        self.max_history = 1000
        self.alert_callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check"""
        with self._lock:
            self.health_checks[health_check.name] = health_check
    
    def register_component_health_check(self, component_id: str, component: Any) -> None:
        """Register health check for a component"""
        health_check = ComponentHealthCheck(component_id, component)
        self.register_health_check(health_check)
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous health monitoring"""
        self.check_interval = interval
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                results = self.run_all_checks()
                self._process_results(results)
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        with self._lock:
            for name, health_check in self.health_checks.items():
                try:
                    result = health_check.run()
                    results[name] = result
                except Exception as e:
                    results[name] = HealthCheckResult(
                        status=HealthStatus.CRITICAL,
                        message=f"Health check '{name}' threw exception: {str(e)}"
                    )
        
        return results
    
    def _process_results(self, results: Dict[str, HealthCheckResult]) -> None:
        """Process health check results"""
        timestamp = datetime.now()
        
        # Store in history
        history_entry = {
            'timestamp': timestamp,
            'results': {name: result.__dict__ for name, result in results.items()},
            'overall_status': self._calculate_overall_status(results)
        }
        
        with self._lock:
            self.health_history.append(history_entry)
            
            # Limit history size
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)
        
        # Check for alerts
        for name, result in results.items():
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._trigger_alert(name, result)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                for name, result in results.items():
                    if result.status != HealthStatus.HEALTHY:
                        callback(result)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def _calculate_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Calculate overall system health status"""
        if not results:
            return HealthStatus.UNKNOWN.value
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL.value
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING.value
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY.value
        else:
            return HealthStatus.WARNING.value
    
    def _trigger_alert(self, check_name: str, result: HealthCheckResult) -> None:
        """Trigger alert for unhealthy check"""
        alert = {
            'timestamp': datetime.now(),
            'check_name': check_name,
            'status': result.status.value,
            'message': result.message,
            'details': result.details
        }
        
        with self._lock:
            self.alerts.append(alert)
            
            # Limit alerts history
            if len(self.alerts) > 100:
                self.alerts.pop(0)
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        results = self.run_all_checks()
        
        return {
            'timestamp': datetime.now(),
            'overall_status': self._calculate_overall_status(results),
            'check_results': {name: result.__dict__ for name, result in results.items()},
            'summary': self._get_health_summary(results)
        }
    
    def _get_health_summary(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Get health summary statistics"""
        status_counts = {}
        total_checks = len(results)
        
        for result in results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_checks': total_checks,
            'healthy_checks': status_counts.get('healthy', 0),
            'warning_checks': status_counts.get('warning', 0),
            'critical_checks': status_counts.get('critical', 0),
            'unknown_checks': status_counts.get('unknown', 0),
            'health_percentage': (status_counts.get('healthy', 0) / max(1, total_checks)) * 100
        }
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_history = [
                entry for entry in self.health_history
                if entry['timestamp'] > cutoff_time
            ]
        
        if not recent_history:
            return {'message': 'No recent health data available'}
        
        # Calculate trends
        health_percentages = []
        timestamps = []
        
        for entry in recent_history:
            timestamps.append(entry['timestamp'])
            
            total_checks = len(entry['results'])
            healthy_checks = sum(1 for result in entry['results'].values() 
                               if result['status'] == 'healthy')
            
            health_percentage = (healthy_checks / max(1, total_checks)) * 100
            health_percentages.append(health_percentage)
        
        return {
            'period_hours': hours,
            'data_points': len(recent_history),
            'current_health_percentage': health_percentages[-1] if health_percentages else 0,
            'average_health_percentage': sum(health_percentages) / len(health_percentages) if health_percentages else 0,
            'min_health_percentage': min(health_percentages) if health_percentages else 0,
            'max_health_percentage': max(health_percentages) if health_percentages else 0,
            'trend_data': list(zip([t.isoformat() for t in timestamps], health_percentages))
        }

class HealthMonitoringCapability(Capability):
    """Adds health monitoring to components"""
    
    def get_name(self) -> str:
        return "health_monitoring"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'health_check'):
            component.health_check = lambda: self._default_health_check(component)
        
        if not hasattr(component, 'get_health_metrics'):
            component.get_health_metrics = lambda: self._default_health_metrics(component)
        
        # Register with health monitor during initialization
        original_init = getattr(component, 'initialize', lambda ctx: None)
        def enhanced_init(ctx):
            health_monitor = ctx.resolve("health_monitor")
            component_id = spec.get('name', component.__class__.__name__)
            health_monitor.register_component_health_check(component_id, component)
            original_init(ctx)
        component.initialize = enhanced_init
        
        return component
    
    def _default_health_check(self, component: Any) -> HealthCheckResult:
        """Default health check implementation"""
        try:
            # Basic checks
            if hasattr(component, '_lifecycle'):
                if not component._lifecycle.initialized:
                    return HealthCheckResult(
                        status=HealthStatus.CRITICAL,
                        message="Component not initialized"
                    )
                
                if component._lifecycle.state == ComponentState.DISPOSED:
                    return HealthCheckResult(
                        status=HealthStatus.CRITICAL,
                        message="Component has been disposed"
                    )
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Component is healthy"
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}"
            )
    
    def _default_health_metrics(self, component: Any) -> Dict[str, Any]:
        """Default health metrics"""
        metrics = {
            'type': component.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        if hasattr(component, '_lifecycle'):
            metrics['lifecycle_state'] = component._lifecycle.state.value
            metrics['initialized'] = component._lifecycle.initialized
            metrics['running'] = component._lifecycle.running
        
        return metrics
```

### 3.4 Health Monitoring Configuration

```yaml
# Configuration
components:
  health_monitor:
    class: "HealthMonitor"
    capabilities: ["lifecycle"]
    params:
      check_interval: 30
      max_history: 1000
      enable_alerts: true
    
  portfolio_manager:
    class: "Portfolio"
    capabilities: ["lifecycle", "events", "health_monitoring"]
    
  trend_strategy:
    class: "TrendStrategy"
    capabilities: ["lifecycle", "events", "health_monitoring"]

system:
  health_monitoring:
    enabled: true
    check_interval: 30
    alert_thresholds:
      cpu_percent: 80
      memory_percent: 85
      disk_percent: 90
    trading_thresholds:
      max_drawdown: 0.20
      max_data_lag_minutes: 5
    
    # Alert destinations
    alerts:
      email:
        enabled: true
        recipients: ["trader@example.com"]
        smtp_server: "smtp.example.com"
      
      webhook:
        enabled: true
        url: "https://api.example.com/alerts"
        
      slack:
        enabled: false
        webhook_url: "https://hooks.slack.com/..."
```

### 3.5 Usage Examples

```python
# Set up health monitoring
health_monitor = HealthMonitor()

# Register system-level health checks
health_monitor.register_health_check(SystemResourcesHealthCheck())
health_monitor.register_health_check(EventSystemHealthCheck(event_bus))

# Register trading-specific health checks
if portfolio and data_handler:
    health_monitor.register_health_check(
        TradingSystemHealthCheck(portfolio, data_handler)
    )

# Add alert callback
def alert_handler(result: HealthCheckResult):
    if result.status == HealthStatus.CRITICAL:
        print(f"CRITICAL ALERT: {result.message}")
        # Send email, webhook, etc.

health_monitor.add_alert_callback(alert_handler)

# Start monitoring
health_monitor.start_monitoring(interval=30)

# Get current status
status = health_monitor.get_current_health_status()
print(f"System health: {status['overall_status']}")

# Get health trends
trends = health_monitor.get_health_trends(hours=24)
print(f"Average health over 24h: {trends['average_health_percentage']:.1f}%")
```

This completes the documentation for the five missing critical features. Each section provides comprehensive protocols, implementations, and usage examples that integrate seamlessly with the Protocol + Composition architecture while addressing production-critical needs for state management, visualization, health monitoring, caching, and component discovery.
