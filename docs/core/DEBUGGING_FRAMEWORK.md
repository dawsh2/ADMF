# Debugging Framework

This document outlines the design for the ADMF-Trader Debugging Framework, which provides comprehensive tools for inspecting, tracing, visualizing, and troubleshooting the trading system.

## 1. Overview

The Debugging Framework provides a suite of tools to help developers understand system behavior, identify issues, and improve the reliability of the ADMF-Trader system. The framework focuses on four key capabilities:

1. **Execution Tracing**: Capture and analyze the flow of events through the system, especially the signal → order → fill path
2. **State Inspection**: Examine component states during execution and detect state changes
3. **Event Recording/Replay**: Record system events and replay them for debugging
4. **Visualization**: Generate visual representations of system state and event flow

## 2. Core Architecture

The Debugging Framework consists of five primary components:

```
DebuggingFramework
  ├── ExecutionTracer
  ├── StateInspector
  ├── EventRecorder
  ├── DebugVisualizer
  └── DebugManager
```

### 2.1 ExecutionTracer

The ExecutionTracer tracks the flow of events through the system, focusing particularly on the signal → order → fill path.

```python
class ExecutionTracer(Component):
    def __init__(self, name="execution_tracer", config=None):
        super().__init__(name, config)
        self.traces = {}                # Active traces by ID
        self.active_trace_ids = set()   # Currently active trace IDs
        self.trace_levels = {
            "detail": 3,    # Full details with all events and state changes
            "standard": 2,  # Standard level with key events only
            "minimal": 1    # Minimal info for high-level overview
        }
        self.current_level = self.get_parameter("trace_level", "standard")
        
    def start_trace(self, trace_id=None, description=""):
        """Start a new execution trace with optional custom ID."""
        trace_id = trace_id or f"trace_{uuid.uuid4()}"
        self.traces[trace_id] = {
            "id": trace_id,
            "description": description,
            "start_time": time.time(),
            "events": [],
            "state_snapshots": {},
            "completed": False
        }
        self.active_trace_ids.add(trace_id)
        return trace_id
        
    def end_trace(self, trace_id):
        """Complete and finalize a trace."""
        if trace_id in self.active_trace_ids:
            self.traces[trace_id]["completed"] = True
            self.traces[trace_id]["end_time"] = time.time()
            self.active_trace_ids.remove(trace_id)
            
    def on_event(self, event):
        """Record an event in all active traces."""
        for trace_id in self.active_trace_ids:
            self.record_event(trace_id, event)
            
    def record_event(self, trace_id, event):
        """Record an event in a specific trace."""
        if trace_id not in self.traces:
            return
            
        trace_level = self.trace_levels.get(self.current_level, 2)
        
        # For all levels, capture key events
        if event.get_type() in [EventType.SIGNAL, EventType.ORDER, EventType.FILL]:
            self.traces[trace_id]["events"].append({
                "timestamp": time.time(),
                "event_type": event.get_type().value,
                "event_data": event.get_data(),
                "source": event.get_metadata().get("source", "unknown")
            })
```

Key capabilities:
- Start and end execution traces with user-defined descriptions
- Record events at configurable detail levels (minimal, standard, detail)
- Focus on tracking the signal → order → fill path
- Capture state snapshots at key points in execution
- Query and analyze captured traces

### 2.2 StateInspector

The StateInspector provides capabilities to examine component states during execution and detect changes.

```python
class StateInspector(Component):
    def __init__(self, name="state_inspector", config=None):
        super().__init__(name, config)
        self.snapshots = {}             # State snapshots by ID
        self.watch_points = {}          # Configured watchpoints
        self.component_registry = {}    # Registered components for inspection
        self.history_retention = self.get_parameter("history_retention", 10)
        
    def capture_snapshot(self, component_name=None):
        """Capture a snapshot of component state."""
        timestamp = time.time()
        snapshot = {
            "timestamp": timestamp,
            "components": {}
        }
        
        # If specific component requested, snapshot just that one
        if component_name:
            if component_name in self.component_registry:
                component = self.component_registry[component_name]
                try:
                    snapshot["components"][component_name] = component._get_debug_state()
                except Exception as e:
                    snapshot["components"][component_name] = f"ERROR: {str(e)}"
        else:
            # Snapshot all registered components
            for name, component in self.component_registry.items():
                try:
                    snapshot["components"][name] = component._get_debug_state()
                except Exception as e:
                    snapshot["components"][name] = f"ERROR: {str(e)}"
                    
        # Store snapshot
        snapshot_id = f"snapshot_{uuid.uuid4()}"
        self.snapshots[snapshot_id] = snapshot
        
        # Clean up old snapshots if needed
        self._cleanup_old_snapshots()
        
        return snapshot_id
        
    def compare_snapshots(self, snapshot_id1, snapshot_id2):
        """Compare two snapshots and return the differences."""
        if snapshot_id1 not in self.snapshots or snapshot_id2 not in self.snapshots:
            return None
            
        snapshot1 = self.snapshots[snapshot_id1]
        snapshot2 = self.snapshots[snapshot_id2]
        
        differences = {
            "timestamp1": snapshot1["timestamp"],
            "timestamp2": snapshot2["timestamp"],
            "components": {}
        }
        
        # Compare each component that exists in both snapshots
        all_components = set(snapshot1["components"].keys()) | set(snapshot2["components"].keys())
        
        for component_name in all_components:
            if component_name not in snapshot1["components"]:
                differences["components"][component_name] = "Added in snapshot2"
            elif component_name not in snapshot2["components"]:
                differences["components"][component_name] = "Removed in snapshot2"
            else:
                # Compare attributes
                comp_diff = self._compare_component_states(
                    snapshot1["components"][component_name],
                    snapshot2["components"][component_name]
                )
                if comp_diff:
                    differences["components"][component_name] = comp_diff
                    
        return differences
```

Key capabilities:
- Capture state snapshots of components during execution
- Register components for state inspection
- Add watchpoints to monitor specific attributes
- Compare snapshots to detect state changes
- Track history of component states

### 2.3 EventRecorder

The EventRecorder captures events for later analysis and supports event replay for debugging.

```python
class EventRecorder(Component):
    def __init__(self, name="event_recorder", config=None):
        super().__init__(name, config)
        self.recordings = {}            # Stored recordings by ID
        self.active_recording = None    # Currently active recording ID
        self.record_all_events = self.get_parameter("record_all_events", False)
        self.filtered_event_types = set(self.get_parameter("event_types", [
            EventType.SIGNAL.value,
            EventType.ORDER.value,
            EventType.FILL.value
        ]))
        
    def start_recording(self, recording_id=None, description=""):
        """Start a new event recording."""
        recording_id = recording_id or f"recording_{uuid.uuid4()}"
        
        self.recordings[recording_id] = {
            "id": recording_id,
            "description": description,
            "start_time": time.time(),
            "events": [],
            "completed": False
        }
        
        self.active_recording = recording_id
        return recording_id
        
    def stop_recording(self):
        """Stop the active recording."""
        if self.active_recording:
            self.recordings[self.active_recording]["completed"] = True
            self.recordings[self.active_recording]["end_time"] = time.time()
            recording_id = self.active_recording
            self.active_recording = None
            return recording_id
        return None
        
    def on_event(self, event):
        """Record an event if recording is active."""
        if not self.active_recording:
            return
            
        # Only record if it's in our filtered list or we're recording all
        event_type = event.get_type().value
        if self.record_all_events or event_type in self.filtered_event_types:
            # Store event with timestamp
            self.recordings[self.active_recording]["events"].append({
                "timestamp": time.time(),
                "event_type": event_type,
                "event_data": event.get_data(),
                "metadata": event.get_metadata()
            })
            
    def replay_recording(self, recording_id, replay_mode="simulated"):
        """Replay a recording through the event bus."""
        if recording_id not in self.recordings:
            return False
            
        recording = self.recordings[recording_id]
        
        if replay_mode == "simulated":
            # Simulate timing between events
            last_timestamp = recording["events"][0]["timestamp"] if recording["events"] else None
            
            for event_record in recording["events"]:
                # Calculate delay to match original timing
                if last_timestamp:
                    delay = event_record["timestamp"] - last_timestamp
                    if delay > 0:
                        time.sleep(delay)
                        
                # Create and publish event
                event_type = event_record["event_type"]
                event_data = event_record["event_data"]
                metadata = event_record["metadata"]
                
                event = Event(event_type, event_data, metadata)
                self.event_bus.publish(event)
                
                last_timestamp = event_record["timestamp"]
                
        elif replay_mode == "immediate":
            # Replay all events immediately
            for event_record in recording["events"]:
                event_type = event_record["event_type"]
                event_data = event_record["event_data"]
                metadata = event_record["metadata"]
                
                event = Event(event_type, event_data, metadata)
                self.event_bus.publish(event)
                
        return True
```

Key capabilities:
- Record events with configurable filtering
- Save and load recordings from disk
- Replay events in simulated time or immediately
- Filter events by type
- Capture complete event data and metadata

### 2.4 DebugVisualizer

The DebugVisualizer generates visual representations of system state and event flow.

```python
class DebugVisualizer(Component):
    def __init__(self, name="debug_visualizer", config=None):
        super().__init__(name, config)
        self.data_sources = {}  # Registered data sources
        self.output_dir = self.get_parameter("output_dir", "./debug_output")
        self.format = self.get_parameter("format", "html")
        
    def generate_event_flow_diagram(self, trace_id, output_path=None):
        """Generate a visualization of event flow from a trace."""
        # Get the execution tracer from the context
        execution_tracer = self._get_execution_tracer()
        if not execution_tracer:
            return None
            
        trace = execution_tracer.get_trace(trace_id)
        if not trace:
            return None
            
        # Generate diagram using events in the trace
        if self.format == "html":
            return self._generate_html_event_flow(trace, output_path)
        elif self.format == "json":
            return self._generate_json_event_flow(trace, output_path)
        else:
            return None
            
    def generate_component_state_diagram(self, snapshot_id, output_path=None):
        """Generate a visualization of component state."""
        # Get the state inspector from the context
        state_inspector = self._get_state_inspector()
        if not state_inspector:
            return None
            
        snapshot = state_inspector.get_snapshot(snapshot_id)
        if not snapshot:
            return None
            
        # Generate diagram using component states in the snapshot
        if self.format == "html":
            return self._generate_html_state_diagram(snapshot, output_path)
        elif self.format == "json":
            return self._generate_json_state_diagram(snapshot, output_path)
        else:
            return None
```

Key capabilities:
- Generate visual representations of event flows
- Create component state diagrams
- Support multiple output formats (HTML, JSON)
- Export visualizations to disk
- Integrate with other debugging components

### 2.5 DebugManager

The DebugManager coordinates debugging activities and provides a central interface.

```python
class DebugManager(Component):
    def __init__(self, name="debug_manager", config=None):
        super().__init__(name, config)
        self.debug_mode = self.get_parameter("debug_mode", False)
        self.active_components = {}     # Active debugging components
        self.breakpoints = {}           # Configured breakpoints
        self.active_trace_id = None     # Currently active trace
        
    def start_debugging_session(self, description=""):
        """Start a debugging session with coordinated component activation."""
        if not self.debug_mode:
            return None
            
        # Start Execution Tracer
        if "execution_tracer" in self.active_components:
            self.active_trace_id = self.active_components["execution_tracer"].start_trace(
                description=description
            )
            
        # Start Event Recording
        if "event_recorder" in self.active_components:
            self.active_components["event_recorder"].start_recording(
                description=description
            )
            
        # Capture initial state snapshot
        if "state_inspector" in self.active_components:
            self.active_components["state_inspector"].capture_snapshot()
            
        return self.active_trace_id
        
    def end_debugging_session(self):
        """End the current debugging session."""
        if not self.debug_mode or not self.active_trace_id:
            return None
            
        result = {
            "trace_id": self.active_trace_id,
            "recording_id": None,
            "final_snapshot_id": None
        }
        
        # End Execution Tracing
        if "execution_tracer" in self.active_components:
            self.active_components["execution_tracer"].end_trace(self.active_trace_id)
            
        # End Event Recording
        if "event_recorder" in self.active_components:
            result["recording_id"] = self.active_components["event_recorder"].stop_recording()
            
        # Capture final state snapshot
        if "state_inspector" in self.active_components:
            result["final_snapshot_id"] = self.active_components["state_inspector"].capture_snapshot()
            
        temp_id = self.active_trace_id
        self.active_trace_id = None
        
        return result
        
    def set_breakpoint(self, event_type, condition=None):
        """Set a breakpoint to pause execution when a condition is met."""
        breakpoint_id = f"bp_{uuid.uuid4()}"
        
        self.breakpoints[breakpoint_id] = {
            "event_type": event_type,
            "condition": condition,
            "enabled": True
        }
        
        # Register event handler for this breakpoint
        self.event_bus.subscribe(event_type, lambda e: self._check_breakpoint(breakpoint_id, e))
        
        return breakpoint_id
        
    def generate_debug_report(self, trace_id, output_path=None):
        """Generate a comprehensive debug report."""
        if not self.debug_mode:
            return None
            
        # Get components
        tracer = self.active_components.get("execution_tracer")
        inspector = self.active_components.get("state_inspector")
        visualizer = self.active_components.get("debug_visualizer")
        
        if not tracer or not visualizer:
            return None
            
        # Get trace data
        trace = tracer.get_trace(trace_id)
        if not trace:
            return None
            
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(visualizer.output_dir, f"debug_report_{timestamp}.html")
            
        # Create and save the report
        # (Implementation details omitted for brevity)
        
        return output_path
```

Key capabilities:
- Start and end debugging sessions
- Coordinate multiple debugging components
- Set and manage breakpoints
- Generate comprehensive debug reports
- Control debug mode globally

## 3. Integration with Existing Architecture

### 3.1 Component Lifecycle Integration

The debugging framework integrates with the existing component lifecycle to ensure proper initialization, operation, and cleanup:

```python
# Example of integration with BacktestCoordinator
class BacktestCoordinator(Component):
    def initialize(self, context):
        # Existing initialization code...
        
        # Initialize debug manager if present
        if context.has("debug_manager"):
            self.debug_manager = context.get("debug_manager")
            self.debug_mode = self.debug_manager.debug_mode
        else:
            self.debug_manager = None
            self.debug_mode = False
            
    def run(self):
        # Start debug session if in debug mode
        trace_id = None
        if self.debug_mode and self.debug_manager:
            trace_id = self.debug_manager.start_debugging_session(
                description=f"Backtest run: {self.config.get('name', 'unnamed')}"
            )
            
        # Existing backtest run code...
        
        # End debug session if active
        if trace_id:
            self.debug_manager.end_debugging_session()
            
        # Generate debug report if requested
        if trace_id and self.config.get("generate_debug_report", False):
            report_path = self.debug_manager.generate_debug_report(trace_id)
            print(f"Debug report generated: {report_path}")
```

### 3.2 EventBus Integration

The debugging framework hooks into the EventBus to capture and analyze events:

```python
# Enhanced EventBus with debugging support
class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.debug_mode = False
        self.debug_hooks = []
        
    def set_debug_mode(self, debug_mode):
        """Enable or disable debug mode."""
        self.debug_mode = debug_mode
        
    def register_debug_hook(self, hook):
        """Register a function to be called for each event in debug mode."""
        self.debug_hooks.append(hook)
        
    def publish(self, event):
        """Publish an event to all subscribers."""
        event_type = event.get_type()
        
        # Call debug hooks if in debug mode
        if self.debug_mode:
            for hook in self.debug_hooks:
                try:
                    hook(event)
                except Exception as e:
                    # Log error but continue
                    print(f"Debug hook error: {str(e)}")
                    
        # Normal event delivery
        # ... existing publication code ...
```

## 4. Use Cases

### 4.1 Tracing Signal → Order → Fill Path

```python
# Example usage to trace a specific signal path
def debug_signal_path(strategy, signal_id):
    """Debug the flow of a specific signal through the system."""
    # Get debug components
    debug_mgr = container.get("debug_manager")
    tracer = container.get("execution_tracer")
    
    # Start tracing session
    trace_id = debug_mgr.start_debugging_session(f"Signal flow: {signal_id}")
    
    # Run strategy
    strategy.on_bar(some_bar_event)
    
    # End tracing session
    result = debug_mgr.end_debugging_session()
    
    # Get the trace
    trace = tracer.get_trace(trace_id)
    
    # Analyze signal path
    signal_events = [e for e in trace["events"] if e["event_type"] == "SIGNAL" 
                     and e["event_data"].get("signal_id") == signal_id]
    order_events = [e for e in trace["events"] if e["event_type"] == "ORDER" 
                    and e["event_data"].get("signal_id") == signal_id]
    fill_events = [e for e in trace["events"] if e["event_type"] == "FILL" 
                   and e["event_data"].get("signal_id") == signal_id]
    
    # Print analysis
    print(f"Signal generated at: {signal_events[0]['timestamp'] if signal_events else 'N/A'}")
    print(f"Order created at: {order_events[0]['timestamp'] if order_events else 'N/A'}")
    print(f"Fill executed at: {fill_events[0]['timestamp'] if fill_events else 'N/A'}")
    
    # Calculate latencies
    if signal_events and order_events:
        signal_to_order = order_events[0]['timestamp'] - signal_events[0]['timestamp']
        print(f"Signal → Order latency: {signal_to_order:.6f}s")
        
    if order_events and fill_events:
        order_to_fill = fill_events[0]['timestamp'] - order_events[0]['timestamp']
        print(f"Order → Fill latency: {order_to_fill:.6f}s")
```

### 4.2 Inspecting Component State

```python
# Example usage to inspect and compare component states
def debug_component_state_change(portfolio, before_action, after_action):
    """Capture and compare portfolio state before and after an action."""
    # Get state inspector
    inspector = container.get("state_inspector")
    
    # Capture initial state
    before_snapshot_id = inspector.capture_snapshot("portfolio")
    
    # Perform the action
    before_action()
    
    # Capture intermediate state
    during_snapshot_id = inspector.capture_snapshot("portfolio")
    
    # Perform second action
    after_action()
    
    # Capture final state
    after_snapshot_id = inspector.capture_snapshot("portfolio")
    
    # Compare states
    before_during_diff = inspector.compare_snapshots(before_snapshot_id, during_snapshot_id)
    during_after_diff = inspector.compare_snapshots(during_snapshot_id, after_snapshot_id)
    
    # Print differences
    print("Changes after first action:")
    for component, diff in before_during_diff["components"].items():
        if diff:
            print(f"  {component}:")
            for attr, change in diff.items():
                print(f"    {attr}: {change}")
```

### 4.3 Recording and Replaying Events

```python
# Example usage for event recording and replay
def record_and_replay_scenario(scenario_name):
    """Record a backtest scenario and replay it for debugging."""
    # Get debug components
    recorder = container.get("event_recorder")
    
    # Start recording
    recording_id = recorder.start_recording(description=f"Scenario: {scenario_name}")
    
    # Run scenario
    run_backtest()
    
    # Stop recording
    recorder.stop_recording()
    
    # Save recording to disk
    recording_path = f"./recordings/{scenario_name}.json"
    recorder.save_recording(recording_id, recording_path)
    print(f"Scenario recorded to {recording_path}")
    
    # Reset system for replay
    reset_all_components()
    
    # Load recording
    loaded_id = recorder.load_recording(recording_path)
    
    # Replay recording
    recorder.replay_recording(loaded_id, replay_mode="immediate")
    print("Scenario replayed successfully")
```

### 4.4 Setting Breakpoints

```python
# Example usage for breakpoints
def debug_with_breakpoints():
    """Use breakpoints to debug specific conditions."""
    # Get debug manager
    debug_mgr = container.get("debug_manager")
    
    # Set breakpoint for large orders
    debug_mgr.set_breakpoint(
        event_type=EventType.ORDER,
        condition=lambda e: e.get_data().get('quantity', 0) > 1000
    )
    
    # Set breakpoint for specific symbol signal
    debug_mgr.set_breakpoint(
        event_type=EventType.SIGNAL,
        condition=lambda e: e.get_data().get('symbol') == 'AAPL'
    )
    
    # Run system - breakpoints will trigger when conditions are met
    run_system()
```

## 5. Configuration

The debugging framework can be enabled and configured through the system configuration:

```yaml
# Example configuration
debug:
  enabled: true  # Enable debug mode
  components:
    execution_tracer:
      trace_level: "detail"  # detail, standard, minimal
    state_inspector:
      history_retention: 20
    event_recorder:
      record_all_events: false
      event_types:  # Only record these event types
        - "SIGNAL"
        - "ORDER"
        - "FILL"
    debug_visualizer:
      output_dir: "./debug_output"
      format: "html"  # html, json
  breakpoints:
    - event_type: "ORDER"
      condition: "lambda e: e.get_data().get('quantity', 0) > 1000"
      enabled: true
  generate_reports: true
```

## 6. Implementation Plan

### 6.1 Phase 1: Core Infrastructure

1. Implement the `ExecutionTracer` component for event flow tracking
2. Implement the `StateInspector` component for component state analysis
3. Implement the `DebugManager` for central coordination
4. Update the `Component` base class with debugging hooks
5. Enhance the `EventBus` with debugging support

### 6.2 Phase 2: Advanced Features

1. Implement the `EventRecorder` for recording and replay capabilities
2. Implement the `DebugVisualizer` for visual representations
3. Develop HTML/JavaScript templates for interactive visualizations
4. Create JSON export formats for external tool integration

### 6.3 Phase 3: Integration

1. Integrate with existing components (Backtest, Strategy, etc.)
2. Add configuration options for debugging features
3. Extend CLI to support debugging commands
4. Create comprehensive documentation and usage examples
5. Develop tutorials for common debugging scenarios

## 7. Performance and Design Considerations

### 7.1 Performance Impact

- Components are only created when debug mode is enabled
- Event hooks can be selectively enabled or disabled
- Configurable verbosity levels to control data collection
- Separate visualization tasks from core data collection

### 7.2 Storage Management

- Configurable retention policies for historical snapshots
- Ability to filter recorded events by type
- Compression options for stored data
- Automatic cleanup of old debug data

### 7.3 Thread Safety

- Atomic operations for state capture
- Thread-safe collections for storing debug data
- Support for both single-threaded and multi-threaded execution models
- Proper synchronization for shared debugging resources

## 8. Key Benefits

1. **Transparency**: Provides clear visibility into system operations
2. **Traceability**: Enables tracking of events through the system
3. **Reproducibility**: Allows replay of recorded scenarios
4. **Analysis**: Facilitates detection of issues and bottlenecks
5. **Validation**: Helps verify correct system behavior
6. **Documentation**: Creates visual artifacts for understanding
7. **Development Efficiency**: Speeds up debugging and problem resolution