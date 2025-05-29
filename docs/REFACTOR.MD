# ADMF-Trader: Protocol + Composition Architecture Documentation

## Table of Contents

1. [Core Architecture & Protocols](#1-core-architecture--protocols)
2. [Bootstrap & Configuration](#2-bootstrap--configuration)
3. [Capability Framework](#3-capability-framework)
4. [Event System](#4-event-system)
5. [Data Module](#5-data-module)
6. [Strategy Module](#6-strategy-module)
7. [Risk Module](#7-risk-module)
8. [Execution Module](#8-execution-module)
9. [Optimization Framework](#9-optimization-framework)
10. [Infrastructure Services](#10-infrastructure-services)

---

## 1. Core Architecture & Protocols

### 1.1 Foundational Principles

The ADMF-Trader system is built on **Protocol + Composition** architecture, providing:

- **Protocol-based contracts** instead of inheritance hierarchies
- **Composable capabilities** that components opt into as needed
- **Zero-overhead simplicity** for basic components
- **Configuration-driven assembly** of complex systems
- **Duck typing** infrastructure that works with any component

### 1.2 Core Protocols

```python
from typing import Protocol, runtime_checkable, Dict, Any, Optional
from abc import abstractmethod

@runtime_checkable
class Component(Protocol):
    """Lifecycle management protocol for system components"""
    
    @abstractmethod
    def initialize(self, context: 'SystemContext') -> None:
        """Initialize component with system dependencies"""
        ...
        
    @abstractmethod
    def start(self) -> None:
        """Begin component operation"""
        ...
        
    @abstractmethod
    def stop(self) -> None:
        """Stop component operation"""
        ...
        
    @abstractmethod
    def reset(self) -> None:
        """Reset component state for reuse"""
        ...
        
    @abstractmethod
    def teardown(self) -> None:
        """Release all resources"""
        ...

@runtime_checkable
class EventSubscriber(Protocol):
    """Protocol for components that handle events"""
    
    @abstractmethod
    def initialize_event_subscriptions(self) -> None:
        """Set up event subscriptions"""
        ...

@runtime_checkable
class Optimizable(Protocol):
    """Protocol for components that can be optimized"""
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return parameter space for optimization"""
        ...
        
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameter values"""
        ...
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        ...

@runtime_checkable
class Resettable(Protocol):
    """Protocol for components that can reset state"""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset to clean state"""
        ...

@runtime_checkable
class Configurable(Protocol):
    """Protocol for components with configuration"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings"""
        ...
        
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...

@runtime_checkable
class Validatable(Protocol):
    """Protocol for components that can validate themselves"""
    
    @abstractmethod
    def validate(self) -> tuple[bool, str]:
        """Validate component state/configuration"""
        ...
```

### 1.3 System Context

```python
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum

class RunMode(Enum):
    BACKTEST_SINGLE = "backtest_single"
    BACKTEST_PARALLEL = "backtest_parallel"
    OPTIMIZATION = "optimization"
    LIVE_TRADING = "live_trading"
    PAPER_TRADING = "paper_trading"

@dataclass
class SystemContext:
    """Shared context passed to all components during initialization"""
    config: 'Config'
    event_bus: 'EventBus'
    container: 'Container'
    logger: 'Logger'
    run_mode: RunMode
    metadata: Optional[Dict[str, Any]] = None
    
    def resolve(self, service_name: str) -> Any:
        """Convenience method to resolve services from container"""
        return self.container.resolve(service_name)
```

---

## 2. Bootstrap & Configuration

### 2.1 Component Factory

```python
import importlib
from typing import Dict, Any, List, Callable

class ComponentFactory:
    """Creates and configures components based on specifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.capability_registry = CapabilityRegistry()
        self._register_standard_capabilities()
    
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create component from specification
        
        Args:
            spec: Component specification containing:
                - class/function: Component implementation
                - params: Constructor/configuration parameters
                - capabilities: List of capabilities to add
                - profile: Predefined capability profile
        """
        # Create base component
        component = self._create_base_component(spec)
        
        # Apply capabilities
        capabilities = self._resolve_capabilities(spec)
        for capability_name in capabilities:
            capability = self.capability_registry.get(capability_name)
            component = capability.apply(component, spec)
        
        return component
    
    def _create_base_component(self, spec: Dict[str, Any]) -> Any:
        """Create the base component instance"""
        if 'class' in spec:
            return self._create_from_class(spec)
        elif 'function' in spec:
            return self._create_from_function(spec)
        elif 'factory' in spec:
            return self._create_from_factory(spec)
        else:
            raise ValueError("Component spec must include 'class', 'function', or 'factory'")
    
    def _create_from_class(self, spec: Dict[str, Any]) -> Any:
        """Create component from class specification"""
        class_path = spec['class']
        params = spec.get('params', {})
        
        # Dynamic import
        if '.' in class_path:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        else:
            # Assume in current namespace or registered
            cls = self._resolve_class(class_path)
        
        return cls(**params)
    
    def _create_from_function(self, spec: Dict[str, Any]) -> Any:
        """Wrap function as component"""
        func_path = spec['function']
        params = spec.get('params', {})
        
        func = self._resolve_function(func_path)
        return FunctionWrapper(func, params, spec.get('name', func_path))

class FunctionWrapper:
    """Wraps a function to make it behave like a component"""
    
    def __init__(self, func: Callable, params: Dict[str, Any], name: str):
        self.func = func
        self.params = params
        self.name = name
        self._result = None
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable"""
        merged_kwargs = {**self.params, **kwargs}
        self._result = self.func(*args, **merged_kwargs)
        return self._result
    
    def get_result(self):
        """Get last function result"""
        return self._result
```

### 2.2 Bootstrap System

```python
class Bootstrap:
    """Configuration-driven system bootstrap using duck typing"""
    
    def __init__(self, config_path: str):
        self.config = Config.load(config_path)
        self.factory = ComponentFactory(self.config)
        self.context: Optional[SystemContext] = None
        self.components: Dict[str, Any] = {}
        self.dependency_graph = DependencyGraph()
    
    def initialize(self) -> SystemContext:
        """Initialize core system services"""
        # Create core services
        event_bus = EventBus()
        container = Container()
        logger = self._setup_logging()
        
        # Register core services
        container.register_instance("event_bus", event_bus)
        container.register_instance("container", container)
        container.register_instance("logger", logger)
        container.register_instance("config", self.config)
        
        # Create system context
        run_mode = RunMode(self.config.get('system.run_mode', 'backtest_single'))
        self.context = SystemContext(
            config=self.config,
            event_bus=event_bus,
            container=container,
            logger=logger,
            run_mode=run_mode
        )
        
        return self.context
    
    def build_system(self) -> Dict[str, Any]:
        """Build complete system from configuration"""
        if not self.context:
            self.initialize()
        
        # Create all components
        component_specs = self.config.get('components', {})
        for name, spec in component_specs.items():
            self.components[name] = self.factory.create_component(spec)
            self.container.register_instance(name, self.components[name])
            
            # Build dependency graph
            dependencies = spec.get('dependencies', [])
            self.dependency_graph.add_component(name, spec)
            for dep in dependencies:
                self.dependency_graph.add_dependency(name, dep)
        
        # Validate dependencies
        cycles = self.dependency_graph.detect_cycles()
        if cycles:
            raise ValueError(f"Circular dependencies detected: {cycles}")
        
        # Initialize components in dependency order
        init_order = self.dependency_graph.get_initialization_order()
        for component_name in init_order:
            self._initialize_component(self.components[component_name])
        
        return self.components
    
    def _initialize_component(self, component: Any) -> None:
        """Initialize component using duck typing"""
        # Initialize if component supports it
        if hasattr(component, 'initialize') and callable(getattr(component, 'initialize')):
            component.initialize(self.context)
        
        # Set up event subscriptions if supported
        if hasattr(component, 'initialize_event_subscriptions'):
            component.initialize_event_subscriptions()
        
        # Configure if component supports it
        if hasattr(component, 'configure') and hasattr(component, '_config_spec'):
            config_section = component._config_spec
            config_data = self.config.get(config_section, {})
            component.configure(config_data)
    
    def start_system(self) -> None:
        """Start all components that support starting"""
        start_order = self.dependency_graph.get_initialization_order()
        for component_name in start_order:
            component = self.components[component_name]
            if hasattr(component, 'start') and callable(getattr(component, 'start')):
                component.start()
    
    def stop_system(self) -> None:
        """Stop all components in reverse order"""
        stop_order = reversed(self.dependency_graph.get_initialization_order())
        for component_name in stop_order:
            component = self.components[component_name]
            if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                component.stop()
    
    def teardown_system(self) -> None:
        """Teardown all components"""
        teardown_order = reversed(self.dependency_graph.get_initialization_order())
        for component_name in teardown_order:
            component = self.components[component_name]
            if hasattr(component, 'teardown') and callable(getattr(component, 'teardown')):
                component.teardown()
```

### 2.3 Configuration System

```yaml
# config/system.yaml
system:
  run_mode: "backtest_single"
  log_level: "INFO"
  
capability_profiles:
  minimal:
    description: "Just calculation, no overhead"
    capabilities: []
  
  basic:
    description: "Basic lifecycle management"
    capabilities: ["lifecycle"]
  
  trading:
    description: "Full trading component"
    capabilities: ["lifecycle", "events", "reset"]
    events: ["BAR", "FILL", "PORTFOLIO_UPDATE"]
  
  optimizable:
    description: "Can be optimized and backtested"
    capabilities: ["lifecycle", "events", "optimization", "reset"]

components:
  data_handler:
    class: "HistoricalDataHandler"
    profile: "trading"
    params:
      data_dir: "data"
      symbols: ["EURUSD", "GBPUSD"]
    dependencies: []
  
  simple_ma:
    function: "ta.SMA"
    profile: "minimal"
    params:
      timeperiod: 20
  
  trend_strategy:
    class: "TrendFollowingStrategy"
    profile: "optimizable"
    params:
      fast_period: 10
      slow_period: 30
    dependencies: ["data_handler"]
  
  risk_manager:
    class: "RiskManager"
    profile: "trading"
    params:
      max_position_size: 10000
    dependencies: ["portfolio_manager"]
  
  portfolio_manager:
    class: "Portfolio"
    profile: "trading"
    params:
      initial_cash: 100000
    dependencies: []
```

---

## 3. Capability Framework

### 3.1 Capability System

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Capability(ABC):
    """Base class for composable capabilities"""
    
    @abstractmethod
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply this capability to a component"""
        ...
    
    @abstractmethod
    def get_name(self) -> str:
        """Get capability name"""
        ...

class LifecycleCapability(Capability):
    """Adds lifecycle management to components"""
    
    def get_name(self) -> str:
        return "lifecycle"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_lifecycle'):
            name = spec.get('name', component.__class__.__name__)
            component._lifecycle = ComponentLifecycle(name)
            
            # Add lifecycle methods if they don't exist
            self._add_lifecycle_methods(component)
        
        return component
    
    def _add_lifecycle_methods(self, component: Any) -> None:
        """Add lifecycle methods to component"""
        if not hasattr(component, 'initialize'):
            component.initialize = lambda ctx: component._lifecycle.transition_to_initialized()
        
        if not hasattr(component, 'start'):
            component.start = lambda: component._lifecycle.transition_to_running()
        
        if not hasattr(component, 'stop'):
            component.stop = lambda: component._lifecycle.transition_to_stopped()
        
        if not hasattr(component, 'teardown'):
            component.teardown = lambda: component._lifecycle.transition_to_disposed()

class EventCapability(Capability):
    """Adds event handling to components"""
    
    def get_name(self) -> str:
        return "events"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_events'):
            component._events = EventSubscriptions()
            
            # Enhance initialize method to set up events
            original_init = getattr(component, 'initialize', lambda ctx: None)
            def enhanced_init(ctx):
                component._events.initialize_with_bus(ctx.event_bus)
                original_init(ctx)
                if hasattr(component, 'setup_subscriptions'):
                    component.setup_subscriptions()
            component.initialize = enhanced_init
            
            # Add event subscription helper
            component.subscribe = component._events.subscribe
        
        return component

class OptimizationCapability(Capability):
    """Adds optimization support to components"""
    
    def get_name(self) -> str:
        return "optimization"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, '_optimization'):
            component._optimization = OptimizationSupport()
            
            # Add optimization methods if they don't exist
            if not hasattr(component, 'get_parameter_space'):
                component.get_parameter_space = component._optimization.get_parameter_space
            
            if not hasattr(component, 'set_parameters'):
                component.set_parameters = component._optimization.set_parameters
            
            if not hasattr(component, 'get_parameters'):
                component.get_parameters = component._optimization.get_parameters
        
        return component

class ResetCapability(Capability):
    """Adds reset functionality to components"""
    
    def get_name(self) -> str:
        return "reset"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'reset'):
            # Try to find state attributes to reset
            state_attrs = spec.get('reset_attributes', [])
            
            def reset_method():
                for attr in state_attrs:
                    if hasattr(component, attr):
                        if isinstance(getattr(component, attr), list):
                            getattr(component, attr).clear()
                        elif isinstance(getattr(component, attr), dict):
                            getattr(component, attr).clear()
                        else:
                            setattr(component, attr, None)
            
            component.reset = reset_method
        
        return component

class CapabilityRegistry:
    """Registry for all available capabilities"""
    
    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self._register_standard_capabilities()
    
    def register(self, capability: Capability) -> None:
        """Register a capability"""
        self.capabilities[capability.get_name()] = capability
    
    def get(self, name: str) -> Capability:
        """Get capability by name"""
        if name not in self.capabilities:
            raise ValueError(f"Unknown capability: {name}")
        return self.capabilities[name]
    
    def _register_standard_capabilities(self) -> None:
        """Register standard capabilities"""
        self.register(LifecycleCapability())
        self.register(EventCapability())
        self.register(OptimizationCapability())
        self.register(ResetCapability())
```

### 3.2 Composable Support Classes

```python
from enum import Enum
from typing import Optional, Dict, Any, List
import threading

class ComponentState(Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    DISPOSED = "disposed"

class ComponentLifecycle:
    """Composable lifecycle management"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.CREATED
        self.initialized = False
        self.running = False
        self._lock = threading.RLock()
    
    def transition_to_initialized(self) -> None:
        with self._lock:
            if self.state != ComponentState.CREATED:
                raise ValueError(f"Cannot initialize {self.name} from state {self.state}")
            self.state = ComponentState.INITIALIZED
            self.initialized = True
    
    def transition_to_running(self) -> None:
        with self._lock:
            if self.state not in [ComponentState.INITIALIZED, ComponentState.STOPPED]:
                raise ValueError(f"Cannot start {self.name} from state {self.state}")
            self.state = ComponentState.RUNNING
            self.running = True
    
    def transition_to_stopped(self) -> None:
        with self._lock:
            if self.state != ComponentState.RUNNING:
                raise ValueError(f"Cannot stop {self.name} from state {self.state}")
            self.state = ComponentState.STOPPED
            self.running = False
    
    def transition_to_disposed(self) -> None:
        with self._lock:
            self.state = ComponentState.DISPOSED
            self.initialized = False
            self.running = False

class EventSubscriptions:
    """Composable event handling"""
    
    def __init__(self):
        self.event_bus: Optional['EventBus'] = None
        self.subscription_manager: Optional['SubscriptionManager'] = None
        self.subscriptions: List[tuple] = []
    
    def initialize_with_bus(self, event_bus: 'EventBus') -> None:
        """Initialize with event bus from system context"""
        self.event_bus = event_bus
        self.subscription_manager = SubscriptionManager(event_bus)
    
    def subscribe(self, event_type: str, handler) -> None:
        if not self.subscription_manager:
            # Store for later when bus is available
            self.subscriptions.append((event_type, handler))
        else:
            self.subscription_manager.subscribe(event_type, handler)
    
    def unsubscribe_all(self) -> None:
        if self.subscription_manager:
            self.subscription_manager.unsubscribe_all()

class OptimizationSupport:
    """Composable optimization capabilities"""
    
    def __init__(self):
        self._parameters: Dict[str, Any] = {}
        self._parameter_space: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def set_parameter_space(self, space: Dict[str, Any]) -> None:
        with self._lock:
            self._parameter_space = space
    
    def get_parameter_space(self) -> Dict[str, Any]:
        with self._lock:
            return self._parameter_space.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        with self._lock:
            self._parameters.update(params)
    
    def get_parameters(self) -> Dict[str, Any]:
        with self._lock:
            return self._parameters.copy()
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Override in specific components for validation"""
        return True, ""
```

---

## 4. Event System

### 4.1 Event Architecture

The event system remains largely the same but works with any component that implements the `EventSubscriber` protocol:

```python
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime
from enum import Enum

class EventType(Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    CLASSIFICATION = "CLASSIFICATION"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"

class Event:
    """Event message structure"""
    
    def __init__(self, 
                 event_type: Union[EventType, str],
                 payload: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime] = None,
                 context: Optional[Any] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.event_type = event_type
        self.payload = payload or {}
        self.timestamp = timestamp or datetime.now()
        self.context = context
        self.metadata = metadata or {}

class EventBus:
    """Protocol-agnostic event bus"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: Union[EventType, str], handler: Callable) -> None:
        """Subscribe to events - works with any callable"""
        event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            if event_type_str not in self.subscribers:
                self.subscribers[event_type_str] = []
            self.subscribers[event_type_str].append(handler)
    
    def publish(self, event: Event) -> None:
        """Publish event to all subscribers"""
        event_type_str = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
        
        with self._lock:
            handlers = self.subscribers.get(event_type_str, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error in event handler {handler}: {e}")

class SubscriptionManager:
    """Manages subscriptions for a component"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.subscriptions: List[tuple] = []
    
    def subscribe(self, event_type: Union[EventType, str], handler: Callable) -> None:
        """Subscribe and track the subscription"""
        self.event_bus.subscribe(event_type, handler)
        self.subscriptions.append((event_type, handler))
    
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all tracked subscriptions"""
        # Implementation depends on EventBus unsubscribe method
        for event_type, handler in self.subscriptions:
            self.event_bus.unsubscribe(event_type, handler)
        self.subscriptions.clear()
```

---

## 5. Data Module

### 5.1 Data Handler Protocol

```python
from typing import Protocol, List, Optional, Dict, Any
from abc import abstractmethod

@runtime_checkable
class DataProvider(Protocol):
    """Protocol for data provision"""
    
    @abstractmethod
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols"""
        ...
    
    @abstractmethod
    def get_next_bar(self) -> Optional['Bar']:
        """Get next available bar across all symbols"""
        ...
    
    @abstractmethod
    def has_more_data(self) -> bool:
        """Check if more data is available"""
        ...

@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for train/test splitting"""
    
    @abstractmethod
    def setup_split(self, method: str = 'ratio', **kwargs) -> None:
        """Set up train/test split"""
        ...
    
    @abstractmethod
    def set_active_split(self, split_name: Optional[str]) -> None:
        """Set active data split (train/test/None)"""
        ...

class Bar:
    """Market data bar"""
    
    def __init__(self, symbol: str, timestamp: datetime, 
                 open_price: float, high: float, low: float, 
                 close: float, volume: float = 0):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
```

### 5.2 Historical Data Handler Implementation

```python
class HistoricalDataHandler:
    """Data handler using composition for capabilities"""
    
    def __init__(self, data_dir: str = "data", symbols: List[str] = None):
        self.data_dir = data_dir
        self.symbols = symbols or []
        
        # Core data state
        self.data: Dict[str, Any] = {}
        self.current_indices: Dict[str, int] = {}
        self.splits: Dict[str, Dict[str, Any]] = {}
        self.active_split: Optional[str] = None
        
        # Composed capabilities (added by capability system)
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
        
        # Data processing
        self.csv_loader = CSVLoader()
        self.splitter = TimeSeriesSplitter()
    
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols"""
        self.symbols = symbols
        success = True
        
        for symbol in symbols:
            file_path = f"{self.data_dir}/{symbol}.csv"
            try:
                symbol_data = self.csv_loader.load(file_path)
                self.data[symbol] = symbol_data
                self.current_indices[symbol] = 0
            except Exception as e:
                print(f"Failed to load data for {symbol}: {e}")
                success = False
        
        return success
    
    def setup_split(self, method: str = 'ratio', train_ratio: float = 0.7, 
                   split_date: Optional[str] = None) -> None:
        """Set up train/test data splits"""
        for symbol in self.symbols:
            if symbol in self.data:
                train_data, test_data = self.splitter.split(
                    self.data[symbol], method, train_ratio, split_date
                )
                self.splits[symbol] = {
                    'train': train_data,
                    'test': test_data,
                    'full': self.data[symbol]
                }
    
    def set_active_split(self, split_name: Optional[str]) -> None:
        """Set active data split"""
        self.active_split = split_name
        # Reset indices for new split
        for symbol in self.symbols:
            self.current_indices[symbol] = 0
    
    def get_next_bar(self) -> Optional[Bar]:
        """Get next bar across all symbols"""
        if not self.symbols:
            return None
        
        # Find earliest next timestamp across all symbols
        next_candidates = []
        active_data = self._get_active_data()
        
        for symbol in self.symbols:
            if symbol in active_data:
                current_idx = self.current_indices[symbol]
                symbol_data = active_data[symbol]
                
                if current_idx < len(symbol_data):
                    row = symbol_data.iloc[current_idx]
                    next_candidates.append((row['timestamp'], symbol, current_idx, row))
        
        if not next_candidates:
            return None
        
        # Sort by timestamp and get earliest
        next_candidates.sort(key=lambda x: x[0])
        timestamp, symbol, idx, row = next_candidates[0]
        
        # Update index
        self.current_indices[symbol] = idx + 1
        
        # Create and return bar
        bar = Bar(
            symbol=symbol,
            timestamp=timestamp,
            open_price=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row.get('volume', 0)
        )
        
        # Emit event if event capability is available
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.BAR, bar.to_dict())
            self._events.event_bus.publish(event)
        
        return bar
    
    def has_more_data(self) -> bool:
        """Check if more data is available"""
        active_data = self._get_active_data()
        
        for symbol in self.symbols:
            if symbol in active_data:
                current_idx = self.current_indices[symbol]
                if current_idx < len(active_data[symbol]):
                    return True
        
        return False
    
    def _get_active_data(self) -> Dict[str, Any]:
        """Get currently active dataset"""
        if self.active_split and self.splits:
            return {symbol: self.splits[symbol][self.active_split] 
                   for symbol in self.symbols if symbol in self.splits}
        return self.data
    
    def reset(self) -> None:
        """Reset handler state"""
        for symbol in self.symbols:
            self.current_indices[symbol] = 0
```

### 5.3 Configuration

```yaml
# Data handler configuration
components:
  data_handler:
    class: "HistoricalDataHandler"
    profile: "trading"
    params:
      data_dir: "data"
      symbols: ["EURUSD", "GBPUSD", "USDJPY"]
    capabilities: ["lifecycle", "events", "reset"]
    config_section: "data"

data:
  train_test_split:
    method: "ratio"
    train_ratio: 0.7
  csv_format:
    timestamp_column: "timestamp"
    price_columns: ["open", "high", "low", "close"]
    volume_column: "volume"
```

---

## 6. Strategy Module

### 6.1 Strategy Protocols

```python
from typing import Protocol, List, Dict, Any, Optional
from abc import abstractmethod

@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for components that generate trading signals"""
    
    @abstractmethod
    def generate_signal(self, data: Any) -> Optional[Dict[str, Any]]:
        """Generate trading signal from market data"""
        ...

@runtime_checkable
class Indicator(Protocol):
    """Protocol for technical indicators"""
    
    @abstractmethod
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate indicator value"""
        ...
    
    @property
    @abstractmethod
    def value(self) -> Optional[float]:
        """Current indicator value"""
        ...
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether indicator has enough data"""
        ...

@runtime_checkable
class TradingRule(Protocol):
    """Protocol for trading rules"""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> tuple[bool, float]:
        """Evaluate rule and return (triggered, strength)"""
        ...
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Rule weight for ensemble strategies"""
        ...
```

### 6.2 Strategy Implementation

```python
class TrendFollowingStrategy:
    """Strategy using composition - no inheritance"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, 
                 signal_threshold: float = 0.02):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_threshold = signal_threshold
        
        # Strategy state
        self.prices: List[float] = []
        self.fast_ma_values: List[float] = []
        self.slow_ma_values: List[float] = []
        self.last_signal: Optional[str] = None
        
        # Composed indicators
        self.fast_ma = SimpleMovingAverage(fast_period)
        self.slow_ma = SimpleMovingAverage(slow_period)
        
        # Will be added by capability system
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
        self._optimization: Optional[OptimizationSupport] = None
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions (called by event capability)"""
        if self._events:
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Handle new price bar"""
        bar_data = event.payload
        price = bar_data['close']
        timestamp = bar_data['timestamp']
        
        # Update indicators
        fast_value = self.fast_ma.calculate(price, timestamp)
        slow_value = self.slow_ma.calculate(price, timestamp)
        
        if fast_value is not None and slow_value is not None:
            # Generate signal
            signal = self._evaluate_signal(fast_value, slow_value, price, timestamp)
            if signal:
                self._emit_signal(signal)
    
    def _evaluate_signal(self, fast_ma: float, slow_ma: float, 
                        price: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Evaluate trading signal"""
        if not self.fast_ma.ready or not self.slow_ma.ready:
            return None
        
        # Calculate signal strength
        ma_diff = (fast_ma - slow_ma) / slow_ma
        
        if ma_diff > self.signal_threshold and self.last_signal != "BUY":
            self.last_signal = "BUY"
            return {
                'symbol': 'EURUSD',  # Should come from bar data
                'direction': 'BUY',
                'strength': abs(ma_diff),
                'price': price,
                'timestamp': timestamp,
                'strategy': 'trend_following',
                'reason': f'Fast MA ({fast_ma:.5f}) > Slow MA ({slow_ma:.5f})'
            }
        
        elif ma_diff < -self.signal_threshold and self.last_signal != "SELL":
            self.last_signal = "SELL"
            return {
                'symbol': 'EURUSD',
                'direction': 'SELL',
                'strength': abs(ma_diff),
                'price': price,
                'timestamp': timestamp,
                'strategy': 'trend_following',
                'reason': f'Fast MA ({fast_ma:.5f}) < Slow MA ({slow_ma:.5f})'
            }
        
        return None
    
    def _emit_signal(self, signal: Dict[str, Any]) -> None:
        """Emit trading signal"""
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.SIGNAL, signal)
            self._events.event_bus.publish(event)
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.prices.clear()
        self.fast_ma_values.clear()
        self.slow_ma_values.clear()
        self.last_signal = None
        self.fast_ma.reset()
        self.slow_ma.reset()
    
    # Optimization protocol methods (added by optimization capability)
    def get_parameter_space(self) -> Dict[str, Any]:
        """Define optimizable parameters"""
        return {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50],
            'signal_threshold': [0.01, 0.02, 0.03, 0.05]
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply optimization parameters"""
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
            self.fast_ma = SimpleMovingAverage(self.fast_period)
        
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
            self.slow_ma = SimpleMovingAverage(self.slow_period)
        
        if 'signal_threshold' in params:
            self.signal_threshold = params['signal_threshold']
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_threshold': self.signal_threshold
        }

class SimpleMovingAverage:
    """Simple moving average indicator"""
    
    def __init__(self, period: int):
        self.period = period
        self.values: List[float] = []
        self._current_value: Optional[float] = None
    
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate MA value"""
        self.values.append(value)
        
        if len(self.values) > self.period:
            self.values.pop(0)
        
        if self.ready:
            self._current_value = sum(self.values) / len(self.values)
            return self._current_value
        
        return None
    
    @property
    def value(self) -> Optional[float]:
        return self._current_value
    
    @property
    def ready(self) -> bool:
        return len(self.values) >= self.period
    
    def reset(self) -> None:
        self.values.clear()
        self._current_value = None
```

### 6.3 Ensemble Strategy

```python
class EnsembleStrategy:
    """Ensemble of multiple strategies/rules"""
    
    def __init__(self, aggregation_method: str = "weighted_average"):
        self.aggregation_method = aggregation_method
        self.components: List[tuple] = []  # (component, weight)
        self.last_signals: Dict[str, Any] = {}
        
        # Capabilities added by system
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
        self._optimization: Optional[OptimizationSupport] = None
    
    def add_component(self, component: Any, weight: float = 1.0) -> None:
        """Add strategy component with weight"""
        self.components.append((component, weight))
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event) -> None:
        """Process bar through all components"""
        signals = []
        
        # Collect signals from all components
        for component, weight in self.components:
            if hasattr(component, 'generate_signal'):
                signal = component.generate_signal(event.payload)
                if signal:
                    signals.append((signal, weight))
            elif hasattr(component, 'on_bar'):
                # Let component process bar, collect any signals it emits
                component.on_bar(event)
        
        # Aggregate signals
        if signals:
            aggregated_signal = self._aggregate_signals(signals)
            if aggregated_signal:
                self._emit_signal(aggregated_signal)
    
    def _aggregate_signals(self, signals: List[tuple]) -> Optional[Dict[str, Any]]:
        """Aggregate multiple signals"""
        if not signals:
            return None
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(signals)
        elif self.aggregation_method == "majority_vote":
            return self._majority_vote_aggregation(signals)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _weighted_average_aggregation(self, signals: List[tuple]) -> Optional[Dict[str, Any]]:
        """Aggregate using weighted average"""
        total_weight = 0
        weighted_strength = 0
        directions = []
        
        for signal, weight in signals:
            total_weight += weight
            strength = signal.get('strength', 1.0)
            direction_mult = 1.0 if signal['direction'] == 'BUY' else -1.0
            weighted_strength += strength * direction_mult * weight
            directions.append(signal['direction'])
        
        if total_weight == 0:
            return None
        
        final_strength = abs(weighted_strength / total_weight)
        final_direction = 'BUY' if weighted_strength > 0 else 'SELL'
        
        # Only emit if strength is above threshold
        if final_strength > 0.1:  # Configurable threshold
            return {
                'symbol': signals[0][0]['symbol'],
                'direction': final_direction,
                'strength': final_strength,
                'price': signals[0][0]['price'],
                'timestamp': signals[0][0]['timestamp'],
                'strategy': 'ensemble',
                'component_count': len(signals),
                'reason': f'Ensemble of {len(signals)} components'
            }
        
        return None
```

---

## 7. Risk Module

### 7.1 Risk Management Protocols

```python
@runtime_checkable
class PositionSizer(Protocol):
    """Protocol for position sizing algorithms"""
    
    @abstractmethod
    def calculate_size(self, signal: Dict[str, Any], 
                      portfolio: 'Portfolio', 
                      current_position: float = 0) -> int:
        """Calculate position size for signal"""
        ...

@runtime_checkable
class RiskLimit(Protocol):
    """Protocol for risk limits"""
    
    @abstractmethod
    def check(self, signal: Dict[str, Any], 
              quantity: int, 
              portfolio: 'Portfolio') -> tuple[bool, str]:
        """Check if trade passes risk limit"""
        ...
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get limit priority for conflict resolution"""
        ...

@runtime_checkable
class Portfolio(Protocol):
    """Protocol for portfolio management"""
    
    @abstractmethod
    def get_position(self, symbol: str) -> 'Position':
        """Get position for symbol"""
        ...
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        ...
    
    @abstractmethod
    def update_position(self, symbol: str, quantity: int, 
                       price: float, commission: float = 0) -> None:
        """Update position from trade"""
        ...
```

### 7.2 Risk Manager Implementation

```python
class RiskManager:
    """Risk manager using composition"""
    
    def __init__(self, default_position_sizer: str = "fixed",
                 max_position_size: int = 1000):
        self.default_position_sizer = default_position_sizer
        self.max_position_size = max_position_size
        
        # State
        self.order_id_counter = 0
        self.position_sizers: Dict[str, Any] = {}
        self.risk_limits: List[Any] = []
        
        # Dependencies (injected during initialization)
        self.portfolio: Optional['Portfolio'] = None
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.SIGNAL, self.on_signal)
    
    def initialize(self, context: SystemContext) -> None:
        """Initialize with dependencies"""
        self.portfolio = context.resolve("portfolio_manager")
        
        # Set up default position sizers
        self.position_sizers['default'] = FixedPositionSizer(self.max_position_size)
        
        # Set up default risk limits
        self.risk_limits = [
            MaxPositionLimit(self.max_position_size),
            MaxExposureLimit(0.1)  # 10% max exposure per position
        ]
    
    def on_signal(self, event: Event) -> None:
        """Process trading signal"""
        signal = event.payload
        
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
            
            # Calculate position size
            symbol = signal['symbol']
            current_position = self.portfolio.get_position(symbol).quantity
            
            sizer = self.position_sizers.get(symbol, self.position_sizers['default'])
            quantity = sizer.calculate_size(signal, self.portfolio, current_position)
            
            if quantity == 0:
                return
            
            # Check risk limits
            passed, reason = self._check_risk_limits(signal, quantity)
            if not passed:
                print(f"Trade rejected by risk limits: {reason}")
                return
            
            # Create and emit order
            order = self._create_order(signal, quantity)
            self._emit_order(order)
            
        except Exception as e:
            print(f"Error processing signal: {e}")
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal structure"""
        required_fields = ['symbol', 'direction', 'price', 'timestamp']
        return all(field in signal for field in required_fields)
    
    def _check_risk_limits(self, signal: Dict[str, Any], 
                          quantity: int) -> tuple[bool, str]:
        """Check all risk limits"""
        for limit in self.risk_limits:
            passed, reason = limit.check(signal, quantity, self.portfolio)
            if not passed:
                return False, reason
        return True, ""
    
    def _create_order(self, signal: Dict[str, Any], quantity: int) -> Dict[str, Any]:
        """Create order from signal and quantity"""
        self.order_id_counter += 1
        
        # Adjust quantity for direction
        if signal['direction'] == 'SELL':
            quantity = -quantity
        
        return {
            'order_id': f"ORDER_{self.order_id_counter:06d}",
            'symbol': signal['symbol'],
            'quantity': quantity,
            'order_type': 'MARKET',
            'price': signal['price'],
            'timestamp': signal['timestamp'],
            'strategy': signal.get('strategy', 'unknown'),
            'signal_strength': signal.get('strength', 1.0)
        }
    
    def _emit_order(self, order: Dict[str, Any]) -> None:
        """Emit order event"""
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.ORDER, order)
            self._events.event_bus.publish(event)

class FixedPositionSizer:
    """Fixed position size strategy"""
    
    def __init__(self, size: int):
        self.size = size
    
    def calculate_size(self, signal: Dict[str, Any], 
                      portfolio: 'Portfolio', 
                      current_position: float = 0) -> int:
        """Calculate fixed position size"""
        if signal['direction'] == 'BUY':
            return self.size
        else:
            return self.size  # Absolute size, direction handled by RiskManager

class MaxPositionLimit:
    """Maximum position size limit"""
    
    def __init__(self, max_size: int, priority: int = 100):
        self.max_size = max_size
        self.priority = priority
    
    def check(self, signal: Dict[str, Any], 
              quantity: int, 
              portfolio: 'Portfolio') -> tuple[bool, str]:
        """Check position size limit"""
        symbol = signal['symbol']
        current_pos = portfolio.get_position(symbol).quantity
        new_position = current_pos + quantity
        
        if abs(new_position) > self.max_size:
            return False, f"Position size {abs(new_position)} exceeds limit {self.max_size}"
        
        return True, ""
    
    def get_priority(self) -> int:
        return self.priority
```

---

## 8. Execution Module

### 8.1 Execution Protocols

```python
@runtime_checkable
class OrderProcessor(Protocol):
    """Protocol for order processing"""
    
    @abstractmethod
    def process_order(self, order: Dict[str, Any]) -> None:
        """Process incoming order"""
        ...

@runtime_checkable
class Broker(Protocol):
    """Protocol for broker implementations"""
    
    @abstractmethod
    def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute order and return fill if successful"""
        ...

@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage models"""
    
    @abstractmethod
    def apply_slippage(self, order: Dict[str, Any], 
                      market_price: float) -> float:
        """Apply slippage to get execution price"""
        ...

@runtime_checkable
class CommissionModel(Protocol):
    """Protocol for commission models"""
    
    @abstractmethod
    def calculate_commission(self, order: Dict[str, Any], 
                           fill_price: float) -> float:
        """Calculate commission for order"""
        ...
```

### 8.2 Simulated Broker Implementation

```python
class SimulatedBroker:
    """Simulated broker for backtesting"""
    
    def __init__(self, slippage_model: str = "fixed", 
                 commission_model: str = "fixed"):
        # State
        self.pending_orders: Dict[str, Dict] = {}
        self.latest_prices: Dict[str, float] = {}
        
        # Models
        self.slippage_model = self._create_slippage_model(slippage_model)
        self.commission_model = self._create_commission_model(commission_model)
        
        # Capabilities
        self._lifecycle: Optional[ComponentLifecycle] = None
        self._events: Optional[EventSubscriptions] = None
    
    def setup_subscriptions(self) -> None:
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe(EventType.ORDER, self.on_order)
            self._events.subscribe(EventType.BAR, self.on_bar)
    
    def on_order(self, event: Event) -> None:
        """Handle incoming order"""
        order = event.payload
        self.process_order(order)
    
    def on_bar(self, event: Event) -> None:
        """Handle new price bar"""
        bar_data = event.payload
        symbol = bar_data['symbol']
        
        # Update latest prices
        self.latest_prices[symbol] = bar_data['close']
        
        # Process pending orders for this symbol
        self._process_pending_orders(symbol, bar_data)
    
    def process_order(self, order: Dict[str, Any]) -> None:
        """Process incoming order"""
        order_type = order.get('order_type', 'MARKET')
        symbol = order['symbol']
        
        if order_type == 'MARKET':
            # Execute immediately if price available
            if symbol in self.latest_prices:
                fill = self.execute_order(order)
                if fill:
                    self._emit_fill(fill)
            else:
                # Store as pending
                self.pending_orders[order['order_id']] = order
        
        elif order_type == 'LIMIT':
            # Always store limit orders as pending
            self.pending_orders[order['order_id']] = order
        
        else:
            print(f"Unsupported order type: {order_type}")
    
    def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute order and return fill"""
        symbol = order['symbol']
        
        if symbol not in self.latest_prices:
            return None
        
        market_price = self.latest_prices[symbol]
        
        # Apply slippage
        execution_price = self.slippage_model.apply_slippage(order, market_price)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, execution_price)
        
        # Create fill
        fill = {
            'fill_id': f"FILL_{order['order_id']}",
            'order_id': order['order_id'],
            'symbol': symbol,
            'quantity': order['quantity'],
            'price': execution_price,
            'commission': commission,
            'timestamp': datetime.now(),
            'strategy': order.get('strategy', 'unknown')
        }
        
        return fill
    
    def _process_pending_orders(self, symbol: str, bar_data: Dict[str, Any]) -> None:
        """Process pending orders for symbol"""
        executed_orders = []
        
        for order_id, order in self.pending_orders.items():
            if order['symbol'] != symbol:
                continue
            
            if self._should_execute_order(order, bar_data):
                fill = self.execute_order(order)
                if fill:
                    self._emit_fill(fill)
                    executed_orders.append(order_id)
        
        # Remove executed orders
        for order_id in executed_orders:
            del self.pending_orders[order_id]
    
    def _should_execute_order(self, order: Dict[str, Any], 
                             bar_data: Dict[str, Any]) -> bool:
        """Check if order should be executed based on bar data"""
        order_type = order.get('order_type', 'MARKET')
        
        if order_type == 'MARKET':
            return True
        
        elif order_type == 'LIMIT':
            limit_price = order['price']
            quantity = order['quantity']
            
            if quantity > 0:  # Buy order
                return bar_data['low'] <= limit_price
            else:  # Sell order
                return bar_data['high'] >= limit_price
        
        return False
    
    def _emit_fill(self, fill: Dict[str, Any]) -> None:
        """Emit fill event"""
        if hasattr(self, '_events') and self._events and self._events.event_bus:
            event = Event(EventType.FILL, fill)
            self._events.event_bus.publish(event)
    
    def _create_slippage_model(self, model_type: str):
        """Create slippage model"""
        if model_type == "fixed":
            return FixedSlippageModel(0.0001)  # 1 pip
        else:
            raise ValueError(f"Unknown slippage model: {model_type}")
    
    def _create_commission_model(self, model_type: str):
        """Create commission model"""
        if model_type == "fixed":
            return FixedCommissionModel(2.0)  # $2 per trade
        else:
            raise ValueError(f"Unknown commission model: {model_type}")

class FixedSlippageModel:
    """Fixed slippage model"""
    
    def __init__(self, slippage: float):
        self.slippage = slippage
    
    def apply_slippage(self, order: Dict[str, Any], market_price: float) -> float:
        """Apply fixed slippage"""
        quantity = order['quantity']
        
        if quantity > 0:  # Buy order - pay more
            return market_price + self.slippage
        else:  # Sell order - receive less
            return market_price - self.slippage

class FixedCommissionModel:
    """Fixed commission model"""
    
    def __init__(self, commission: float):
        self.commission = commission
    
    def calculate_commission(self, order: Dict[str, Any], fill_price: float) -> float:
        """Calculate fixed commission"""
        return self.commission
```

---

## 9. Optimization Framework

### 9.1 Optimization Protocols

```python
@runtime_checkable
class Optimizer(Protocol):
    """Protocol for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, objective_func: Callable, 
                parameter_space: Dict[str, Any],
                n_trials: int = 100) -> Dict[str, Any]:
        """Run optimization and return best parameters"""
        ...

@runtime_checkable
class Objective(Protocol):
    """Protocol for optimization objectives"""
    
    @abstractmethod
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate objective value from backtest results"""
        ...
    
    @abstractmethod
    def get_direction(self) -> str:
        """Get optimization direction ('maximize' or 'minimize')"""
        ...
```

### 9.2 Component Optimizer

```python
class ComponentOptimizer:
    """Optimizes any component that implements Optimizable protocol"""
    
    def __init__(self, optimizer_type: str = "grid", 
                 objective_type: str = "sharpe"):
        self.optimizer_type = optimizer_type
        self.objective_type = objective_type
        
        # Will be created based on configuration
        self.optimizer: Optional[Any] = None
        self.objective: Optional[Any] = None
        self.backtest_runner: Optional[Callable] = None
        
    def initialize(self, context: SystemContext) -> None:
        """Initialize optimizer components"""
        self.optimizer = self._create_optimizer(self.optimizer_type)
        self.objective = self._create_objective(self.objective_type)
        self.backtest_runner = context.resolve("backtest_runner")
    
    def optimize_component(self, component: Any, 
                          override_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize any component that implements Optimizable protocol"""
        
        # Check if component is optimizable
        if not isinstance(component, Optimizable):
            raise ValueError(f"Component {component} does not implement Optimizable protocol")
        
        # Get parameter space
        parameter_space = override_space or component.get_parameter_space()
        if not parameter_space:
            print(f"Component {component} has no parameters to optimize")
            return {}
        
        # Define objective function
        def objective_func(params: Dict[str, Any]) -> float:
            # Apply parameters to component
            component.set_parameters(params)
            
            # Run backtest
            results = self.backtest_runner(component)
            
            # Calculate objective
            return self.objective.calculate(results)
        
        # Run optimization
        best_params = self.optimizer.optimize(
            objective_func, 
            parameter_space,
            n_trials=100
        )
        
        # Apply best parameters
        component.set_parameters(best_params)
        
        return best_params
    
    def _create_optimizer(self, optimizer_type: str):
        """Create optimizer instance"""
        if optimizer_type == "grid":
            return GridOptimizer()
        elif optimizer_type == "genetic":
            return GeneticOptimizer()
        elif optimizer_type == "bayesian":
            return BayesianOptimizer()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_objective(self, objective_type: str):
        """Create objective instance"""
        if objective_type == "sharpe":
            return SharpeObjective()
        elif objective_type == "return":
            return ReturnObjective()
        elif objective_type == "drawdown":
            return DrawdownObjective()
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")

class GridOptimizer:
    """Grid search optimizer"""
    
    def optimize(self, objective_func: Callable, 
                parameter_space: Dict[str, Any],
                n_trials: int = 100) -> Dict[str, Any]:
        """Run grid search optimization"""
        
        # Generate all parameter combinations
        combinations = self._generate_combinations(parameter_space)
        
        best_score = float('-inf')
        best_params = {}
        
        for params in combinations[:n_trials]:  # Limit trials
            try:
                score = objective_func(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                continue
        
        return best_params
    
    def _generate_combinations(self, space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        import itertools
        
        keys = list(space.keys())
        values = [space[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations

class SharpeObjective:
    """Sharpe ratio objective"""
    
    def calculate(self, results: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio from backtest results"""
        returns = results.get('returns', [])
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        
        if np.std(returns_array) == 0:
            return 0.0
        
        return np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)  # Annualized
    
    def get_direction(self) -> str:
        return "maximize"
```

### 9.3 Joint Optimizer

```python
class JointOptimizer:
    """Optimizes multiple components simultaneously"""
    
    def __init__(self, optimizer_type: str = "genetic", 
                 objective_type: str = "sharpe"):
        self.optimizer_type = optimizer_type
        self.objective_type = objective_type
        self.components: List[tuple] = []  # (component, prefix)
        
    def add_component(self, component: Any, prefix: str = "") -> None:
        """Add component to joint optimization"""
        if not isinstance(component, Optimizable):
            raise ValueError(f"Component {component} does not implement Optimizable protocol")
        
        self.components.append((component, prefix))
    
    def optimize(self, backtest_runner: Callable) -> Dict[str, Any]:
        """Run joint optimization"""
        if not self.components:
            raise ValueError("No components added for optimization")
        
        # Build combined parameter space
        combined_space = {}
        for component, prefix in self.components:
            component_space = component.get_parameter_space()
            for param_name, param_values in component_space.items():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                combined_space[full_name] = param_values
        
        # Create objective function
        def objective_func(params: Dict[str, Any]) -> float:
            # Distribute parameters to components
            for component, prefix in self.components:
                component_params = {}
                prefix_with_dot = f"{prefix}." if prefix else ""
                
                for param_name, value in params.items():
                    if param_name.startswith(prefix_with_dot):
                        actual_param = param_name[len(prefix_with_dot):]
                        component_params[actual_param] = value
                
                if component_params:
                    component.set_parameters(component_params)
            
            # Run backtest with all components
            results = backtest_runner()
            
            # Calculate objective
            objective = self._create_objective(self.objective_type)
            return objective.calculate(results)
        
        # Run optimization
        optimizer = self._create_optimizer(self.optimizer_type)
        best_params = optimizer.optimize(objective_func, combined_space)
        
        # Apply best parameters
        for component, prefix in self.components:
            component_params = {}
            prefix_with_dot = f"{prefix}." if prefix else ""
            
            for param_name, value in best_params.items():
                if param_name.startswith(prefix_with_dot):
                    actual_param = param_name[len(prefix_with_dot):]
                    component_params[actual_param] = value
            
            if component_params:
                component.set_parameters(component_params)
        
        return best_params
```

### 9.4 Configuration-Driven Optimization

```yaml
# config/optimization.yaml
optimization:
  workflows:
    single_strategy:
      type: "component"
      target: "trend_strategy"
      optimizer: "grid"
      objective: "sharpe"
      trials: 100
    
    joint_optimization:
      type: "joint"
      components:
        - component: "trend_strategy"
          prefix: "strategy"
        - component: "risk_manager.position_sizer"
          prefix: "sizing"
      optimizer: "genetic"
      objective: "composite"
      objectives:
        sharpe: { weight: 0.7 }
        drawdown: { weight: 0.3, direction: "minimize" }
    
    regime_based:
      type: "regime"
      regime_detector: "volatility_regime"
      target: "adaptive_strategy"
      optimizer: "bayesian"
      objective: "sharpe"
      min_samples_per_regime: 10

  optimizers:
    grid:
      class: "GridOptimizer"
      max_combinations: 1000
    
    genetic:
      class: "GeneticOptimizer"
      population_size: 50
      generations: 20
      mutation_rate: 0.1
    
    bayesian:
      class: "BayesianOptimizer"
      acquisition_function: "expected_improvement"
      n_initial_points: 10

  objectives:
    sharpe:
      class: "SharpeObjective"
      direction: "maximize"
    
    return:
      class: "ReturnObjective"
      direction: "maximize"
    
    drawdown:
      class: "DrawdownObjective"
      direction: "minimize"
    
    composite:
      class: "CompositeObjective"
      components:
        - objective: "sharpe"
          weight: 0.6
        - objective: "drawdown"
          weight: 0.4
          direction: "minimize"
```

---

## 10. Infrastructure Services

### 10.1 Logging Framework

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import threading

class StructuredLogger:
    """Structured logging with consistent format"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Structured formatter
        formatter = StructuredFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Context storage
        self.context = threading.local()
    
    def set_context(self, **kwargs) -> None:
        """Set logging context for current thread"""
        if not hasattr(self.context, 'data'):
            self.context.data = {}
        self.context.data.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context"""
        if hasattr(self.context, 'data'):
            self.context.data.clear()
    
    def _get_context(self) -> Dict[str, Any]:
        """Get current context"""
        if hasattr(self.context, 'data'):
            return self.context.data.copy()
        return {}
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        extra = {**self._get_context(), **kwargs}
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context"""
        extra = {**self._get_context(), **kwargs}
        self.logger.error(message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        extra = {**self._get_context(), **kwargs}
        self.logger.debug(message, extra=extra)

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in log_data and not key.startswith('_'):
                    log_data[key] = value
        
        return json.dumps(log_data)

class LoggingCapability(Capability):
    """Adds structured logging to components"""
    
    def get_name(self) -> str:
        return "logging"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'logger'):
            logger_name = spec.get('name', component.__class__.__name__)
            log_level = spec.get('log_level', 'INFO')
            component.logger = StructuredLogger(logger_name, log_level)
        
        return component
```

### 10.2 Monitoring Framework

```python
@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection"""
    
    @abstractmethod
    def record_value(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        ...
    
    @abstractmethod
    def record_counter(self, name: str, increment: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record counter increment"""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        ...

class SimpleMetricsCollector:
    """Simple in-memory metrics collector"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def record_value(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = {
                    'type': 'gauge',
                    'values': [],
                    'tags': tags or {}
                }
            
            self.metrics[name]['values'].append({
                'value': value,
                'timestamp': datetime.now(),
                'tags': tags or {}
            })
    
    def record_counter(self, name: str, increment: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record counter increment"""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = {
                    'type': 'counter',
                    'count': 0,
                    'tags': tags or {}
                }
            
            self.metrics[name]['count'] += increment
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return self.metrics.copy()

class MonitoringCapability(Capability):
    """Adds monitoring to components"""
    
    def get_name(self) -> str:
        return "monitoring"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'metrics'):
            component.metrics = SimpleMetricsCollector()
            
            # Add convenience methods
            component.record_metric = component.metrics.record_value
            component.increment_counter = component.metrics.record_counter
        
        return component
```

### 10.3 Error Handling Framework

```python
from typing import Type, List, Callable, Optional
import traceback
from contextlib import contextmanager

class ErrorBoundary:
    """Context manager for error handling"""
    
    def __init__(self, 
                 component_name: str,
                 logger: Optional[StructuredLogger] = None,
                 event_bus: Optional['EventBus'] = None,
                 reraise: bool = True):
        self.component_name = component_name
        self.logger = logger
        self.event_bus = event_bus
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[Any]) -> bool:
        if exc_type is not None:
            # Log error
            if self.logger:
                self.logger.error(
                    f"Error in component {self.component_name}",
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    traceback=traceback.format_exc()
                )
            
            # Emit error event
            if self.event_bus:
                error_event = Event(EventType.ERROR, {
                    'component': self.component_name,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val),
                    'timestamp': datetime.now()
                })
                self.event_bus.publish(error_event)
            
            # Don't suppress exception if reraise is True
            return not self.reraise
        
        return False

class ErrorHandlingCapability(Capability):
    """Adds error handling to components"""
    
    def get_name(self) -> str:
        return "error_handling"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        if not hasattr(component, 'error_boundary'):
            component_name = spec.get('name', component.__class__.__name__)
            
            def create_error_boundary(reraise: bool = True):
                logger = getattr(component, 'logger', None)
                event_bus = getattr(component, '_events', None)
                if event_bus:
                    event_bus = event_bus.event_bus
                
                return ErrorBoundary(component_name, logger, event_bus, reraise)
            
            component.error_boundary = create_error_boundary
        
        return component
```

### 10.4 Complete System Configuration

```yaml
# config/complete_system.yaml
system:
  name: "ADMF-Trader"
  version: "2.0.0"
  run_mode: "backtest_single"

capability_profiles:
  minimal:
    capabilities: []
  
  basic:
    capabilities: ["lifecycle", "logging"]
  
  trading:
    capabilities: ["lifecycle", "events", "reset", "logging", "monitoring"]
  
  optimizable:
    capabilities: ["lifecycle", "events", "optimization", "reset", "logging"]
  
  production:
    capabilities: ["lifecycle", "events", "reset", "logging", "monitoring", "error_handling"]

components:
  # Data components
  data_handler:
    class: "HistoricalDataHandler"
    profile: "trading"
    params:
      data_dir: "data"
      symbols: ["EURUSD", "GBPUSD"]
    dependencies: []
  
  # Strategy components
  simple_ma:
    function: "ta.SMA"
    profile: "minimal"
    params:
      timeperiod: 20
  
  trend_strategy:
    class: "TrendFollowingStrategy"
    profile: "optimizable"
    params:
      fast_period: 10
      slow_period: 30
      signal_threshold: 0.02
    dependencies: ["data_handler"]
  
  ensemble_strategy:
    class: "EnsembleStrategy"
    profile: "optimizable"
    params:
      aggregation_method: "weighted_average"
    components:
      - component_ref: "trend_strategy"
        weight: 0.7
      - component_ref: "simple_ma"
        weight: 0.3
    dependencies: ["data_handler"]
  
  # Risk management
  portfolio_manager:
    class: "Portfolio"
    profile: "trading"
    params:
      initial_cash: 100000
    dependencies: []
  
  risk_manager:
    class: "RiskManager"
    profile: "trading"
    params:
      default_position_sizer: "fixed"
      max_position_size: 1000
    dependencies: ["portfolio_manager"]
  
  # Execution
  simulated_broker:
    class: "SimulatedBroker"
    profile: "trading"
    params:
      slippage_model: "fixed"
      commission_model: "fixed"
    dependencies: ["portfolio_manager"]

workflows:
  simple_backtest:
    type: "backtest"
    components: ["data_handler", "trend_strategy", "risk_manager", "simulated_broker", "portfolio_manager"]
    
  optimization_run:
    type: "optimization"
    workflow: "single_strategy"
    target: "trend_strategy"
    
  production_trading:
    type: "live_trading"
    components: ["live_data_handler", "ensemble_strategy", "risk_manager", "live_broker", "portfolio_manager"]

# Logging configuration
logging:
  level: "INFO"
  format: "structured"
  handlers:
    - type: "console"
    - type: "file"
      filename: "logs/admf-trader.log"
      rotation: "daily"

# Monitoring configuration
monitoring:
  enabled: true
  collection_interval: 60  # seconds
  metrics:
    - "system.cpu_usage"
    - "system.memory_usage"
    - "trading.signals_generated"
    - "trading.orders_placed"
    - "trading.fills_received"
```

---

## Summary

This Protocol + Composition architecture provides:

### **Key Benefits:**

1. **Maximum Flexibility**: Components choose exactly what capabilities they need
2. **Zero Overhead**: Simple components have no framework overhead
3. **Composability**: Mix any type of component (functions, classes, external libraries)
4. **Testability**: Easy to test components in isolation
5. **Configuration-Driven**: Complete system behavior controlled by configuration
6. **Protocol Safety**: Runtime checking ensures components implement required protocols
7. **Gradual Migration**: Can coexist with existing inheritance-based components

### **Architecture Principles:**

- **Protocols define contracts** instead of inheritance hierarchies
- **Capabilities are composed** rather than inherited
- **Duck typing** allows any component to participate
- **Configuration drives assembly** of complex systems
- **Components opt into only needed functionality**

### **Implementation Strategy:**

1. **Start with protocols** - Define the contracts your system needs
2. **Build capability classes** - Create composable functionality
3. **Implement component factory** - Configuration-driven component creation
4. **Update infrastructure** - Bootstrap, container, optimization to use duck typing
5. **Create new components** - Using composition approach
6. **Migrate gradually** - Old and new approaches can coexist

This architecture gives you the foundation for a highly flexible, maintainable, and extensible algorithmic trading system that can adapt to any trading strategy or market condition.
