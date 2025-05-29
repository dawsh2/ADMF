# Protocol + Composition Migration Plan for ADMF

## Overview

This document outlines a practical migration strategy to adopt Protocol + Composition architecture while maintaining backward compatibility with the existing ComponentBase system.

## Phase 1: Foundation (2-3 weeks)

### 1.1 Define Core Protocols

```python
# src/core/protocols.py
from typing import Protocol, runtime_checkable, Dict, Any
from abc import abstractmethod

@runtime_checkable
class Component(Protocol):
    """Core component protocol - minimal lifecycle"""
    @abstractmethod
    def initialize(self, context: 'SystemContext') -> None: ...
    @abstractmethod
    def teardown(self) -> None: ...

@runtime_checkable
class Lifecycle(Protocol):
    """Full lifecycle management"""
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...

@runtime_checkable
class EventSubscriber(Protocol):
    """Event subscription capability"""
    @abstractmethod
    def initialize_event_subscriptions(self) -> None: ...

@runtime_checkable
class Optimizable(Protocol):
    """Optimization support"""
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]: ...
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None: ...
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]: ...

@runtime_checkable
class Calculator(Protocol):
    """Simple calculation interface"""
    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any: ...
```

### 1.2 Create Capability Mixins

```python
# src/core/capabilities.py
from enum import Enum
from typing import Optional, Dict, Any
import threading

class ComponentState(Enum):
    CREATED = "created"
    INITIALIZED = "initialized" 
    RUNNING = "running"
    STOPPED = "stopped"
    DISPOSED = "disposed"

class LifecycleCapability:
    """Composable lifecycle management"""
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.CREATED
        self._lock = threading.RLock()
        
    def transition_to_initialized(self):
        with self._lock:
            if self.state != ComponentState.CREATED:
                raise ValueError(f"Cannot initialize from {self.state}")
            self.state = ComponentState.INITIALIZED
            
    def transition_to_running(self):
        with self._lock:
            if self.state != ComponentState.INITIALIZED:
                raise ValueError(f"Cannot start from {self.state}")
            self.state = ComponentState.RUNNING
            
    # ... other transitions

class EventCapability:
    """Composable event handling"""
    def __init__(self):
        self.event_bus: Optional['EventBus'] = None
        self.subscription_manager: Optional['SubscriptionManager'] = None
        
    def initialize_events(self, event_bus: 'EventBus'):
        self.event_bus = event_bus
        from src.core.subscription_manager import SubscriptionManager
        self.subscription_manager = SubscriptionManager(event_bus)
        
    def subscribe(self, event_type: str, handler):
        if not self.subscription_manager:
            raise ValueError("Events not initialized")
        self.subscription_manager.subscribe(event_type, handler)
        
    def unsubscribe_all(self):
        if self.subscription_manager:
            self.subscription_manager.unsubscribe_all()

class OptimizationCapability:
    """Composable optimization support"""
    def __init__(self):
        self._parameters: Dict[str, Any] = {}
        self._parameter_space: Dict[str, Any] = {}
        
    def set_parameter_space(self, space: Dict[str, Any]):
        self._parameter_space = space
        
    def get_parameter_space(self) -> Dict[str, Any]:
        return self._parameter_space.copy()
        
    def set_parameters(self, params: Dict[str, Any]):
        self._parameters.update(params)
        
    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()
```

### 1.3 Create Adapter for ComponentBase

```python
# src/core/component_adapter.py
from src.core.component import ComponentBase
from src.core.protocols import Component, Lifecycle, EventSubscriber, Optimizable

class ProtocolAdapter:
    """Makes ComponentBase work with protocol-based system"""
    
    @staticmethod
    def adapt(component: ComponentBase) -> Component:
        """ComponentBase already implements all protocols"""
        return component
    
    @staticmethod
    def is_lifecycle(obj: Any) -> bool:
        return isinstance(obj, Lifecycle)
    
    @staticmethod
    def is_optimizable(obj: Any) -> bool:
        return isinstance(obj, Optimizable)
    
    @staticmethod
    def is_event_subscriber(obj: Any) -> bool:
        return isinstance(obj, EventSubscriber)
```

## Phase 2: Enhanced Bootstrap (1-2 weeks)

### 2.1 Update Bootstrap to Support Both Approaches

```python
# src/core/bootstrap.py updates
class Bootstrap:
    def initialize_component(self, component: Any, context: SystemContext):
        """Initialize any component that implements Component protocol"""
        # Works with both ComponentBase and protocol-based components
        if isinstance(component, Component):
            component.initialize(context)
        elif hasattr(component, 'initialize'):
            component.initialize(context)
            
        # Initialize event subscriptions if supported
        if isinstance(component, EventSubscriber):
            component.initialize_event_subscriptions()
        elif hasattr(component, 'initialize_event_subscriptions'):
            component.initialize_event_subscriptions()
    
    def start_component(self, component: Any):
        """Start component if it supports lifecycle"""
        if isinstance(component, Lifecycle):
            component.start()
        elif hasattr(component, 'start'):
            component.start()
```

### 2.2 Component Factory

```python
# src/core/component_factory.py
import importlib
from typing import Dict, Any, List
from src.core.capabilities import LifecycleCapability, EventCapability, OptimizationCapability

class ComponentFactory:
    """Creates components with configured capabilities"""
    
    def __init__(self):
        self.capability_builders = {
            'lifecycle': self._add_lifecycle,
            'events': self._add_events,
            'optimization': self._add_optimization,
        }
    
    def create_from_config(self, config: Dict[str, Any]) -> Any:
        """Create component based on configuration"""
        
        # Create base component
        if 'class' in config:
            component = self._create_from_class(config)
        elif 'function' in config:
            component = self._create_from_function(config)
        else:
            raise ValueError("Config must specify 'class' or 'function'")
        
        # Add configured capabilities
        for capability in config.get('capabilities', []):
            if capability in self.capability_builders:
                self.capability_builders[capability](component, config)
        
        return component
    
    def _create_from_class(self, config: Dict[str, Any]) -> Any:
        """Create instance from class name"""
        module_name = config.get('module', 'src.strategy')
        class_name = config['class']
        params = config.get('params', {})
        
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        
        # Check if it's already a ComponentBase
        if issubclass(cls, ComponentBase):
            return cls(config.get('name', class_name), 
                      config.get('config_key', class_name.lower()),
                      **params)
        else:
            return cls(**params)
    
    def _create_from_function(self, config: Dict[str, Any]) -> Any:
        """Wrap function as component"""
        func_path = config['function']
        params = config.get('params', {})
        
        # Import function
        if '.' in func_path:
            module_name, func_name = func_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        else:
            func = globals().get(func_path)
        
        # Create wrapper
        class FunctionComponent:
            def __init__(self):
                self.func = func
                self.params = params
                
            def calculate(self, *args, **kwargs):
                return self.func(*args, **{**self.params, **kwargs})
        
        return FunctionComponent()
    
    def _add_lifecycle(self, component: Any, config: Dict[str, Any]):
        """Add lifecycle capability to component"""
        if not hasattr(component, '_lifecycle'):
            component._lifecycle = LifecycleCapability(config.get('name', 'component'))
            
            # Add lifecycle methods
            if not hasattr(component, 'start'):
                component.start = lambda: component._lifecycle.transition_to_running()
            if not hasattr(component, 'stop'):
                component.stop = lambda: component._lifecycle.transition_to_stopped()
            if not hasattr(component, 'reset'):
                component.reset = lambda: None  # Default no-op
    
    def _add_events(self, component: Any, config: Dict[str, Any]):
        """Add event capability to component"""
        if not hasattr(component, '_events'):
            component._events = EventCapability()
            
            # Wrap initialize to set up events
            original_init = getattr(component, 'initialize', lambda ctx: None)
            def enhanced_init(ctx):
                component._events.initialize_events(ctx.event_bus)
                original_init(ctx)
            component.initialize = enhanced_init
            
            # Add event subscription method
            if not hasattr(component, 'initialize_event_subscriptions'):
                component.initialize_event_subscriptions = lambda: None
    
    def _add_optimization(self, component: Any, config: Dict[str, Any]):
        """Add optimization capability to component"""
        if not hasattr(component, '_optimization'):
            component._optimization = OptimizationCapability()
            
            # Set parameter space from config
            if 'parameter_space' in config:
                component._optimization.set_parameter_space(config['parameter_space'])
            
            # Add optimization methods
            if not hasattr(component, 'get_parameter_space'):
                component.get_parameter_space = component._optimization.get_parameter_space
            if not hasattr(component, 'set_parameters'):
                component.set_parameters = component._optimization.set_parameters
            if not hasattr(component, 'get_parameters'):
                component.get_parameters = component._optimization.get_parameters
```

## Phase 3: Example Implementations (2-3 weeks)

### 3.1 Simple Indicator (No Framework Overhead)

```python
# src/indicators/simple_rsi.py
class SimpleRSI:
    """Pure calculation component - no framework overhead"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.gains = []
        self.losses = []
        
    def calculate(self, price: float, prev_price: float) -> float:
        """Calculate RSI value"""
        change = price - prev_price
        
        if change > 0:
            self.gains.append(change)
            self.losses.append(0)
        else:
            self.gains.append(0)
            self.losses.append(abs(change))
            
        if len(self.gains) > self.period:
            self.gains.pop(0)
            self.losses.pop(0)
            
        if len(self.gains) < self.period:
            return 50.0  # Neutral until enough data
            
        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def reset(self):
        """Optional reset method"""
        self.gains.clear()
        self.losses.clear()
```

### 3.2 Strategy Using Composition

```python
# src/strategy/flexible_ma_strategy.py
from typing import List, Optional, Dict, Any

class FlexibleMAStrategy:
    """Strategy using composition for only needed capabilities"""
    
    def __init__(self, name: str, fast_period: int = 10, slow_period: int = 30):
        self.name = name
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Only compose what we need
        self._lifecycle = LifecycleCapability(name)
        self._events: Optional[EventCapability] = None
        self._optimization = OptimizationCapability()
        
        # Strategy state
        self.prices: List[float] = []
        self.position: Optional[str] = None
        
        # Set up optimization
        self._optimization.set_parameter_space({
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50]
        })
        self._optimization.set_parameters({
            "fast_period": fast_period,
            "slow_period": slow_period
        })
    
    # Implement Component protocol
    def initialize(self, context):
        """Initialize with context"""
        self._events = EventCapability()
        self._events.initialize_events(context.event_bus)
        self._lifecycle.transition_to_initialized()
    
    def teardown(self):
        """Cleanup"""
        if self._events:
            self._events.unsubscribe_all()
        self._lifecycle.transition_to_disposed()
    
    # Implement Lifecycle protocol
    def start(self):
        self._lifecycle.transition_to_running()
    
    def stop(self):
        self._lifecycle.transition_to_stopped()
    
    def reset(self):
        self.prices.clear()
        self.position = None
    
    # Implement EventSubscriber protocol
    def initialize_event_subscriptions(self):
        if self._events:
            self._events.subscribe("BAR", self.on_bar)
    
    # Implement Optimizable protocol
    def get_parameter_space(self) -> Dict[str, Any]:
        return self._optimization.get_parameter_space()
    
    def set_parameters(self, params: Dict[str, Any]):
        self._optimization.set_parameters(params)
        if "fast_period" in params:
            self.fast_period = params["fast_period"]
        if "slow_period" in params:
            self.slow_period = params["slow_period"]
    
    def get_parameters(self) -> Dict[str, Any]:
        return self._optimization.get_parameters()
    
    # Strategy logic
    def on_bar(self, event):
        """Handle new price bar"""
        price = event.payload.get('close')
        self.prices.append(price)
        
        if len(self.prices) < self.slow_period:
            return
            
        fast_ma = sum(self.prices[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period:]) / self.slow_period
        
        if fast_ma > slow_ma and self.position != "LONG":
            self._emit_signal("BUY", price)
            self.position = "LONG"
        elif fast_ma < slow_ma and self.position != "SHORT":
            self._emit_signal("SELL", price)
            self.position = "SHORT"
    
    def _emit_signal(self, direction: str, price: float):
        """Emit trading signal"""
        if self._events and self._events.event_bus:
            from src.core.event import Event
            signal_event = Event("SIGNAL", {
                "symbol": "EURUSD",
                "direction": direction,
                "price": price,
                "strategy": self.name
            })
            self._events.event_bus.publish(signal_event)
```

### 3.3 Ensemble Strategy Mixing Components

```python
# src/strategy/ensemble_strategy.py
from typing import List, Any, Dict
import ta  # Technical analysis library

class EnsembleStrategy:
    """Mix components from different sources"""
    
    def __init__(self, name: str):
        self.name = name
        self._lifecycle = LifecycleCapability(name)
        self._events: Optional[EventCapability] = None
        
        # Mix different component types
        self.components = []
        self.weights = []
        self.price_history = []
    
    def add_component(self, component: Any, weight: float = 1.0):
        """Add any component that can generate signals"""
        self.components.append(component)
        self.weights.append(weight)
    
    def initialize(self, context):
        """Initialize ensemble and components"""
        self._events = EventCapability()
        self._events.initialize_events(context.event_bus)
        
        # Initialize only components that need it
        for component in self.components:
            if hasattr(component, 'initialize'):
                component.initialize(context)
        
        self._lifecycle.transition_to_initialized()
    
    def initialize_event_subscriptions(self):
        """Set up event subscriptions"""
        if self._events:
            self._events.subscribe("BAR", self.on_bar)
            
        # Subscribe components that support events
        for component in self.components:
            if hasattr(component, 'initialize_event_subscriptions'):
                component.initialize_event_subscriptions()
    
    def on_bar(self, event):
        """Process bar and collect signals from all components"""
        price = event.payload.get('close')
        self.price_history.append(price)
        
        # Collect signals from all components
        signals = []
        for i, component in enumerate(self.components):
            signal = None
            
            # Different components expose signals differently
            if hasattr(component, 'on_bar'):
                # Strategy-style component
                component.on_bar(event)
                if hasattr(component, 'last_signal'):
                    signal = component.last_signal
                    
            elif hasattr(component, 'calculate'):
                # Indicator-style component
                if len(self.price_history) > 1:
                    value = component.calculate(price, self.price_history[-2])
                    # Convert indicator value to signal
                    if hasattr(component, 'threshold'):
                        if value > component.threshold:
                            signal = "BUY"
                        else:
                            signal = "SELL"
                            
            elif callable(component):
                # Function-style component
                try:
                    signal = component(self.price_history)
                except:
                    pass  # Skip if function fails
            
            if signal:
                signals.append((signal, self.weights[i]))
        
        # Combine signals with weighted voting
        if signals:
            combined_signal = self._combine_signals(signals)
            if combined_signal:
                self._emit_signal(combined_signal, price)
    
    def _combine_signals(self, signals: List[tuple]) -> Optional[str]:
        """Weighted voting on signals"""
        buy_weight = sum(w for s, w in signals if s == "BUY")
        sell_weight = sum(w for s, w in signals if s == "SELL")
        
        if buy_weight > sell_weight:
            return "BUY"
        elif sell_weight > buy_weight:
            return "SELL"
        return None
    
    def _emit_signal(self, direction: str, price: float):
        """Emit consensus signal"""
        if self._events and self._events.event_bus:
            from src.core.event import Event
            signal_event = Event("SIGNAL", {
                "symbol": "EURUSD",
                "direction": direction,
                "price": price,
                "strategy": self.name,
                "type": "ensemble"
            })
            self._events.event_bus.publish(signal_event)
    
    def teardown(self):
        """Cleanup all components"""
        for component in self.components:
            if hasattr(component, 'teardown'):
                component.teardown()
        
        if self._events:
            self._events.unsubscribe_all()
        self._lifecycle.transition_to_disposed()
```

## Phase 4: Configuration System (1 week)

### 4.1 Enhanced Configuration Schema

```yaml
# config/components.yaml
capability_profiles:
  minimal:
    description: "Pure calculation, no framework overhead"
    capabilities: []
    
  basic:
    description: "Lifecycle management only"
    capabilities: ["lifecycle"]
    
  trading:
    description: "Full trading component"
    capabilities: ["lifecycle", "events", "reset"]
    
  optimizable:
    description: "Full featured with optimization"
    capabilities: ["lifecycle", "events", "reset", "optimization"]

components:
  # Simple indicator - no overhead
  rsi_calculator:
    class: "SimpleRSI"
    module: "src.indicators.simple_rsi"
    profile: "minimal"
    params:
      period: 14
  
  # External library function
  ta_macd:
    function: "ta.MACD"
    profile: "minimal"
    params:
      fast: 12
      slow: 26
      signal: 9
  
  # Full strategy
  ma_crossover:
    class: "FlexibleMAStrategy"
    module: "src.strategy.flexible_ma_strategy"
    profile: "optimizable"
    params:
      fast_period: 10
      slow_period: 30
    parameter_space:
      fast_period: [5, 10, 15, 20]
      slow_period: [20, 30, 40, 50]
  
  # Ensemble mixing everything
  ensemble:
    class: "EnsembleStrategy"
    module: "src.strategy.ensemble_strategy"
    profile: "trading"
    components:
      - ref: "ma_crossover"
        weight: 0.5
      - ref: "rsi_calculator"
        weight: 0.3
        threshold: 70
      - function: "lambda prices: 'BUY' if prices[-1] > sum(prices[-20:])/20 else 'SELL'"
        weight: 0.2
```

### 4.2 Bootstrap Integration

```python
# src/core/bootstrap.py additions
class Bootstrap:
    def __init__(self):
        super().__init__()
        self.component_factory = ComponentFactory()
    
    def load_components_from_config(self, config_path: str):
        """Load components using new factory system"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create components
        for name, component_config in config.get('components', {}).items():
            # Apply profile if specified
            if 'profile' in component_config:
                profile = config['capability_profiles'][component_config['profile']]
                component_config['capabilities'] = profile['capabilities']
            
            # Create component
            component = self.component_factory.create_from_config(component_config)
            
            # Register in container
            self.container.register(name, component, is_instance=True)
```

## Phase 5: Testing & Migration Tools (2 weeks)

### 5.1 Protocol Compliance Testing

```python
# tests/test_protocol_compliance.py
import unittest
from src.core.protocols import Component, Lifecycle, Optimizable

class ProtocolComplianceTest(unittest.TestCase):
    """Test that components properly implement protocols"""
    
    def test_component_protocol(self, component):
        """Verify Component protocol implementation"""
        self.assertTrue(hasattr(component, 'initialize'))
        self.assertTrue(hasattr(component, 'teardown'))
        
        # Test with mock context
        mock_context = Mock()
        component.initialize(mock_context)
        component.teardown()
    
    def test_lifecycle_protocol(self, component):
        """Verify Lifecycle protocol implementation"""
        if isinstance(component, Lifecycle):
            self.assertTrue(hasattr(component, 'start'))
            self.assertTrue(hasattr(component, 'stop'))
            self.assertTrue(hasattr(component, 'reset'))
```

### 5.2 Migration Helper

```python
# src/tools/migration_helper.py
class MigrationHelper:
    """Tools to help migrate ComponentBase components to composition"""
    
    @staticmethod
    def analyze_component(component_class):
        """Analyze what capabilities a ComponentBase uses"""
        used_capabilities = []
        
        # Check method implementations
        instance = component_class("test", "test")
        
        # Check lifecycle usage
        if hasattr(instance, 'start') and instance.start != ComponentBase.start:
            used_capabilities.append('lifecycle')
            
        # Check event usage
        if hasattr(instance, 'initialize_event_subscriptions') and \
           instance.initialize_event_subscriptions != ComponentBase.initialize_event_subscriptions:
            used_capabilities.append('events')
            
        # Check optimization
        if hasattr(instance, 'get_parameter_space') and \
           instance.get_parameter_space() != {}:
            used_capabilities.append('optimization')
            
        return used_capabilities
    
    @staticmethod
    def generate_composition_code(component_class):
        """Generate composition-based equivalent"""
        capabilities = MigrationHelper.analyze_component(component_class)
        
        # Generate new class code
        code = f"""
class {component_class.__name__}Composed:
    def __init__(self, {self._get_init_params(component_class)}):
        # Initialize capabilities
        {self._generate_capability_init(capabilities)}
        
        # Component-specific initialization
        {self._get_init_body(component_class)}
    
    {self._generate_protocol_methods(capabilities)}
    
    {self._get_business_methods(component_class)}
"""
        return code
```

## Migration Timeline

### Month 1: Foundation
- Week 1-2: Implement protocols and capabilities
- Week 3: Update Bootstrap and create ComponentFactory
- Week 4: Create example components and test

### Month 2: Integration
- Week 1-2: Enhanced configuration system
- Week 3: Migration tools and helpers
- Week 4: Documentation and training

### Month 3-6: Gradual Migration
- Migrate components module by module
- Start with new components using composition
- Gradually convert existing components
- Maintain backward compatibility throughout

## Key Benefits Realized

1. **Zero Overhead**: Simple components have no framework baggage
2. **Mix Anything**: Integrate sklearn, TA-Lib, custom functions seamlessly
3. **Granular Control**: Configure exactly what each component needs
4. **Better Testing**: Test business logic without framework setup
5. **Future Proof**: Easy to integrate new libraries and approaches

## Risk Mitigation

1. **Backward Compatibility**: Both approaches work together
2. **Gradual Migration**: No big-bang rewrite needed
3. **Testing Coverage**: Protocol compliance tests ensure correctness
4. **Documentation**: Clear examples and migration guides
5. **Tooling Support**: Migration helpers ease transition

This approach gives you all the benefits of composition while minimizing disruption to the existing system.