# ADMF-Trader: Universal Scoped Container Architecture with Protocol + Composition

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Universal Container Pattern](#2-universal-container-pattern)
3. [Container Lifecycle Management](#3-container-lifecycle-management)
4. [Use Case Implementations](#4-use-case-implementations)
5. [Performance & Resource Management](#5-performance--resource-management)
6. [Integration Patterns](#6-integration-patterns)
7. [Best Practices & Guidelines](#7-best-practices--guidelines)

---

## 1. Architectural Overview

The Universal Scoped Container Architecture provides a consistent isolation mechanism across all execution contexts in ADMF-Trader. By combining Protocol + Composition with scoped containers, we achieve complete flexibility and isolation without inheritance overhead.

### 1.1 Core Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADMF-Trader System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Shared Read-Only Layer                      │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │  │
│  │  │ Market Data │ │Configuration│ │ Historical Data │   │  │
│  │  │    Feed     │ │   Service   │ │     Store       │   │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴─────────────────────────────┐  │
│  │              Container Orchestration Layer               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │  │
│  │  │ Optimization│ │   Backtest  │ │  Live Trading   │   │  │
│  │  │Orchestrator │ │Orchestrator │ │  Orchestrator   │   │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴─────────────────────────────┐  │
│  │                  Scoped Container Layer                  │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │  Container 1  │  │  Container 2  │  │  Container N  │ │  │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │ │  │
│  │  │ │ Strategy │ │  │ │ Strategy │ │  │ │ Strategy │ │ │  │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │ │  │
│  │  │ │   Risk   │ │  │ │   Risk   │ │  │ │   Risk   │ │ │  │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │ │  │
│  │  │ │Portfolio │ │  │ │Portfolio │ │  │ │Portfolio │ │ │  │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │ │  │
│  │  │ │  Broker  │ │  │ │  Broker  │ │  │ │  Broker  │ │ │  │
│  │  │ ├──────────┤ │  │ ├──────────┤ │  │ ├──────────┤ │ │  │
│  │  │ │Event Bus │ │  │ │Event Bus │ │  │ │Event Bus │ │ │  │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Principles

1. **Universal Isolation**: Every execution context uses the same container pattern
2. **Protocol-Based Components**: Components implement protocols, not inherit from base classes
3. **Shared Nothing Architecture**: Containers share only read-only data
4. **Lifecycle Independence**: Each container manages its own component lifecycles
5. **Resource Efficiency**: Shared read-only layer minimizes memory usage

---

## 2. Universal Container Pattern

### 2.1 Container Protocol Definition

```python
from typing import Protocol, Any, Dict, Optional, List
from abc import abstractmethod

@runtime_checkable
class ScopedContainer(Protocol):
    """Protocol for scoped containers"""
    
    @abstractmethod
    def register_shared_service(self, name: str, service: Any) -> None:
        """Register a shared read-only service"""
        ...
    
    @abstractmethod
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create and register a component within this scope"""
        ...
    
    @abstractmethod
    def resolve(self, name: str) -> Any:
        """Resolve a component or service"""
        ...
    
    @abstractmethod
    def initialize_scope(self) -> None:
        """Initialize all components in dependency order"""
        ...
    
    @abstractmethod
    def teardown_scope(self) -> None:
        """Teardown all components in reverse order"""
        ...

class UniversalScopedContainer:
    """Universal implementation of scoped container"""
    
    def __init__(self, container_id: str, shared_services: Dict[str, Any] = None):
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.local_components: Dict[str, Any] = {}
        self.component_factory = ComponentFactory()
        self.dependency_graph = DependencyGraph()
        self._lock = threading.RLock()
        
        # Container-specific services
        self.event_bus = EventBus()  # Each container gets its own
        self.logger = StructuredLogger(f"container.{container_id}")
        
    def register_shared_service(self, name: str, service: Any) -> None:
        """Register a shared read-only service"""
        with self._lock:
            self.shared_services[name] = service
    
    def create_component(self, spec: Dict[str, Any]) -> Any:
        """Create component with automatic protocol detection"""
        component = self.component_factory.create_component(spec)
        component_name = spec.get('name', component.__class__.__name__)
        
        with self._lock:
            self.local_components[component_name] = component
            
            # Track dependencies
            dependencies = spec.get('dependencies', [])
            self.dependency_graph.add_component(component_name, spec)
            for dep in dependencies:
                self.dependency_graph.add_dependency(component_name, dep)
        
        return component
    
    def resolve(self, name: str) -> Any:
        """Resolve from local components first, then shared services"""
        with self._lock:
            # Check local components first
            if name in self.local_components:
                return self.local_components[name]
            
            # Check container services
            if name == "event_bus":
                return self.event_bus
            if name == "logger":
                return self.logger
            
            # Fall back to shared services
            if name in self.shared_services:
                return self.shared_services[name]
            
            raise ValueError(f"Component '{name}' not found in container {self.container_id}")
    
    def initialize_scope(self) -> None:
        """Initialize all components respecting dependencies"""
        init_order = self.dependency_graph.get_initialization_order()
        
        context = SystemContext(
            config=self.resolve("config") if "config" in self.shared_services else Config(),
            event_bus=self.event_bus,
            container=self,
            logger=self.logger,
            run_mode=self._determine_run_mode()
        )
        
        for component_name in init_order:
            component = self.local_components[component_name]
            
            # Initialize if component supports it (duck typing)
            if hasattr(component, 'initialize') and callable(getattr(component, 'initialize')):
                component.initialize(context)
            
            # Set up event subscriptions if supported
            if hasattr(component, 'initialize_event_subscriptions'):
                component.initialize_event_subscriptions()
    
    def teardown_scope(self) -> None:
        """Teardown all components in reverse order"""
        teardown_order = reversed(self.dependency_graph.get_initialization_order())
        
        for component_name in teardown_order:
            component = self.local_components.get(component_name)
            if component and hasattr(component, 'teardown'):
                try:
                    component.teardown()
                except Exception as e:
                    self.logger.error(f"Error tearing down {component_name}: {e}")
        
        # Clear local components
        self.local_components.clear()
        self.dependency_graph = DependencyGraph()
```

### 2.2 Container Factory

```python
class ContainerFactory:
    """Factory for creating specialized containers"""
    
    def __init__(self, shared_services_provider: 'SharedServicesProvider'):
        self.shared_services_provider = shared_services_provider
        self.container_count = 0
        self._lock = threading.Lock()
    
    def create_optimization_container(self, parameters: Dict[str, Any]) -> UniversalScopedContainer:
        """Create container for optimization trial"""
        container_id = self._generate_container_id("opt")
        shared_services = self.shared_services_provider.get_optimization_services()
        
        container = UniversalScopedContainer(container_id, shared_services)
        
        # Create components with specific parameters
        container.create_component({
            'name': 'strategy',
            'class': parameters['strategy_class'],
            'params': parameters['strategy_params'],
            'capabilities': ['lifecycle', 'events', 'optimization']
        })
        
        container.create_component({
            'name': 'portfolio',
            'class': 'Portfolio',
            'params': {'initial_cash': 100000},
            'capabilities': ['lifecycle', 'events', 'reset']
        })
        
        # ... create other components ...
        
        return container
    
    def create_live_trading_container(self, strategy_spec: Dict[str, Any]) -> UniversalScopedContainer:
        """Create container for live trading strategy"""
        container_id = self._generate_container_id("live")
        shared_services = self.shared_services_provider.get_live_trading_services()
        
        container = UniversalScopedContainer(container_id, shared_services)
        
        # Create components for live trading
        container.create_component({
            'name': 'strategy',
            'class': strategy_spec['class'],
            'params': strategy_spec['params'],
            'capabilities': ['lifecycle', 'events', 'monitoring']
        })
        
        # Virtual portfolio for tracking
        container.create_component({
            'name': 'portfolio',
            'class': 'VirtualPortfolio',
            'params': {
                'strategy_id': container_id,
                'initial_capital': strategy_spec.get('allocated_capital', 100000)
            }
        })
        
        return container
    
    def create_test_container(self, test_spec: Dict[str, Any]) -> UniversalScopedContainer:
        """Create container for testing/development"""
        container_id = self._generate_container_id("test")
        shared_services = self.shared_services_provider.get_test_services()
        
        container = UniversalScopedContainer(container_id, shared_services)
        
        # Flexible component creation for testing
        for component_spec in test_spec.get('components', []):
            container.create_component(component_spec)
        
        return container
    
    def _generate_container_id(self, prefix: str) -> str:
        """Generate unique container ID"""
        with self._lock:
            self.container_count += 1
            return f"{prefix}_{self.container_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### 2.3 Shared Services Provider

```python
class SharedServicesProvider:
    """Manages shared read-only services across containers"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialize_core_services()
    
    def _initialize_core_services(self) -> None:
        """Initialize core shared services"""
        # Configuration service
        self._services['config'] = Config.load("config/system.yaml")
        
        # Market data service (read-only interface)
        self._services['market_data'] = MarketDataService()
        
        # Historical data store
        self._services['historical_data'] = HistoricalDataStore()
        
        # System monitor (read-only metrics)
        self._services['system_monitor'] = SystemMonitor()
    
    def get_optimization_services(self) -> Dict[str, Any]:
        """Get services for optimization containers"""
        return {
            'config': self._services['config'],
            'historical_data': self._services['historical_data'],
            'market_data': ReadOnlyDataView(self._services['market_data'])
        }
    
    def get_live_trading_services(self) -> Dict[str, Any]:
        """Get services for live trading containers"""
        return {
            'config': self._services['config'],
            'market_data': self._services['market_data'],
            'order_router': self._services.get('order_router'),  # Thread-safe router
            'system_monitor': self._services['system_monitor']
        }
    
    def get_test_services(self) -> Dict[str, Any]:
        """Get services for test containers"""
        return {
            'config': self._services['config'],
            'historical_data': self._services['historical_data']
        }
```

---

## 3. Container Lifecycle Management

### 3.1 Lifecycle Flow Diagram

```
┌─────────────────┐
│   Container     │
│   Creation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Register Shared │
│   Services      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Create Components│
│  (Lazy Init)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Dep.     │
│    Graph        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Initialize    │
│   Components    │
│ (Dependency     │
│    Order)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Active       │
│  (Processing)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Teardown     │
│  (Reverse       │
│   Order)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Container     │
│   Destroyed     │
└─────────────────┘
```

### 3.2 Lifecycle Manager

```python
class ContainerLifecycleManager:
    """Manages lifecycle of multiple containers"""
    
    def __init__(self):
        self.active_containers: Dict[str, UniversalScopedContainer] = {}
        self.container_factory = ContainerFactory(SharedServicesProvider())
        self._lock = threading.RLock()
    
    def create_and_start_container(self, container_type: str, 
                                  spec: Dict[str, Any]) -> str:
        """Create, initialize, and start a container"""
        # Create appropriate container
        if container_type == "optimization":
            container = self.container_factory.create_optimization_container(spec)
        elif container_type == "live_trading":
            container = self.container_factory.create_live_trading_container(spec)
        elif container_type == "test":
            container = self.container_factory.create_test_container(spec)
        else:
            raise ValueError(f"Unknown container type: {container_type}")
        
        # Initialize container
        container.initialize_scope()
        
        # Start components
        self._start_container_components(container)
        
        # Track active container
        with self._lock:
            self.active_containers[container.container_id] = container
        
        return container.container_id
    
    def stop_and_destroy_container(self, container_id: str) -> None:
        """Stop and destroy a container"""
        with self._lock:
            if container_id not in self.active_containers:
                return
            
            container = self.active_containers[container_id]
        
        # Stop components
        self._stop_container_components(container)
        
        # Teardown
        container.teardown_scope()
        
        # Remove from tracking
        with self._lock:
            del self.active_containers[container_id]
    
    def _start_container_components(self, container: UniversalScopedContainer) -> None:
        """Start all components that support starting"""
        for name, component in container.local_components.items():
            if hasattr(component, 'start') and callable(getattr(component, 'start')):
                try:
                    component.start()
                except Exception as e:
                    container.logger.error(f"Error starting {name}: {e}")
    
    def _stop_container_components(self, container: UniversalScopedContainer) -> None:
        """Stop all components in reverse order"""
        for name in reversed(list(container.local_components.keys())):
            component = container.local_components[name]
            if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                try:
                    component.stop()
                except Exception as e:
                    container.logger.error(f"Error stopping {name}: {e}")
```

---

## 4. Use Case Implementations

### 4.1 Parallel Optimization

```python
class ParallelOptimizationOrchestrator:
    """Orchestrates parallel optimization using containers"""
    
    def __init__(self, optimization_config: Dict[str, Any]):
        self.config = optimization_config
        self.lifecycle_manager = ContainerLifecycleManager()
        self.results_collector = OptimizationResultsCollector()
        self.shared_data_handler = self._create_shared_data_handler()
    
    def run_optimization(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run parallel optimization with scoped containers"""
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_space)
        
        # Determine batch size based on resources
        batch_size = self._calculate_optimal_batch_size(len(param_combinations))
        
        # Process in batches
        all_results = []
        
        for batch_start in range(0, len(param_combinations), batch_size):
            batch_end = min(batch_start + batch_size, len(param_combinations))
            batch_params = param_combinations[batch_start:batch_end]
            
            # Run batch in parallel
            batch_results = self._run_optimization_batch(batch_params)
            all_results.extend(batch_results)
            
            # Memory cleanup between batches
            gc.collect()
        
        # Find best parameters
        return self._analyze_optimization_results(all_results)
    
    def _run_optimization_batch(self, batch_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a batch of optimizations in parallel containers"""
        with ThreadPoolExecutor(max_workers=len(batch_params)) as executor:
            futures = []
            
            for params in batch_params:
                future = executor.submit(self._run_single_optimization, params)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Optimization failed: {e}")
                    results.append({'error': str(e), 'params': params})
            
            return results
    
    def _run_single_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run single optimization in isolated container"""
        # Create container
        container_id = self.lifecycle_manager.create_and_start_container(
            "optimization", 
            parameters
        )
        
        try:
            container = self.lifecycle_manager.active_containers[container_id]
            
            # Get components
            strategy = container.resolve("strategy")
            portfolio = container.resolve("portfolio")
            data_handler = container.resolve("data_handler")
            
            # Run backtest
            results = self._execute_backtest(container)
            
            # Add container metrics
            results['container_id'] = container_id
            results['parameters'] = parameters
            results['memory_usage'] = self._get_container_memory_usage(container)
            
            return results
            
        finally:
            # Always cleanup container
            self.lifecycle_manager.stop_and_destroy_container(container_id)
```

### 4.2 Out-of-Sample Testing

```python
class OutOfSampleTestRunner:
    """Runs OOS tests using optimization insights"""
    
    def __init__(self, optimization_results: Dict[str, Any]):
        self.optimization_results = optimization_results
        self.lifecycle_manager = ContainerLifecycleManager()
        self.strategy_generator = OOSStrategyGenerator(optimization_results)
    
    def run_oos_tests(self, test_data: Any) -> Dict[str, Any]:
        """Run out-of-sample tests with generated strategies"""
        # Generate OOS strategies based on optimization data
        oos_strategies = self.strategy_generator.generate_strategies()
        
        # Run each strategy in isolated container
        test_results = {}
        
        with ProcessPoolExecutor() as executor:
            futures = {}
            
            for strategy_name, strategy_spec in oos_strategies.items():
                future = executor.submit(
                    self._run_oos_strategy,
                    strategy_name,
                    strategy_spec,
                    test_data
                )
                futures[future] = strategy_name
            
            # Collect results
            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    result = future.result()
                    test_results[strategy_name] = result
                except Exception as e:
                    test_results[strategy_name] = {'error': str(e)}
        
        return self._analyze_oos_results(test_results)
    
    def _run_oos_strategy(self, strategy_name: str, 
                         strategy_spec: Dict[str, Any],
                         test_data: Any) -> Dict[str, Any]:
        """Run single OOS strategy in isolated container"""
        # Create test container
        test_spec = {
            'components': [
                strategy_spec,
                {'name': 'portfolio', 'class': 'Portfolio', 'params': {'initial_cash': 100000}},
                {'name': 'risk_manager', 'class': 'RiskManager'},
                {'name': 'broker', 'class': 'SimulatedBroker'}
            ]
        }
        
        container_id = self.lifecycle_manager.create_and_start_container("test", test_spec)
        
        try:
            # Run test
            return self._execute_test(container_id, test_data)
        finally:
            self.lifecycle_manager.stop_and_destroy_container(container_id)

class OOSStrategyGenerator:
    """Generates out-of-sample test strategies"""
    
    def __init__(self, optimization_results: Dict[str, Any]):
        self.optimization_results = optimization_results
    
    def generate_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Generate diverse OOS strategies"""
        strategies = {}
        
        # 1. Best parameters strategy
        strategies['best_params'] = {
            'name': 'best_params_strategy',
            'class': self.optimization_results['strategy_class'],
            'params': self.optimization_results['best_parameters']
        }
        
        # 2. Robust parameters (good across multiple regimes)
        strategies['robust_params'] = self._generate_robust_strategy()
        
        # 3. Ensemble of top N
        strategies['ensemble_top5'] = self._generate_ensemble_strategy(top_n=5)
        
        # 4. Contrarian for worst regime
        strategies['contrarian'] = self._generate_contrarian_strategy()
        
        # 5. Adaptive strategy switching between parameter sets
        strategies['adaptive'] = self._generate_adaptive_strategy()
        
        return strategies
```

### 4.3 Live Trading Multi-Strategy

```python
class LiveTradingOrchestrator:
    """Manages multiple live trading strategies in isolation"""
    
    def __init__(self, broker_connection: Any):
        self.broker_connection = broker_connection
        self.lifecycle_manager = ContainerLifecycleManager()
        self.position_aggregator = PositionAggregator()
        self.risk_monitor = SystemRiskMonitor()
        self.strategy_threads: Dict[str, Thread] = {}
    
    def add_live_strategy(self, strategy_id: str, 
                         strategy_spec: Dict[str, Any]) -> None:
        """Add a live trading strategy"""
        # Create live trading container
        container_id = self.lifecycle_manager.create_and_start_container(
            "live_trading",
            strategy_spec
        )
        
        # Map strategy ID to container
        strategy_spec['container_id'] = container_id
        
        # Start strategy thread
        thread = Thread(
            target=self._run_live_strategy,
            args=(strategy_id, container_id),
            name=f"LiveStrategy_{strategy_id}"
        )
        thread.daemon = True
        self.strategy_threads[strategy_id] = thread
        thread.start()
        
        self.logger.info(f"Started live strategy {strategy_id} in container {container_id}")
    
    def _run_live_strategy(self, strategy_id: str, container_id: str) -> None:
        """Run live strategy in its container"""
        try:
            container = self.lifecycle_manager.active_containers[container_id]
            strategy = container.resolve("strategy")
            
            # Strategy main loop
            while self._should_run_strategy(strategy_id):
                try:
                    # Each strategy processes its own events
                    # via its isolated event bus
                    time.sleep(0.001)
                    
                    # Periodic position reporting
                    if self._should_report_positions():
                        self._report_strategy_positions(strategy_id, container)
                    
                except Exception as e:
                    self.logger.error(f"Error in strategy {strategy_id}: {e}")
                    self._handle_strategy_error(strategy_id, e)
                    
        except Exception as e:
            self.logger.critical(f"Fatal error in strategy {strategy_id}: {e}")
        finally:
            self.lifecycle_manager.stop_and_destroy_container(container_id)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get aggregated system state across all strategies"""
        state = {
            'active_strategies': len(self.strategy_threads),
            'total_positions': {},
            'strategy_states': {},
            'system_metrics': self.risk_monitor.get_metrics()
        }
        
        # Aggregate positions across all containers
        for strategy_id, thread in self.strategy_threads.items():
            if thread.is_alive():
                container_id = self._get_container_id(strategy_id)
                container = self.lifecycle_manager.active_containers.get(container_id)
                
                if container:
                    portfolio = container.resolve("portfolio")
                    positions = portfolio.get_all_positions()
                    
                    # Add to aggregated positions
                    for symbol, position in positions.items():
                        if symbol not in state['total_positions']:
                            state['total_positions'][symbol] = 0
                        state['total_positions'][symbol] += position.quantity
                    
                    # Strategy-specific state
                    state['strategy_states'][strategy_id] = {
                        'positions': positions,
                        'portfolio_value': portfolio.get_portfolio_value(),
                        'container_id': container_id
                    }
        
        return state
```

### 4.4 A/B Testing in Production

```python
class ProductionABTestManager:
    """Manages A/B tests in production using containers"""
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.lifecycle_manager = ContainerLifecycleManager()
        self.test_results: Dict[str, Any] = {}
        self.test_start_time = datetime.now()
    
    def setup_ab_test(self, test_config: Dict[str, Any]) -> None:
        """Set up A/B test with control and variants"""
        control_allocation = test_config['control_allocation']  # e.g., 80%
        variant_allocations = test_config['variant_allocations']  # e.g., {'v1': 10%, 'v2': 10%}
        
        # Set up control strategy
        control_capital = self.total_capital * (control_allocation / 100)
        self._setup_test_strategy(
            'control',
            test_config['control_strategy'],
            control_capital
        )
        
        # Set up variant strategies
        for variant_id, allocation in variant_allocations.items():
            variant_capital = self.total_capital * (allocation / 100)
            self._setup_test_strategy(
                variant_id,
                test_config['variants'][variant_id],
                variant_capital
            )
    
    def _setup_test_strategy(self, test_id: str, 
                           strategy_spec: Dict[str, Any],
                           allocated_capital: float) -> None:
        """Set up individual test strategy in container"""
        # Enhance spec with test metadata
        enhanced_spec = strategy_spec.copy()
        enhanced_spec['allocated_capital'] = allocated_capital
        enhanced_spec['test_id'] = test_id
        enhanced_spec['test_start'] = self.test_start_time
        
        # Create container
        container_id = self.lifecycle_manager.create_and_start_container(
            "live_trading",
            enhanced_spec
        )
        
        # Track test
        self.test_results[test_id] = {
            'container_id': container_id,
            'start_time': datetime.now(),
            'allocated_capital': allocated_capital,
            'performance_metrics': []
        }
    
    def get_test_results(self) -> Dict[str, Any]:
        """Get current A/B test results"""
        results = {
            'test_duration': (datetime.now() - self.test_start_time).total_seconds() / 3600,
            'variants': {}
        }
        
        for test_id, test_info in self.test_results.items():
            container = self.lifecycle_manager.active_containers.get(test_info['container_id'])
            
            if container:
                portfolio = container.resolve("portfolio")
                current_value = portfolio.get_portfolio_value()
                
                results['variants'][test_id] = {
                    'current_value': current_value,
                    'return': (current_value - test_info['allocated_capital']) / test_info['allocated_capital'],
                    'sharpe_ratio': self._calculate_sharpe(test_id),
                    'max_drawdown': self._calculate_max_drawdown(test_id),
                    'trade_count': len(portfolio.trades)
                }
        
        # Statistical significance testing
        if len(results['variants']) > 1:
            results['statistical_analysis'] = self._perform_statistical_tests(results['variants'])
        
        return results
```

---

## 5. Performance & Resource Management

### 5.1 Memory Management Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Memory Layout                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Shared Read-Only Memory                 │  │
│  │   ┌─────────────┐      ┌──────────────┐        │  │
│  │   │ Market Data │      │ Historical   │        │  │
│  │   │   (mmap)    │      │ Data (mmap)  │        │  │
│  │   └─────────────┘      └──────────────┘        │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Per-Container Memory                  │  │
│  │                                                  │  │
│  │  Container 1          Container 2         ...   │  │
│  │  ┌──────────┐        ┌──────────┐              │  │
│  │  │ Strategy │        │ Strategy │              │  │
│  │  │   State  │        │   State  │              │  │
│  │  ├──────────┤        ├──────────┤              │  │
│  │  │Portfolio │        │Portfolio │              │  │
│  │  │  State   │        │  State   │              │  │
│  │  ├──────────┤        ├──────────┤              │  │
│  │  │  Event   │        │  Event   │              │  │
│  │  │  Queue   │        │  Queue   │              │  │
│  │  └──────────┘        └──────────┘              │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Resource Manager

```python
class ContainerResourceManager:
    """Manages resources across all containers"""
    
    def __init__(self, max_memory_gb: float = 16.0, 
                 max_containers: int = 100):
        self.max_memory_gb = max_memory_gb
        self.max_containers = max_containers
        self.container_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def can_create_container(self) -> bool:
        """Check if resources available for new container"""
        with self._lock:
            current_containers = len(self.container_metrics)
            current_memory = self._get_total_memory_usage()
            
            # Check limits
            if current_containers >= self.max_containers:
                return False
            
            # Estimate memory for new container (conservative)
            estimated_new_container_memory = 0.1  # 100MB estimate
            
            if current_memory + estimated_new_container_memory > self.max_memory_gb:
                return False
            
            return True
    
    def register_container(self, container_id: str) -> None:
        """Register new container for monitoring"""
        with self._lock:
            self.container_metrics[container_id] = {
                'created_at': datetime.now(),
                'memory_usage': 0,
                'cpu_usage': 0,
                'event_count': 0
            }
    
    def update_container_metrics(self, container_id: str, 
                               metrics: Dict[str, Any]) -> None:
        """Update container resource metrics"""
        with self._lock:
            if container_id in self.container_metrics:
                self.container_metrics[container_id].update(metrics)
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for resource optimization"""
        with self._lock:
            total_memory = self._get_total_memory_usage()
            container_count = len(self.container_metrics)
            
            recommendations = {
                'optimal_batch_size': self._calculate_optimal_batch_size(),
                'memory_pressure': total_memory / self.max_memory_gb,
                'suggested_actions': []
            }
            
            # High memory pressure
            if recommendations['memory_pressure'] > 0.8:
                recommendations['suggested_actions'].append(
                    "Consider reducing batch size or using disk-based data storage"
                )
            
            # Many idle containers
            idle_containers = self._find_idle_containers()
            if len(idle_containers) > container_count * 0.2:
                recommendations['suggested_actions'].append(
                    f"Consider removing {len(idle_containers)} idle containers"
                )
            
            return recommendations
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on resources"""
        available_memory = self.max_memory_gb - self._get_total_memory_usage()
        avg_container_memory = 0.1  # 100MB average
        
        # Leave 20% buffer
        usable_memory = available_memory * 0.8
        optimal_batch = int(usable_memory / avg_container_memory)
        
        # Apply reasonable limits
        return max(1, min(optimal_batch, 50))
```

### 5.3 Performance Monitoring

```python
class ContainerPerformanceMonitor:
    """Monitors performance across containers"""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.aggregated_metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def record_container_metric(self, container_id: str, 
                              metric_name: str, 
                              value: float) -> None:
        """Record performance metric for container"""
        with self._lock:
            if container_id not in self.metrics_buffer:
                self.metrics_buffer[container_id] = []
            
            self.metrics_buffer[container_id].append({
                'timestamp': datetime.now(),
                'metric': metric_name,
                'value': value
            })
            
            # Limit buffer size
            if len(self.metrics_buffer[container_id]) > 1000:
                self.metrics_buffer[container_id].pop(0)
    
    def get_container_performance_summary(self, container_id: str) -> Dict[str, Any]:
        """Get performance summary for specific container"""
        with self._lock:
            if container_id not in self.metrics_buffer:
                return {}
            
            metrics = self.metrics_buffer[container_id]
            
            # Group by metric name
            grouped = {}
            for entry in metrics:
                metric_name = entry['metric']
                if metric_name not in grouped:
                    grouped[metric_name] = []
                grouped[metric_name].append(entry['value'])
            
            # Calculate statistics
            summary = {}
            for metric_name, values in grouped.items():
                if values:
                    summary[metric_name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95)
                    }
            
            return summary
    
    def get_system_performance_dashboard(self) -> Dict[str, Any]:
        """Get system-wide performance dashboard"""
        with self._lock:
            active_containers = len(self.metrics_buffer)
            
            # Aggregate metrics across all containers
            all_metrics = []
            for container_metrics in self.metrics_buffer.values():
                all_metrics.extend(container_metrics)
            
            # Calculate system-wide statistics
            if all_metrics:
                recent_metrics = [m for m in all_metrics 
                                if (datetime.now() - m['timestamp']).seconds < 300]  # Last 5 minutes
                
                return {
                    'active_containers': active_containers,
                    'total_events_processed': len(recent_metrics),
                    'events_per_second': len(recent_metrics) / 300,
                    'container_performance': self._aggregate_container_performance(),
                    'system_health': self._calculate_system_health()
                }
            
            return {'active_containers': 0, 'status': 'no_data'}
```

---

## 6. Integration Patterns

### 6.1 Component Discovery and Registration

```python
class ContainerComponentRegistry:
    """Registry for components that can be used in containers"""
    
    def __init__(self):
        self.component_specs: Dict[str, Dict[str, Any]] = {}
        self.protocol_implementations: Dict[str, List[str]] = {}
        self._scan_for_components()
    
    def _scan_for_components(self) -> None:
        """Scan for available components"""
        # Scan built-in components
        self._register_builtin_components()
        
        # Scan for user components
        self._scan_user_components()
    
    def _register_builtin_components(self) -> None:
        """Register built-in components"""
        # Strategies
        self.register_component('TrendFollowingStrategy', {
            'class': 'strategies.TrendFollowingStrategy',
            'protocols': ['SignalGenerator', 'Optimizable', 'Resettable'],
            'default_params': {'fast_period': 10, 'slow_period': 30},
            'capabilities': ['lifecycle', 'events', 'optimization']
        })
        
        # Risk Management
        self.register_component('RiskManager', {
            'class': 'risk.RiskManager',
            'protocols': ['EventSubscriber', 'Configurable'],
            'default_params': {'max_position_size': 1000},
            'capabilities': ['lifecycle', 'events']
        })
        
        # Portfolio
        self.register_component('Portfolio', {
            'class': 'risk.Portfolio',
            'protocols': ['EventSubscriber', 'Resettable'],
            'default_params': {'initial_cash': 100000},
            'capabilities': ['lifecycle', 'events', 'reset']
        })
    
    def register_component(self, name: str, spec: Dict[str, Any]) -> None:
        """Register a component specification"""
        self.component_specs[name] = spec
        
        # Track protocol implementations
        for protocol in spec.get('protocols', []):
            if protocol not in self.protocol_implementations:
                self.protocol_implementations[protocol] = []
            self.protocol_implementations[protocol].append(name)
    
    def find_components_by_protocol(self, protocol: str) -> List[str]:
        """Find all components implementing a protocol"""
        return self.protocol_implementations.get(protocol, [])
    
    def get_component_spec(self, name: str) -> Dict[str, Any]:
        """Get component specification"""
        return self.component_specs.get(name, {})
```

### 6.2 Inter-Container Communication

```python
class InterContainerMessageBus:
    """Enables controlled communication between containers"""
    
    def __init__(self):
        self.channels: Dict[str, List[Callable]] = {}
        self.container_subscriptions: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def create_channel(self, channel_name: str, 
                      access_policy: Dict[str, Any]) -> None:
        """Create communication channel with access policy"""
        with self._lock:
            self.channels[channel_name] = {
                'subscribers': [],
                'policy': access_policy,
                'created_at': datetime.now()
            }
    
    def subscribe_container(self, container_id: str, 
                          channel_name: str,
                          handler: Callable) -> bool:
        """Subscribe container to channel if allowed"""
        with self._lock:
            if channel_name not in self.channels:
                return False
            
            # Check access policy
            policy = self.channels[channel_name]['policy']
            if not self._check_access_policy(container_id, policy):
                return False
            
            # Add subscription
            self.channels[channel_name]['subscribers'].append({
                'container_id': container_id,
                'handler': handler
            })
            
            # Track container subscriptions
            if container_id not in self.container_subscriptions:
                self.container_subscriptions[container_id] = []
            self.container_subscriptions[container_id].append(channel_name)
            
            return True
    
    def publish_to_channel(self, channel_name: str, 
                         message: Dict[str, Any],
                         sender_container_id: str) -> None:
        """Publish message to channel"""
        with self._lock:
            if channel_name not in self.channels:
                return
            
            # Add sender info
            message['sender_container'] = sender_container_id
            message['timestamp'] = datetime.now()
            
            # Deliver to subscribers
            for subscription in self.channels[channel_name]['subscribers']:
                if subscription['container_id'] != sender_container_id:  # No self-delivery
                    try:
                        subscription['handler'](message)
                    except Exception as e:
                        print(f"Error delivering message: {e}")
    
    def _check_access_policy(self, container_id: str, 
                           policy: Dict[str, Any]) -> bool:
        """Check if container meets access policy"""
        # Example policy checks
        if 'allowed_types' in policy:
            container_type = self._get_container_type(container_id)
            if container_type not in policy['allowed_types']:
                return False
        
        if 'max_subscribers' in policy:
            current_subs = len(self.channels[channel_name]['subscribers'])
            if current_subs >= policy['max_subscribers']:
                return False
        
        return True
```

---

## 7. Best Practices & Guidelines

### 7.1 Container Design Patterns

#### 7.1.1 Minimal Container Pattern
```python
# For simple, single-purpose operations
def create_minimal_container(strategy_func: Callable) -> UniversalScopedContainer:
    """Create minimal container for simple strategies"""
    container = UniversalScopedContainer("minimal")
    
    # Only essential components
    container.create_component({
        'name': 'strategy',
        'function': strategy_func,
        'capabilities': []  # No framework overhead
    })
    
    container.create_component({
        'name': 'data_handler',
        'class': 'SimpleDataHandler',
        'capabilities': ['reset']  # Only what's needed
    })
    
    return container
```

#### 7.1.2 Full-Featured Container Pattern
```python
# For complex production strategies
def create_production_container(strategy_config: Dict[str, Any]) -> UniversalScopedContainer:
    """Create fully-featured container for production"""
    container = UniversalScopedContainer("production")
    
    # All production capabilities
    components = [
        {
            'name': 'strategy',
            'class': strategy_config['class'],
            'params': strategy_config['params'],
            'capabilities': ['lifecycle', 'events', 'optimization', 
                           'monitoring', 'error_handling']
        },
        {
            'name': 'risk_manager',
            'class': 'EnhancedRiskManager',
            'capabilities': ['lifecycle', 'events', 'monitoring']
        },
        {
            'name': 'portfolio',
            'class': 'ProductionPortfolio',
            'capabilities': ['lifecycle', 'events', 'reset', 'monitoring']
        },
        {
            'name': 'performance_tracker',
            'class': 'PerformanceTracker',
            'capabilities': ['lifecycle', 'monitoring']
        }
    ]
    
    for component_spec in components:
        container.create_component(component_spec)
    
    return container
```

### 7.2 Resource Management Guidelines

1. **Memory Limits**: Set explicit memory limits per container type
2. **Batch Sizing**: Use adaptive batch sizing based on available resources
3. **Cleanup**: Always use try/finally blocks for container cleanup
4. **Monitoring**: Track container metrics for optimization
5. **Pooling**: Consider container pooling for frequently created/destroyed patterns

### 7.3 Testing Patterns

```python
class ContainerTestFramework:
    """Framework for testing containerized components"""
    
    def test_strategy_in_isolation(self, strategy_class: type) -> None:
        """Test strategy in complete isolation"""
        # Create test container
        container = UniversalScopedContainer("test")
        
        # Create strategy with test dependencies
        container.create_component({
            'name': 'strategy',
            'class': strategy_class,
            'params': {'test_mode': True}
        })
        
        # Mock dependencies
        container.register_shared_service('market_data', MockMarketData())
        
        # Initialize and test
        container.initialize_scope()
        
        try:
            strategy = container.resolve('strategy')
            # Run tests...
        finally:
            container.teardown_scope()
    
    def test_container_isolation(self) -> None:
        """Verify containers are truly isolated"""
        container1 = UniversalScopedContainer("test1")
        container2 = UniversalScopedContainer("test2")
        
        # Create same component in both
        for container in [container1, container2]:
            container.create_component({
                'name': 'counter',
                'class': 'SimpleCounter'
            })
        
        # Verify isolation
        counter1 = container1.resolve('counter')
        counter2 = container2.resolve('counter')
        
        counter1.increment()
        assert counter1.value == 1
        assert counter2.value == 0  # Isolated!
```

### 7.4 Production Deployment Checklist

- [ ] Container resource limits configured
- [ ] Monitoring enabled for all containers
- [ ] Error boundaries implemented
- [ ] Graceful shutdown handlers registered
- [ ] Inter-container communication policies defined
- [ ] Performance baselines established
- [ ] Disaster recovery procedures documented
- [ ] Container health checks implemented
- [ ] Audit logging enabled
- [ ] Security policies enforced

---

## 8. Optimization Pattern: Shared Indicator Containers

### 8.1 The Redundant Calculation Problem

When optimizing strategies with overlapping indicators, the naive approach recalculates the same indicators multiple times:

```
Traditional Approach:
- Strategy MA(5,10):  Container calculates MA(5), MA(10), RSI(14)
- Strategy MA(10,20): Container calculates MA(10), MA(20), RSI(14)
- Strategy MA(5,20):  Container calculates MA(5), MA(20), RSI(14)

Problem: MA(5), MA(10), MA(20), and RSI(14) are each calculated multiple times!
```

### 8.2 Shared Indicator Architecture

The solution is to use the scoped container architecture intelligently by separating stateless calculations (indicators) from stateful components (portfolios):

```python
class SharedIndicatorArchitecture:
    """Optimized architecture for strategies with overlapping indicators"""
    
    def __init__(self, parameter_space: List[Dict[str, Any]]):
        self.parameter_space = parameter_space
        self.indicator_container = None
        self.strategy_containers = []
        
    def setup_containers(self):
        """Set up shared indicator container and strategy containers"""
        
        # Step 1: Identify all unique indicators needed
        unique_indicators = self._extract_unique_indicators(self.parameter_space)
        
        # Step 2: Create shared indicator container
        self.indicator_container = UniversalScopedContainer("shared_indicators")
        self.indicator_container.create_component({
            'name': 'indicator_hub',
            'class': 'IndicatorHub',
            'params': {'indicators': unique_indicators},
            'capabilities': ['lifecycle', 'events']
        })
        
        # Step 3: Create strategy containers with isolated state
        for i, params in enumerate(self.parameter_space):
            container = UniversalScopedContainer(f"strategy_{i}")
            
            # Isolated stateful components
            container.create_component({
                'name': 'portfolio',
                'class': 'Portfolio',
                'params': {'initial_cash': 100000},
                'capabilities': ['lifecycle', 'events', 'reset']
            })
            
            container.create_component({
                'name': 'risk_manager',
                'class': 'RiskManager',
                'params': {'max_position_size': 1000},
                'capabilities': ['lifecycle', 'events']
            })
            
            container.create_component({
                'name': 'strategy',
                'class': params['strategy_class'],
                'params': params,
                'capabilities': ['lifecycle', 'events', 'optimization']
            })
            
            # Link to shared indicators (read-only)
            container.register_shared_service(
                'indicator_hub', 
                self.indicator_container.resolve('indicator_hub')
            )
            
            self.strategy_containers.append(container)
    
    def _extract_unique_indicators(self, param_space):
        """Extract unique indicators from parameter space"""
        indicators = set()
        
        for params in param_space:
            if 'ma_fast' in params:
                indicators.add(('SMA', params['ma_fast']))
            if 'ma_slow' in params:
                indicators.add(('SMA', params['ma_slow']))
            if 'rsi_period' in params:
                indicators.add(('RSI', params['rsi_period']))
            # ... extract other indicators
            
        return list(indicators)
```

### 8.3 Indicator Hub Implementation

```python
class IndicatorHub:
    """Centralized indicator calculation hub"""
    
    def __init__(self, indicators: List[Tuple[str, int]]):
        self.indicators = {}
        self.latest_values = {}
        
        # Initialize all required indicators
        for ind_type, period in indicators:
            key = f"{ind_type.lower()}_{period}"
            if ind_type == 'SMA':
                self.indicators[key] = SimpleMovingAverage(period)
            elif ind_type == 'RSI':
                self.indicators[key] = RSI(period)
            elif ind_type == 'EMA':
                self.indicators[key] = ExponentialMovingAverage(period)
            # ... other indicator types
    
    def initialize(self, context: SystemContext):
        """Initialize with event subscriptions"""
        self.event_bus = context.event_bus
        self.event_bus.subscribe(EventType.BAR, self.on_bar)
    
    def on_bar(self, event: Event):
        """Calculate all indicators on new bar"""
        bar_data = event.payload
        
        # Calculate all indicators once
        for key, indicator in self.indicators.items():
            value = indicator.calculate(bar_data['close'], bar_data['timestamp'])
            self.latest_values[key] = value
        
        # Publish indicator snapshot
        self.event_bus.publish(Event(
            EventType.INDICATOR_UPDATE,
            {
                'timestamp': bar_data['timestamp'],
                'values': self.latest_values.copy()
            }
        ))
    
    def get_value(self, indicator_key: str) -> Optional[float]:
        """Get latest value for an indicator"""
        return self.latest_values.get(indicator_key)
    
    def get_indicator(self, indicator_key: str) -> Optional[Any]:
        """Get indicator instance for direct access"""
        return self.indicators.get(indicator_key)
```

### 8.4 Strategy Using Shared Indicators

```python
class OptimizedTrendStrategy:
    """Strategy that uses shared indicators"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.last_signal = None
        
    def initialize(self, context: SystemContext):
        """Initialize with shared indicator access"""
        self.indicator_hub = context.resolve('indicator_hub')
        self.portfolio = context.resolve('portfolio')
        self.event_bus = context.event_bus
        
        # Subscribe to indicator updates
        self.event_bus.subscribe(EventType.INDICATOR_UPDATE, self.on_indicators)
    
    def on_indicators(self, event: Event):
        """Process indicator updates"""
        indicator_values = event.payload['values']
        
        # Get our specific indicators
        fast_key = f"sma_{self.fast_period}"
        slow_key = f"sma_{self.slow_period}"
        
        if fast_key in indicator_values and slow_key in indicator_values:
            fast_ma = indicator_values[fast_key]
            slow_ma = indicator_values[slow_key]
            
            # Generate signal based on shared indicator values
            signal = self._evaluate_signal(fast_ma, slow_ma)
            if signal:
                self.event_bus.publish(Event(EventType.SIGNAL, signal))
```

### 8.5 Orchestration Pattern

```python
class SharedIndicatorOrchestrator:
    """Orchestrates optimization with shared indicators"""
    
    def run_optimization(self, data_stream, parameter_space):
        # Set up containers
        architecture = SharedIndicatorArchitecture(parameter_space)
        architecture.setup_containers()
        
        # Initialize all containers
        architecture.indicator_container.initialize_scope()
        for container in architecture.strategy_containers:
            container.initialize_scope()
        
        # Process event stream ONCE
        for event in data_stream:
            # Indicators process event once
            architecture.indicator_container.resolve('indicator_hub').on_bar(event)
            
            # All strategies receive indicator updates automatically
            # and process with their isolated state
        
        # Collect results
        results = []
        for i, container in enumerate(architecture.strategy_containers):
            results.append({
                'params': parameter_space[i],
                'performance': container.resolve('portfolio').get_performance_metrics(),
                'trades': container.resolve('portfolio').get_trades()
            })
        
        return results
```

### 8.6 Benefits of This Pattern

1. **Efficiency**: Each indicator calculated only once per market event
2. **Isolation**: Portfolios, risk, and state remain properly isolated
3. **Scalability**: Easily handles hundreds of parameter combinations
4. **Simplicity**: Still uses the same container architecture
5. **Flexibility**: Can mix shared and strategy-specific indicators

### 8.7 When to Use This Pattern

Use shared indicator containers when:
- Optimizing multiple variations of similar strategies
- Indicators are computationally expensive
- There's significant overlap in indicator usage
- Memory efficiency is important

Continue using fully isolated containers when:
- Strategies are fundamentally different
- Indicators have strategy-specific configurations
- Complete fault isolation is critical
- Debugging simplicity is prioritized

### 8.8 Advanced: Dynamic Indicator Management

```python
class DynamicIndicatorHub:
    """Dynamically manages indicators based on active strategies"""
    
    def __init__(self):
        self.indicator_registry = {}
        self.usage_count = defaultdict(int)
        self.active_indicators = {}
    
    def request_indicator(self, indicator_type: str, params: Dict) -> str:
        """Request an indicator, create if needed"""
        indicator_key = f"{indicator_type}_{params.get('period', '')}"
        
        self.usage_count[indicator_key] += 1
        
        if indicator_key not in self.active_indicators:
            # Create indicator on first request
            self.active_indicators[indicator_key] = self._create_indicator(
                indicator_type, params
            )
        
        return indicator_key
    
    def release_indicator(self, indicator_key: str):
        """Release indicator reference, destroy if unused"""
        self.usage_count[indicator_key] -= 1
        
        if self.usage_count[indicator_key] == 0:
            # No strategies using this indicator
            del self.active_indicators[indicator_key]
            del self.usage_count[indicator_key]
```

This pattern represents a powerful optimization within the scoped container architecture, demonstrating how the same architectural principles can be applied intelligently to eliminate redundant computation while maintaining proper isolation where it matters.

---

## 9. Advanced Indicator Patterns

Building on the Shared Indicator Container pattern, these advanced patterns provide even more sophisticated optimization techniques for large-scale strategy optimization.

### 9.1 Hierarchical Indicator Containers

For complex strategies using composed indicators, a hierarchical organization prevents redundant calculations and makes dependencies explicit:

```python
class HierarchicalIndicatorHub:
    """Manages indicators in a hierarchical dependency structure"""
    
    def __init__(self):
        self.levels = {
            'base': UniversalScopedContainer("base_indicators"),
            'derived': UniversalScopedContainer("derived_indicators"), 
            'complex': UniversalScopedContainer("complex_indicators")
        }
        self.dependency_graph = {}
        self.calculation_order = []
    
    def setup_hierarchy(self, indicator_specs: List[Dict]):
        """Set up indicator hierarchy based on dependencies"""
        # Create base indicators (no dependencies)
        for spec in indicator_specs:
            if spec['level'] == 'base':
                self.levels['base'].create_component({
                    'name': spec['name'],
                    'class': spec['class'],
                    'params': spec['params'],
                    'capabilities': ['calculate']
                })
        
        # Create derived indicators (depend on base)
        for spec in indicator_specs:
            if spec['level'] == 'derived':
                self.levels['derived'].create_component({
                    'name': spec['name'],
                    'class': spec['class'],
                    'params': spec['params'],
                    'dependencies': spec['dependencies'],
                    'capabilities': ['calculate']
                })
                # Register access to base level
                self.levels['derived'].register_shared_service(
                    'base_indicators',
                    self.levels['base']
                )
        
        # Create complex indicators (depend on derived)
        for spec in indicator_specs:
            if spec['level'] == 'complex':
                self.levels['complex'].create_component({
                    'name': spec['name'],
                    'class': spec['class'],
                    'params': spec['params'],
                    'dependencies': spec['dependencies'],
                    'capabilities': ['calculate']
                })
                # Register access to lower levels
                self.levels['complex'].register_shared_service(
                    'derived_indicators',
                    self.levels['derived']
                )
    
    def on_bar(self, bar_data: Dict):
        """Process bar through hierarchy"""
        # Level 1: Base indicators (price, volume transformations)
        self.levels['base'].publish('price', bar_data['close'])
        self.levels['base'].publish('volume', bar_data['volume'])
        self.levels['base'].publish('high', bar_data['high'])
        self.levels['base'].publish('low', bar_data['low'])
        
        # Calculate all base indicators
        for name, component in self.levels['base'].local_components.items():
            if hasattr(component, 'calculate'):
                value = component.calculate(bar_data)
                self.levels['base'].publish(f"{name}_value", value)
        
        # Level 2: Derived indicators (MA, momentum from price)
        base_data = self._collect_level_data('base')
        for name, component in self.levels['derived'].local_components.items():
            if hasattr(component, 'calculate'):
                # Component can access base indicators through shared service
                value = component.calculate(base_data)
                self.levels['derived'].publish(f"{name}_value", value)
        
        # Level 3: Complex indicators (MACD from MAs, composite signals)
        derived_data = self._collect_level_data('derived')
        for name, component in self.levels['complex'].local_components.items():
            if hasattr(component, 'calculate'):
                value = component.calculate(derived_data)
                self.levels['complex'].publish(f"{name}_value", value)
        
        # Publish complete indicator snapshot
        self._publish_indicator_snapshot(bar_data['timestamp'])
    
    def _collect_level_data(self, level: str) -> Dict:
        """Collect all calculated values from a level"""
        container = self.levels[level]
        data = {}
        for key in container.local_components:
            value_key = f"{key}_value"
            if container.event_bus.has_data(value_key):
                data[key] = container.event_bus.get_data(value_key)
        return data
```

**Benefits**:
- Clear dependency management
- Natural calculation order (no circular dependencies)
- Easy to debug and visualize data flow
- Efficient caching at each level
- Reusable intermediate calculations

### 9.2 Copy-on-Write for Historical Windows

When strategies need different historical window sizes, COW prevents memory duplication:

```python
class COWIndicatorWindow:
    """Copy-on-Write window for efficient memory usage"""
    
    def __init__(self, shared_data: np.ndarray, window_size: int):
        self.shared_data = shared_data  # Immutable reference
        self.window_size = window_size
        self.local_modifications = {}
        self.is_modified = False
        self._view_cache = {}
    
    def get_window(self, end_idx: int) -> np.ndarray:
        """Get window ending at end_idx"""
        start_idx = max(0, end_idx - self.window_size)
        
        # Check cache first
        cache_key = (start_idx, end_idx)
        if cache_key in self._view_cache and not self.is_modified:
            return self._view_cache[cache_key]
        
        if not self.is_modified:
            # Return view of shared data (zero-copy)
            view = self.shared_data[start_idx:end_idx]
            self._view_cache[cache_key] = view
            return view
        else:
            # Build merged view with modifications
            result = np.empty(end_idx - start_idx)
            for i, idx in enumerate(range(start_idx, end_idx)):
                if idx in self.local_modifications:
                    result[i] = self.local_modifications[idx]
                else:
                    result[i] = self.shared_data[idx]
            return result
    
    def modify(self, index: int, value: float):
        """Modify a value (triggers COW)"""
        if not self.is_modified:
            self.is_modified = True
            self._view_cache.clear()  # Invalidate cache
        self.local_modifications[index] = value
    
    def append(self, value: float):
        """Append new value (shared operation)"""
        # This would be handled by the shared data manager
        # to update all windows atomically
        pass

class COWDataManager:
    """Manages shared data with COW windows"""
    
    def __init__(self, max_history: int = 10000):
        self.shared_buffer = np.zeros(max_history)
        self.current_index = 0
        self.windows: Dict[str, COWIndicatorWindow] = {}
        self._lock = threading.RLock()
    
    def create_window(self, window_id: str, size: int) -> COWIndicatorWindow:
        """Create a COW window for a strategy"""
        with self._lock:
            window = COWIndicatorWindow(self.shared_buffer, size)
            self.windows[window_id] = window
            return window
    
    def append_bar(self, bar_data: Dict):
        """Append new bar to shared buffer"""
        with self._lock:
            if self.current_index >= len(self.shared_buffer):
                # Ring buffer behavior or resize
                self._resize_buffer()
            
            self.shared_buffer[self.current_index] = bar_data['close']
            self.current_index += 1
            
            # Notify all windows of new data
            for window in self.windows.values():
                window.append(bar_data['close'])
```

**Benefits**:
- Massive memory savings for read-only access
- Strategies can have different window sizes without duplication
- Modifications are isolated per strategy
- Zero-copy views for unmodified data
- Efficient cache management

### 9.3 Lazy Indicator Evaluation

Only calculate indicators that have active subscribers:

```python
class LazyIndicatorHub:
    """Calculates indicators only when requested"""
    
    def __init__(self):
        self.indicator_factories = {}
        self.active_indicators = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.calculated_this_bar = set()
        self.current_bar = None
        self.calculation_count = defaultdict(int)
    
    def register_indicator_type(self, indicator_type: str, factory: Callable):
        """Register factory for creating indicators"""
        self.indicator_factories[indicator_type] = factory
    
    def subscribe(self, strategy_id: str, indicator_key: str):
        """Strategy subscribes to an indicator"""
        self.subscriptions[indicator_key].add(strategy_id)
        
        # Lazy creation - indicator created on first subscription
        if indicator_key not in self.active_indicators:
            self._create_indicator(indicator_key)
    
    def unsubscribe(self, strategy_id: str, indicator_key: str):
        """Strategy unsubscribes from an indicator"""
        self.subscriptions[indicator_key].discard(strategy_id)
        
        # Destroy indicator if no subscribers
        if not self.subscriptions[indicator_key]:
            self._destroy_indicator(indicator_key)
    
    def on_bar(self, bar: Dict):
        """Process new bar - but don't calculate anything yet!"""
        self.current_bar = bar
        self.calculated_this_bar.clear()
        
        # No calculations here - completely lazy
    
    def get_value(self, strategy_id: str, indicator_key: str) -> Optional[float]:
        """Get indicator value - calculates if needed"""
        if strategy_id not in self.subscriptions[indicator_key]:
            return None  # Not subscribed
        
        if indicator_key not in self.calculated_this_bar:
            # First request this bar - calculate now
            self._calculate_indicator(indicator_key)
            self.calculated_this_bar.add(indicator_key)
            self.calculation_count[indicator_key] += 1
        
        return self.active_indicators[indicator_key].value
    
    def _calculate_indicator(self, indicator_key: str):
        """Calculate indicator on current bar"""
        if indicator_key not in self.active_indicators:
            return
        
        indicator = self.active_indicators[indicator_key]
        indicator.update(self.current_bar)
    
    def _create_indicator(self, indicator_key: str):
        """Create indicator instance"""
        # Parse indicator key (e.g., "sma_20" -> type="sma", period=20)
        parts = indicator_key.split('_')
        ind_type = parts[0]
        params = {'period': int(parts[1])} if len(parts) > 1 else {}
        
        if ind_type in self.indicator_factories:
            indicator = self.indicator_factories[ind_type](**params)
            self.active_indicators[indicator_key] = indicator
    
    def _destroy_indicator(self, indicator_key: str):
        """Destroy unused indicator"""
        if indicator_key in self.active_indicators:
            del self.active_indicators[indicator_key]
            del self.subscriptions[indicator_key]
    
    def get_statistics(self) -> Dict:
        """Get lazy evaluation statistics"""
        total_possible = len(self.indicator_factories) * len(self.calculated_this_bar)
        total_calculated = sum(self.calculation_count.values())
        
        return {
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'active_indicators': len(self.active_indicators),
            'calculations_saved': total_possible - total_calculated,
            'efficiency_ratio': 1 - (total_calculated / max(total_possible, 1)),
            'calculation_counts': dict(self.calculation_count)
        }
```

**Benefits**:
- Only compute what's actually used
- Dynamic subscription management  
- Automatic cleanup of unused indicators
- Huge savings in optimization where many strategies are pruned early
- Perfect for phased execution where analysis phase might not need all indicators

### 9.4 Combining All Patterns

These patterns work synergistically to create an extremely efficient optimization system:

```python
class AdvancedIndicatorArchitecture:
    """Combines all advanced indicator patterns"""
    
    def __init__(self):
        # Hierarchical organization for complex indicators
        self.hierarchy = HierarchicalIndicatorHub()
        
        # COW for efficient historical data management
        self.data_manager = COWDataManager()
        
        # Lazy evaluation for on-demand calculation
        self.lazy_hub = LazyIndicatorHub()
        
        # Adaptive caching for frequently accessed values
        self.cache = AdaptiveCache(max_memory_mb=512)
    
    def setup_for_optimization(self, strategy_configs: List[Dict]):
        """Set up optimized indicator system"""
        
        # 1. Analyze indicator dependencies and build hierarchy
        indicator_specs = self._analyze_indicator_requirements(strategy_configs)
        self.hierarchy.setup_hierarchy(indicator_specs)
        
        # 2. Create COW windows for each strategy
        for i, config in enumerate(strategy_configs):
            window_size = config.get('lookback_period', 100)
            window = self.data_manager.create_window(f"strategy_{i}", window_size)
            config['data_window'] = window
        
        # 3. Set up lazy evaluation subscriptions
        for i, config in enumerate(strategy_configs):
            strategy_id = f"strategy_{i}"
            for indicator in config.get('indicators', []):
                self.lazy_hub.subscribe(strategy_id, indicator)
    
    def on_market_data(self, bar: Dict):
        """Process new market data through all systems"""
        
        # 1. Update shared data (COW)
        self.data_manager.append_bar(bar)
        
        # 2. Process through hierarchy (only if needed)
        if self._has_active_hierarchical_subscribers():
            self.hierarchy.on_bar(bar)
        
        # 3. Update lazy hub (doesn't calculate yet)
        self.lazy_hub.on_bar(bar)
        
        # 4. Clear cache for outdated values
        self.cache.invalidate_by_timestamp(bar['timestamp'])
    
    def get_indicator_value(self, strategy_id: str, indicator_key: str) -> Optional[float]:
        """Get indicator value with full optimization stack"""
        
        # 1. Check cache first
        cache_key = f"{strategy_id}:{indicator_key}"
        if cached := self.cache.get(cache_key):
            return cached
        
        # 2. Try lazy evaluation
        if value := self.lazy_hub.get_value(strategy_id, indicator_key):
            self.cache.put(cache_key, value)
            return value
        
        # 3. Check hierarchical indicators
        if value := self.hierarchy.get_value(indicator_key):
            self.cache.put(cache_key, value)
            return value
        
        return None
    
    def cleanup_strategy(self, strategy_id: str):
        """Clean up resources when strategy is eliminated"""
        # Unsubscribe from all indicators
        for indicator_key in list(self.lazy_hub.subscriptions.keys()):
            self.lazy_hub.unsubscribe(strategy_id, indicator_key)
        
        # Remove COW window
        self.data_manager.remove_window(strategy_id)
        
        # Clear cache entries
        self.cache.invalidate_by_prefix(f"{strategy_id}:")
```

### 9.5 Performance Impact

With these advanced patterns, optimization performance improves dramatically:

```python
# Traditional approach (1000 strategies, 20 indicators each)
# Calculations per bar: 1000 * 20 = 20,000

# With shared indicators (Section 8)
# Calculations per bar: ~50 unique indicators

# With lazy evaluation (many strategies eliminated early)
# Calculations per bar: ~10-20 active indicators

# With hierarchical organization (reusing intermediate values)
# Calculations per bar: ~5-10 base + derived calculations

# With COW (memory usage)
# Traditional: 1000 strategies * 1000 bars * 8 bytes = 8MB per indicator
# With COW: 1000 bars * 8 bytes = 8KB shared + minimal overhead
```

### 9.6 When to Use Each Pattern

**Hierarchical Indicators**:
- Complex indicators with clear dependencies
- Strategies using indicator combinations
- Need to debug/visualize calculation flow

**Copy-on-Write Windows**:
- Many strategies with different lookback periods
- Memory-constrained environments
- Most strategies only read historical data

**Lazy Evaluation**:
- Large parameter spaces with early pruning
- Phased optimization with analysis steps
- Strategies use different indicator subsets

**Combined Approach**:
- Large-scale optimization (1000+ parameter combinations)
- Complex strategies with many indicators
- Production systems requiring maximum efficiency

These advanced patterns demonstrate how the scoped container architecture can be pushed to achieve extreme optimization while maintaining clean separation of concerns and architectural integrity.

### 9.7 Optimal Container Organization: Split by Calculation Pattern

While the previous examples showed splitting indicators by type (RSI, MA, etc.), a more sophisticated approach splits by computational characteristics:

```python
class CalculationPatternIndicatorArchitecture:
    """Split indicators by how they compute, not what they compute"""
    
    def __init__(self):
        self.containers = {
            'stateless': UniversalScopedContainer("stateless_indicators"),
            'rolling': UniversalScopedContainer("rolling_indicators"),
            'cumulative': UniversalScopedContainer("cumulative_indicators"),
            'recursive': UniversalScopedContainer("recursive_indicators")
        }
        self._setup_specialized_infrastructure()
    
    def _setup_specialized_infrastructure(self):
        """Set up optimized infrastructure for each pattern"""
        
        # Stateless: Pure functions, no history needed
        self.containers['stateless'].create_component({
            'name': 'calculator',
            'class': 'StatelessCalculator',
            'params': {
                'cache_results': False,  # No point caching
                'parallel_safe': True    # Can parallelize freely
            }
        })
        
        # Rolling: Fixed-size windows
        self.containers['rolling'].create_component({
            'name': 'window_manager',
            'class': 'RingBufferManager',
            'params': {
                'default_size': 200,
                'overflow_strategy': 'rotate'
            }
        })
        
        # Cumulative: Incremental updates
        self.containers['cumulative'].create_component({
            'name': 'accumulator',
            'class': 'IncrementalAccumulator',
            'params': {
                'checkpoint_interval': 1000,
                'compression': True
            }
        })
        
        # Recursive: Depends on previous values
        self.containers['recursive'].create_component({
            'name': 'state_manager',
            'class': 'RecursiveStateManager',
            'params': {
                'state_history': 2,  # How many previous states to keep
                'warm_up_periods': 50
            }
        })
```

#### Indicator Classification

```python
# Stateless Indicators (current bar only)
STATELESS_INDICATORS = {
    'price_transform': lambda bar: (bar.high + bar.low + bar.close) / 3,
    'true_range': lambda bar, prev_close: max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close)
    ),
    'pivot_points': lambda bar: {
        'pivot': (bar.high + bar.low + bar.close) / 3,
        'r1': (2 * pivot) - bar.low,
        's1': (2 * pivot) - bar.high
    }
}

# Rolling Window Indicators (fixed lookback)
ROLLING_INDICATORS = {
    'sma': {'window': 'fixed', 'computation': 'mean'},
    'std_dev': {'window': 'fixed', 'computation': 'std'},
    'rsi': {'window': 'fixed', 'computation': 'momentum'},
    'bollinger': {'window': 'fixed', 'computation': 'statistical'},
    'min_max': {'window': 'fixed', 'computation': 'extremes'}
}

# Cumulative Indicators (all history)
CUMULATIVE_INDICATORS = {
    'vwap': {'update': 'incremental', 'reset': 'daily'},
    'cumulative_volume': {'update': 'additive', 'reset': 'never'},
    'realized_variance': {'update': 'incremental', 'reset': 'periodic'},
    'trade_count': {'update': 'counter', 'reset': 'session'}
}

# Recursive Indicators (depend on own previous values)
RECURSIVE_INDICATORS = {
    'ema': {'dependency': 'previous_value', 'decay': 'exponential'},
    'kama': {'dependency': 'adaptive', 'efficiency_ratio': True},
    'vidya': {'dependency': 'volatility_adjusted'},
    'mesa': {'dependency': 'phase_adjusted'}
}
```

#### Optimized Processing for Each Pattern

```python
class StatelessProcessor:
    """Optimized for stateless calculations"""
    
    def process_batch(self, bars: List[Bar]) -> Dict[str, List[float]]:
        """Process multiple bars in parallel"""
        with ThreadPoolExecutor() as executor:
            # Can parallelize since no dependencies
            futures = []
            for indicator_name, func in self.indicators.items():
                future = executor.submit(self._calculate_all, func, bars)
                futures.append((indicator_name, future))
            
            results = {}
            for name, future in futures:
                results[name] = future.result()
            return results

class RollingWindowProcessor:
    """Optimized for rolling window calculations"""
    
    def __init__(self, max_window: int = 200):
        # Pre-allocate ring buffers
        self.buffers = {}
        self.max_window = max_window
    
    def add_bar(self, bar: Bar):
        """Efficiently update all rolling windows"""
        # Single pass to update all buffers
        for buffer in self.buffers.values():
            buffer.append(bar.close)  # O(1) with ring buffer
        
        # Calculate all indicators with their specific windows
        results = {}
        for name, indicator in self.indicators.items():
            window_size = indicator.period
            data = buffer.get_window(window_size)  # O(1) view
            results[name] = indicator.calculate(data)
        
        return results

class CumulativeProcessor:
    """Optimized for cumulative calculations"""
    
    def __init__(self):
        self.accumulators = {}
        self.checkpoint_counter = 0
    
    def add_bar(self, bar: Bar):
        """Incrementally update cumulative values"""
        results = {}
        
        for name, accumulator in self.accumulators.items():
            # O(1) incremental update instead of O(n) recalculation
            accumulator.update(bar)
            results[name] = accumulator.value
        
        # Periodic checkpointing for recovery
        self.checkpoint_counter += 1
        if self.checkpoint_counter % 1000 == 0:
            self._save_checkpoint()
        
        return results

class RecursiveProcessor:
    """Optimized for recursive calculations"""
    
    def __init__(self):
        self.state_history = {}  # Track previous states
        self.warm_up_counter = {}
    
    def add_bar(self, bar: Bar):
        """Update recursive indicators with state management"""
        results = {}
        
        for name, indicator in self.indicators.items():
            # Get previous state
            prev_state = self.state_history.get(name, indicator.initial_state())
            
            # Calculate new value based on previous
            new_value, new_state = indicator.update(bar, prev_state)
            
            # Store state for next iteration
            self.state_history[name] = new_state
            
            # Only return value after warm-up
            if self.warm_up_counter[name] >= indicator.warm_up_periods:
                results[name] = new_value
            else:
                self.warm_up_counter[name] += 1
                
        return results
```

#### When to Use This Pattern

**Use calculation-based splitting when:**
- You have many indicators with different computational patterns
- Performance is critical and you need specialized optimizations
- You want to scale different calculation types independently
- You need different failure recovery strategies per type

**Benefits over type-based splitting:**
- Better CPU cache utilization (similar calculations together)
- Can optimize data structures per calculation pattern
- Easier to parallelize (stateless can use all cores)
- More efficient memory usage (rolling windows can share buffers)

**Example optimization gains:**
```python
# Traditional: Every indicator maintains its own data
# 100 SMAs with different periods = 100 separate arrays
memory_traditional = 100 * 200 * 8  # 160KB per bar

# Calculation-pattern based: Shared ring buffer
# 1 ring buffer + 100 pointers to different windows
memory_optimized = 1 * 200 * 8 + 100 * 16  # 3.2KB per bar

# 50x memory reduction!
```

This pattern represents the evolution from naive splitting (by type) to intelligent splitting (by computational characteristics), maximizing the benefits of the container architecture while minimizing resource usage.

### 9.8 Advanced Memory Optimization Patterns

Building on the COW pattern, here are additional memory-efficient implementations:

```python
class MemoryEfficientIndicator:
    """Base class for memory-efficient indicators using circular buffers"""
    
    def __init__(self, period: int, dtype: np.dtype = np.float32):
        self.period = period
        self.dtype = dtype
        self.circular_buffer = np.zeros(period, dtype=dtype)
        self.buffer_idx = 0
        self.is_ready = False
        self.count = 0
        
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value"""
        self.circular_buffer[self.buffer_idx] = value
        self.buffer_idx = (self.buffer_idx + 1) % self.period
        self.count += 1
        
        if self.count >= self.period:
            self.is_ready = True
            
        return self.calculate() if self.is_ready else None
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        return self.circular_buffer.nbytes

class MemoryEfficientSMA(MemoryEfficientIndicator):
    """SMA with O(1) updates using running sum"""
    
    def __init__(self, period: int):
        super().__init__(period)
        self.sum = 0.0
        
    def update(self, value: float) -> Optional[float]:
        """Update with running sum optimization"""
        if self.is_ready:
            # Remove oldest value from sum
            oldest_idx = self.buffer_idx
            self.sum -= self.circular_buffer[oldest_idx]
            
        self.circular_buffer[self.buffer_idx] = value
        self.sum += value
        
        self.buffer_idx = (self.buffer_idx + 1) % self.period
        self.count += 1
        
        if self.count >= self.period:
            self.is_ready = True
            
        return self.calculate() if self.is_ready else None
    
    def calculate(self) -> float:
        """O(1) calculation using running sum"""
        return self.sum / self.period
```

### 9.9 Event Synchronization Patterns

For ensuring all strategies process events in lockstep:

```python
class BarrierSynchronizedOrchestrator:
    """Ensures all strategies complete processing before next event"""
    
    def __init__(self, indicator_hub: Any, 
                 strategy_containers: List[UniversalScopedContainer]):
        self.indicator_hub = indicator_hub
        self.strategy_containers = strategy_containers
        self.processing_barrier = threading.Barrier(len(strategy_containers) + 1)
        self._setup_strategy_callbacks()
        
    def _setup_strategy_callbacks(self):
        """Inject barrier synchronization into strategies"""
        for container in self.strategy_containers:
            strategy = container.resolve('strategy')
            original_on_indicators = strategy.on_indicators
            
            def wrapped_on_indicators(event):
                result = original_on_indicators(event)
                self.processing_barrier.wait()  # Signal completion
                return result
                
            strategy.on_indicators = wrapped_on_indicators
    
    def process_market_event_stream(self, event_stream):
        """Process events with guaranteed synchronization"""
        for market_event in event_stream:
            # Indicator hub processes and distributes
            self.indicator_hub.on_bar(market_event)
            
            # Wait for all strategies to complete
            self.processing_barrier.wait()
            
            # All strategies have processed - safe to continue
```

### 9.10 Performance Measurement Utilities

Utilities to quantify the benefits of shared indicator architecture:

```python
def calculate_memory_savings(n_strategies: int, n_indicators: int, 
                           lookback_period: int) -> Dict[str, Any]:
    """Calculate memory savings from shared indicator approach"""
    
    # Traditional: each strategy stores all indicators
    traditional_memory = (
        n_strategies * n_indicators * lookback_period * 8  # 8 bytes per float64
    )
    
    # Shared: single storage + strategy-specific state
    shared_indicator_memory = n_indicators * lookback_period * 8
    strategy_state_memory = n_strategies * 1024  # ~1KB per strategy state
    shared_memory = shared_indicator_memory + strategy_state_memory
    
    savings = traditional_memory - shared_memory
    savings_percent = (savings / traditional_memory) * 100
    
    return {
        'traditional_memory_mb': traditional_memory / (1024 * 1024),
        'shared_memory_mb': shared_memory / (1024 * 1024),
        'savings_mb': savings / (1024 * 1024),
        'savings_percent': savings_percent,
        'break_even_strategies': shared_indicator_memory / (
            n_indicators * lookback_period * 8 - 1024
        )
    }

def measure_computation_savings(n_strategies: int, n_indicators: int,
                              indicator_compute_time_ms: float = 0.1) -> Dict[str, Any]:
    """Calculate computation time savings"""
    
    # Traditional: each strategy computes all indicators
    traditional_time = n_strategies * n_indicators * indicator_compute_time_ms
    
    # Shared: compute once
    shared_time = n_indicators * indicator_compute_time_ms
    
    savings = traditional_time - shared_time
    speedup = traditional_time / shared_time if shared_time > 0 else float('inf')
    
    return {
        'traditional_time_ms': traditional_time,
        'shared_time_ms': shared_time,
        'savings_ms': savings,
        'speedup_factor': speedup,
        'strategies_for_2x_speedup': 2  # Always ≥2x with 2+ strategies
    }

# Example: 100 strategies, 10 indicators, 1000 lookback
# Memory: ~75MB saved (98.7% reduction)
# Computation: 100x speedup
```

### 9.11 Practical Implementation Checklist

When implementing shared indicator containers:

**Memory Optimization**:
- [ ] Use appropriate data types (float32 vs float64)
- [ ] Implement circular buffers for rolling windows
- [ ] Use COW for strategy-specific modifications
- [ ] Force garbage collection between optimization batches

**Synchronization**:
- [ ] Use sequence numbers for event ordering
- [ ] Implement timeout handling for barriers
- [ ] Handle missing indicator values gracefully
- [ ] Monitor synchronization overhead

**Performance**:
- [ ] Profile memory usage before/after optimization
- [ ] Measure computation time improvements
- [ ] Track cache hit rates for lazy evaluation
- [ ] Monitor container creation/destruction overhead

**Testing**:
- [ ] Verify indicator calculations match traditional approach
- [ ] Test with various parameter space sizes
- [ ] Validate memory savings at scale
- [ ] Ensure proper cleanup prevents memory leaks

---

## 10. Advanced Memory Optimization Patterns

### 10.1 Memory-Efficient Indicator Storage

Building on the shared indicator patterns, we can optimize memory usage even further with specialized storage patterns:

```python
class MemoryEfficientIndicator:
    """Base class for memory-efficient indicators"""
    
    def __init__(self, period: int, dtype: np.dtype = np.float32):
        self.period = period
        self.dtype = dtype
        self.circular_buffer = np.zeros(period, dtype=dtype)
        self.buffer_idx = 0
        self.is_ready = False
        self.count = 0
        
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value"""
        self.circular_buffer[self.buffer_idx] = value
        self.buffer_idx = (self.buffer_idx + 1) % self.period
        self.count += 1
        
        if self.count >= self.period:
            self.is_ready = True
            
        return self.calculate() if self.is_ready else None
    
    def calculate(self) -> float:
        """Calculate current indicator value"""
        raise NotImplementedError
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        return self.circular_buffer.nbytes

class MemoryEfficientSMA(MemoryEfficientIndicator):
    """Memory-efficient Simple Moving Average"""
    
    def __init__(self, period: int):
        super().__init__(period)
        self.sum = 0.0
        
    def update(self, value: float) -> Optional[float]:
        """Update with running sum optimization"""
        if self.is_ready:
            # Remove oldest value from sum
            oldest_idx = self.buffer_idx
            self.sum -= self.circular_buffer[oldest_idx]
            
        self.circular_buffer[self.buffer_idx] = value
        self.sum += value
        
        self.buffer_idx = (self.buffer_idx + 1) % self.period
        self.count += 1
        
        if self.count >= self.period:
            self.is_ready = True
            
        return self.calculate() if self.is_ready else None
    
    def calculate(self) -> float:
        """O(1) calculation using running sum"""
        return self.sum / self.period
```

### 10.2 Event Synchronization Patterns

Ensuring proper synchronization across containers is critical for accurate results:

```python
class BarrierSynchronizedOrchestrator:
    """Ensures all strategies complete processing before next event"""
    
    def __init__(self, indicator_hub: SynchronizedIndicatorHub, 
                 strategy_containers: List[UniversalScopedContainer]):
        self.indicator_hub = indicator_hub
        self.strategy_containers = strategy_containers
        self.processing_barrier = threading.Barrier(len(strategy_containers) + 1)
        self._setup_strategy_callbacks()
        
    def _setup_strategy_callbacks(self):
        """Set up processing completion callbacks"""
        for container in self.strategy_containers:
            strategy = container.resolve('strategy')
            # Inject barrier wait into strategy processing
            original_on_indicators = strategy.on_indicators
            
            def wrapped_on_indicators(event):
                result = original_on_indicators(event)
                self.processing_barrier.wait()  # Signal completion
                return result
                
            strategy.on_indicators = wrapped_on_indicators
    
    def process_market_event_stream(self, event_stream):
        """Process events with barrier synchronization"""
        for market_event in event_stream:
            # Indicator hub processes and distributes
            self.indicator_hub.on_bar(market_event)
            
            # Wait for all strategies to complete
            self.processing_barrier.wait()
            
            # All strategies have processed - safe to continue
```

### 10.3 Performance Measurement Utilities

To quantify the benefits of these optimization patterns:

```python
def calculate_memory_savings(n_strategies: int, n_indicators: int, 
                           lookback_period: int, data_points: int) -> Dict[str, Any]:
    """Calculate memory savings from shared indicator approach"""
    
    # Traditional approach: each strategy stores all indicators
    traditional_memory = (
        n_strategies * n_indicators * lookback_period * 8  # 8 bytes per float64
    )
    
    # Shared approach: single storage + strategy-specific state
    shared_indicator_memory = n_indicators * lookback_period * 8
    strategy_state_memory = n_strategies * 1024  # Assume 1KB per strategy state
    shared_memory = shared_indicator_memory + strategy_state_memory
    
    savings = traditional_memory - shared_memory
    savings_percent = (savings / traditional_memory) * 100
    
    return {
        'traditional_memory_mb': traditional_memory / (1024 * 1024),
        'shared_memory_mb': shared_memory / (1024 * 1024),
        'savings_mb': savings / (1024 * 1024),
        'savings_percent': savings_percent,
        'break_even_strategies': shared_indicator_memory / (
            n_indicators * lookback_period * 8 - 1024
        )
    }

def measure_computation_savings(n_strategies: int, n_indicators: int,
                              indicator_compute_time_ms: float = 0.1) -> Dict[str, Any]:
    """Calculate computation time savings"""
    
    # Traditional: each strategy computes all indicators
    traditional_time = n_strategies * n_indicators * indicator_compute_time_ms
    
    # Shared: compute once
    shared_time = n_indicators * indicator_compute_time_ms
    
    savings = traditional_time - shared_time
    speedup = traditional_time / shared_time if shared_time > 0 else float('inf')
    
    return {
        'traditional_time_ms': traditional_time,
        'shared_time_ms': shared_time,
        'savings_ms': savings,
        'speedup_factor': speedup,
        'strategies_for_2x_speedup': 2  # Always 2x with 2+ strategies
    }
```

---

## Summary

The Universal Scoped Container Architecture with Protocol + Composition provides:

1. **Complete Isolation**: Every execution context is isolated
2. **Maximum Flexibility**: Any component can be used in any container
3. **Resource Efficiency**: Shared read-only, isolated mutable state
4. **Universal Pattern**: Same architecture for all use cases
5. **Production Ready**: Built-in monitoring, error handling, and resource management
6. **Optimization Patterns**: Smart container organization for efficiency
7. **Advanced Memory Management**: COW, circular buffers, and efficient storage
8. **Synchronized Processing**: Barrier synchronization for consistent results

Key Performance Benefits:
- **Memory Reduction**: 90%+ savings with shared indicators and COW
- **Computation Speedup**: Near-linear with number of strategies
- **Scalability**: Handles thousands of parameter combinations efficiently

This architecture enables ADMF-Trader to scale from simple backtests to complex production deployments while maintaining clean separation of concerns and optimal resource utilization.
